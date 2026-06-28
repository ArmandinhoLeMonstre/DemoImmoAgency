"""Outil d'enregistrement d'un VENDEUR dans Odoo, en remplissage progressif.

  1. crée ou met à jour le contact (res.partner)  -> dédoublonnage email/téléphone
  2. crée un lead minimal (crm.lead) dans le pipeline "Mandats" dès le nom + un canal
  3. enrichit le lead au fil de l'appel (update_lead) : priorité et description
     entièrement recalculées à chaque passage.

Le schéma Pydantic `SellerLead` décrit tout ce qu'on peut capturer sur un vendeur.
Le câblage dans l'agent (handlers Pipecat Flows) se fera à l'étape 2 ; ici le
client est testable en isolation via `python odoo_seller_tool.py`.

Connexion via XML-RPC (le protocole standard d'Odoo, déjà utilisé côté agent).

Config via un fichier .env (chargé par python-dotenv) :
    ODOO_URL            ex: https://immoai.odoo.com
    ODOO_DB             ex: immoai
    ODOO_USER           ex: api@immoai.be
    ODOO_PASSWORD       (clé API Odoo de préférence, pas le mot de passe)
    ODOO_MANDATS_TEAM   (optionnel, défaut: "Mandats")
"""

from __future__ import annotations

import logging
import os
import xmlrpc.client
from datetime import date, timedelta
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator

try:
    from dotenv import load_dotenv

    load_dotenv()  # charge automatiquement un fichier .env
except ImportError:
    pass  # dotenv optionnel : si absent, on lit l'env système

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("odoo_seller")


# --------------------------------------------------------------------------- #
#  Schéma vendeur  (= schéma d'arguments pour l'agent IA)                      #
# --------------------------------------------------------------------------- #


class Timeframe(str, Enum):
    """Délai de vente souhaité par le vendeur."""

    asap = "asap"  # le plus vite possible
    one_month = "1_month"
    three_months = "3_months"
    six_months_plus = "6_months_plus"
    unsure = "unsure"


class PropertyType(str, Enum):
    """Type de bien immobilier."""

    appartement = "appartement"
    maison = "maison"
    terrain = "terrain"
    commerce = "commerce"
    autre = "autre"


class SellerLead(BaseModel):
    """Tout ce que l'agent doit capturer sur un vendeur."""

    # --- contact ---
    last_name: str = Field(..., description="Nom du vendeur")
    first_name: str = Field("", description="Prénom du vendeur")
    phone: Optional[str] = Field(None, description="Téléphone, idéalement E.164 (+32...)")
    email: Optional[str] = Field(None, description="Email du vendeur")
    language: str = Field("fr", description="Langue: fr / nl / en")
    best_callback_time: Optional[str] = Field(None, description="Meilleur moment pour rappeler")

    # --- le bien à vendre ---
    property_type: Optional[PropertyType] = None
    city: Optional[str] = Field(None, description="Commune / ville du bien")
    address: Optional[str] = Field(None, description="Rue + numéro")
    zip: Optional[str] = None
    bedrooms: Optional[int] = None
    surface_m2: Optional[float] = None
    expected_price: Optional[float] = Field(None, description="Prix espéré par le vendeur, en €")

    # --- qualifiers (pilotent le score chaud/tiède/froid) ---
    timeframe: Timeframe = Field(Timeframe.unsure, description="Délai de vente souhaité")
    reason: Optional[str] = Field(
        None, description="Raison de la vente (succession, déménagement...)"
    )
    sole_owner: Optional[bool] = Field(None, description="Seul propriétaire / seul décideur ?")
    already_mandated: Optional[bool] = Field(None, description="Bien déjà confié à une agence ?")
    peb: Optional[str] = Field(None, description="Certificat PEB (déjà disponible ?)")

    # --- trace ---
    call_summary: Optional[str] = Field(None, description="Résumé de l'appel en 2-3 phrases")

    @model_validator(mode="after")
    def _need_a_channel(self):
        if not self.phone and not self.email:
            raise ValueError("Au moins un téléphone OU un email est requis.")
        return self


# --------------------------------------------------------------------------- #
#  Config                                                                      #
# --------------------------------------------------------------------------- #


class OdooConfig(BaseModel):
    """Configuration de connexion et de mapping Odoo."""

    url: str
    db: str
    username: str
    password: str
    mandats_team: str = "Mandats"
    seller_tag: str = "Vendeur"
    commission_rate: float = 0.03  # 3 % -> sert à estimer expected_revenue
    # Mapping optionnel attribut SellerLead -> champ custom Odoo (x_...).
    # Tant que tu n'as pas créé les champs custom, laisse vide :
    # tout part dans la description, rien ne casse.
    custom_field_map: dict = {}

    @classmethod
    def from_env(cls) -> "OdooConfig":
        """Construit la config depuis les variables d'environnement (.env)."""
        required = {
            "url": os.getenv("ODOO_URL"),
            "db": os.getenv("ODOO_DB"),
            "username": os.getenv("ODOO_USER"),
            "password": os.getenv("ODOO_PASSWORD"),
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            env_names = {
                "url": "ODOO_URL",
                "db": "ODOO_DB",
                "username": "ODOO_USER",
                "password": "ODOO_PASSWORD",
            }
            raise RuntimeError(
                "Variables manquantes dans ton .env : " + ", ".join(env_names[k] for k in missing)
            )
        return cls(
            url=required["url"].rstrip("/"),
            db=required["db"],
            username=required["username"],
            password=required["password"],
            mandats_team=os.getenv("ODOO_MANDATS_TEAM", "Mandats"),
        )


# --------------------------------------------------------------------------- #
#  Client Odoo                                                                 #
# --------------------------------------------------------------------------- #


class OdooClient:
    """Client Odoo (XML-RPC) : upsert contact + création/mise à jour de leads."""

    def __init__(self, cfg: OdooConfig):
        """Ouvre la connexion XML-RPC et authentifie l'utilisateur."""
        self.cfg = cfg
        common = xmlrpc.client.ServerProxy(f"{cfg.url}/xmlrpc/2/common")
        self.uid = common.authenticate(cfg.db, cfg.username, cfg.password, {})
        if not self.uid:
            raise RuntimeError("Auth Odoo échouée — vérifie URL / DB / user / password.")
        self.models = xmlrpc.client.ServerProxy(f"{cfg.url}/xmlrpc/2/object")
        log.info("Connecté à Odoo (uid=%s)", self.uid)

    # -- helper générique execute_kw --
    def _kw(self, model: str, method: str, args: list, kwargs: dict | None = None):
        return self.models.execute_kw(
            self.cfg.db, self.uid, self.cfg.password, model, method, args, kwargs or {}
        )

    def _id_by_name(self, model: str, name: str, extra=None) -> Optional[int]:
        domain = [("name", "=", name)] + (extra or [])
        res = self._kw(model, "search", [domain], {"limit": 1})
        return res[0] if res else None

    def _get_or_create_tag(self, name: str) -> int:
        return self._id_by_name("crm.tag", name) or self._kw("crm.tag", "create", [{"name": name}])

    def _country_be(self) -> Optional[int]:
        res = self._kw("res.country", "search", [[("code", "=", "BE")]], {"limit": 1})
        return res[0] if res else None

    # ------------------------------------------------------------------ #
    #  1) Upsert contact                                                  #
    # ------------------------------------------------------------------ #
    def upsert_partner(self, s: SellerLead) -> tuple[int, bool]:
        """Retourne (partner_id, created?). Dédoublonne sur email puis téléphone."""
        domain = []
        if s.email:
            domain = [("email", "=ilike", s.email)]
        elif s.phone:
            domain = [("phone", "=", s.phone)]
        existing = self._kw("res.partner", "search", [domain], {"limit": 1}) if domain else []

        vals = {
            "name": f"{s.first_name} {s.last_name}".strip(),
            "phone": s.phone,
            "email": s.email,
            "street": s.address,
            "city": s.city,
            "zip": s.zip,
        }
        be = self._country_be()
        if be:
            vals["country_id"] = be
        vals = {k: v for k, v in vals.items() if v not in (None, "")}

        if existing:
            pid = existing[0]
            self._kw("res.partner", "write", [[pid], vals])
            log.info("Contact mis à jour: id=%s", pid)
            return pid, False

        pid = self._kw("res.partner", "create", [vals])
        log.info("Contact créé: id=%s", pid)
        return pid, True

    # ------------------------------------------------------------------ #
    #  Scoring & description                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _priority(s: SellerLead) -> str:
        """Renvoie la priorité Odoo '0'..'3' (les étoiles de la carte)."""
        score = 0
        if s.timeframe in (Timeframe.asap, Timeframe.one_month):
            score += 2
        elif s.timeframe == Timeframe.three_months:
            score += 1
        if s.sole_owner:
            score += 1
        if s.already_mandated is False:  # explicitement PAS déjà sous mandat
            score += 1
        if s.reason:
            score += 1
        if score >= 4:
            return "3"  # 🔥 chaud
        if score >= 2:
            return "2"  # tiède+
        if score >= 1:
            return "1"  # tiède
        return "0"  # froid

    @staticmethod
    def _description(s: SellerLead) -> str:
        lines = ["Lead VENDEUR — qualifié par agent IA", ""]

        def add(label, val):
            if val not in (None, ""):
                lines.append(f"- {label}: {val}")

        loc = ", ".join(x for x in [s.address, s.zip, s.city] if x)
        add("Type de bien", s.property_type.value if s.property_type else None)
        add("Localisation", loc)
        add("Chambres", s.bedrooms)
        add("Surface", f"{s.surface_m2:g} m2" if s.surface_m2 else None)
        add("Prix espéré", f"{s.expected_price:,.0f} EUR" if s.expected_price else None)
        add("Délai", s.timeframe.value)
        add("Raison", s.reason)
        add("Propriétaire unique", {True: "Oui", False: "Non"}.get(s.sole_owner))
        add("Déjà sous mandat", {True: "Oui", False: "Non"}.get(s.already_mandated))
        add("PEB", s.peb)
        add("Meilleur moment rappel", s.best_callback_time)
        add("Langue", s.language)
        if s.call_summary:
            lines += ["", "Résumé appel:", s.call_summary]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  2) Création du lead vendeur                                         #
    # ------------------------------------------------------------------ #
    def add_seller(self, s: SellerLead) -> dict:
        """Crée le contact puis le lead vendeur complet en une seule passe."""
        partner_id, partner_created = self.upsert_partner(s)

        team_id = self._id_by_name("crm.team", self.cfg.mandats_team)
        if not team_id:
            raise RuntimeError(
                f"Équipe '{self.cfg.mandats_team}' introuvable. "
                "Crée-la dans CRM -> Configuration -> Équipes commerciales, "
                "ou change ODOO_MANDATS_TEAM."
            )

        tag_id = self._get_or_create_tag(self.cfg.seller_tag)

        expected_revenue = (
            round(s.expected_price * self.cfg.commission_rate, 2) if s.expected_price else 0.0
        )

        opp_name = (
            f"Vendeur — {s.property_type.value if s.property_type else 'bien'} {s.city or ''}"
        ).strip()

        vals = {
            "name": opp_name,
            "type": "opportunity",  # pour apparaître dans le pipeline Kanban
            "partner_id": partner_id,
            "contact_name": f"{s.first_name} {s.last_name}".strip(),
            "email_from": s.email,
            "phone": s.phone,
            "team_id": team_id,
            "tag_ids": [(6, 0, [tag_id])],
            "priority": self._priority(s),
            "expected_revenue": expected_revenue,
            "description": self._description(s),
        }

        horizon = {
            Timeframe.asap: 14,
            Timeframe.one_month: 30,
            Timeframe.three_months: 90,
            Timeframe.six_months_plus: 180,
        }.get(s.timeframe)
        if horizon:
            vals["date_deadline"] = (date.today() + timedelta(days=horizon)).isoformat()

        # Champs custom (seulement si tu les as créés et mappés dans la config)
        for attr, odoo_field in self.cfg.custom_field_map.items():
            val = getattr(s, attr, None)
            if isinstance(val, Enum):
                val = val.value
            if val not in (None, ""):
                vals[odoo_field] = val

        vals = {k: v for k, v in vals.items() if v not in (None, "")}
        lead_id = self._kw("crm.lead", "create", [vals])
        log.info(
            "Lead vendeur créé: id=%s (partner=%s, priorité=%s)",
            lead_id,
            partner_id,
            vals["priority"],
        )

        return {
            "ok": True,
            "lead_id": lead_id,
            "partner_id": partner_id,
            "partner_created": partner_created,
            "priority": vals["priority"],
            "expected_revenue": expected_revenue,
        }

    # ------------------------------------------------------------------ #
    #  Helpers partagés (création minimale + update progressif)           #
    # ------------------------------------------------------------------ #
    def _resolve_team_id(self) -> int:
        """Retourne l'id de l'équipe Mandats, ou lève si elle n'existe pas."""
        team_id = self._id_by_name("crm.team", self.cfg.mandats_team)
        if not team_id:
            raise RuntimeError(
                f"Équipe '{self.cfg.mandats_team}' introuvable. "
                "Crée-la dans CRM -> Configuration -> Équipes commerciales, "
                "ou change ODOO_MANDATS_TEAM."
            )
        return team_id

    def _expected_revenue(self, s: SellerLead) -> float:
        """Estime la commission attendue à partir du prix espéré (0 si inconnu)."""
        return round(s.expected_price * self.cfg.commission_rate, 2) if s.expected_price else 0.0

    @staticmethod
    def _deadline_iso(s: SellerLead) -> Optional[str]:
        """Date d'échéance ISO dérivée du délai de vente, ou None si inconnu."""
        horizon = {
            Timeframe.asap: 14,
            Timeframe.one_month: 30,
            Timeframe.three_months: 90,
            Timeframe.six_months_plus: 180,
        }.get(s.timeframe)
        return (date.today() + timedelta(days=horizon)).isoformat() if horizon else None

    @staticmethod
    def _opp_name(s: SellerLead) -> str:
        """Nom de l'opportunité (carte Kanban), recalculé selon les infos connues."""
        type_label = s.property_type.value if s.property_type else "bien"
        return f"Vendeur — {type_label} {s.city or ''}".strip()

    # ------------------------------------------------------------------ #
    #  Création allégée du lead (dès le nom + un canal)                    #
    # ------------------------------------------------------------------ #
    def create_lead_minimal(self, s: SellerLead) -> dict:
        """Crée un lead vendeur minimal et renvoie lead_id + partner_id.

        Ne requiert que last_name + un canal (phone ou email, garanti par
        SellerLead). priority et description sont calculées sur les infos déjà
        connues (souvent faibles à ce stade — c'est normal) ; elles seront
        recalculées par update_lead à mesure que l'appel se précise.
        """
        partner_id, partner_created = self.upsert_partner(s)
        team_id = self._resolve_team_id()
        tag_id = self._get_or_create_tag(self.cfg.seller_tag)

        vals = {
            "name": self._opp_name(s),
            "type": "opportunity",
            "partner_id": partner_id,
            "contact_name": f"{s.first_name} {s.last_name}".strip(),
            "email_from": s.email,
            "phone": s.phone,
            "team_id": team_id,
            "tag_ids": [(6, 0, [tag_id])],
            "priority": self._priority(s),
            "description": self._description(s),
        }
        vals = {k: v for k, v in vals.items() if v not in (None, "")}
        lead_id = self._kw("crm.lead", "create", [vals])
        log.info(
            "Lead vendeur (minimal) créé: id=%s (partner=%s, priorité=%s)",
            lead_id,
            partner_id,
            vals["priority"],
        )

        return {
            "ok": True,
            "lead_id": lead_id,
            "partner_id": partner_id,
            "partner_created": partner_created,
            "priority": vals["priority"],
        }

    # ------------------------------------------------------------------ #
    #  Update progressif du lead                                          #
    # ------------------------------------------------------------------ #
    def update_lead(self, lead_id: int, s: SellerLead) -> dict:
        """Met à jour un lead vendeur existant à partir d'un SellerLead complet.

        On ne fusionne pas des champs bruts : priority ET description sont
        ENTIÈREMENT recalculées depuis `s`, et le nom de l'opportunité rafraîchi.
        expected_revenue et date_deadline sont posés dès que l'info nécessaire
        (prix / délai) est disponible.
        """
        vals = {
            "name": self._opp_name(s),
            "priority": self._priority(s),
            "description": self._description(s),
        }
        revenue = self._expected_revenue(s)
        if revenue:
            vals["expected_revenue"] = revenue
        deadline = self._deadline_iso(s)
        if deadline:
            vals["date_deadline"] = deadline

        self._kw("crm.lead", "write", [[lead_id], vals])
        log.info(
            "Lead %s mis à jour (priorité=%s, revenue=%s, échéance=%s)",
            lead_id,
            vals["priority"],
            vals.get("expected_revenue"),
            vals.get("date_deadline"),
        )

        return {
            "ok": True,
            "lead_id": lead_id,
            "priority": vals["priority"],
            "expected_revenue": vals.get("expected_revenue", 0.0),
            "date_deadline": vals.get("date_deadline"),
        }


# --------------------------------------------------------------------------- #
#  Client partagé (singleton)                                                  #
# --------------------------------------------------------------------------- #

_client_singleton: Optional[OdooClient] = None


def _client() -> OdooClient:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = OdooClient(OdooConfig.from_env())
    return _client_singleton


def get_client() -> OdooClient:
    """Retourne le client Odoo partagé (singleton, connexion paresseuse au 1er usage).

    Point d'entrée public pour les appelants externes (ex. handlers Pipecat Flows) :
    aucune connexion n'est ouverte tant que cette fonction n'est pas appelée.
    """
    return _client()


# --------------------------------------------------------------------------- #
#  Test manuel — parcours PROGRESSIF (create_lead_minimal puis 2x update_lead) #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    client = _client()

    # 1) Création minimale : on n'a que le nom + un téléphone.
    minimal = SellerLead(last_name="Martin", phone="+32471980011")
    created = client.create_lead_minimal(minimal)
    lead_id = created["lead_id"]
    print("1) create_lead_minimal ->", created)

    # 2) Update #1 : type de bien, ville, prix espéré (priorité recalculée).
    step1 = SellerLead(
        last_name="Martin",
        phone="+32471980011",
        property_type=PropertyType.maison,
        city="Wavre",
        expected_price=325000,
    )
    print("2) update_lead #1   ->", client.update_lead(lead_id, step1))

    # 3) Update #2 : délai, raison, sole_owner, résumé -> priorité finale (🔥 "3").
    step2 = SellerLead(
        last_name="Martin",
        phone="+32471980011",
        property_type=PropertyType.maison,
        city="Wavre",
        expected_price=325000,
        timeframe=Timeframe.asap,
        reason="Succession",
        sole_owner=True,
        call_summary="Vendeur pressé suite à une succession, seul décideur.",
    )
    print("3) update_lead #2   ->", client.update_lead(lead_id, step2))
