# CLAUDE.md

## Projet

Agent vocal **inbound** pour agences immobilières belges. Appel entrant en
**français** : l'agent accueille, qualifie le prospect (vendeur / acheteur /
location / estimation), propose un RDV, et écrit dans le CRM (Odoo). Construit sur
**Pipecat** + **Pipecat Flows 1.0**. Auto-hébergé en EU (argument de vente : RGPD /
AI Act / hébergement européen).

## Discipline des phases — RÈGLE LA PLUS IMPORTANTE

Le projet se construit par **phases incrémentales et testables**. Tu implémentes
**une seule phase à la fois**. Tu n'anticipes pas, tu ne scaffoldes pas les phases
suivantes, tu n'ajoutes pas de dépendances « pour plus tard ». Chaque phase doit
**tourner et être testée** avant de passer à la suivante.

0. Bootstrap projet (uv, deps, structure, `.env.example` — pas de bot)
1. Voix qui marche (`bot.py` : pipeline STT→LLM→TTS, SANS Flows)
2. Premier node Flows (greeting seul)
3. Deux nodes + une transition
4. Router d'intention (branching)
5. Premier tool Odoo (qualif vendeur)
6. Les 3 autres tools (acheteur, RDV, notification agence)
7. Global functions + RGPD
8. Téléphonie (numéro entrant)
9. Déploiement EU

**Phase courante : Phase 0.** Ne code rien des phases ≥ 1 sans qu'on te le demande
explicitement.

## Stack

- **Transport** : SmallWebRTC en dev local (test navigateur). Daily / téléphonie plus tard.
- **STT** : Gladia, `language="fr"`.
- **LLM** : OpenAI `gpt-4o-mini` (Phase 1). Un LLM EU pourra le remplacer plus tard.
- **TTS** : Cartesia, modèle **Sonic**, voix française.
- **VAD** : Silero (dès la Phase 1).
- **CRM** : Odoo (à partir de la Phase 5 — pas avant).
- Backend FastAPI : phases ultérieures. Phase 1 utilise le dev runner Pipecat.

## Fonctionnement de Pipecat — SOURCE UNIQUE : la doc via MCP

- Pour TOUT ce qui touche au fonctionnement de Pipecat / Pipecat Flows (imports,
  signatures, noms de classes, paramètres, noms d'extras, patterns, exemples), ta
  **SEULE** source autorisée est le MCP **pipecat-docs**. Rien d'autre : ni ta mémoire
  ou tes données d'entraînement, ni des blogs / articles / exemples trouvés ailleurs,
  ni des suppositions « au plus probable ».
- Avant d'écrire la moindre ligne touchant à Pipecat, vérifie dans pipecat-docs.
- Si une info est introuvable dans la doc via le MCP, tu **t'arrêtes** et tu me le
  signales. Tu n'inventes pas, tu ne combles pas avec ta mémoire, tu ne déduis pas.

## Pipecat Flows 1.0 — patterns à respecter (et pièges 0.0.x à éviter)

À utiliser :
- **Dynamic flows uniquement.** Des fonctions `create_*_node()` qui retournent un
  `NodeConfig`, et démarrage via `flow_manager.initialize(create_initial_node())`.
- **Transitions = valeur de retour du handler.** Un handler retourne un tuple
  `(result, next_node)` (l'un ou l'autre peut être `None`).
- **Fonctions** : `FlowsFunctionSchema(name, description, properties, required, handler)`
  OU direct functions (signature + docstring génèrent le schema).
- Contrôle d'interruption sur une direct function : décorateur
  `@flows_tool_options(cancel_on_interruption=False)` (l'ancien alias
  `@flows_direct_function` est déprécié).
- **`global_functions=[...]`** passé au `FlowManager` pour les fonctions dispo sur
  tous les nodes (ex. `transfert_humain`, `hors_scope`).
- **`role_message`** : une **string** (pas une liste), system instruction qui persiste
  entre nodes. À poser une seule fois dans le node initial.
- **`task_messages`** : requis, format OpenAI (`{"role": "developer", "content": ...}`),
  l'objectif du node courant.
- **`flow_manager.state`** : dict pour partager des données entre nodes (ex. `lead_id`).
- Inbound : le greeting parle en premier → `respond_immediately=True` (défaut).

À NE PAS utiliser (supprimé en 1.0) :
- `FlowConfig` / `flow_config=` / static flows / transition par nom de node.
- `transition_to=` et `transition_callback=` sur `FlowsFunctionSchema`.
- `flow_manager.set_node(id, config)` → utiliser `set_node_from_config(config)`.
- Dicts de fonctions spécifiques au provider (OpenAI/Anthropic/Gemini).
- `role_messages` (liste, déprécié) → `role_message` (string).
- `FlowManager(tts=...)` → ce paramètre n'existe plus.

## Environnement & conventions

- **Python 3.12** (pinné via uv — version recommandée par Pipecat), gestion des deps via **uv**. `pipecat-ai >= 1.0.0`, `pipecat-ai-flows`.
- Lint/format : **Ruff**, longueur de ligne **100**. Docstrings Google.
- Tout secret (clés API) dans `.env`, jamais commité. `.env` est dans `.claudeignore`.

## Commandes

```bash
uv sync                       # installer les deps
uv run bot.py                 # lancer le bot (Phase 1 : UI navigateur en local)
uv run ruff check --fix .     # lint + autofix
uv run ruff format .          # format
```

## Workflow git

Commits petits et fréquents, un par étape logique. Ne commit jamais `.env` ni `.venv/`.