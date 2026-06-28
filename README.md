# DemoImmoAgency

Agent vocal **inbound** (français) pour agences immobilières belges, construit sur
[Pipecat](https://docs.pipecat.ai/). Auto-hébergé en EU (RGPD / AI Act).

> **Phase courante : Phase 0 — bootstrap.** Pas encore de `bot.py` ni de logique.
> Voir `CLAUDE.MD` pour la discipline des phases.

## Stack

- **Transport** : SmallWebRTC (dev local, test navigateur)
- **STT** : Gladia (`fr`) · **LLM** : OpenAI · **TTS** : Cartesia (Sonic) · **VAD** : Silero
- **Runtime dev** : dev runner Pipecat

## Prérequis

- Python 3.12 (pinné via `.python-version`)
- [uv](https://docs.astral.sh/uv/)

## Installation

```bash
uv sync                  # installe les dépendances
cp .env.example .env     # puis renseigner les clés API
```

Clés requises dans `.env` : `GLADIA_API_KEY`, `OPENAI_API_KEY`, `CARTESIA_API_KEY`.

## Commandes

```bash
uv sync                       # installer les deps
uv run ruff check --fix .     # lint + autofix
uv run ruff format .          # format
```
