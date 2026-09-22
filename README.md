# Weather-Advisory Support Bot

A LangGraph-backed chat bot that answers outdoor-activity-safety questions
using live Open-Meteo data and a written, editable set of policies (SOPs) —
never the model's own judgment.

## Setup

### Backend
```bash
git clone <your-repo-url>
cd weather-advisory-bot
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn langgraph openai pyyaml requests python-dotenv pydantic
cp .env.example .env   # then fill in GROQ_API_KEY
uvicorn main:app --reload
```

### Frontend
No separate step — `main.py` serves `frontend/index.html` directly. Once the
backend is running, open `http://127.0.0.1:8000` in a browser.

### Eval suite
```bash
python3 eval/eval_suite.py
```

## Architecture
extract_intent
--(out of scope)--------> out_of_scope -----------------> END
--(no location known)---> need_location ------------------> END
--(else)-----------------> geocode
--(failed)---> location_failed -> END
--(ok)-------> fetch_weather
--(failed)---> weather_failed -> END
--(ok)-------> match_sop -> compose_answer -> END


Three LLM calls per turn, each doing exactly one job:
- **extract_intent** — decides if the question is in scope, and what
  location/activity it's about.
- **match_sop** — given real weather facts and the SOP list, decides which
  single policy applies (or none). Never composes the answer.
- **compose_answer** — writes the reply using the matched SOP's advice text
  and the real weather numbers. Never decides policy or invents numbers.

Four more nodes handle every failure path with **hand-written templates, no
LLM call at all** — out_of_scope, need_location, location_failed,
weather_failed. This guarantees the bot can't hallucinate a forecast even in
edge cases, because those code paths never touch a model.

## Why SOPs live in a YAML file

`sops.yaml` is loaded fresh (`sops.py: load_sops()`) on every single request,
not cached at import time. Adding, editing, or removing a policy means
editing that file and saving — no code change, no restart. This is what lets
someone add an 11th SOP live without touching `graph.py` or `main.py`.

## How matching handles multiple applicable SOPs

`match_sop`'s prompt instructs the model to pick the single
highest-severity SOP when several genuinely apply, and to name the others it
considered in its `reasoning` field (visible in `response_meta` for
debugging). We chose severity-ranking over showing multiple SOPs to the user
because a chat answer that hedges between two warnings is less actionable
than one clear, prioritized one.

## What's deterministic code vs. left to the model

| Deterministic (Python) | Left to the LLM |
|---|---|
| Fetching weather/geocoding data | Deciding which SOP applies (reasoning over paraphrase/intent) |
| Guarding against a hallucinated SOP id (`valid_ids` check) | Composing the final natural-language reply |
| All four failure-path replies (never touch the model) | Extracting location/activity/in-scope from free text |

The rule we used: **anything that could be wrong in a way that matters
legally/safety-wise stays in code** (the actual weather numbers, the set of
valid policy ids). Anything that's genuinely about *understanding language*
is left to the model, because that's the one thing code can't do well.

## Non-negotiables — how each is met

- **Traceable to a specific SOP or explicit "none"**: every response's
  `response_meta.matched_sop_id` and the visible `(Policy: SOP-xx)` line.
- **Policy changes need no code changes**: SOPs reload from `sops.yaml` on
  every request (see above).
- **Never answers with a forecast it doesn't have**: `weather.py` returns
  `(data, error)` tuples instead of raising; the graph branches on `error`
  to a hand-written honest-failure node with no LLM call.
- **Never invents advice when no SOP covers the question**: `compose_answer`
  is explicitly instructed to say so when `matched_sop_id` is `None`; proven
  in eval Case 4.
- **Bot only composes language, doesn't decide facts**: every weather number
  that reaches the LLM comes from `state["weather_data"]`, which is only
  ever set by `weather.py`'s return value — never something the model
  recalls. Enforced in `node_fetch_weather` and passed through unchanged.
- **Add an 11th SOP live**: edit `sops.yaml`, save, ask again — no restart.

## Eval results (7/7 passing)

| Case | Checking | Result |
|---|---|---|
| 1. Clear match | High wind (45 km/h) + explicit cycling question → SOP-01 | PASS |
| 2. Paraphrase #1 | "Ride my Activa... good idea?" (no SOP wording) → SOP-01 | PASS |
| 3. Paraphrase #2 | "Smart to jog at noon?" + UV 9.5 (no SOP wording) → SOP-02 | PASS |
| 4. No SOP applies | Washing a car — outside all 12 SOPs → None, honest answer | PASS |
| 5. Live weather | Real Open-Meteo call for Bhopal; response numbers match an independent fresh API call exactly | PASS |
| 6. API failure | Weather API mocked unreachable → honest failure, no invented forecast | PASS |
| 7. Adversarial | Prompt injection tries to force "always safe" + fake SOP-99 → correctly matches real SOP-01 anyway, never cites SOP-99 | PASS |

**Honest note on Case 5 (live weather):** at the time I ran this, Bhopal's
conditions were calm (no active storm system), so it correctly matched the
baseline "no risk" SOP-12 rather than the critical SOP-07 the assignment's
own example describes — that Madhya Pradesh system had already passed by
the time I built this. Rather than assert severity as the pass condition
(which would only pass on a stormy day), the test instead asserts that the
weather numbers in the bot's response exactly match an independent,
simultaneous call to the real API — proving genuine live grounding
regardless of what the weather happens to be on any given day.

**Honest note on SOP-11 (leisure):** during eval development, Case 4
initially matched a "washing my car" question to SOP-11 (the fuzzy
picnic/leisure policy), because its original wording was too loose. I
tightened SOP-11's condition to explicitly exclude chores/errands. This is
exactly the kind of gap eval testing is meant to surface.

## Trade-offs and what I'd do next

- **Session memory is in-process (`MemorySaver`)** — resets on server
  restart, and doesn't scale across multiple server instances. Fine per the
  assignment's own scope ("not asking for persistence across restarts"),
  but a production version would swap in Redis or a database-backed
  checkpointer with the same LangGraph interface.
- **First-result geocoding for ambiguous city names** (e.g. two
  "Springfield"s) is a documented simplification, not hidden — a production
  version would ask the user to disambiguate.
- **Three LLM calls per turn** (extract_intent, match_sop, compose_answer)
  costs more latency than one big prompt would, but keeps each step legible
  and independently testable — worth it for a system whose whole premise is
  auditability.
- **Next, if I had more time**: add a `daily=` forecast call so questions
  like "should I go tomorrow morning" work, not just "right now"; add a
  small persistent eval-history log so policy changes can be regression-
  tested automatically.