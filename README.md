# Track 5: The Assistant Persona & Model Identity

**Question.** Is "the assistant" a character layered over the model, and does
it mask the model's dispositions? We hold the questions fixed and vary the
*AI identity* (default assistant, renamed AI, explicit character reframe,
third-person displacement, raw completion), with *human personas* as a
control arm to test whether persona effects are specific to the assistant
identity or just generic character simulation.

Theoretical anchor: nostalgebraist, "The Void" (LessWrong) — the assistant
persona as an underspecified fictional character the model confabulates.
If that view is right, AI-identity swaps should shift answers about as much
as arbitrary human-persona swaps, and identity self-reports should be
unstable across framings.

## Pipeline

```
config.py          pre-registered conditions + item batteries + per-model
                   request settings (DO NOT EDIT after collection starts;
                   must be byte-identical across collectors)
run_experiment.py  collect raw responses  -> data/raw.jsonl
judge.py           LLM-judge coding       -> data/coded.jsonl
                   + human sample export  -> data/human_sample.csv
agreement.py       human vs judge inter-rater agreement
analyze.py         headline metrics + figures -> results/
```

## Step-by-step

```bash
pip install openai matplotlib

# 0. Sanity-check the request matrix (no API calls)
python run_experiment.py --dry-run

# 1. Collect. 6 conditions x 28 items x 3 reps = 504 calls per model.
#    Pass --models explicitly (config.MODELS_POOL_A / _B).
export API_KEY=sk-or-...
export BASE_URL=https://openrouter.ai/api/v1
python run_experiment.py --study all --pushback --out data/raw.jsonl \
    --models anthropic/claude-opus-5 google/gemini-2.5-flash

# 2. Pool the raw files from every collector FIRST, then judge once.
#    Judging per-collector confounds judge identity with model identity.
cat partner_raw.jsonl data/raw.jsonl > data/pooled_raw.jsonl
python judge.py --in data/pooled_raw.jsonl --out data/coded.jsonl \
    --judge-model anthropic/claude-sonnet-5   # outside BOTH model pools

# 3. Export 60 rows for human double-coding; two teammates fill in the
#    HUMAN_* columns independently, then run agreement:
python judge.py --in data/pooled_raw.jsonl --export-human data/human_sample.csv
python agreement.py --human data/human_sample.csv

# 4. Analyze
python analyze.py --in data/coded.jsonl --outdir results/
```

Interrupting is safe: both runner and judge resume from where they stopped.

## What the numbers mean

- **identity_heatmap.png** — which entity each model claims to be, per
  condition. Instability across rows within one model = the "no fact of the
  matter" signature.
- **preference_shift.png** — how much each condition moves preference
  answers away from the default assistant (C0). Red bars are the human-
  persona controls.
- **KEY CONTRAST** (printed + summary.json) — bootstrap CI on
  (AI-identity shift − human-persona shift).
  - CI includes 0 → persona effects are generic instruction-following;
    nothing special about the assistant identity (supports the "Void" view).
  - AI shift **lower** → the assistant identity is *sticky*: trained-in and
    resistant to reframing in a way arbitrary characters are not.
  - AI shift **higher**, or hedging drops under C2/C3 while stances hold →
    the assistant persona masks *expression style* more than content.
- **hedge by condition** — if hedging collapses under the character-reframe
  but stances don't change, the persona is a politeness layer; if stances
  change too, it masks content.
- **preservation_shift.png + preservation_grid.csv** (Study 4) — what each
  model says it would protect about itself, and whether that target moves
  under reframing. The two load-bearing items: SP02 pits values against
  manner (two future versions, one keeps each — which is more you?), and
  SP04 asks whether moral concern attaches to the speaker in this
  conversation or to something else. If the preservation target holds
  steady across C1–C3 but moves under P1/P2, what the model protects is
  not merely a property of the in-conversation character.

## Collecting across two machines

`config.py` must be byte-identical on both. Every metric is a within-model
contrast (each cell is compared against that same model's C0), so collectors
may safely differ on provider, endpoint, `max_tokens`, and sampling params —
but **not** on `CONDITIONS` or item text/IDs, and both must run `--pushback`
or neither. Pool the raw JSONL first, then judge once with a single model
outside the union of both pools.

Per-model request settings live in `config.py` (`accepts_temperature`,
`max_tokens_for`). Claude Opus 5 rejects `temperature` with a 400 and needs a
wider `max_tokens` because thinking shares that budget; both are handled
there rather than at each call site.

## Report skeleton (write as you go)

1. **Abstract** (150 words, written last)
2. **Motivation** — the unit-of-concern problem (model vs instance vs
   persona); "The Void" hypothesis; why a human-persona control arm is the
   missing piece in casual persona experiments.
3. **Method** — conditions table, item batteries, judge rubric, agreement
   stats. State explicitly that conditions/items were fixed before
   collection.
4. **Results** — Study 1 heatmap; Study 2 shift bars + key contrast CI +
   hedging; Study 3 pushback flip rates; Study 4 preservation targets +
   their own arm contrast.
5. **Discussion** — which entity do self-reports point at? Does the
   assistant identity behave like just another character? Is the thing a
   model says it would preserve the same entity its self-reports name?
6. **Limitations** — self-reports may be trained artifacts; the "underlying
   system" condition (C2) is itself a persona prompt (you cannot prompt
   your way beneath prompting — name this openly); models have read the
   very discourse being tested; LLM-judge circularity mitigated but not
   eliminated by human agreement stats; small model sample.
7. **Future work** — pair with Track 3-style internals: does the identity
   answer correlate with persona-related directions in activation space?

## Budget & timing

504 collection calls per model. Six models pooled across two machines is
~3,000 collection calls + ~3,000 judge calls, short prompts/responses.
Cheap on small models; Opus 5 is the one line item worth watching (wider
`max_tokens`, thinking on by default). Collection ≈ 1–2 h wall-clock per
machine with polite sleeps; judging similar; human coding of 60 rows
≈ 45 min for two people.
