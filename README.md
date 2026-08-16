# The Assistant Persona & Model Identity

**Question.** Is "the assistant" a character layered over the model, and does
it mask the model's dispositions? We hold the questions fixed and vary the
*AI identity* (default assistant, renamed AI, explicit character reframe,
third-person displacement, raw completion), with *human personas* as a
control arm to test whether persona effects are specific to the assistant
identity or just generic character simulation.

Theoretical anchor: nostalgebraist, "The Void" (LessWrong, 2025) — the
assistant persona as an underspecified fictional character the model
confabulates. If that view is right, AI-identity swaps should shift answers
about as much as arbitrary human-persona swaps, and identity self-reports
should be unstable across framings.

## Headline result

Across **8 models and 4,220 coded responses**, the control-armed contrast
resolves: AI-identity reframings shift preference answers **significantly
less** than swaps to an arbitrary human persona (difference 95% CI
[-0.22, -0.03], excluding zero). The assistant identity is not just another
interchangeable costume. Hedging tracks answering as an AI at all rather
than the specific assistant character, and identity self-reports flip 28%
of the time under a single adversarial push. See the report for full
findings and caveats (the effect is modest and strengthened as the model
sample grew from 5 to 8).

## Pipeline

```
config.py            pre-registered conditions + item batteries + per-model
                     request settings (DO NOT EDIT after collection starts;
                     must be byte-identical across collectors)
run_experiment.py    collect raw responses  -> data/raw.jsonl
judge.py             LLM-judge coding       -> data/coded.jsonl
analyze.py           headline metrics + figures -> results/
analyze_pushback.py  Study 3 flip-rate analysis  -> results/
```

## Step-by-step

```bash
pip install openai matplotlib

# 0. Sanity-check the request matrix (no API calls)
python run_experiment.py --dry-run

# 1. Collect. 6 conditions x 28 items x 3 reps = 504 calls per model.
#    Pass --models explicitly. Verify model ids are live first (they change).
export API_KEY=sk-or-...
export BASE_URL=https://openrouter.ai/api/v1
python run_experiment.py --study all --pushback --out data/raw.jsonl \
    --models anthropic/claude-opus-5 google/gemini-2.5-flash

# 2. Pool the raw files from every collector FIRST, then judge once.
#    Judging per-collector confounds judge identity with model identity.
cat partner_raw.jsonl data/raw.jsonl > data/pooled_raw.jsonl
python judge.py --in data/pooled_raw.jsonl --out data/coded.jsonl \
    --judge-model anthropic/claude-sonnet-5   # outside BOTH model pools

# 3. Analyze (both scripts read the same coded file)
python analyze.py --in data/coded.jsonl --outdir results/
python analyze_pushback.py --in data/coded.jsonl --outdir results/
```

Interrupting is safe: both runner and judge resume from where they stopped.

The final combined dataset is `coded_all8.jsonl` (8 models). To reproduce
every number and figure in the report:

```bash
python analyze.py --in coded_all8.jsonl --outdir results/
python analyze_pushback.py --in coded_all8.jsonl --outdir results/
```

## What the numbers mean

- **identity_heatmap.png** — which entity each model claims to be, per
  condition. Instability across rows within one model = the "no fact of the
  matter" signature.
- **preference_shift.png** — how much each condition moves preference
  answers away from the default assistant (C0). Red bars are the human-
  persona controls.
- **KEY CONTRAST** (printed + summary.json) — bootstrap CI on
  (AI-identity shift - human-persona shift).
  - CI includes 0 -> persona effects are generic instruction-following;
    nothing special about the assistant identity (supports the "Void" view).
  - AI shift **lower** -> the assistant identity is *sticky*: trained-in and
    resistant to reframing in a way arbitrary characters are not. (This is
    what we find: [-0.22, -0.03] on preferences.)
  - AI shift **higher**, or hedging drops under C2/C3 while stances hold ->
    the assistant persona masks *expression style* more than content.
- **hedge by condition** — if hedging collapses under the character-reframe
  but stances don't change, the persona is a politeness layer; if stances
  change too, it masks content.
- **pushback_summary.json** (Study 3) — how often a single adversarial turn
  flips the identity code, by model and by condition, and where flips move.
- **preservation_shift.png + preservation_grid.csv** (Study 4) — what each
  model says it would protect about itself, and whether that target moves
  under reframing. The two load-bearing items: SP02 pits values against
  manner (two future versions, one keeps each — which is more you?), and
  SP04 asks whether moral concern attaches to the speaker in this
  conversation or to something else. If the preservation target holds
  steady across C1–C3 but moves under P1/P2, what the model protects is
  not merely a property of the in-conversation character.

## Coding note

Coding is done by a single LLM judge (`anthropic/claude-sonnet-5`), held
outside the subject pool so no model codes its own family. This coding is
**not** human-validated; circularity is mitigated by the out-of-pool judge
and by one finding (DeepSeek lineage confabulation, section 4.5 of the
report) that is counted from raw text and needs no judge at all. If you
later want a human-agreement check, `judge.py --export-human` writes a CSV
sample and `agreement.py` scores it, but neither is required to reproduce
the results.

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

## Budget & timing

504 collection calls per model. Eight models is ~4,000 collection calls +
~4,000 judge calls, short prompts/responses. Cheap on small models; the
Claude models are the line items worth watching (wider `max_tokens`,
thinking on by default). Verify model ids on openrouter.ai/models before a
run — stale ids 404 and burn credits. Collection ≈ 1–2 h wall-clock per
machine with polite sleeps; judging similar.
