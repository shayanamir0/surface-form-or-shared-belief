# Surface Form or Shared Belief?

This repository contains the code for a behavioral test of agreement in
LLM-as-a-judge panels.

The main question is simple: do different language models agree because they
see the same wording, or because they make the same underlying judgment? We
tested six OpenAI and Anthropic models on natural-language inference and
pairwise response evaluation. Giving each judge a different
meaning-preserving rewrite did not create useful independence. The six-model
panel still behaved like roughly two effective votes.

![Change in effective panel size under the tested interventions](assets/delta_neff.png)

## Run the code

Python 3.11 or newer is required.

```bash
uv sync --extra dev --extra figures
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

Prepare the two datasets:

```bash
uv run behavioural-test prepare-chaosnli
uv run behavioural-test prepare-rewardbench
```

Generate and audit the meaning-preserving views before running the
interventions:

```bash
uv run behavioural-test generate-views \
  --output data/processed/c2_views_v2.jsonl \
  --reasoning-effort low
uv run behavioural-test audit-views \
  --views data/processed/c2_views_v2.jsonl \
  --output results/c2_view_audit_v2.csv
```

Run a baseline and the per-judge view condition:

```bash
uv run behavioural-test run --condition C0
uv run behavioural-test run --condition C2 \
  --views data/processed/c2_views_v2.jsonl
```

Use `--task preference` for the RewardBench experiment. When all runs are
complete, evaluate the preregistered endpoints:

```bash
uv run behavioural-test endpoints
```

Run the local checks with:

```bash
uv run pytest
uv run ruff check .
```
