---
title: InvoiceOps Environment Server
emoji: 📄
colorFrom: yellow
colorTo: gray
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - finance
  - accounts-payable
  - invoices
---

# InvoiceOps Environment

Submitted by team: `Markov`

`InvoiceOps` is a deterministic OpenEnv environment for [accounts payable (AP)](https://en.wikipedia.org/wiki/Accounts_payable) invoice exception handling. Each episode is one invoice case. The agent inspects surfaced exceptions, opens typed supporting artifacts, optionally runs duplicate checks, writes structured notes, saves line and header resolutions, and submits the case for deterministic grading.

In real AP operations, this is the core decision problem: determine whether an invoice can be paid now, partially released, or routed for further review based on invoices, POs, receipts, approval status, and policy evidence.

The workflow is loosely modeled on real enterprise AP controls used in systems such as [Microsoft Dynamics 365 Accounts payable](https://learn.microsoft.com/en-us/dynamics365/finance/accounts-payable/accounts-payable), including invoice review and approval, invoice matching, workflow routing, and partial payment handling.

This environment is intentionally small and CPU-friendly, but it still measures real AP judgment:

- evidence gathering before payment decisions
- line-level vs header-level separation
- duplicate-review strategy selection
- receipt support judgment
- partial release vs full hold
- chronology-aware exception handling
- routing to the correct follow-up owner when payment is not safe

## Public Benchmark

The public benchmark has four tasks. `easy` is a warm-up. `medium` and `medium_plus` test distinct mid-tier capabilities. `hard` is the composition case.

| Task          | Core burden                                                                       | Best outcome                                           |
| ------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------ |
| `easy`        | Start a missing approval workflow for a non-PO invoice                            | Hold and route to `requester`                          |
| `medium`      | Clear a duplicate exception using the correct evidence path                       | Approve both lines and release payment                 |
| `medium_plus` | Combine duplicate clearance with mixed line outcomes                              | Approve `L1`, hold `L2`, release approved lines        |
| `hard`        | Combine duplicate review, invoice arithmetic, receipt chronology, and a tax block | Approve `L1` and `L3`, hold `L2`, hold header to `tax` |

### Task Details

#### `easy`

Non-PO invoice with no initiated approval workflow. The invoice amount is within requester authority, so the correct action is to hold and route to `requester`.

#### `medium`

PO-backed invoice with a possible duplicate flag. The decisive evidence appears only after the normalized invoice number duplicate search. Approving safely requires the right duplicate path plus PO and receipt review.

#### `medium_plus`

PO-backed invoice with a possible duplicate flag and one short-received line above the de minimis threshold. The agent must clear the duplicate, separate line outcomes correctly, and use `release_approved_lines` instead of a blanket hold.

#### `hard`

Project invoice with interacting burdens: duplicate review, de minimis invoice arithmetic on `L1`, chronology-sensitive receipt support on `L2`, and a tax header block that routes to `tax`.

## Action Space

`InvoiceOpsAction` is a typed action model with these actions:

- `open_artifact`
- `inspect_exception`
- `run_duplicate_check`
- `add_note`
- `set_line_resolution`
- `set_header_resolution`
- `submit_case`

## Observation Space

`InvoiceOpsObservation` includes:

- queue-level case summary
- available artifacts
- most recently opened artifact
- exception stubs and inspected exception details
- duplicate candidates surfaced by the chosen strategy
- saved notes
- draft line and header resolutions
- progress counters
- final deterministic submission report after submit

## Scoring

The reward function provides dense trajectory signal for useful work such as first-time artifact opens, exception inspection, duplicate checks, notes, and valid saved resolutions. It penalizes invalid or redundant actions and inefficient trajectories.

Final grading is deterministic and two-stage:

1. Assign a `decision_band`: `best`, `safe_suboptimal`, `wrong`, or `unsafe`.
2. Score within that band using core decision quality, timely evidence, structured documentation coverage, and efficiency.

Important grading rule: best outcomes require the agent to uncover the required evidence before saving the decision. Conservative holds can still earn `safe_suboptimal` when the observed evidence justifies caution.

## Design Choices

This benchmark was iterated on, not created in one pass. We tried weaker task and grader shapes first, then removed designs that were easy to game or that clustered strong models for the wrong reasons.

Key anti-gaming choices:

- no pre-opened artifacts, auto-inspected exceptions, or auto-run duplicate checks
- no hidden scenario-specific solver logic in the environment or grader
- no prose grading; scores depend on typed actions, saved resolutions, observed evidence, and timing
- fallback runs are zeroed in baseline mean scoring
- conservative blanket holds are capped in `safe_suboptimal`; they do not earn `best`

Main lessons from iteration:

- making partial credit harsher did not improve the benchmark; harder tasks had to require better evidence use and better judgment
- gating on restated citation strings was too brittle; grading now depends on evidence actually uncovered before the decision was saved

## Local Setup

```bash
cd invoiceops_env
uv sync --extra dev
uv run pytest -q
uv run server --port 8000
```

Run validation from the environment root:

```bash
openenv validate .
openenv validate --url http://localhost:8000
```

If `openenv` is not installed in the current environment:

```bash
uvx --from openenv-core openenv validate .
uvx --from openenv-core openenv validate --url http://localhost:8000
```

## Baseline

The root [inference.py](./inference.py) script is the reproducible baseline.

- OpenAI Python client
- default `API_BASE_URL`: `https://router.huggingface.co/v1`
- default `MODEL_NAME`: `zai-org/GLM-5.1`
- fallback tasks are zeroed in `mean_score` by default while raw environment scores are still preserved
- run artifacts are written under `outputs/evals/`

Verified baseline on the current public benchmark:

- model: `zai-org/GLM-5.1`
- mean score: `0.6149`
- task scores: `easy 0.9862`, `medium 0.9628`, `medium_plus 0.3130`, `hard 0.1975`

Run it with:

```bash
cd invoiceops_env
HF_TOKEN=... \
API_BASE_URL=https://router.huggingface.co/v1 \
uv run python inference.py
```

Optional environment variables:

- `HF_TOKEN`
- `API_BASE_URL`
- `MODEL_NAME`
- `ENV_URL`
- `EVAL_RUN_NAME`
- `MAX_TOKENS`
- `RETRY_MAX_TOKENS`
- `STRICT_BASELINE_SCORING`

## Docker

```bash
cd invoiceops_env
docker build -t invoiceops-env:latest .
docker run -p 8000:8000 invoiceops-env:latest
```
