# Intelligent Procurement AI Agent

A prototype agent that finds risky purchase orders in a (simulated) SAP
procurement landscape, investigates them with SAP-like tools, explains the
risks, recommends actions, and executes only the actions that are allowed -
asking a human for approval when needed and logging every step.

> **Prototype scope.** All SAP data and SAP interactions are simulated (CSV files
> and Python functions). Nothing connects to SAP, SAP BTP or Joule; no email is
> sent and no data is modified.

## What it does

| Capability | How |
|---|---|
| Detect risky purchase orders | Business rules + a supplier statistic (historical late rate) |
| Prioritize | `priority_score` = risk x material criticality, order value, urgency |
| Investigate | 7 mock SAP tools (purchase order, supplier performance, stock, alternatives, actions) |
| Explain and recommend | LLM (GPT-OSS on Groq) **or** deterministic templates - same facts, same actions |
| Act safely | Default-deny policy; sensitive or high-value actions need explicit human approval |
| Trace | Append-only `outputs/audit_log.jsonl`, plus `risk_report.md` and `prioritized_risks.csv` |



## Project structure

```
src/
  config.py           all paths, dates, thresholds, weights and policies
  generate_data.py    synthetic SAP-like data + injected anomalies
  data_processing.py  validation, cleaning, quality summary, supplier statistics
  risk_engine.py      rules, supplier statistic, composite and priority scores
  mock_sap_api.py     7 simulated SAP tools
  llm.py              Groq LLM call + template fallback
  audit.py            append-only JSONL audit log
  agent.py            workflow, guardrail, human approval
  report.py           risk_report.md + prioritized_risks.csv
  main.py             command-line interface
app.py                Flask web app (same code as the CLI)
templates/, static/   web app pages and stylesheet
start_dashboard.bat   Windows launcher for the dashboard
tests/                25 pytest tests (no network)
data/raw/             generated data (with anomalies) - see data/README.md
data/processed/       cleaned data, closed orders, open orders
outputs/              reports, audit log
presentation/         slides and demo 
```

## Installation

Prerequisites: Python 3.10+ (developed with 3.12).

```powershell
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Optional, for LLM explanations: copy `.env.example` to `.env` and set
`GROQ_API_KEY`. Without it, everything works with templates.

## Commands

```powershell
python -m src.main setup                          # generate the data and clean it
python -m src.main rank --top 10                  # ranked open orders (+ prioritized_risks.csv)
python -m src.main run --top 3                    # run the agent, approvals asked in the terminal
python -m src.main run --top 3 --no-llm           # same, with templates instead of the LLM
python -m src.main run --top 3 --approval deny    # unattended: every approval is refused
python -m pytest -q                               # run the tests (25)
```

### Dashboard (optional)

```powershell
python app.py               # opens http://localhost:5000
```

On Windows you can also double-click **`start_dashboard.bat`**. It installs the
dependencies and runs `setup` the first time if needed, starts the server (on this
computer only) and opens the browser. If the server is already running, it just
opens the browser. Close its window to stop the server.

Five pages, all built on the same code as the CLI:

| Page | Content |
|---|---|
| Overview | Orders to look at first, key figures, data quality |
| Risk ranking | Top 15 by priority, risk mix, risk drivers, value at risk, full scored table; filter by risk level |
| Agent investigation | Pick an order, investigate it, then Approve/Reject each sensitive action with a reason (every action starts as Pending) |
| Audit log | Every audit entry, filterable by run, PO and event; full JSON per row; CSV download |
| Tests | Runs pytest and shows each test result |

The pages are Jinja templates in `templates/`; the look is in `static/style.css`.
Every page is marked "do not translate", so browsers leave it in English.
Investigations in progress are kept in the server's memory: restarting the
server forgets them (the audit log keeps everything that was decided).

`setup` can also be run step by step: `generate`, then `process`.
`run` accepts `--audit-log PATH` to write the audit trail elsewhere.

### With and without an LLM

| Situation | Explanations | Everything else |
|---|---|---|
| `GROQ_API_KEY` set | GPT-OSS (`openai/gpt-oss-20b` by default, `GROQ_MODEL` to change) | identical |
| `--no-llm`, no key, API error, or unusable answer | Python templates; the reason is printed and logged | identical |

The LLM receives only a JSON of verified facts (scores, tool results, decided
actions). It cannot change scores, choose actions or approve anything.

## Results (measured on the synthetic data, seed 42)

**Data**

- 500 purchase orders, 25 suppliers, 1,200 historical deliveries, 20 materials.
- After cleaning: 488 purchase orders. 8 duplicates were removed and 12 orders were rejected (unknown supplier, promised date before order date, delivered without a date). 15 missing values were imputed and 60 supplier names normalized.
- 1,607 closed orders (30.5% late), used for the supplier statistics; 68 open orders to score, 6 of them overdue.

**Risk scoring of the 68 open orders:** 4 High, 5 Medium, 59 Low.

`composite = 100 x (0.25 x supplier late rate + 0.45 x overdue + 0.30 x rule score / 100)`
(weights in `config.COMPOSITE_WEIGHTS`). The top order, PO 4500000337, is 3 days
overdue for a High-criticality proximity sensor; its supplier delivered 64% of
its past orders late (36% on time) and its stock is below safety stock.
Composite risk 80.6, priority score 75.5.



**Agent run** (`run --top 3 --approval deny`): 3 supplier reminders were sent
automatically (simulated). The supplier replacement on PO 4500000337 and the
reschedule on PO 4500000417 (both over 10,000 EUR, so Director approval) were
refused and not executed. All of it is in the audit log.

**Tests:** 25 passed (5 pipeline tests + 20 guardrail cases).


## Limitations

- **Synthetic data**: patterns were designed by the generator. The prototype shows
  that the method works end to end, not how it would perform on real SAP data.
- **Hand-picked weights**: rule points, composite weights and thresholds are
  business judgment, not fitted. With real data they would be reviewed with buyers.
- **Supplier reliability counts twice**: once in the late rate, once in the
  unreliable-supplier rule (25 points), on purpose, to flag the worst suppliers.
- **Simplified stock rule**: compares one PO with the material's stock and ignores
  other open POs for the same material.
- **Single-item POs**, no schedule lines, no partial deliveries.
- **The approval is enforced by the agent only**; the mock tools do not check
  authorisation themselves. A real SAP system would enforce it server-side.
- **The LLM can add mild interpretation** (e.g. "risk of a production halt"). It
  never affects scores or actions, but its text should be read as a draft.
- No real authentication, user roles or second-person approval.




