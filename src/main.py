"""Command-line interface of the Intelligent Procurement AI Agent.

    python -m src.main setup                    # generate + process
    python -m src.main generate                 # synthetic raw data
    python -m src.main process                  # cleaning + tables for scoring
    python -m src.main rank --top 10            # show the prioritized open orders
    python -m src.main run --top 3              # run the agent (asks for approvals)
    python -m src.main run --top 3 --no-llm     # same, with templates instead of the LLM
    python -m src.main run --approval deny      # unattended: every approval is refused
"""

import argparse
import os
import sys
from pathlib import Path

from src import agent, config, data_processing, generate_data, llm, report
from src.audit import AuditLog


def cmd_generate(args):
    generate_data.main()


def cmd_process(args):
    data_processing.main()


def cmd_setup(args):
    cmd_generate(args)
    cmd_process(args)


def cmd_rank(args):
    scored = agent.detect_and_prioritize()
    path = report.write_prioritized_csv(scored)
    columns = ["rank", "po_id", "material", "criticality_of_material", "order_value",
               "supplier_late_rate", "rule_risk_score", "composite_risk_score",
               "risk_level", "priority_score"]
    print(scored[columns].head(args.top).to_string(index=False))
    print(f"\nFull ranking: {path}")


def cmd_run(args):
    use_llm = not args.no_llm
    llm.load_env_file()
    if use_llm and os.environ.get("GROQ_API_KEY"):
        explanations = f"LLM ({os.environ.get('GROQ_MODEL', config.DEFAULT_GROQ_MODEL)} via Groq)"
    elif use_llm:
        explanations = "templates (GROQ_API_KEY not set)"
    else:
        explanations = "templates (LLM disabled with --no-llm)"

    approver = agent.terminal_approver if args.approval == "ask" else agent.deny_all_approver
    audit = AuditLog(args.audit_log)
    print(f"Explanations: {explanations} | approval mode: {args.approval}")

    scored, results = agent.run_agent(top_n=args.top, use_llm=use_llm, approver=approver,
                                      audit=audit)
    run_info = {"run_id": audit.run_id, "explanations": explanations,
                "approval_mode": args.approval}
    csv_path = report.write_prioritized_csv(scored)
    report_path = report.write_risk_report(scored, results, run_info)
    print(f"Reports: {report_path}, {csv_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Intelligent Procurement AI Agent (prototype)")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("setup", help="generate the data and process it")
    commands.add_parser("generate", help="generate the synthetic raw data")
    commands.add_parser("process", help="clean the data and build the tables for scoring")

    rank = commands.add_parser("rank", help="show open orders ranked by priority")
    rank.add_argument("--top", type=int, default=10)

    run = commands.add_parser("run", help="run the agent on the top N orders")
    run.add_argument("--top", type=int, default=config.DEFAULT_TOP_N)
    run.add_argument("--no-llm", action="store_true", help="use templates instead of the LLM")
    run.add_argument("--approval", choices=["ask", "deny"], default="ask",
                     help="ask in the terminal (default) or refuse every approval")
    run.add_argument("--audit-log", type=Path, default=config.AUDIT_LOG_PATH)
    return parser


COMMANDS = {"setup": cmd_setup, "generate": cmd_generate, "process": cmd_process,
            "rank": cmd_rank, "run": cmd_run}


def main():
    # LLM text can contain characters a Windows console cannot display:
    # replace them instead of crashing in the middle of a demo.
    sys.stdout.reconfigure(errors="replace")
    args = build_parser().parse_args()

    if args.command in ("rank", "run") and not (config.PROCESSED_DIR / "open_orders.csv").exists():
        sys.exit("No processed data found. Run:  python -m src.main setup")
    COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
