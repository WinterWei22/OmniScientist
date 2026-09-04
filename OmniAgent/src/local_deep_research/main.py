import argparse
import asyncio
import logging
import sys
import os
import time
from datetime import datetime
from typing import Dict

from .config import settings
from .search_system import AdvancedSearchSystem


async def run_research(query: str, mode: str = "quick", using_model: str | None = None):
    """Run research with specified query and mode"""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    using_model = using_model or settings.default_model
    logger.info(f"\n🔬 Starting {mode} research for query: {query}")
    
    # Create session log directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_log_dir = os.path.join(project_root, "logs", run_ts)
    if not os.path.exists(session_log_dir):
        os.makedirs(session_log_dir, exist_ok=True)
    print(f"Session logs will be saved to: {session_log_dir}")

    if mode == "quick":
        print("\nResearching... This may take a few minutes.\n")
        system = AdvancedSearchSystem(max_iterations=settings.quick.iteration, 
                                      questions_per_iteration=settings.quick.questions_per_iteration,
                                      is_report = False,
                                      using_model=using_model,
                                      session_log_dir=session_log_dir)
    else:
        print(
            "\nGenerating detailed report...  Please be patient as this enables deeper analysis.\n"
        )
        system = AdvancedSearchSystem(max_iterations=settings.detailed.iteration, 
                                      questions_per_iteration=settings.detailed.questions_per_iteration,
                                      is_report = True,
                                      using_model=using_model,
                                      session_log_dir=session_log_dir)
    await system.initialize()
    
    # Use 'single_run' or similar as ID for manual runs
    results = await system.analyze_topic(query, question_id="manual_run")
    print("Research analysis completed")
    
    # Also ensure index.csv is created for consistency
    import csv
    csv_path = os.path.join(session_log_dir, "index.csv")
    full_trace = os.path.abspath(os.path.join(session_log_dir, f"trace_manual_run_full.md"))
    clean_trace = os.path.abspath(os.path.join(session_log_dir, f"trace_manual_run_clean.md"))
    case_json = os.path.abspath(os.path.join(session_log_dir, f"case_manual_run.json"))
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'status', 'query', 'answer', 'full_trace', 'clean_trace', 'case_json'])
        # Truncate answer
        answer = results.get("current_knowledge", "")
        csv_answer = answer[:1000] + "..." if len(answer) > 1000 else answer
        writer.writerow(['manual_run', 'success', query, csv_answer, full_trace, clean_trace, case_json])
        
    return results


async def main():
    """Main function with both interactive and CLI support"""
    # Check if running with command line arguments
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="OriAgent Local Deep Research")
        parser.add_argument("query", help="Research query")
        parser.add_argument(
            "--mode",
            choices=["quick", "detailed"],
            default="quick",
            help="Research mode (default: quick)",
        )
        parser.add_argument(
            "--deep",
            action="store_true",
            help="Use detailed research mode (equivalent to --mode detailed)",
        )

        parser.add_argument("--model", choices=["gpt4_1", "gpt4_1_mini", "deepseek", "claude", "qwen"], default=None)

        args = parser.parse_args()

        # Handle --deep flag
        mode = "detailed" if args.deep else args.mode

        # Run research with CLI arguments
        await run_research(args.query, mode, args.model)
        return

    # Interactive mode (original behavior)
    print("Welcome to the Advanced Research System")
    print("Type 'quit' to exit")

    while True:
        print("\nSelect output type:")
        print("1) Analysis (Few minutes, answer questions, summarize findings)")
        print("2) Detailed Report (More time, generates comprehensive report with deep analysis)"
        )
        choice = input("Enter number (1 or 2): ").strip()
        if choice in ("quit", "q", "exit"):
            break
        while choice not in ["1", "2"]:
            print("\nInvalid input. Please enter 1 or 2:")
            print("1) Analysis (Few minutes, answer questions, summarize findings)")
            print("2) Detailed Report (More time, generates comprehensive report with deep analysis)")
            choice = input("Enter number (1 or 2): ").strip()
            if choice in ("quit", "q", "exit"):
                break
        query = input("\nEnter your research query: ").strip()

        if query.lower() == "quit":
            break

        mode = "quick" if choice == "1" else "detailed"
        await run_research(query, mode)


if __name__ == "__main__":
    asyncio.run(main(), debug=True)
