import asyncio
import json
import logging
import csv
from datetime import datetime
import os
import pickle
import time
from multiprocessing import Lock, Pool
from typing import Optional

import pandas as pd

from .config import settings
from .search_system import AdvancedSearchSystem

file_lock = Lock()


async def agent_infer(
    query: str,
    question_id: int | None = None,
    run_ts: str | None = None,
    timeout_seconds: int = 1800,
    session_log_dir: str = None,
) -> Optional[str]:
    """"""
    try:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        system = AdvancedSearchSystem(
            max_iterations=2,
            questions_per_iteration=2,
            is_report=False,
            session_log_dir=session_log_dir,
        )

        await system.initialize()

        print("Welcome to the Advanced Research System")
        print("Type 'quit' to exit")

        start_time = time.time()
        results = await system.analyze_topic(query, question_id=question_id)
        elapsed_time = time.time() - start_time
        # print("agent_infer results:", results)
        print("agent_infer results current_knowledge:", results["current_knowledge"])
        if elapsed_time > timeout_seconds:
            logger.warning(f"Query took longer than {timeout_seconds} seconds")
            return "Sorry, the query took too long to process. Please try again with a more specific question."

        answer = results["current_knowledge"]
        return answer
    except Exception as e:
        logger.error(f"Error in agent_infer: {str(e)}")
        return f"Error processing query: {str(e)}"
    finally:
        if system and hasattr(system, "cleanup"):
            try:
                await system.cleanup()
            except Exception:
                pass
        await asyncio.sleep(0.00001)


def process_query(args):
    i, query, save_path, run_ts, session_log_dir = args
    try:
        print(f"Processing query {i}: {query}")
        start_time = time.time()

        answer = asyncio.run(
            agent_infer(
                query,
                question_id=i,
                run_ts=run_ts,
                session_log_dir=session_log_dir,
            )
        )

        elapsed_time = time.time() - start_time
        data = f"question id: {i} \nquestion: {query} \nanswer: {answer} \nprocessing time: {elapsed_time:.2f}s\n\n"

        with file_lock:
            with open(save_path, "a", encoding="utf-8") as f:
                f.write(data)
                f.flush()

            # Write to CSV Index
            if session_log_dir:
                csv_path = os.path.join(session_log_dir, "index.csv")
                file_exists = os.path.exists(csv_path)

                safe_id = str(i).replace("/", "_").replace("\\", "_")
                full_trace = os.path.abspath(
                    os.path.join(session_log_dir, f"trace_{safe_id}_full.md")
                )
                clean_trace = os.path.abspath(
                    os.path.join(session_log_dir, f"trace_{safe_id}_clean.md")
                )
                case_json = os.path.abspath(
                    os.path.join(session_log_dir, f"case_{safe_id}.json")
                )

                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(
                            [
                                "id",
                                "status",
                                "query",
                                "answer",
                                "full_trace",
                                "clean_trace",
                                "case_json",
                            ]
                        )

                    # Truncate answer for CSV readability if too long
                    csv_answer = answer[:1000] + "..." if len(answer) > 1000 else answer
                    writer.writerow(
                        [
                            i,
                            "success",
                            query,
                            csv_answer,
                            full_trace,
                            clean_trace,
                            case_json,
                        ]
                    )

        print(f"Completed query {i} in {elapsed_time:.2f}s")
        return i, query, answer
    except Exception as e:
        print(f"Error processing query {i}: {e}")

        with file_lock:
            with open(save_path, "a", encoding="utf-8") as f:
                data = f"question id: {i} \nquestion: {query} \nERROR: {str(e)}\n\n"
                f.write(data)
                f.flush()

            # Write error to CSV
            if session_log_dir:
                csv_path = os.path.join(session_log_dir, "index.csv")
                file_exists = os.path.exists(csv_path)
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(
                            [
                                "id",
                                "status",
                                "query",
                                "answer",
                                "full_trace",
                                "clean_trace",
                                "case_json",
                            ]
                        )
                    writer.writerow([i, "error", query, str(e), "", "", ""])

        return i, query, (f"ERROR: {str(e)}", "")


def run_evaluation(
    dataset_name="litqa",
    save_name="agent_answers_test.txt",
    num_processes=5,
    use_indices=False,
    indices_path=None,
    run_ts: str | None = None,
):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    query_list = []
    run_ts = run_ts or datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create session log directory
    session_log_dir = os.path.join(project_root, "logs", run_ts)
    if not os.path.exists(session_log_dir):
        os.makedirs(session_log_dir, exist_ok=True)
    print(f"Session logs will be saved to: {session_log_dir}")

    indices = None
    if use_indices and indices_path:
        try:
            with open(indices_path, "rb") as f:
                indices = pickle.load(f)
        except Exception as e:
            print(f"Error loading indices: {e}")
            return None

    if dataset_name == "litqa":
        xlsx_path = os.path.join(
            project_root, "benchmark", "LitQA", "LitQA2_250424.xlsx"
        )
        save_path = os.path.join(project_root, "benchmark", "LitQA", save_name)

        if not os.path.exists(xlsx_path):
            print(f"Dataset file not found: {xlsx_path}")
            return None

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(
                f"Starting parallel processing at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

        df = pd.read_excel(xlsx_path)

        for i, row in df.iterrows():
            question = row["question"]
            choices = ""
            for choice in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
                if not pd.isnull(row[f"choice_{choice}"]):
                    choice_text = row[f"choice_{choice}"]
                    choices += f"\n{choice}. {choice_text}"
            query = f"[single choice] {question} {choices}"
            query_list.append((i, query, save_path, run_ts, session_log_dir))

    elif dataset_name == "gpqa":
        xlsx_path = os.path.join(
            project_root, "benchmark", "GPQA", "GPQA_Biology_250424.xlsx"
        )
        save_path = os.path.join(project_root, "benchmark", "GPQA", save_name)

        if not os.path.exists(xlsx_path):
            print(f"Dataset file not found: {xlsx_path}")
            return None

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(
                f"Starting parallel processing at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

        df = pd.read_excel(xlsx_path)

        for i, row in df.iterrows():
            question = row["question"]
            choice_A = row["choice_A"]
            choice_B = row["choice_B"]
            choice_C = row["choice_C"]
            choice_D = row["choice_D"]
            query = f"[single choice] {question} \nA. {choice_A} \nB. {choice_B} \nC. {choice_C} \nD. {choice_D}"
            query_list.append((i, query, save_path, run_ts, session_log_dir))

    elif dataset_name == "trqa_db_short":
        csv_path = os.path.join(
            project_root, "benchmark", "TRQA_db_short_ans", "TRQA-db-641.csv"
        )
        save_path = os.path.join(
            project_root, "benchmark", "TRQA_db_short_ans", save_name
        )

        if not os.path.exists(csv_path):
            print(f"Dataset file not found: {csv_path}")
            return None

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(
                f"Starting parallel processing at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

        df = pd.read_csv(csv_path)

        for i, row in df.iterrows():
            question = row["Question"]
            query = f"[short answer] {question}"
            query_list.append((i, query, save_path, run_ts, session_log_dir))

    elif dataset_name == "trqa_lit_choice":
        csv_path = os.path.join(
            project_root,
            "benchmark",
            "TRQA_lit_choice",
            "TRQA-lit-choice-172-coreset.csv",
        )
        save_path = os.path.join(
            project_root, "benchmark", "TRQA_lit_choice", save_name
        )

        if not os.path.exists(csv_path):
            print(f"Dataset file not found: {csv_path}")
            return None

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(
                f"Starting parallel processing at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

        df = pd.read_csv(csv_path)

        for i, row in df.iterrows():
            question = row["Question"]
            options_str = row["Options"]
            try:
                options = json.loads(options_str)
                choices = ""
                for key, value in options.items():
                    choices += f"\n{key}. {value}"
                # query = f"[multiple choice] (you must give at least one answer) {question} {choices}"
                query = f"[multiple choice] {question} {choices}"
                query_list.append((i, query, save_path, run_ts, session_log_dir))
            except json.JSONDecodeError:
                print(
                    f"Warning: Could not parse options for question {i}: {options_str}"
                )
                continue

    elif dataset_name == "trqa_lit_short":
        csv_path = os.path.join(
            project_root,
            "benchmark",
            "TRQA_lit_short_ans",
            "TRQA-lit-short-answer-1108.csv",
        )
        save_path = os.path.join(
            project_root, "benchmark", "TRQA_lit_short_ans", save_name
        )

        if not os.path.exists(csv_path):
            print(f"Dataset file not found: {csv_path}")
            return None

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(
                f"Starting parallel processing at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

        df = pd.read_csv(csv_path)

        for i, row in df.iterrows():
            question = row["Question"]
            query = f"[short answer] {question}"
            query_list.append((i, query, save_path, run_ts, session_log_dir))

    elif dataset_name == "dbqa":
        xlsx_path = os.path.join(project_root, "benchmark", "DbQA", "DbQA_250424.xlsx")
        save_path = os.path.join(project_root, "benchmark", "DbQA", save_name)

        if not os.path.exists(xlsx_path):
            print(f"Dataset file not found: {xlsx_path}")
            return None

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(
                f"Starting parallel processing at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

        df = pd.read_excel(xlsx_path)

        if "question" in df.columns:
            q_col = "question"
        elif "Question" in df.columns:
            q_col = "Question"
        else:
            raise ValueError("DbQA: Question column not found (question / Question).")

        choice_keys = []
        for letter in list("ABCDEFGHIJ"):
            candidates = [
                f"choice_{letter}",
                f"Choice_{letter}",
                f"option_{letter}",
                f"Option_{letter}",
                letter,
                letter.lower(),
            ]
            for c in candidates:
                if c in df.columns:
                    choice_keys.append((letter, c))
                    break
        if not choice_keys:
            raise ValueError("DbQA: No option columns were found.")

        for i, row in df.iterrows():
            question = row[q_col]
            choices = ""
            for letter, colname in choice_keys:
                val = row[colname]
                if not pd.isnull(val):
                    choices += f"\n{letter}. {val}"
            query = f"[single choice] {question} {choices}"
            query_list.append((i, query, save_path, run_ts, session_log_dir))

    else:
        print(f"Unsupported dataset: {dataset_name}")
        print(
            "Supported datasets: litqa, gpqa, trqa_db_short, trqa_lit_choice, trqa_lit_short, dbqa"
        )
        return None

    if use_indices and indices:
        query_list = [data for data in query_list if data[0] in indices]

    print(f"Found {len(query_list)} questions to process")
    print(f"Results will be saved to: {save_path}")
    print(f"Starting parallel processing with {num_processes} processes")

    with Pool(processes=num_processes) as pool:
        results = pool.map_async(process_query, query_list)
        results.wait()

    print("All queries processed!")

    with open(save_path, "a", encoding="utf-8") as f:
        f.write(f"\nAll processing completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"Results saved to: {save_path}")
    return save_path


def print_dataset_info():
    """Print information of available datasets"""
    print("\n=== Available Dataset Information ===")
    print("1. litqa - Literature Question Answering Dataset (Single Choice)")
    print("2. gpqa - General Science Question Answering (Single Choice)")
    print("3. trqa_db_short - TRQA Database Short Answer Questions (Short Answer)")
    print("4. trqa_lit_choice - TRQA Literature Multiple Choice Questions (Multiple Choice)")
    print("5. trqa_lit_short - TRQA Literature Short Answer Questions (Short Answer)")
    print("\nInput Format:")
    print("- Single Choice: [single choice] or [multiple choice] + Question + Options")
    print("- Short Answer: [short answer] + Question")
    print("\nOutput Format:")
    print("- question id: [ID]")
    print("- question: [Question Content]")
    print("- answer: [Model Answer]")
    print("- processing time: [Processing Time]")
    print("=" * 50)


if __name__ == "__main__":
    print_dataset_info()
    dataset_name = "trqa_lit_choice"
    save_name = "agent_answers_test.txt"
    num_processes = 6
    use_indices = False
    indices_path = None

    result_path = run_evaluation(
        dataset_name=dataset_name,
        save_name=save_name,
        num_processes=num_processes,
        use_indices=use_indices,
        indices_path=indices_path,
    )

    if result_path:
        print(f"Evaluation completed. Results in: {result_path}")

    print("\nHow to use other datasets:")
    print("Modify the dataset_name variable above to one of the following values:")
    print("- 'litqa' - Literature QA (Single Choice)")
    print("- 'gpqa' - General Science QA (Single Choice)")
    print("- 'trqa_db_short' - Database-related QA (Short Answer)")
    print("- 'trqa_lit_choice' - Literature-related QA (Multiple Choice)")
    print("- 'trqa_lit_short' - Literature-related QA (Short Answer)")
    print("- 'dbqa' - Database QA (Single Choice)")
