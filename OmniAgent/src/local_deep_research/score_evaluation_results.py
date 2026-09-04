#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OriAgent evaluation result scoring script - integrated solution

Features:
1. read the original output file of evaluate_local.py
2. built-in data conversion function
3. call Orieval system scoring
4. output final accuracy

using LLM: DeepSeek-V3 (through Volcano Engine API)

usage:
python score_evaluation_results.py \
    --agent_results benchmark/TRQA_lit_choice/agent_answers_test.txt \
    --original_data benchmark/TRQA_lit_choice/TRQA-lit-choice-172-coreset.csv \
    --model_name "OriAgent"
"""
import argparse
import asyncio
import json
import os
import re
from pathlib import Path
import tomllib
from .config import settings
import pandas as pd
from langchain_openai import ChatOpenAI

# dataset configuration
DATASET_CONFIGS = {
    # TRQA dataset
    "TRQA-lit-choice-172-coreset.csv": "choice",
    "TRQA-db-641.csv": "keyword",
    "TRQA-lit-short-answer-1108.csv": "keyword",

    # original dataset
    "LitQA2_250424.xlsx": "choice",
    "DbQA.csv": "choice",
    # keyword matching rules
    "choice": "choice",
    "short": "keyword",
    "lit_choice": "choice",
    "db_short": "keyword",
    "lit_short": "keyword",
    "qa": "keyword",
}
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH  = PROJECT_ROOT / "_settings" / ".secrets.toml"
with CONFIG_PATH.open("rb") as f:
    secrets = tomllib.load(f)
endpoint_openai_api_base_url   = secrets["openai"]["api_base"]
endpoint_openai_api_key        = secrets["openai"]["api_key"]
LLM_CONFIG = {
    "base_url": endpoint_openai_api_base_url ,
    "model": "o4-mini",  
    "api_key": endpoint_openai_api_key,  
}


def get_question_type(original_data_path):
    """get question type according to configuration"""
    filename = Path(original_data_path).name

    # exact match
    if filename in DATASET_CONFIGS:
        question_type = DATASET_CONFIGS[filename]
        print(f"🎯 exact match: {filename} -> {question_type}")
        return question_type

    # keyword match
    filename_lower = filename.lower()
    for keyword, qtype in DATASET_CONFIGS.items():
        if keyword in filename_lower:
            print(f"🎯 keyword match: {filename} contains '{keyword}' -> {qtype}")
            return qtype

    # default to short answer
    print(f"⚠️  using default type: {filename} -> keyword")
    return "keyword"


def parse_agent_results(file_path):
    """parse agent result file"""
    questions = []
    answers = []

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # split by question
    question_blocks = re.split(r"question id: \d+", content)[1:]

    for block in question_blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue

        question_line = None
        answer_start = -1

        for i, line in enumerate(lines):
            if line.startswith("question:"):
                question_line = line[10:].strip()
            elif line.startswith("answer:"):
                answer_start = i
                break

        if question_line and answer_start >= 0:
            answer_lines = []
            for i in range(answer_start + 1, len(lines)):
                if lines[i].startswith("processing time:"):
                    break
                answer_lines.append(lines[i])

            answer = "\n".join(answer_lines).strip()
            questions.append(question_line)
            answers.append(answer)

    return questions, answers


def load_original_data(file_path, dataset_type):
    """load original data"""
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("unsupported file format")

    questions = []
    answers = []

    if dataset_type == "choice" and "Options" in df.columns:
        # process choice question format
        for _, row in df.iterrows():
            question = row["Question"]
            options_str = row["Options"]
            answer = row["Answer"]

            try:
                options = json.loads(options_str)
                choices = ""
                for key, value in options.items():
                    choices += f"\n{key}. {value}"
                formatted_question = f"[multiple choice] {question} {choices}"
                questions.append(formatted_question)
                answers.append(answer)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse options: {options_str}")
                continue
    else:
        # process short answer or general format
        for _, row in df.iterrows():
            question = row["Question"] if "Question" in df.columns else row.iloc[0]
            answer = row["Answer"] if "Answer" in df.columns else row.iloc[1]

            if dataset_type == "choice":
                formatted_question = f"[multiple choice] {question}"
            else:
                formatted_question = f"[short answer] {question}"

            questions.append(formatted_question)
            answers.append(answer)

    return questions, answers


def create_evaluation_data(
    agent_questions, agent_answers, original_questions, original_answers, model_name
):
    """create evaluation data"""
    matched_data = []

    for i, agent_q in enumerate(agent_questions):
        if i < len(agent_answers):
            # find matching original question
            original_answer = "Unknown"

            for j, orig_q in enumerate(original_questions):
                agent_core = re.sub(
                    r"\[(single choice|multiple choice|short answer)\]\s*", "", agent_q
                )
                orig_core = re.sub(
                    r"\[(single choice|multiple choice|short answer)\]\s*", "", orig_q
                )

                if (
                    agent_core.strip() in orig_core.strip()
                    or orig_core.strip() in agent_core.strip()
                ):
                    if j < len(original_answers):
                        original_answer = original_answers[j]
                    break

            matched_data.append(
                {
                    "Question": agent_q,
                    "Answer": original_answer,
                    f"{model_name}_Answer": agent_answers[i],
                }
            )

    return pd.DataFrame(matched_data)


async def invoke_with_timeout_and_retry(
    llm, messages, timeout=90.0, max_retries=3, retry_delay=60.0
):
    """async call LLM, with timeout and retry"""
    for attempt in range(max_retries):
        try:
            response = await asyncio.wait_for(llm.ainvoke(messages), timeout=timeout)
            return response
        except asyncio.TimeoutError:
            print(f"Attempt {attempt + 1}/{max_retries} timed out")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            else:
                raise Exception(f"Failed after {max_retries} attempts: {e}")
    raise Exception("Failed to get response after multiple attempts")


async def evaluate_choice_question(question, correct_answer, user_answer, llm):
    """evaluate choice question"""
    try:
        # extract options from user answer
        extract_messages = [
            {
                "role": "system",
                "content": """You are an expert at analyzing multiple-choice question responses.
Extract which option(s) the user has chosen from their answer.
Respond ONLY with the letter(s) of the chosen option(s) (e.g., "A", "B", "C", "D").
Do not include any explanations.""",
            },
            {
                "role": "user",
                "content": f"""Question: {question}
User's answer: {user_answer}

Extract ONLY the letter(s) of the option(s) chosen. Reply with just the letter(s) like "A" or "A,B".""",
            },
        ]

        extract_output = await invoke_with_timeout_and_retry(
            llm, extract_messages, timeout=30.0
        )
        extracted_choice = extract_output.content.strip().upper()

        # judge if correct
        judge_messages = [
            {
                "role": "system",
                "content": """Compare the user's selected option(s) with the correct answer(s).
Respond ONLY with "Correct" or "Incorrect".""",
            },
            {
                "role": "user",
                "content": f"""Correct answer: {correct_answer}
User's choice: {extracted_choice}

Are they exactly matching? Reply ONLY with "Correct" or "Incorrect".""",
            },
        ]

        judge_output = await invoke_with_timeout_and_retry(
            llm, judge_messages, timeout=30.0
        )
        judge_result = judge_output.content.strip()

        if (
            "correct" in judge_result.lower()
            and "incorrect" not in judge_result.lower()
        ):
            return 1, extracted_choice
        else:
            return 0, extracted_choice

    except Exception as e:
        print(f"Error evaluating choice question: {e}")
        return 0, "Error"


def extract_json_from_string(text):
    """extract JSON from string"""
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            return json.loads(text)
    except Exception as e:
        return {
            "evaluation": [],
            "summary": {
                "total_keywords": 0,
                "covered_keywords": 0,
                "coverage_percentage": 0,
                "overall_assessment": f"Failed to parse: {str(e)}",
            },
        }


async def extract_keywords(question, answer, llm):
    """extract keywords from standard answer"""
    try:
        messages = [
            {
                "role": "system",
                "content": """Extract key terms/concepts from the standard answer for evaluation.
Output them as a Python list format: ["term1", "term2", "term3"]""",
            },
            {
                "role": "user",
                "content": f"""Question: {question}
Standard Answer: {answer}

Extract key terms and concepts as a Python list.""",
            },
        ]

        output = await invoke_with_timeout_and_retry(llm, messages)

        # extract list
        list_pattern = r"\[.*?\]"
        list_match = re.search(list_pattern, output.content, re.DOTALL)

        if list_match:
            list_str = list_match.group()
            try:
                import ast

                result = ast.literal_eval(list_str)
                if isinstance(result, list):
                    return result
            except:
                pass

        return ["extraction_error"]

    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return ["extraction_error"]


async def evaluate_keyword_coverage(question, keywords, user_answer, llm):
    """evaluate keyword coverage"""
    try:
        if isinstance(keywords, str):
            try:
                keywords_list = json.loads(keywords)
                if not isinstance(keywords_list, list):
                    keywords_list = [k.strip() for k in keywords.split(",")]
            except:
                keywords_list = [k.strip() for k in keywords.split(",")]
        else:
            keywords_list = keywords

        messages = [
            {
                "role": "system",
                "content": """Evaluate how well an answer covers key medical concepts.
Return a JSON object with this format:
{
    "evaluation": [
        {"keyword": "concept1", "covered": true/false, "explanation": "explanation"},
        ...
    ],
    "summary": {
        "total_keywords": number,
        "covered_keywords": number,
        "coverage_percentage": number,
        "overall_assessment": "assessment"
    }
}""",
            },
            {
                "role": "user",
                "content": f"""Question: {question}

Key Concepts: {", ".join(keywords_list)}

Answer to Evaluate: {user_answer}

Assess coverage of each concept.""",
            },
        ]

        response = await invoke_with_timeout_and_retry(llm, messages)
        coverage_data = extract_json_from_string(response.content)

        if "evaluation" not in coverage_data or "summary" not in coverage_data:
            evaluation_list = []
            for keyword in keywords_list:
                evaluation_list.append(
                    {
                        "keyword": keyword,
                        "covered": False,
                        "explanation": "Evaluation format error",
                    }
                )

            coverage_data = {
                "evaluation": evaluation_list,
                "summary": {
                    "total_keywords": len(keywords_list),
                    "covered_keywords": 0,
                    "coverage_percentage": 0,
                    "overall_assessment": "Format error",
                },
            }

        # ensure coverage percentage
        if "coverage_percentage" not in coverage_data["summary"]:
            covered_count = sum(
                1 for item in coverage_data["evaluation"] if item.get("covered", False)
            )
            total_count = len(coverage_data["evaluation"])
            coverage_data["summary"]["coverage_percentage"] = (
                (covered_count / total_count * 100) if total_count > 0 else 0
            )

        recall = coverage_data["summary"].get("coverage_percentage", 0) / 100

        return {
            "coverage_percentage": coverage_data["summary"]["coverage_percentage"],
            "recall": recall,
            "total_keywords": coverage_data["summary"]["total_keywords"],
            "covered_keywords": coverage_data["summary"]["covered_keywords"],
        }

    except Exception as e:
        print(f"Error evaluating keyword coverage: {e}")
        return {
            "coverage_percentage": 0,
            "recall": 0,
            "total_keywords": len(keywords_list) if "keywords_list" in locals() else 0,
            "covered_keywords": 0,
        }


async def evaluate_questions(df, question_type, model_name, dataset_name, concurrent):
    llm = ChatOpenAI(**LLM_CONFIG)

    total_questions = len(df)
    print(f"\n📊 begin to evaluate {total_questions} questions...")

    
    semaphore = asyncio.Semaphore(concurrent)  

    if question_type == "choice":
        async def evaluate_single_choice(row, index):
            async with semaphore:
                print(f"evaluating progress: {index + 1}/{total_questions}", end="\r")
                score, extracted = await evaluate_choice_question(
                    row["Question"], row["Answer"], row[f"{model_name}_Answer"], llm
                )
                return index, {
                    "Question": row["Question"],
                    "Answer": row["Answer"],
                    f"{model_name}_Answer": row[f"{model_name}_Answer"],
                    "Evaluation": score,
                    "Extracted_Answer": extracted,
                }


        tasks = [evaluate_single_choice(row, i) for i, row in df.iterrows()]

       
        results_with_index = await asyncio.gather(*tasks)

        
        results_with_index.sort(key=lambda x: x[0])
        results = [result for _, result in results_with_index]

        
        correct_count = sum(1 for r in results if r["Evaluation"] == 1)
        accuracy = (correct_count / total_questions * 100) if total_questions > 0 else 0

        print("\n\n" + "=" * 80)
        print(f"📊 【{dataset_name}】choice question evaluation result - model: {model_name}")
        print("=" * 80)
        print(f"   dataset: {dataset_name}")
        print("   type: choice (Multiple Choice)")
        print(f"   total questions: {total_questions}")
        print(f"   correct questions: {correct_count}")
        print(f"   accuracy: {accuracy:.2f}%")
        print(f"   concurrent evaluation: ✅ (max {concurrent} concurrent)")
        print("=" * 80)

        return results, {
            "dataset": dataset_name,
            "model": model_name,
            "type": "choice",
            "accuracy": accuracy,
            "correct": correct_count,
            "total": total_questions,
        }

    else:

        async def evaluate_single_keyword(row, index):
            async with semaphore:
                print(f"evaluating progress: {index + 1}/{total_questions} (extract keywords)", end="\r")

                keywords = await extract_keywords(row["Question"], row["Answer"], llm)

                print(f"evaluating progress: {index + 1}/{total_questions} (evaluate coverage)", end="\r")

                coverage_result = await evaluate_keyword_coverage(
                    row["Question"], keywords, row[f"{model_name}_Answer"], llm
                )

                return index, {
                    "Question": row["Question"],
                    "Answer": row["Answer"],
                    f"{model_name}_Answer": row[f"{model_name}_Answer"],
                    "Keywords": ", ".join(keywords)
                    if isinstance(keywords, list)
                    else str(keywords),
                    "Coverage_Percentage": coverage_result["coverage_percentage"],
                    "Recall": coverage_result["recall"],
                    "Total_Keywords": coverage_result["total_keywords"],
                    "Covered_Keywords": coverage_result["covered_keywords"],
                }

        tasks = [evaluate_single_keyword(row, i) for i, row in df.iterrows()]

        results_with_index = await asyncio.gather(*tasks)

        results_with_index.sort(key=lambda x: x[0])
        results = [result for _, result in results_with_index]

        avg_coverage = (
            sum(r["Coverage_Percentage"] for r in results) / len(results)
            if results
            else 0
        )
        avg_recall = sum(r["Recall"] for r in results) / len(results) if results else 0

        print("\n\n" + "=" * 80)
        print(f"📊 【{dataset_name}】short answer evaluation result - model: {model_name}")
        print("=" * 80)
        print(f"   dataset: {dataset_name}")
        print("   type: short answer (Short Answer)")
        print(f"   total questions: {total_questions}")
        print(f"   average keyword coverage: {avg_coverage:.2f}%")
        print(f"   average recall: {avg_recall:.3f}")
        print(f"   concurrent evaluation: ✅ (max {concurrent} concurrent)")
        print("=" * 80)

        return results, {
            "dataset": dataset_name,
            "model": model_name,
            "type": "keyword",
            "avg_coverage": avg_coverage,
            "avg_recall": avg_recall,
            "total": total_questions,
        }


def main():
    parser = argparse.ArgumentParser(description="OriAgent evaluation result scoring")
    parser.add_argument("--agent_results", required=True, help="Agent evaluation result file path")
    parser.add_argument("--original_data", required=True, help="original data file path")
    parser.add_argument("--model_name", default="OriAgent", help="model name")
    parser.add_argument(
        "--concurrent", type=int, default=10, help="concurrent requests (default: 10)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.agent_results):
        print(f"❌ Agent result file not found: {args.agent_results}")
        return

    if not os.path.exists(args.original_data):
        print(f"❌ original data file not found: {args.original_data}")
        return

   
    question_type = get_question_type(args.original_data)
    dataset_name = Path(args.original_data).stem  

    print("=" * 80)
    print("OriAgent evaluation result scoring system")
    print("=" * 80)
    print(f"🤖 model name: {args.model_name}")
    print(f"📊 dataset: {dataset_name}")
    print(f"📂 agent result: {args.agent_results}")
    print(f"📂 original data: {args.original_data}")
    print(f"📋 question type: {question_type}")
    print("🔧 using LLM: DeepSeek-V3")
    print(f"⚡ concurrent: {args.concurrent}")

    try:
        # parse data
        print("\n🔄 step 1: parse agent evaluation result...")
        agent_questions, agent_answers = parse_agent_results(args.agent_results)
        print(f"✅ parsed {len(agent_questions)} agent answers")

        print("\n🔄 step 2: load original data")
        original_questions, original_answers = load_original_data(
            args.original_data, question_type
        )
        print(f"✅ loaded {len(original_questions)} original questions")

        print("\n🔄 step 3: create evaluation data...")
        df = create_evaluation_data(
            agent_questions,
            agent_answers,
            original_questions,
            original_answers,
            args.model_name,
        )
        print(f"✅ matched {len(df)} questions")

        print("\n🔄 step 4: begin LLM evaluation...")
        results, summary = asyncio.run(
            evaluate_questions(
                df, question_type, args.model_name, dataset_name, args.concurrent
            )
        )

        # save results
        output_dir = f"evaluation_results_{dataset_name}_{args.model_name}"
        os.makedirs(output_dir, exist_ok=True)

        results_df = pd.DataFrame(results)
        results_file = f"{output_dir}/detailed_results_{dataset_name}.xlsx"
        print("results file path:", results_file)
        print("directory exists:", os.path.exists(os.path.dirname(results_file)))
        print(
            "remaining space:",
            os.statvfs(os.path.dirname(results_file)).f_bavail
            * os.statvfs(os.path.dirname(results_file)).f_frsize
            // 1024
            // 1024,
            "MB",
        )
        print("write permission:", os.access(os.path.dirname(results_file), os.W_OK))
        results_df.to_excel(results_file, index=False)

        summary_file = f"{output_dir}/summary_{dataset_name}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 【{dataset_name}】evaluation completed!")
        print(f"📄 detailed results: {results_file}")
        print(f"📊 summary: {summary_file}")

        # generate summary report file
        report_file = f"{output_dir}/REPORT_{dataset_name}_{args.model_name}.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("OriAgent evaluation report\n")
            f.write("=" * 50 + "\n")
            f.write(f"dataset: {dataset_name}\n")
            f.write(f"model: {args.model_name}\n")
            f.write(f"type: {'choice' if question_type == 'choice' else 'short'}\n")
            f.write(f"concurrent: {args.concurrent}\n")
            f.write(f"evaluation time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n")

            if question_type == "choice":
                f.write(f"total questions: {summary['total']}\n")
                f.write(f"correct questions: {summary['correct']}\n")
                f.write(f"accuracy: {summary['accuracy']:.2f}%\n")
            else:
                f.write(f"total questions: {summary['total']}\n")
                f.write(f"average keyword coverage: {summary['avg_coverage']:.2f}%\n")
                f.write(f"average recall: {summary['avg_recall']:.3f}\n")

        print(f"📋 evaluation report: {report_file}")

    except Exception as e:
        print(f"❌ error during evaluation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
