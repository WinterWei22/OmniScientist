from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Protocol

from .contracts import (
    A1TaskResult,
    AnalysisResult,
    ExperimentPlan,
    ResearchState,
    WorkflowEvaluation,
)
from .result_payload import has_material_result


class WorkflowEvaluator(Protocol):
    def evaluate(
        self,
        state: ResearchState,
        plan: ExperimentPlan,
        executions: list[A1TaskResult],
        analysis: AnalysisResult,
    ) -> WorkflowEvaluation: ...


class BenchmarkWorkflowEvaluator:
    """Deterministic Harness feedback; this does not grade hidden benchmark truth."""

    def evaluate(
        self,
        state: ResearchState,
        plan: ExperimentPlan,
        executions: list[A1TaskResult],
        analysis: AnalysisResult,
    ) -> WorkflowEvaluation:
        task_id = str(state.task_manifest.get("id", "")).lower()
        benchmark = str(
            state.task_manifest.get("task_parameters", {}).get("benchmark", "")
        ).lower()
        if task_id.startswith("smdd_") or benchmark == "smdd":
            return self._evaluate_smdd(state, executions, analysis)
        if benchmark == "drugdiscoverybench":
            return self._evaluate_drug_discovery(state, executions, analysis)
        if benchmark == "kg_link_prediction":
            return self._evaluate_kg_link_prediction(state, executions, analysis)
        return self._evaluate_generic(executions, analysis)

    @staticmethod
    def _process_metrics(
        executions: list[A1TaskResult], analysis: AnalysisResult
    ) -> dict[str, float]:
        total = len(executions)
        successful = sum(1 for item in executions if item.success)
        return {
            "execution_success_rate": successful / total if total else 0.0,
            "observation_count": float(len(analysis.observations)),
            "numeric_metric_count": float(len(analysis.metrics)),
            "execution_error_count": float(
                sum(len(item.errors) for item in executions)
            ),
        }

    def _evaluate_generic(
        self, executions: list[A1TaskResult], analysis: AnalysisResult
    ) -> WorkflowEvaluation:
        metrics = self._process_metrics(executions, analysis)
        observation_present = float(metrics["observation_count"] > 0)
        metric_present = float(metrics["numeric_metric_count"] > 0)
        score = (
            0.6 * metrics["execution_success_rate"]
            + 0.2 * observation_present
            + 0.2 * metric_present
        )
        failed = []
        if metrics["execution_success_rate"] < 1.0:
            failed.append("Not every planned workflow step executed successfully.")
        if not observation_present:
            failed.append("No structured observation was produced.")
        if not metric_present:
            failed.append("No deterministic numeric metric was produced.")
        return WorkflowEvaluation(
            evaluator_id="generic_workflow_v1",
            score=score,
            metrics=metrics,
            satisfied_criteria=[] if failed else ["Workflow result is structured."],
            failed_criteria=failed,
            evidence=[{"analysis_summary": analysis.summary}],
            errors=[
                error for result in executions for error in result.errors
            ],
            retryable=True,
        )

    def _evaluate_kg_link_prediction(
        self,
        state: ResearchState,
        executions: list[A1TaskResult],
        analysis: AnalysisResult,
    ) -> WorkflowEvaluation:
        metrics = self._process_metrics(executions, analysis)
        evidence_records = list(state.scientific_state.evidence.values())
        has_kg_mcp_evidence = any(
            record.source_backend == "mcp"
            and "knowledge_graph" in record.source_capability_id
            for record in evidence_records
        )
        has_verified_prediction = any(
            "biomedical_kg_link_prediction"
            in json.dumps(record.payload, ensure_ascii=False, default=str)
            for record in evidence_records
        )
        metrics.update(
            {
                "kg_mcp_evidence_present": float(has_kg_mcp_evidence),
                "verified_link_prediction_present": float(has_verified_prediction),
            }
        )
        failed: list[str] = []
        if not has_kg_mcp_evidence:
            failed.append("Collect verified KG structural evidence through MCP first.")
        if not has_verified_prediction:
            failed.append(
                "Use the verified KG evidence in a later A1 relation-prediction round."
            )
        return WorkflowEvaluation(
            evaluator_id="kg_link_prediction_v1",
            score=(1.0 if has_verified_prediction and has_kg_mcp_evidence else 0.45 if has_kg_mcp_evidence else 0.0),
            metrics=metrics,
            satisfied_criteria=(
                ["Masked KG link prediction has MCP evidence and passed the local verifier."]
                if not failed
                else []
            ),
            failed_criteria=failed,
            evidence=[{"analysis_summary": analysis.summary}],
            errors=[error for result in executions for error in result.errors],
            retryable=True,
        )

    def _evaluate_smdd(
        self,
        state: ResearchState,
        executions: list[A1TaskResult],
        analysis: AnalysisResult,
    ) -> WorkflowEvaluation:
        workspace = Path(state.workspace).resolve()
        output_name = str(
            state.task_manifest.get("output_config", {}).get(
                "file_path", "solution.py"
            )
        )
        solution = (workspace / output_name).resolve()
        metrics = self._process_metrics(executions, analysis)
        metrics.update(
            {
                "output_exists": float(solution.is_file()),
                "contract_valid": 0.0,
                "sensitivity": 0.0,
                "specificity": 0.0,
                "balanced_accuracy": 0.0,
            }
        )
        failed: list[str] = []
        errors: list[str] = [
            error for result in executions for error in result.errors
        ]

        if not solution.is_relative_to(workspace):
            failed.append("Configured output path escapes the isolated workspace.")
        elif not solution.is_file():
            failed.append(f"Required output artifact is missing: {output_name}")
        else:
            try:
                result = self._run_smdd_public_evaluation(
                    solution,
                    workspace / "actives.smi",
                    workspace / "inactives.smi",
                    workspace,
                )
                metrics.update(
                    {
                        "contract_valid": 1.0,
                        "sensitivity": float(result["sensitivity"]),
                        "specificity": float(result["specificity"]),
                        "balanced_accuracy": float(result["balanced_accuracy"]),
                        "true_positive": float(result["true_positive"]),
                        "true_negative": float(result["true_negative"]),
                        "false_positive": float(result["false_positive"]),
                        "false_negative": float(result["false_negative"]),
                    }
                )
            except Exception as exc:
                failed.append(
                    "solution.py could not be evaluated through the required function."
                )
                errors.append(f"{type(exc).__name__}: {exc}")

        if metrics["contract_valid"] < 1.0:
            failed.append(
                "check_pharmacophore(smiles: str) -> bool is not executable."
            )
        if metrics["balanced_accuracy"] < 0.8:
            failed.append(
                "Public-example balanced accuracy is below the 0.80 workflow target."
            )
        if metrics["execution_success_rate"] < 1.0:
            failed.append("The A1 workflow step did not complete successfully.")

        score = (
            0.1 * metrics["execution_success_rate"]
            + 0.1 * float(metrics["observation_count"] > 0)
            + 0.1 * metrics["contract_valid"]
            + 0.7 * metrics["balanced_accuracy"]
        )
        satisfied = []
        if metrics["output_exists"]:
            satisfied.append("The required solution.py artifact exists.")
        if metrics["contract_valid"]:
            satisfied.append("The required callable contract is executable.")
        if metrics["balanced_accuracy"] >= 0.8:
            satisfied.append(
                "Public-example balanced accuracy reaches the workflow target."
            )
        return WorkflowEvaluation(
            evaluator_id="smdd_public_workflow_v1",
            score=min(1.0, max(0.0, score)),
            metrics=metrics,
            satisfied_criteria=satisfied,
            failed_criteria=list(dict.fromkeys(failed)),
            evidence=[
                {
                    "workspace_output": output_name,
                    "evaluation_visibility": "public_inputs_only",
                }
            ],
            errors=errors,
            retryable=True,
        )

    @staticmethod
    def _run_smdd_public_evaluation(
        solution: Path,
        actives: Path,
        inactives: Path,
        workspace: Path,
    ) -> dict[str, float]:
        runner = """
import importlib.util
import json
import pathlib
import sys

solution_path = pathlib.Path(sys.argv[1])
actives_path = pathlib.Path(sys.argv[2])
inactives_path = pathlib.Path(sys.argv[3])
spec = importlib.util.spec_from_file_location("candidate_solution", solution_path)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load candidate solution")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
check = getattr(module, "check_pharmacophore", None)
if not callable(check):
    raise TypeError("check_pharmacophore is missing or not callable")

def load_smiles(path):
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            values.append(text.split()[0])
    return values

active_values = load_smiles(actives_path)
inactive_values = load_smiles(inactives_path)
active_predictions = [bool(check(item)) for item in active_values]
inactive_predictions = [bool(check(item)) for item in inactive_values]
tp = sum(active_predictions)
fn = len(active_predictions) - tp
fp = sum(inactive_predictions)
tn = len(inactive_predictions) - fp
sensitivity = tp / len(active_predictions) if active_predictions else 0.0
specificity = tn / len(inactive_predictions) if inactive_predictions else 0.0
payload = {
    "true_positive": tp,
    "true_negative": tn,
    "false_positive": fp,
    "false_negative": fn,
    "sensitivity": sensitivity,
    "specificity": specificity,
    "balanced_accuracy": (sensitivity + specificity) / 2.0,
}
print("__OMNIAGENT_EVAL__" + json.dumps(payload, sort_keys=True))
"""
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                runner,
                str(solution),
                str(actives),
                str(inactives),
            ],
            cwd=workspace,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        marker = "__OMNIAGENT_EVAL__"
        line = next(
            (
                item[len(marker) :]
                for item in reversed(completed.stdout.splitlines())
                if item.startswith(marker)
            ),
            "",
        )
        if completed.returncode != 0 or not line:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(detail[-2000:] or "candidate evaluation failed")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError("candidate evaluator returned a non-object")
        return value

    def _evaluate_drug_discovery(
        self,
        state: ResearchState,
        executions: list[A1TaskResult],
        analysis: AnalysisResult,
    ) -> WorkflowEvaluation:
        workspace = Path(state.workspace).resolve()
        output_name = str(
            state.task_manifest.get("output_config", {}).get(
                "file_path", "final_submission.json"
            )
        )
        output = (workspace / output_name).resolve()
        metrics = self._process_metrics(executions, analysis)
        metrics.update(
            {
                "output_exists": float(output.is_file()),
                "contract_valid": 0.0,
                "evidence_complete": 0.0,
                "terminal_contract": 0.0,
            }
        )
        failed: list[str] = []
        errors = [error for result in executions for error in result.errors]

        if not output.is_relative_to(workspace):
            failed.append("Configured output path escapes the isolated workspace.")
        elif not output.is_file():
            failed.append(f"Required output artifact is missing: {output_name}")
        else:
            try:
                payload = json.loads(output.read_text(encoding="utf-8"))
                metrics["evidence_complete"] = float(
                    self._validate_drug_submission(payload)
                )
                metrics["contract_valid"] = 1.0
                if isinstance(payload, dict) and payload.get("status") == "INSUFFICIENT_EVIDENCE":
                    metrics["terminal_contract"] = 1.0
            except Exception as exc:
                failed.append("The DrugDiscoveryBench output contract is invalid.")
                errors.append(f"{type(exc).__name__}: {exc}")

        if metrics["execution_success_rate"] < 1.0:
            failed.append("The A1 workflow step did not complete successfully.")
        if metrics["observation_count"] < 1.0:
            failed.append("No structured scientific observation was returned.")
        if metrics["contract_valid"] < 1.0:
            failed.append(
                "Return a valid structured task result through Analyzer final_output, or "
                "an explicit INSUFFICIENT_EVIDENCE result. The Harness materializes the "
                "final artifact."
            )

        score = (
            0.15 * metrics["execution_success_rate"]
            + 0.15 * float(metrics["observation_count"] > 0)
            + 0.35 * metrics["contract_valid"]
            + 0.35 * metrics["evidence_complete"]
        )
        satisfied = []
        if metrics["execution_success_rate"] == 1.0:
            satisfied.append("The bounded A1 workflow step completed.")
        if metrics["observation_count"] > 0:
            satisfied.append("A structured observation is available for replanning.")
        if metrics["contract_valid"]:
            satisfied.append("The public task-specific result contract is valid.")
        if metrics["evidence_complete"]:
            satisfied.append("The final submission contains a complete task-specific result.")
        return WorkflowEvaluation(
            evaluator_id="drugdiscoverybench_harness_process_v2",
            score=min(1.0, max(0.0, score)),
            metrics=metrics,
            satisfied_criteria=satisfied,
            failed_criteria=list(dict.fromkeys(failed)),
            evidence=[
                {
                    "workspace_output": output_name,
                    "evaluation_visibility": "public_contract_only",
                }
            ],
            errors=errors,
            retryable=metrics["terminal_contract"] < 1.0,
        )

    @staticmethod
    def _validate_drug_submission(payload: object) -> bool:
        if not isinstance(payload, dict):
            raise TypeError("final submission must be a JSON object")
        if payload.get("status") == "INSUFFICIENT_EVIDENCE":
            if not str(payload.get("reason", "")).strip():
                raise ValueError("INSUFFICIENT_EVIDENCE requires a concise reason")
            if payload.get("metabolism_table"):
                raise ValueError(
                    "INSUFFICIENT_EVIDENCE must not include a metabolism table"
                )
            return False
        if "answer" in payload:
            answer = payload.get("answer")
            if answer is None or (isinstance(answer, str) and not answer.strip()):
                raise ValueError("answer must be non-empty when provided")
            return True
        rows = payload.get("metabolism_table")
        if rows is not None:
            if not isinstance(rows, list) or not rows:
                raise ValueError("metabolism_table must be a non-empty list")
            for row in rows:
                if not isinstance(row, dict) or not str(row.get("compound", "")).strip():
                    raise ValueError("every row requires a compound")
                value = row.get("cyp3a4_metabolism_percentage")
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError("every row requires a numeric metabolism percentage")
                if not 0.0 <= float(value) <= 100.0:
                    raise ValueError("metabolism percentage must lie in [0, 100]")
            return True
        if has_material_result(payload):
            return True
        raise ValueError("final submission requires a non-empty task-specific result field")
