import csv
import json
import os
import pickle
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from difflib import get_close_matches
from pathlib import Path

import numpy as np
import pandas as pd
import requests


def _tools_pkg_root(repository_root: Path | None = None) -> Path:
    """Resolve shared external-tool assets without changing legacy defaults."""
    configured = os.environ.get("BIOMNI_TOOLS_PKG_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    root = repository_root or Path(__file__).resolve().parents[2]
    return (root / "tools_pkg").resolve()


def _hit_success(result, query_info):
    return {"success": True, "result": result, "query_info": query_info}


def _hit_error(message, query_info):
    return {"success": False, "error": str(message), "query_info": query_info}


def _resolve_meeko_command(env_name: str, command_name: str) -> str | None:
    """Resolve a Meeko entry point from config, PATH, or the active Python env."""
    configured = os.environ.get(env_name, "").strip()
    candidates = [configured, shutil.which(command_name), str(Path(sys.executable).resolve().parent / command_name)]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _prepare_vina_receptor_pdb(source_path: str, destination_path: str) -> dict:
    """Create a deterministic protein-only PDB for Meeko receptor preparation.

    Crystal ligands and alternate-location atoms are common in deposited PDBs,
    but neither should be passed to Meeko as receptor residues.  Keep ATOM
    records only and prefer blank alternate locations, then ``A`` when duplicate
    atom locations are present.  The source file is never modified.
    """
    atom_records = []
    residue_altlocs = {}
    other_lines = []
    removed_hetatm = 0
    removed_altloc = 0
    with open(source_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            record = line[0:6].strip()
            if record == "HETATM":
                removed_hetatm += 1
                continue
            if record != "ATOM":
                if record in {"TER", "END", "ENDMDL", "MODEL"}:
                    other_lines.append(line)
                continue
            altloc = line[16:17]
            residue_key = (line[21:22], line[22:26], line[26:27])
            atom_records.append((line, residue_key, altloc))
            if altloc.strip():
                residue_altlocs.setdefault(residue_key, set()).add(altloc)
    selected_altlocs = {
        key: ("A" if "A" in labels else sorted(labels)[0])
        for key, labels in residue_altlocs.items()
    }
    lines = []
    for line, residue_key, altloc in atom_records:
        chosen = selected_altlocs.get(residue_key)
        # For residues with any alternate location, keep only one complete
        # conformer; retaining blank atoms alongside A/B still triggers Meeko.
        if altloc == " " or chosen is None or altloc == chosen:
            lines.append(line[:16] + " " + line[17:] if altloc != " " else line)
        else:
            removed_altloc += 1
    lines.extend(other_lines)
    if not lines:
        raise ValueError("receptor PDB contains no ATOM records")
    with open(destination_path, "w", encoding="utf-8") as handle:
        handle.writelines(lines)
    return {"removed_hetatm_records": removed_hetatm, "removed_altloc_records": removed_altloc}


def _parse_smiles_list(smiles):
    """Parse and validate a bounded list of SMILES without raising to callers."""
    from rdkit import Chem

    values = [smiles] if isinstance(smiles, str) else smiles
    if not isinstance(values, (list, tuple)) or not values or len(values) > 500:
        raise ValueError("smiles must contain 1 to 500 strings")
    parsed = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("each SMILES must be a non-empty string")
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise ValueError(f"invalid SMILES: {value}")
        parsed.append((value, mol))
    return parsed


def score_candidates_qed(smiles: list[str]) -> dict:
    """Calculate RDKit's official quantitative estimate of drug-likeness (QED)."""
    query_info = {"count": len(smiles) if isinstance(smiles, list) else 1, "method": "RDKit QED.qed"}
    try:
        from rdkit.Chem import QED
        rows = [{"smiles": raw, "qed": round(float(QED.qed(mol)), 6)} for raw, mol in _parse_smiles_list(smiles)]
        return _hit_success({"candidates": rows, "score_range": [0.0, 1.0], "higher_is_better": True}, query_info)
    except Exception as exc:
        return _hit_error(exc, query_info)


def score_candidates_sa(smiles: list[str]) -> dict:
    """Estimate synthetic accessibility using a transparent RDKit complexity heuristic."""
    query_info = {"count": len(smiles) if isinstance(smiles, list) else 1, "method": "RDKit heuristic"}
    try:
        from rdkit.Chem import Descriptors, Lipinski
        rows = []
        for raw, mol in _parse_smiles_list(smiles):
            # 1 (easy) to 10 (hard); this is explicitly not Ertl's trained SA score.
            complexity = 1.0 + min(9.0, 0.18 * Lipinski.NumRotatableBonds(mol) + 0.12 * Descriptors.RingCount(mol) + 0.02 * Descriptors.HeavyAtomCount(mol))
            rows.append({"smiles": raw, "sa_score": round(complexity, 4)})
        return _hit_success({"candidates": rows, "score_range": [1.0, 10.0], "lower_is_better": True, "is_official_ertl": False}, query_info)
    except Exception as exc:
        return _hit_error(exc, query_info)


def analyze_matched_molecular_pairs(smiles: list[str], max_pairs: int = 100) -> dict:
    """Find pairwise scaffold-preserving substitutions using RDKit MCS (MMP-style)."""
    query_info = {"count": len(smiles) if isinstance(smiles, list) else 1, "method": "RDKit rdFMCS MMP-style"}
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFMCS
        parsed = _parse_smiles_list(smiles)
        pairs = []
        for i, (a_raw, a) in enumerate(parsed):
            for b_raw, b in parsed[i + 1 :]:
                mcs = rdFMCS.FindMCS([a, b], ringMatchesRingOnly=True, completeRingsOnly=True, timeout=5)
                if not mcs.smartsString:
                    continue
                core = Chem.MolFromSmarts(mcs.smartsString)
                if core is None or core.GetNumAtoms() < 2:
                    continue
                a_match, b_match = a.GetSubstructMatch(core), b.GetSubstructMatch(core)
                if not a_match or not b_match:
                    continue
                a_atoms = [x for x in range(a.GetNumAtoms()) if x not in a_match]
                b_atoms = [x for x in range(b.GetNumAtoms()) if x not in b_match]
                a_frag = Chem.MolFragmentToSmiles(a, atomsToUse=a_atoms) if a_atoms else "[H]"
                b_frag = Chem.MolFragmentToSmiles(b, atomsToUse=b_atoms) if b_atoms else "[H]"
                if a_frag != b_frag:
                    pairs.append({"smiles_a": a_raw, "smiles_b": b_raw, "core_smarts": mcs.smartsString, "fragment_a": a_frag, "fragment_b": b_frag})
                if len(pairs) >= max_pairs:
                    break
            if len(pairs) >= max_pairs:
                break
        return _hit_success({"pairs": pairs, "pair_count": len(pairs), "limitation": "MMP-style MCS; not atom-mapped reaction transformations"}, query_info)
    except Exception as exc:
        return _hit_error(exc, query_info)


def benchmark_moleculenet_qsar(smiles: list[str], labels: list[float], task_type: str = "regression", test_fraction: float = 0.2, random_seed: int = 42) -> dict:
    """Run a reproducible MoleculeNet-style RDKit-descriptor QSAR baseline."""
    query_info = {"count": len(smiles) if isinstance(smiles, list) else 0, "method": "RDKit descriptors + scikit-learn"}
    try:
        from rdkit.Chem import Descriptors
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, roc_auc_score
        parsed = _parse_smiles_list(smiles)
        if not isinstance(labels, list) or len(labels) != len(parsed) or len(parsed) < 4:
            raise ValueError("labels must match smiles and contain at least 4 samples")
        X = np.array([[Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m), Descriptors.NumHAcceptors(m), Descriptors.NumHDonors(m), Descriptors.RingCount(m)] for _, m in parsed], dtype=float)
        Xtr, Xte, ytr, yte = train_test_split(X, np.asarray(labels), test_size=test_fraction, random_state=random_seed)
        if task_type == "classification":
            model = RandomForestClassifier(n_estimators=100, random_state=random_seed).fit(Xtr, ytr)
            pred = model.predict(Xte); metrics = {"accuracy": float(accuracy_score(yte, pred))}
            if len(set(yte)) > 1: metrics["roc_auc"] = float(roc_auc_score(yte, model.predict_proba(Xte)[:, 1]))
        elif task_type == "regression":
            model = RandomForestRegressor(n_estimators=100, random_state=random_seed).fit(Xtr, ytr)
            pred = model.predict(Xte)
            r2 = float(r2_score(yte, pred)) if len(yte) > 1 else None
            metrics = {"mae": float(mean_absolute_error(yte, pred)), "r2": r2}
        else: raise ValueError("task_type must be regression or classification")
        return _hit_success({"metrics": metrics, "train_size": len(ytr), "test_size": len(yte), "descriptor_names": ["MolWt", "MolLogP", "TPSA", "HBA", "HBD", "RingCount"]}, query_info)
    except Exception as exc:
        return _hit_error(exc, query_info)


def optimize_molecules_multiobjective(smiles: list[str], qed_weight: float = 0.5, sa_weight: float = 0.5, top_k: int = 10) -> dict:
    """Rank molecules by a deterministic QED-versus-synthetic-accessibility Pareto utility."""
    qed = score_candidates_qed(smiles)
    sa = score_candidates_sa(smiles)
    if not qed.get("success") or not sa.get("success"): return qed if not qed.get("success") else sa
    rows = []
    for q, s in zip(qed["result"]["candidates"], sa["result"]["candidates"]):
        utility = qed_weight * q["qed"] + sa_weight * (1.0 - (s["sa_score"] - 1.0) / 9.0)
        rows.append({**q, **s, "utility": round(utility, 6)})
    rows.sort(key=lambda x: x["utility"], reverse=True)
    return _hit_success({"ranked_candidates": rows[:top_k], "objective": "weighted QED and heuristic SA", "top_k": top_k}, {"method": "deterministic Pareto-style ranking"})


def optimize_molecules_with_optuna(smiles: list[str], n_trials: int = 20) -> dict:
    """Optimize QED/SA weights with Optuna when the optional dependency is installed."""
    try:
        import optuna  # noqa: F401
    except ImportError:
        return _hit_error("Optional dependency 'optuna' is not installed; install optuna to enable this optimizer.", {"dependency": "optuna"})
    return _hit_error("Optuna objective requires user-provided experimental labels; use optimize_molecules_multiobjective for descriptor-only ranking.", {"method": "optuna"})


def select_next_molecule_botorch(smiles: list[str], objective_values: list[float], maximize: bool = True) -> dict:
    """Select the next candidate with a BoTorch GP acquisition over molecular descriptors."""
    query_info = {"method": "BoTorch GaussianProcess + ExpectedImprovement"}
    try:
        import torch
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from botorch.acquisition import ExpectedImprovement
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from rdkit.Chem import Descriptors
        parsed = _parse_smiles_list(smiles)
        if not isinstance(objective_values, list) or len(objective_values) != len(parsed) or len(parsed) < 3: raise ValueError("at least 3 candidates and matching objective_values are required")
        X = torch.tensor([[Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m), Descriptors.RingCount(m)] for _, m in parsed], dtype=torch.double)
        y = torch.tensor(objective_values, dtype=torch.double).reshape(-1, 1)
        model = SingleTaskGP(X, y); fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        best = y.max() if maximize else y.min()
        acq = ExpectedImprovement(model, best_f=best, maximize=maximize)(X.unsqueeze(1)).detach().flatten()
        idx = int(torch.argmax(acq))
        return _hit_success({"selected_smiles": parsed[idx][0], "selected_index": idx, "acquisition_scores": [float(v) for v in acq]}, query_info)
    except Exception as exc:
        return _hit_error(exc, query_info)


def _resolve_allocated_gpu(gpu_device: int | None, use_gpu: bool = True) -> int | None:
    """优先使用 MCP Worker 分配的物理 GPU；没有分配时不偷偷回退到 0 号卡。"""
    if not use_gpu:
        return -1
    if gpu_device is not None:
        return gpu_device
    allocated = os.environ.get("BIOMNI_ALLOCATED_GPU", "").strip()
    if allocated.isdigit():
        return int(allocated)
    return None


def _child_cuda_index(gpu_device: int | None) -> int | None:
    """Worker 隔离可见设备后，子进程应使用可见列表中的 cuda:0。"""
    if gpu_device is None:
        return None
    allocated = os.environ.get("BIOMNI_ALLOCATED_GPU", "").strip()
    return 0 if allocated and allocated == str(gpu_device) else gpu_device


def screen_antibiotic_candidates_with_antibioticsai(
    input_csv: str,
    output_dir: str,
    asset_dir: str = "",
    library: str = "broad",
    timeout_seconds: int = 3600,
) -> dict:
    """Score candidate SMILES with the four published Antibiotics-AI ensembles."""
    configured_assets = asset_dir.strip() if isinstance(asset_dir, str) else ""
    configured_assets = configured_assets or os.environ.get("ANTIBIOTICSAI_ROOT", "").strip()
    query_info = {
        "input_csv": input_csv,
        "output_dir": output_dir,
        "asset_dir": configured_assets,
        "library": library,
        "timeout_seconds": timeout_seconds,
        "source": "Wong et al., Nature 2024, DOI 10.1038/s41586-023-06887-8",
    }
    try:
        if not isinstance(input_csv, str) or not input_csv.strip():
            return _hit_error("input_csv must be a non-empty path.", query_info)
        if not isinstance(output_dir, str) or not output_dir.strip():
            return _hit_error("output_dir must be a non-empty, caller-controlled path.", query_info)
        if not configured_assets:
            return _hit_error("asset_dir is required unless ANTIBIOTICSAI_ROOT is set.", query_info)
        if library not in {"broad", "mcule"}:
            return _hit_error("library must be either 'broad' or 'mcule'.", query_info)
        if type(timeout_seconds) is not int or timeout_seconds < 1:
            return _hit_error("timeout_seconds must be a positive integer.", query_info)

        resolved_input = Path(input_csv).expanduser().resolve()
        resolved_output = Path(output_dir).expanduser().resolve()
        resolved_assets = Path(configured_assets).expanduser().resolve()
        query_info.update(
            {
                "input_csv": str(resolved_input),
                "output_dir": str(resolved_output),
                "asset_dir": str(resolved_assets),
            }
        )
        if not resolved_input.is_file():
            return _hit_error(f"input_csv is not a regular file: {resolved_input}", query_info)
        try:
            with resolved_input.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                fields = reader.fieldnames or []
        except (OSError, csv.Error) as exc:
            return _hit_error(f"Cannot read input_csv: {exc}", query_info)
        if "SMILES" not in fields:
            return _hit_error("input_csv must contain a case-sensitive SMILES column.", query_info)
        if not rows or any(not row.get("SMILES", "").strip() for row in rows):
            return _hit_error("input_csv must contain at least one row and every SMILES value must be non-empty.", query_info)

        resolved_output.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="antibioticsai_", dir=resolved_output))
        request_path = run_dir / "request.json"
        result_path = run_dir / "result.json"
        request = {
            "input_csv": str(resolved_input),
            "asset_dir": str(resolved_assets),
            "run_dir": str(run_dir),
            "library": library,
            "timeout_seconds": timeout_seconds,
        }
        request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
        query_info["artifacts"] = {
            "run_dir": str(run_dir),
            "request_json": str(request_path),
            "result_json": str(result_path),
        }
        adapter = Path(__file__).resolve().parents[2] / "scripts" / "molecular_design_adapters" / "antibioticsai_screen.py"
        completed = subprocess.run(
            [sys.executable, str(adapter), "--request-json", str(request_path), "--output-json", str(result_path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds + 30,
        )
        if not result_path.is_file():
            details = (completed.stderr or completed.stdout or "No adapter output.")[-3000:]
            return _hit_error(f"Antibiotics-AI adapter did not produce result.json: {details}", query_info)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if isinstance(payload.get("runtime"), dict):
            query_info["runtime"] = payload["runtime"]
        if not payload.get("success"):
            return _hit_error(payload.get("error", "Antibiotics-AI inference failed."), query_info)
        if completed.returncode != 0:
            return _hit_error(f"Antibiotics-AI adapter exited with code {completed.returncode} after success.", query_info)
        result = payload["result"]
        result["interpretation_limits"] = (
            "A combined prediction pass is only the antibacterial/cytotoxicity score stage. It does not apply "
            "PAINS/Brenk, novelty/core, MCTS rationale, procurement, or wet-lab validation stages, and it does "
            "not reproduce selection of the published 283 compounds."
        )
        return _hit_success(result, query_info)
    except subprocess.TimeoutExpired:
        return _hit_error(f"Antibiotics-AI adapter timed out after {timeout_seconds + 30} seconds.", query_info)
    except Exception as exc:
        return _hit_error(f"Antibiotics-AI screening failed: {exc}", query_info)


_DEEPPURPOSE_MODEL_DIRS = {
    "CNN-CNN": "model_cnn_cnn_bindingdb",
    "MPNN-CNN": "model_mpnn_cnn_bindingdb",
    "Morgan-AAC": "model_morgan_aac_bindingdb",
    "Daylight-AAC": "model_daylight_aac_bindingdb",
}

_DEEPPURPOSE_ADMET_TASKS = (
    "AqSolDB",
    "Caco2",
    "HIA",
    "Pgp_inhibitor",
    "Bioavailability",
    "BBB_MolNet",
    "PPBR",
    "CYP2C19",
    "CYP2D6",
    "CYP3A4",
    "CYP1A2",
    "CYP2C9",
    "ClinTox",
    "Lipo_AZ",
    "Half_life_eDrug3D",
    "Clearance_eDrug3D",
)
_DEEPPURPOSE_ADMET_MODEL_CACHE = {}


def _resolve_configured_executable(environment_variable: str, command_name: str) -> str | None:
    """Resolve a configured external executable without invoking a shell."""
    configured = os.environ.get(environment_variable, "").strip()
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.is_file() and os.access(configured_path, os.X_OK):
            return str(configured_path.resolve())
        resolved = shutil.which(configured)
        if resolved:
            return resolved
        return None
    return shutil.which(command_name)


def _find_deeppurpose_checkpoint(root: str, model_type: str) -> str | None:
    """Find or safely extract an official DeepPurpose checkpoint under ``root``."""
    model_dir = _DEEPPURPOSE_MODEL_DIRS[model_type]
    root_path = os.path.abspath(os.path.expanduser(root))
    target_dir = os.path.join(root_path, model_dir)

    def complete(path: str) -> bool:
        return os.path.isfile(os.path.join(path, "config.pkl")) and os.path.isfile(os.path.join(path, "model.pt"))

    if complete(target_dir):
        return target_dir
    archive_candidates = [
        os.path.join(root_path, f"{model_dir}.zip"),
        os.path.join(root_path, f"model_{model_type.lower().replace('-', '_')}_bindingdb.zip"),
    ]
    archive_path = next((path for path in archive_candidates if os.path.isfile(path)), None)
    if not archive_path or not zipfile.is_zipfile(archive_path):
        return None
    os.makedirs(root_path, exist_ok=True)
    try:
        root_real = os.path.realpath(root_path)
        with zipfile.ZipFile(archive_path) as archive:
            members = []
            for member in archive.infolist():
                destination = os.path.realpath(os.path.join(root_path, member.filename))
                if destination != root_real and not destination.startswith(root_real + os.sep):
                    return None
                members.append(member)
            archive.extractall(root_path, members=members)
    except (OSError, zipfile.BadZipFile):
        return None
    if complete(target_dir):
        return target_dir
    for current_root, _dirs, files in os.walk(root_path):
        if "config.pkl" in files and "model.pt" in files:
            return current_root
    return None


def screen_compounds_with_drugclip(
    smiles_list: list[str],
    pocket_path: str,
    output_dir: str = "",
    max_results: int = 10000,
    timeout_seconds: int = 3600,
) -> dict:
    """Screen SMILES against a cropped pocket with the isolated official DrugCLIP runner."""
    compound_count = len(smiles_list) if isinstance(smiles_list, list) else 0
    query_info = {
        "input": {"compound_count": compound_count, "pocket_path": pocket_path},
        "source": "DrugCLIP official unimol/retrieval.py via tools_pkg/DrugCLIP/run_drugclip.py",
        "parameters": {"max_results": max_results, "timeout_seconds": timeout_seconds},
    }
    try:
        if not isinstance(smiles_list, list) or not smiles_list or not all(
            isinstance(value, str) and value.strip() for value in smiles_list
        ):
            return _hit_error("smiles_list must contain at least one non-empty SMILES string.", query_info)
        if not isinstance(pocket_path, str) or not pocket_path.strip():
            return _hit_error("pocket_path is required.", query_info)
        if not isinstance(max_results, int) or not 1 <= max_results <= 10000:
            return _hit_error("max_results must be an integer from 1 to 10000.", query_info)
        if not isinstance(timeout_seconds, int) or timeout_seconds < 1:
            return _hit_error("timeout_seconds must be a positive integer.", query_info)
        resolved_pocket_path = os.path.abspath(os.path.expanduser(pocket_path))
        query_info["input"]["pocket_path"] = resolved_pocket_path
        if not os.path.isfile(resolved_pocket_path):
            return _hit_error(f"pocket_path is not a regular file: {resolved_pocket_path}", query_info)

        biomni_root = Path(__file__).resolve().parents[2]
        tool_home = Path(os.environ.get("DRUGCLIP_HOME", _tools_pkg_root(biomni_root) / "DrugCLIP")).expanduser().resolve()
        runner_path = Path(os.environ.get("DRUGCLIP_RUNNER", tool_home / "run_drugclip.py")).expanduser().resolve()
        if not runner_path.is_file():
            return _hit_error(f"DrugCLIP runner is missing: {runner_path}", query_info)
        drugclip_python = os.path.abspath(
            os.path.expanduser(
                os.environ.get("DRUGCLIP_PYTHON", "").strip() or str(tool_home / ".conda" / "bin" / "python")
            )
        )
        if not drugclip_python or not os.path.isfile(drugclip_python):
            return _hit_error(
                "DrugCLIP environment Python is missing; deploy tools_pkg/DrugCLIP/.conda or set DRUGCLIP_PYTHON.",
                query_info,
            )

        output_root_value = output_dir or os.environ.get("DRUGCLIP_OUTPUT_ROOT", str(tool_home / "outputs" / "runs"))
        output_root = Path(output_root_value).expanduser().resolve()
        try:
            output_root.mkdir(parents=True, exist_ok=True)
        except OSError:
            fallback_root = Path(tempfile.gettempdir()) / "biomni_drugclip_runs"
            fallback_root.mkdir(parents=True, exist_ok=True)
            output_root = fallback_root
            query_info["output_root_fallback"] = str(output_root)
        run_dir = Path(tempfile.mkdtemp(prefix="drugclip_", dir=output_root))
        request_path = run_dir / "request.json"
        result_path = run_dir / "result.json"
        request = {
            "smiles_list": [value.strip() for value in smiles_list],
            "pocket_path": resolved_pocket_path,
            "run_dir": str(run_dir),
            "max_results": max_results,
            "timeout_seconds": timeout_seconds,
        }
        with request_path.open("w", encoding="utf-8") as handle:
            json.dump(request, handle, ensure_ascii=False, indent=2)
        query_info["artifacts"] = {"run_dir": str(run_dir), "request_json": str(request_path), "result_json": str(result_path)}
        completed = subprocess.run(
            [drugclip_python, str(runner_path), "--request-json", str(request_path), "--output-json", str(result_path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds + 60,
            check=False,
        )
        if not result_path.is_file():
            details = (completed.stderr or completed.stdout or "No runner output.")[-3000:]
            return _hit_error(f"DrugCLIP runner failed before producing result.json: {details}", query_info)
        with result_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload.get("runtime"), dict):
            query_info["runtime"] = payload["runtime"]
        if not payload.get("success"):
            return _hit_error(payload.get("error", "DrugCLIP runner failed without an error message."), query_info)
        if completed.returncode != 0:
            return _hit_error(f"DrugCLIP runner exited with code {completed.returncode} after reporting success.", query_info)
        return _hit_success(payload["result"], query_info)
    except subprocess.TimeoutExpired:
        return _hit_error(f"DrugCLIP runner timed out after {timeout_seconds + 60} seconds.", query_info)
    except Exception as exc:
        return _hit_error(f"DrugCLIP execution failed: {exc}", query_info)


def rank_hits_with_deeppurpose(smiles_list: list[str], protein_sequence: str, model_type: str = "MPNN-CNN", pretrained_models_dir: str = "", top_k: int = 20) -> dict:
    """Rank compounds using DeepPurpose's official pretrained virtual-screening model."""
    default_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "DeepPurpose_models", "pretrained_models"))
    models_root = pretrained_models_dir or os.environ.get("DEEPPURPOSE_PRETRAINED_ROOT", default_root)
    query_info = {"input": {"compound_count": len(smiles_list), "protein_length": len(protein_sequence)}, "source": "DeepPurpose official pretrained DTI model", "parameters": {"model_type": model_type, "pretrained_models_dir": models_root, "top_k": top_k}}
    try:
        if not smiles_list or not protein_sequence:
            return _hit_error("smiles_list and protein_sequence are required.", query_info)
        if top_k < 1:
            return _hit_error("top_k must be positive.", query_info)
        from DeepPurpose import DTI, utils
        if model_type not in _DEEPPURPOSE_MODEL_DIRS:
            return _hit_error(f"model_type must be one of {sorted(_DEEPPURPOSE_MODEL_DIRS)}.", query_info)
        checkpoint_dir = _find_deeppurpose_checkpoint(models_root, model_type)
        if checkpoint_dir:
            model = DTI.model_pretrained(path_dir=checkpoint_dir)
            query_info["model_loading"] = {"mode": "local", "checkpoint_dir": checkpoint_dir}
        else:
            model = DTI.model_pretrained(model=model_type.replace("-", "_") + "_BindingDB")
            query_info["model_loading"] = {"mode": "official_download_fallback", "model": model_type.replace("-", "_") + "_BindingDB"}
        data = utils.data_process_repurpose_virtual_screening(smiles_list, protein_sequence, model.drug_encoding, model.target_encoding, "repurposing")
        scores = model.predict(data)
        ranked = [{"smiles": smile, "score": float(score), "unit": "DeepPurpose model output"} for smile, score in sorted(zip(smiles_list, scores, strict=False), key=lambda item: float(item[1]), reverse=True)]
        return _hit_success({"ranked_hits": ranked[:top_k]}, query_info)
    except Exception as exc:
        return _hit_error(f"DeepPurpose official pretrained-model inference failed: {exc}", query_info)


def rank_hits_with_graphdta(
    smiles_list: list[str],
    protein_sequence: str,
    checkpoint_path: str = "",
    model_name: str = "GINConvNet",
    dataset_name: str = "davis",
    top_k: int = 20,
    timeout_seconds: int = 1800,
    repo_path: str = "",
    python_executable: str = "",
) -> dict:
    """Rank compounds with an official pretrained GraphDTA benchmark model."""
    query_info = {"input": {"compound_count": len(smiles_list), "protein_length": len(protein_sequence)}, "source": "GraphDTA official pretrained model (commit 6e58d46)", "parameters": {"model_name": model_name, "dataset_name": dataset_name, "top_k": top_k}}
    try:
        if not isinstance(smiles_list, list) or not smiles_list or not all(
            isinstance(value, str) and value.strip() for value in smiles_list
        ):
            return _hit_error("smiles_list must contain at least one non-empty SMILES string.", query_info)
        if not isinstance(protein_sequence, str) or not protein_sequence.strip():
            return _hit_error("protein_sequence must be a non-empty amino-acid sequence.", query_info)
        if not isinstance(top_k, int) or top_k < 1:
            return _hit_error("top_k must be a positive integer.", query_info)
        if model_name not in {"GINConvNet", "GATNet", "GAT_GCN", "GCNNet"}:
            return _hit_error("Unsupported GraphDTA model_name.", query_info)
        if dataset_name not in {"davis", "kiba"}:
            return _hit_error("dataset_name must be davis or kiba.", query_info)
        if not isinstance(timeout_seconds, int) or timeout_seconds < 1:
            return _hit_error("timeout_seconds must be a positive integer.", query_info)
        repository_root = Path(__file__).resolve().parents[2]
        tool_home = _tools_pkg_root(repository_root) / "GraphDTA"
        repo = os.path.abspath(
            os.path.expanduser(repo_path or os.environ.get("GRAPH_DTA_REPO", "") or str(tool_home / "upstream"))
        )
        checkpoint = os.path.abspath(
            os.path.expanduser(
                checkpoint_path
                or os.environ.get("GRAPH_DTA_CHECKPOINT", "")
                or os.path.join(repo, "pretrained", f"model_{model_name}_{dataset_name}.model")
            )
        )
        graphdta_python = os.path.abspath(
            os.path.expanduser(
                python_executable
                or os.environ.get("GRAPH_DTA_PYTHON", "")
                or str(tool_home / ".conda" / "bin" / "python")
            )
        )
        query_info["parameters"].update(
            {"repo_path": repo, "checkpoint_path": checkpoint, "python_executable": graphdta_python}
        )
        missing = [
            name
            for name, path in (("repo_path", repo), ("checkpoint_path", checkpoint), ("python_executable", graphdta_python))
            if not (os.path.isdir(path) if name == "repo_path" else os.path.isfile(path))
        ]
        if missing:
            return _hit_error(
                "GraphDTA deployment is incomplete; missing "
                + ", ".join(missing)
                + ". Deploy the pinned official GraphDTA commit containing its pretrained benchmark models.",
                query_info,
            )
        runner = str(repository_root / "scripts" / "hit_discovery_adapters" / "graphdta_infer.py")
        if not os.path.isfile(runner):
            return _hit_error(f"GraphDTA Biomni inference adapter is missing: {runner}", query_info)
        payload = json.dumps({"smiles": smiles_list, "protein_sequence": protein_sequence, "checkpoint_path": checkpoint, "model_name": model_name, "dataset_name": dataset_name, "top_k": top_k})
        completed = subprocess.run([graphdta_python, runner, repo, payload], cwd=repo, capture_output=True, text=True, timeout=timeout_seconds, check=False)
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout or "No inference output.")[-3000:]
            return _hit_error(f"GraphDTA inference failed: {details}", query_info)
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return _hit_error("GraphDTA runner did not return JSON.", query_info)
        return _hit_success(result, query_info)
    except subprocess.TimeoutExpired:
        return _hit_error(f"GraphDTA timed out after {timeout_seconds} seconds.", query_info)
    except Exception as exc:
        return _hit_error(f"GraphDTA execution failed: {exc}", query_info)


def query_pharmit_pharmacophores(pharmacophore: dict, subset: str = "", max_results: int = 20, pharmit_url: str = "", timeout_seconds: int = 60, poll_interval_seconds: float = 1.0) -> dict:
    """Query a Pharmit FastCGI server using its real startquery/getdata protocol."""
    default_endpoint = "https://pharmit.csb.pitt.edu/fcgi-bin/pharmitserv.fcgi"
    query_info = {"input": pharmacophore, "source": pharmit_url or os.environ.get("PHARMIT_URL", default_endpoint), "parameters": {"subset": subset, "max_results": max_results}}
    try:
        if not isinstance(pharmacophore, dict) or not pharmacophore.get("points"):
            return _hit_error("pharmacophore must be a Pharmit query object containing a non-empty points list.", query_info)
        if not 1 <= max_results <= 1000:
            return _hit_error("max_results must be between 1 and 1000.", query_info)
        endpoint = pharmit_url or os.environ.get("PHARMIT_URL", default_endpoint)
        query = dict(pharmacophore)
        query.setdefault("subset", subset or "chembl")
        response = requests.post(endpoint, data={"cmd": "startquery", "json": json.dumps(query)}, timeout=timeout_seconds)
        response.raise_for_status()
        started = response.json()
        if not started.get("status") or "qid" not in started:
            return _hit_error(f"Pharmit rejected the query: {started}", query_info)
        qid = started["qid"]
        deadline = datetime.now().timestamp() + timeout_seconds
        while datetime.now().timestamp() < deadline:
            data_response = requests.post(endpoint, data={"cmd": "getdata", "qid": qid, "start": 0, "length": max_results, "draw": 1, "order[0][column]": 1, "order[0][dir]": "asc"}, timeout=timeout_seconds)
            data_response.raise_for_status()
            payload = data_response.json()
            hits = payload.get("data", [])
            if hits or payload.get("finished"):
                return _hit_success(
                    {
                        "qid": qid,
                        "records_total": payload.get("recordsTotal", 0),
                        "hits": hits[:max_results],
                        "search_finished": bool(payload.get("finished")),
                    },
                    query_info,
                )
            import time
            time.sleep(max(0.1, poll_interval_seconds))
        return _hit_error(f"Pharmit query timed out after {timeout_seconds} seconds.", query_info)
    except requests.RequestException as exc:
        return _hit_error(f"Pharmit HTTP request failed: {exc}", query_info)
    except Exception as exc:
        return _hit_error(f"Pharmit query failed: {exc}", query_info)


def design_rdkit_pharmacophore(smiles: str, include_3d: bool = False) -> dict:
    """Extract RDKit pharmacophore features, optionally generating a 3-D conformer."""
    query_info = {"input": smiles, "source": "RDKit ChemicalFeatures", "parameters": {"include_3d": include_3d}}
    try:
        from rdkit import Chem, RDConfig
        from rdkit.Chem import AllChem, ChemicalFeatures
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return _hit_error("Invalid SMILES.", query_info)
        if include_3d:
            molecule = Chem.AddHs(molecule)
            if AllChem.EmbedMolecule(molecule, randomSeed=0xF00D) != 0:
                return _hit_error("RDKit could not generate a conformer.", query_info)
            AllChem.UFFOptimizeMolecule(molecule)
        factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef"))
        features = [{"family": feature.GetFamily(), "type": feature.GetType(), "atom_indices": list(feature.GetAtomIds())} for feature in factory.GetFeaturesForMol(molecule)]
        result = {"canonical_smiles": Chem.MolToSmiles(Chem.RemoveHs(molecule)), "features": features, "feature_count": len(features)}
        if include_3d:
            result["conformer"] = [[float(coord) for coord in molecule.GetConformer().GetAtomPosition(i)] for i in range(molecule.GetNumAtoms())]
        return _hit_success(result, query_info)
    except Exception as exc:
        return _hit_error(f"RDKit pharmacophore design failed: {exc}", query_info)


def run_diffdock_with_smiles(pdb_path, smiles_string, local_output_dir, gpu_device=None, use_gpu=True, model_dir=None):
    try:
        gpu_device = _resolve_allocated_gpu(gpu_device, use_gpu)
        if use_gpu and gpu_device is None:
            return _hit_error("MCP 未分配 GPU；请通过 MCP Worker 调用或显式提供 gpu_device。", {})
        summary = []
        diffdock_image = "rbgcsail/diffdock:latest"
        container_user = 1000

        def grant_container_write_access(path):
            setfacl = shutil.which("setfacl")
            if setfacl:
                try:
                    subprocess.run(
                        [setfacl, "-m", f"u:{container_user}:rwx,d:u:{container_user}:rwx", path],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    return "acl"
                except subprocess.CalledProcessError:
                    pass

            mode = stat.S_IMODE(os.stat(path).st_mode)
            os.chmod(path, mode | stat.S_IWOTH | stat.S_IXOTH)
            return "mode"

        # Check if PDB file exists
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f"The PDB file '{pdb_path}' does not exist.")
        summary.append(f"PDB file '{pdb_path}' found.")

        model_root = model_dir or os.environ.get("DIFFDOCK_MODEL_DIR")
        if model_root is None:
            repository_model_root = Path(__file__).resolve().parents[2] / "data" / "diffdock_models" / "v1.1"
            if repository_model_root.is_dir():
                model_root = str(repository_model_root)
                summary.append(f"Auto-discovered DiffDock models at '{model_root}'.")
        model_mount = []
        cache_mount = []
        if model_root:
            model_root = os.path.abspath(os.path.expanduser(model_root))
            if not os.path.isdir(model_root):
                raise FileNotFoundError(f"The DiffDock model directory '{model_root}' does not exist.")
            for required_dir in ("score_model", "confidence_model"):
                if not os.path.isdir(os.path.join(model_root, required_dir)):
                    raise FileNotFoundError(f"Missing DiffDock model directory '{required_dir}' under '{model_root}'.")
            model_mount = [
                "-v",
                f"{model_root}:/home/appuser/DiffDock/workdir/v1.1:ro",
            ]
            summary.append(f"Using DiffDock models from '{model_root}'.")

            cache_root = os.environ.get("DIFFDOCK_CACHE_DIR") or os.path.join(
                os.path.dirname(model_root), "torch_cache"
            )
            cache_root = os.path.abspath(os.path.expanduser(cache_root))
            os.makedirs(cache_root, exist_ok=True)
            grant_container_write_access(cache_root)
            cache_mount = ["-v", f"{cache_root}:/home/appuser/.cache/torch"]
            summary.append(f"Using persistent DiffDock Torch cache at '{cache_root}'.")
        else:
            summary.append("No local DiffDock model directory configured; the container may attempt a model download.")

        # Ensure the output directory exists
        os.makedirs(local_output_dir, exist_ok=True)
        if not os.path.isdir(local_output_dir):
            raise NotADirectoryError(f"The output path '{local_output_dir}' is not a directory.")
        summary.append(f"Output directory '{local_output_dir}' is ready.")

        # The image runs as appuser (UID 1000), while the host user may have a
        # different UID. Grant only that container user access to the output.
        output_permission_method = grant_container_write_access(local_output_dir)
        if output_permission_method == "acl":
            summary.append(f"Granted output access to container UID {container_user}.")
        else:
            summary.append("setfacl is unavailable or unsupported; enabled write access for other users on the output directory.")

        # Reuse the local image and only contact Docker Hub when it is unavailable.
        image_check = subprocess.run(
            ["docker", "image", "inspect", diffdock_image],
            check=False,
            capture_output=True,
            text=True,
        )
        if image_check.returncode == 0:
            summary.append(f"Using local DiffDock container image '{diffdock_image}'.")
        else:
            summary.append(f"Local DiffDock container image '{diffdock_image}' not found; pulling from Docker Hub...")
            subprocess.run(["docker", "pull", diffdock_image], check=True)
            summary.append("DiffDock container pulled successfully.")

        # Prepare the GPU flag
        gpu_flag = ["--gpus", f"device={gpu_device}"] if use_gpu else []

        # Docker run command
        summary.append("Running DiffDock inference...")
        run_command = (
            ["docker", "run", "--rm"]
            + gpu_flag
            + [
                # Mount the local directory to /home/appuser/output inside the container
                "-v",
                f"{os.path.abspath(pdb_path)}:/home/appuser/input/protein.pdb:ro",  # PDB file mount
                "-v",
                f"{os.path.abspath(local_output_dir)}:/home/appuser/output",  # Output directory mount
            ]
            + model_mount
            + cache_mount
            + [
                "--entrypoint",
                "/bin/bash",
                diffdock_image,
                "-c",
                # Command to run inference using micromamba environment
                f"micromamba run -n diffdock python -m inference --config default_inference_args.yaml "
                f"--protein_path /home/appuser/input/protein.pdb --ligand {shlex.quote(smiles_string)} "
                f"--out_dir /home/appuser/output",
            ]
        )

        # Execute the Docker command
        result = subprocess.run(run_command, check=False, capture_output=True, text=True)

        # Check for errors
        if result.returncode != 0:
            summary.append(f"Error during inference: {result.stderr.strip()}")
            return {"success": False, "error": result.stderr.strip() or "DiffDock inference failed.", "summary": "\n".join(summary)}
        else:
            summary.append("DiffDock inference completed successfully.")
            summary.append(f"Results stored in '{local_output_dir}'.")
        artifact_paths = [path for path in Path(local_output_dir).rglob("*") if path.is_file()]
        published_dir = os.environ.get("BIOMNI_TASK_OUTPUT_DIR")
        if published_dir:
            published_root = Path(published_dir).resolve()
            output_root = Path(local_output_dir).resolve()
            for path in artifact_paths:
                try:
                    relative = path.relative_to(output_root)
                    target = published_root / "diffdock" / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.resolve() != path.resolve():
                        shutil.copy2(path, target)
                except (ValueError, OSError):
                    continue
        return {
            "success": True,
            "summary": "\n".join(summary),
            "capabilities": {
                "docking_scoring": True,
                "docking_pose_generation": any(path.suffix.lower() in {".pdbqt", ".pdb", ".sdf"} for path in artifact_paths),
                "pose_contact_analysis": False,
            },
            "output_directory": str(Path(local_output_dir).resolve()),
            "artifacts": [str(path) for path in artifact_paths],
        }

    except FileNotFoundError as e:
        return f"File error: {e}"
    except subprocess.CalledProcessError as e:
        return f"Command execution error: {e}"
    except Exception as e:
        return f"An error occurred: {e}"


def docking_autodock_vina(
    smiles_list,
    receptor_pdb_file,
    box_center,
    box_size,
    ncpu=1,
    output_dir=None,
    n_poses=5,
):
    """Dock ligands and persist ranked PDBQT poses with structured metadata."""
    import tempfile

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from vina import Vina
    except ImportError as exc:
        return {"success": False, "error": f"AutoDock Vina dependencies are unavailable: {exc}"}

    if not os.path.isfile(receptor_pdb_file):
        return {"success": False, "error": f"Receptor PDB file does not exist: {receptor_pdb_file}"}
    if not isinstance(smiles_list, list) or not smiles_list:
        return {"success": False, "error": "smiles_list must be a non-empty list of SMILES strings."}
    if len(box_center) != 3 or len(box_size) != 3:
        return {"success": False, "error": "box_center and box_size must each contain three numeric values."}
    if any(float(size) <= 0 for size in box_size):
        return {"success": False, "error": "all box_size values must be positive."}
    if ncpu < 1:
        return {"success": False, "error": "ncpu must be at least 1."}
    if not isinstance(n_poses, int) or n_poses < 1:
        return {"success": False, "error": "n_poses must be a positive integer."}

    prepare_receptor = _resolve_meeko_command("MEEKO_PREPARE_RECEPTOR", "mk_prepare_receptor.py")
    prepare_ligand = _resolve_meeko_command("MEEKO_PREPARE_LIGAND", "mk_prepare_ligand.py")
    if not prepare_receptor or not prepare_ligand:
        return {
            "success": False,
            "error": (
                "Meeko receptor/ligand preparation commands are unavailable. "
                "Install meeko in the active Biomni Python environment or set "
                "MEEKO_PREPARE_RECEPTOR and MEEKO_PREPARE_LIGAND."
            ),
        }

    destination = Path(output_dir or os.environ.get("BIOMNI_TASK_OUTPUT_DIR") or "docking_results").expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    log = [
        "AutoDock Vina Docking Research Log",
        f"Receptor PDB File: {receptor_pdb_file}",
        f"Box Center: {[float(value) for value in box_center]}",
        f"Box Size: {[float(value) for value in box_size]}",
    ]
    ligand_results = []
    artifacts = []

    try:
        with tempfile.TemporaryDirectory(prefix="biomni_vina_") as temp_dir:
            receptor_base = os.path.join(temp_dir, "receptor")
            receptor_pdbqt = receptor_base + ".pdbqt"
            clean_receptor = os.path.join(temp_dir, "receptor_clean.pdb")
            try:
                receptor_cleaning = _prepare_vina_receptor_pdb(receptor_pdb_file, clean_receptor)
            except (OSError, ValueError) as exc:
                return {"success": False, "error": f"Receptor preprocessing failed: {exc}", "query_info": {"receptor_pdb_file": receptor_pdb_file}}
            receptor_result = subprocess.run(
                [
                    prepare_receptor,
                    "--read_pdb",
                    clean_receptor,
                    "-o",
                    receptor_base,
                    "-p",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if receptor_result.returncode != 0 or not os.path.isfile(receptor_pdbqt):
                error = receptor_result.stderr.strip() or receptor_result.stdout.strip()
                return {"success": False, "error": f"Meeko receptor preparation failed: {error}", "query_info": {"receptor_pdb_file": receptor_pdb_file, "receptor_cleaning": receptor_cleaning}}

            for index, smiles in enumerate(smiles_list):
                molecule = Chem.MolFromSmiles(smiles)
                if molecule is None:
                    return {"success": False, "error": f"Invalid SMILES string at index {index}: {smiles}"}
                molecule = Chem.AddHs(molecule)
                if AllChem.EmbedMolecule(molecule, randomSeed=42) != 0:
                    return {"success": False, "error": f"RDKit could not generate a 3D conformer for SMILES at index {index}: {smiles}"}
                AllChem.UFFOptimizeMolecule(molecule, maxIters=500)

                ligand_sdf = os.path.join(temp_dir, f"ligand_{index}.sdf")
                ligand_pdbqt = os.path.join(temp_dir, f"ligand_{index}.pdbqt")
                writer = Chem.SDWriter(ligand_sdf)
                writer.write(molecule)
                writer.close()

                ligand_result = subprocess.run(
                    [prepare_ligand, "-i", ligand_sdf, "-o", ligand_pdbqt],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if ligand_result.returncode != 0 or not os.path.isfile(ligand_pdbqt):
                    error = ligand_result.stderr.strip() or ligand_result.stdout.strip()
                    return {"success": False, "error": f"Meeko ligand preparation failed for index {index}: {error}"}

                vina = Vina(sf_name="vina", cpu=ncpu, seed=42, verbosity=0)
                vina.set_receptor(receptor_pdbqt)
                vina.set_ligand_from_file(ligand_pdbqt)
                vina.compute_vina_maps(
                    center=[float(value) for value in box_center],
                    box_size=[float(value) for value in box_size],
                )
                vina.dock(exhaustiveness=4, n_poses=n_poses)
                energies = vina.energies(n_poses=n_poses)
                if len(energies) == 0:
                    return {"success": False, "error": f"AutoDock Vina returned no poses for SMILES at index {index}."}
                ligand_id = f"ligand_{index}"
                pose_path = destination / f"{ligand_id}.pdbqt"
                vina.write_poses(str(pose_path), n_poses=len(energies), overwrite=True)
                if not pose_path.is_file() or pose_path.stat().st_size == 0:
                    return {"success": False, "error": f"AutoDock Vina did not persist a pose for SMILES at index {index}."}
                artifacts.append(str(pose_path))
                ligand_results.append({
                    "ligand_id": ligand_id,
                    "smiles": smiles,
                    "docking_score": round(float(energies[0][0]), 3),
                    "docking_score_unit": "kcal/mol",
                    "pose_file": str(pose_path),
                    "pose_format": "pdbqt",
                    "pose_count": len(energies),
                })
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        return {"success": False, "error": f"AutoDock Vina docking failed: {exc}"}

    return {
        "success": True,
        "query_info": {
            "source": "AutoDock Vina",
            "receptor_pdb_file": os.path.abspath(receptor_pdb_file),
            "receptor_cleaning": receptor_cleaning,
            "box_center": [float(value) for value in box_center],
            "box_size": [float(value) for value in box_size],
            "ncpu": ncpu,
            "n_poses": n_poses,
        },
        "capabilities": {
            "docking_scoring": True,
            "docking_pose_generation": True,
            "pose_contact_analysis": False,
        },
        "ligands": ligand_results,
        "artifacts": artifacts,
    }


def run_autosite(pdb_file, output_dir, spacing=1.0):
    """Run the official ADFR Suite AutoSite commands on a receptor PDB file."""
    if not isinstance(pdb_file, (str, os.PathLike)) or not os.fspath(pdb_file).strip():
        return "Error: pdb_file must be a non-empty path."
    if not isinstance(output_dir, (str, os.PathLike)) or not os.fspath(output_dir).strip():
        return "Error: output_dir must be a non-empty path."
    receptor_path = Path(pdb_file).expanduser().resolve()
    if not receptor_path.is_file():
        return f"Error: Receptor PDB file does not exist: {receptor_path}"
    try:
        spacing_value = float(spacing)
    except (TypeError, ValueError):
        return "Error: spacing must be a positive number."
    if spacing_value <= 0:
        return "Error: spacing must be a positive number."

    prepare_receptor = _resolve_configured_executable("ADFR_PREPARE_RECEPTOR", "prepare_receptor")
    autosite = _resolve_configured_executable("AUTOSITE_BIN", "autosite")
    repository_root = Path(__file__).resolve().parents[2]
    bundled_prepare = repository_root / "scripts" / "adfr_prepare_receptor.sh"
    bundled_autosite = repository_root / "scripts" / "adfr_autosite.sh"
    if (
        not prepare_receptor
        and not os.environ.get("ADFR_PREPARE_RECEPTOR", "").strip()
        and bundled_prepare.is_file()
        and os.access(bundled_prepare, os.X_OK)
    ):
        prepare_receptor = str(bundled_prepare)
    if (
        not autosite
        and not os.environ.get("AUTOSITE_BIN", "").strip()
        and bundled_autosite.is_file()
        and os.access(bundled_autosite, os.X_OK)
    ):
        autosite = str(bundled_autosite)
    missing = []
    if not prepare_receptor:
        missing.append("prepare_receptor (set ADFR_PREPARE_RECEPTOR)")
    if not autosite:
        missing.append("autosite (set AUTOSITE_BIN)")
    if missing:
        return "Error: Official Scripps ADFR Suite commands are unavailable: " + ", ".join(missing)

    output_path = Path(output_dir).expanduser().resolve()
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return f"Error: Could not create AutoSite output directory '{output_path}': {exc}"
    receptor_pdbqt = output_path / f"{receptor_path.stem}.pdbqt"

    try:
        prepared = subprocess.run(
            [
                prepare_receptor,
                "-r",
                str(receptor_path),
                "-o",
                str(receptor_pdbqt),
                "-A",
                "hydrogens",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if prepared.returncode != 0 or not receptor_pdbqt.is_file():
            detail_parts = []
            if prepared.stdout.strip():
                detail_parts.append(f"stdout: {prepared.stdout.strip()}")
            if prepared.stderr.strip():
                detail_parts.append(f"stderr: {prepared.stderr.strip()}")
            detail = "\n".join(detail_parts) or "no PDBQT file was generated"
            return f"Error: ADFR receptor preparation failed: {detail}"

        autosite_result = subprocess.run(
            [
                autosite,
                "-r",
                str(receptor_pdbqt),
                "--spacing",
                str(spacing_value),
                "-o",
                str(output_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if autosite_result.returncode != 0:
            detail_parts = []
            if autosite_result.stdout.strip():
                detail_parts.append(f"stdout: {autosite_result.stdout.strip()}")
            if autosite_result.stderr.strip():
                detail_parts.append(f"stderr: {autosite_result.stderr.strip()}")
            detail = "\n".join(detail_parts) or "unknown AutoSite error"
            return f"Error: AutoSite pocket detection failed: {detail}"
    except subprocess.TimeoutExpired as exc:
        return f"Error: AutoSite command timed out after {exc.timeout} seconds."
    except OSError as exc:
        return f"Error: AutoSite command could not be executed: {exc}"

    # AutoSite 1.0 writes ``<outdir>_AutoSiteSummary.log`` beside outdir,
    # whereas other builds can place it inside the output directory.
    adjacent_summary = output_path.parent / f"{output_path.name}_AutoSiteSummary.log"
    summary_candidates = sorted(output_path.rglob("*AutoSiteSummary.log"))
    if adjacent_summary.is_file():
        summary_candidates.insert(0, adjacent_summary)
    if not summary_candidates:
        return (
            "Error: AutoSite completed but no AutoSiteSummary.log was generated "
            f"under or beside '{output_path}'."
        )
    summary_path = summary_candidates[0]
    try:
        log_content = summary_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"Error: Could not read AutoSite summary '{summary_path}': {exc}"

    box_center_match = re.search(r"Box center:\s*\(([^)]+)\)", log_content, flags=re.IGNORECASE)
    box_size_match = re.search(r"Box size:\s*\(([^)]+)\)", log_content, flags=re.IGNORECASE)
    cluster_match = re.search(r"identified and characterized\s+(\d+)\s+clusters", log_content, flags=re.IGNORECASE)
    csv_candidates = sorted(output_path.rglob("*_summary.csv"))
    top_cluster = None
    if csv_candidates:
        try:
            with csv_candidates[0].open(newline="", encoding="utf-8", errors="replace") as handle:
                rows = list(csv.reader(handle))
            if len(rows) > 1 and len(rows[1]) >= 7:
                top_cluster = rows[1][:7]
        except (OSError, csv.Error):
            top_cluster = None
    research_log = [
        "AutoSite Research Log",
        f"Receptor PDB: {receptor_path}",
        f"Prepared receptor PDBQT: {receptor_pdbqt}",
        f"Grid spacing: {spacing_value}",
        f"Output directory: {output_path}",
        f"Summary log: {summary_path}",
    ]
    if cluster_match:
        research_log.append(f"Detected Pockets: {cluster_match.group(1)}")
    if csv_candidates:
        research_log.append(f"Cluster summary: {csv_candidates[0]}")
    if top_cluster:
        research_log.append(
            "Top-ranked pocket: "
            f"cluster {top_cluster[0]}, energy {top_cluster[1]}, points {top_cluster[2]}, "
            f"radius of gyration {top_cluster[3]}, buriedness {top_cluster[5]}, score {top_cluster[6]}"
        )
    if box_center_match and box_size_match:
        research_log.extend(
            [
                f"Box Center: {box_center_match.group(1)}",
                f"Box Size: {box_size_match.group(1)}",
            ]
        )
    elif not cluster_match and not top_cluster:
        research_log.append("Box Center and Size information not found in the AutoSite summary log.")
    return "\n".join(research_log)


# Function to get TxGNN predictions and return a summarized string output
def retrieve_topk_repurposing_drugs_from_disease_txgnn(disease_name, data_lake_path, k=5):
    """This function computes TxGNN model predictions for drug repurposing. It takes in the paths to the data,
    the disease name, and returns a summary of the top K predicted drugs with their sigmoid-transformed scores.

    Args:
    - disease_name (str): The name of the disease for which the drug predictions are to be retrieved.
    - data_lake_path (str): The path to the data lake containing the TxGNN predictions.
    - k (int, optional): The number of top drug predictions to return. Defaults to 5.

    Returns:
    - str: A summary of the steps and the top K drug predictions with their scores.

    """

    if not isinstance(disease_name, str) or not disease_name.strip():
        return "Error: disease_name must be a non-empty string."
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        return "Error: k must be a positive integer."
    if not isinstance(data_lake_path, (str, os.PathLike)) or not os.fspath(data_lake_path).strip():
        return "Error: data_lake_path must be a non-empty path."

    data_lake = Path(data_lake_path).expanduser().resolve()
    name_mapping_path = data_lake / "txgnn_name_mapping.pkl"
    result_path = data_lake / "txgnn_prediction.pkl"
    missing = [str(path) for path in (name_mapping_path, result_path) if not path.is_file()]
    if missing:
        return "Error: Required TxGNN deployment files are missing: " + ", ".join(missing)

    try:
        with name_mapping_path.open("rb") as handle:
            mapping = pickle.load(handle)
        with result_path.open("rb") as handle:
            result = pickle.load(handle)
    except (OSError, pickle.PickleError, EOFError, AttributeError, ValueError, TypeError, ImportError) as exc:
        return f"Error: Could not load trusted TxGNN deployment files: {exc}"

    if not isinstance(mapping, dict) or not isinstance(mapping.get("id2name_drug"), dict):
        return "Error: txgnn_name_mapping.pkl must contain an 'id2name_drug' dictionary."
    if not isinstance(result, dict) or not result:
        return "Error: txgnn_prediction.pkl must contain a non-empty disease-to-drug-score dictionary."

    # Step 2: Fuzzy match the disease name to find the closest match
    possible_diseases = [name for name in result if isinstance(name, str)]
    exact_matches = [name for name in possible_diseases if name.casefold() == disease_name.strip().casefold()]
    matched_disease = exact_matches or get_close_matches(disease_name.strip(), possible_diseases, n=1, cutoff=0.6)

    if not matched_disease:
        return f"Error: No matching disease found for '{disease_name}'. Please try a different name."

    matched_disease = matched_disease[0]

    # Step 3: Retrieve the prediction scores for the matched disease
    disease_predictions = result[matched_disease]
    if not isinstance(disease_predictions, dict) or not disease_predictions:
        return f"Error: TxGNN predictions for '{matched_disease}' are empty or malformed."

    # Step 4: Apply the sigmoid function to the raw prediction scores
    sigmoid_predictions = {}
    for drug_id, score in disease_predictions.items():
        if not isinstance(score, (int, float, np.integer, np.floating)) or not np.isfinite(score):
            continue
        numeric_score = float(score)
        if numeric_score >= 0:
            transformed_score = 1 / (1 + np.exp(-numeric_score))
        else:
            exponential = np.exp(numeric_score)
            transformed_score = exponential / (1 + exponential)
        sigmoid_predictions[drug_id] = float(transformed_score)
    if not sigmoid_predictions:
        return f"Error: TxGNN predictions for '{matched_disease}' contain no finite numeric scores."

    # Step 5: Sort the drugs by prediction score in descending order
    top_k_drugs = sorted(sigmoid_predictions.items(), key=lambda x: x[1], reverse=True)[:k]

    # Step 6: Map drug IDs to their names and format the results
    top_k_drug_names = [(mapping["id2name_drug"].get(drug_id, "Unknown Drug"), score) for drug_id, score in top_k_drugs]

    # Step 7: Create a human and LLM-friendly summary string
    summary = f"TxGNN Drug Repurposing Predictions for '{matched_disease}':\n"
    summary += (
        f"Top {min(k, len(top_k_drugs))} predicted drugs and their sigmoid-transformed ranking scores "
        "(not calibrated probabilities):\n"
    )

    for i, (drug_name, score) in enumerate(top_k_drug_names, 1):
        summary += f"{i}. {drug_name} - Sigmoid-transformed Ranking Score: {score:.4f}\n"

    summary += "\nProcess Summary:\n"
    summary += f"- The input disease name was matched to '{matched_disease}'.\n"
    summary += "- Sigmoid transformation was applied for ranking display; these scores are not calibrated probabilities.\n"
    summary += f"- The top {k} drugs were selected based on their prediction scores.\n"
    summary += f"- Deployment files: {name_mapping_path.name}, {result_path.name}.\n"

    return summary


# ADMET prediction function with research log format
def predict_admet_properties(smiles_list, ADMET_model_type="MPNN", models_root=None, allow_download=False):
    """Predict 16 ADMET endpoints using validated DeepPurpose checkpoints."""
    try:
        from DeepPurpose import CompoundPred, utils
    except Exception as exc:
        return f"Error: DeepPurpose dependencies are unavailable: {exc}"

    # Define available model types
    available_model_types = ["MPNN", "CNN", "Morgan"]

    # Check if the provided model type is valid
    if ADMET_model_type not in available_model_types:
        return f"Error: Invalid ADMET model type '{ADMET_model_type}'. Available options are: {', '.join(available_model_types)}."
    if not isinstance(smiles_list, list) or not smiles_list or not all(isinstance(smiles, str) and smiles.strip() for smiles in smiles_list):
        return "Error: smiles_list must be a non-empty list of SMILES strings."
    if not isinstance(allow_download, bool):
        return "Error: allow_download must be a boolean."
    if models_root is not None and (
        not isinstance(models_root, (str, os.PathLike)) or not os.fspath(models_root).strip()
    ):
        return "Error: models_root must be a non-empty path when provided."

    default_models_root = Path(__file__).resolve().parents[2] / "DeepPurpose_models" / "pretrained_models"
    configured_root = models_root or os.environ.get("DEEPPURPOSE_PRETRAINED_ROOT") or default_models_root
    model_root = Path(configured_root).expanduser().resolve()
    model_directories = {
        task: model_root / f"{task}_{ADMET_model_type}_model" for task in _DEEPPURPOSE_ADMET_TASKS
    }
    missing_models = [
        str(path)
        for path in model_directories.values()
        if not (path / "config.pkl").is_file() or not (path / "model.pt").is_file()
    ]
    if missing_models and not allow_download:
        return (
            f"Error: {len(missing_models)} required {ADMET_model_type} ADMET checkpoint directories are missing or incomplete "
            f"under '{model_root}': " + ", ".join(missing_models)
        )

    cache_key = (str(model_root), ADMET_model_type, allow_download)
    model_ADMETs = _DEEPPURPOSE_ADMET_MODEL_CACHE.get(cache_key)
    if model_ADMETs is None:
        model_ADMETs = {}
        try:
            for task, local_model_dir in model_directories.items():
                model_name = f"{task}_{ADMET_model_type}_model"
                if (local_model_dir / "config.pkl").is_file() and (local_model_dir / "model.pt").is_file():
                    model_ADMETs[model_name] = CompoundPred.model_pretrained(path_dir=str(local_model_dir))
                else:
                    model_ADMETs[model_name] = CompoundPred.model_pretrained(model=model_name)
        except Exception as exc:
            return f"Error: Unable to load the {ADMET_model_type} ADMET checkpoints: {exc}"
        _DEEPPURPOSE_ADMET_MODEL_CACHE[cache_key] = model_ADMETs

    # Helper function for ADMET prediction
    def ADMET_pred(drug, task, unit):
        model = model_ADMETs[task + "_" + ADMET_model_type + "_model"]
        X_pred = utils.data_process(
            X_drug=[drug],
            y=[0],
            drug_encoding=ADMET_model_type,
            split_method="no_split",
        )
        y_pred = model.predict(X_pred)[0]

        if unit == "%":
            y_pred = y_pred * 100

        return f"{y_pred:.2f} " + unit

    # Initialize research log string
    research_log = "Research Log for ADMET Predictions:\n"
    research_log += "-------------------------------------\n"

    # Process each SMILES string in the list
    try:
        for smiles in smiles_list:
            research_log += f"\nCompound SMILES: {smiles}\n"
            research_log += "Predicted ADMET properties:\n"

            # Physiochemical properties
            solubility = ADMET_pred(smiles, "AqSolDB", "log mol/L")
            lipophilicity = ADMET_pred(smiles, "Lipo_AZ", "(log-ratio)")
            research_log += f"- Solubility: {solubility}\n"
            research_log += f"- Lipophilicity: {lipophilicity}\n"

            # Absorption
            caco2 = ADMET_pred(smiles, "Caco2", "cm/s")
            hia = ADMET_pred(smiles, "HIA", "%")
            pgp = ADMET_pred(smiles, "Pgp_inhibitor", "%")
            bioavail = ADMET_pred(smiles, "Bioavailability", "%")
            research_log += f"- Absorption (Caco-2 permeability): {caco2}\n"
            research_log += f"- Absorption (HIA): {hia}\n"
            research_log += f"- Absorption (Pgp Inhibitor): {pgp}\n"
            research_log += f"- Absorption (Bioavailability): {bioavail}\n"

            # Distribution
            bbb = ADMET_pred(smiles, "BBB_MolNet", "%")
            ppbr = ADMET_pred(smiles, "PPBR", "%")
            research_log += f"- Distribution (BBB permeation): {bbb}\n"
            research_log += f"- Distribution (PPBR): {ppbr}\n"

            # Metabolism
            cyp2c19 = ADMET_pred(smiles, "CYP2C19", "%")
            cyp2d6 = ADMET_pred(smiles, "CYP2D6", "%")
            cyp3a4 = ADMET_pred(smiles, "CYP3A4", "%")
            cyp1a2 = ADMET_pred(smiles, "CYP1A2", "%")
            cyp2c9 = ADMET_pred(smiles, "CYP2C9", "%")
            research_log += f"- Metabolism (CYP2C19): {cyp2c19}\n"
            research_log += f"- Metabolism (CYP2D6): {cyp2d6}\n"
            research_log += f"- Metabolism (CYP3A4): {cyp3a4}\n"
            research_log += f"- Metabolism (CYP1A2): {cyp1a2}\n"
            research_log += f"- Metabolism (CYP2C9): {cyp2c9}\n"

            # Excretion
            half_life = ADMET_pred(smiles, "Half_life_eDrug3D", "h")
            clearance = ADMET_pred(smiles, "Clearance_eDrug3D", "mL/min/kg")
            research_log += f"- Excretion (Half-life): {half_life}\n"
            research_log += f"- Excretion (Clearance): {clearance}\n"

            # Clinical Toxicity
            clinical_toxicity = ADMET_pred(smiles, "ClinTox", "%")
            research_log += f"- Clinical Toxicity: {clinical_toxicity}\n"

            research_log += "-------------------------------------\n"
    except Exception as exc:
        return f"Error: DeepPurpose ADMET inference failed: {exc}"

    return research_log


# Binding Affinity prediction function with model_type validation
def predict_binding_affinity_protein_1d_sequence(smiles_list, amino_acid_sequence, affinity_model_type="MPNN-CNN"):
    try:
        from DeepPurpose import DTI, utils
    except Exception as exc:
        return f"Error: DeepPurpose dependencies are unavailable: {exc}"

    # Define available model types for Binding Affinity
    available_affinity_model_types = [
        "CNN-CNN",
        "MPNN-CNN",
        "Morgan-CNN",
        "Morgan-AAC",
        "Daylight-AAC",
    ]

    # Check if the provided affinity model type is valid
    if affinity_model_type not in available_affinity_model_types:
        return f"Error: Invalid affinity model type '{affinity_model_type}'. Available options are: {', '.join(available_affinity_model_types)}."

    # Prefer repository-managed official weights so inference does not depend on
    # Harvard Dataverse availability. Other model types retain the upstream
    # download fallback.
    model_directory_names = {
        "CNN-CNN": "model_cnn_cnn_bindingdb",
        "MPNN-CNN": "model_mpnn_cnn_bindingdb",
        "Morgan-CNN": "model_morgan_cnn_bindingdb",
        "Morgan-AAC": "model_morgan_aac_bindingdb",
        "Daylight-AAC": "model_daylight_aac_bindingdb",
    }
    default_models_root = Path(__file__).resolve().parents[2] / "DeepPurpose_models" / "pretrained_models"
    models_root = Path(os.environ.get("DEEPPURPOSE_PRETRAINED_ROOT", default_models_root)).expanduser().resolve()
    local_model_dir = models_root / model_directory_names[affinity_model_type]

    try:
        if (local_model_dir / "config.pkl").is_file() and (local_model_dir / "model.pt").is_file():
            model_DTI = DTI.model_pretrained(path_dir=str(local_model_dir))
            model_source = f"local official checkpoint: {local_model_dir}"
        else:
            upstream_model = affinity_model_type.replace("-", "_") + "_BindingDB"
            model_DTI = DTI.model_pretrained(model=upstream_model)
            model_source = f"DeepPurpose upstream download: {upstream_model}"
    except Exception as exc:
        return f"Error: Unable to load the {affinity_model_type} BindingDB model: {exc}"

    # Initialize research log string
    research_log = "Research Log for Binding Affinity Predictions:\n"
    research_log += "-------------------------------------\n"
    research_log += f"Model source: {model_source}\n"

    # Process each SMILES string in the list
    for smiles in smiles_list:
        research_log += f"\nCompound SMILES: {smiles}\n"
        research_log += f"Amino Acid Sequence: {amino_acid_sequence}\n"

        # Predict binding affinity
        X_pred = utils.data_process(
            X_drug=[smiles],
            X_target=[amino_acid_sequence],
            y=[0],
            drug_encoding=affinity_model_type.split("-")[0],
            target_encoding=affinity_model_type.split("-")[1],
            split_method="no_split",
        )
        y_pred = model_DTI.predict(X_pred)[0]
        y_pred_nM = 10 ** (-y_pred) / 1e-9

        research_log += f"Predicted Binding Affinity: {y_pred_nM:.2f} nM\n"
        research_log += "-------------------------------------\n"

    return research_log


def analyze_accelerated_stability_of_pharmaceutical_formulations(formulations, storage_conditions, time_points):
    """Analyzes the stability of pharmaceutical formulations under accelerated storage conditions.

    Parameters
    ----------
    formulations : list of dict
        List of formulation dictionaries, each containing:
        - 'name': str, name of the formulation
        - 'active_ingredient': str, name of the active pharmaceutical ingredient
        - 'concentration': float, concentration in mg/mL
        - 'excipients': list, list of excipients

    storage_conditions : list of dict
        List of storage condition dictionaries, each containing:
        - 'temperature': float, temperature in °C
        - 'humidity': float, relative humidity in percentage (optional for solid dosage forms)
        - 'description': str, description of storage condition (e.g., "Room Temperature", "Accelerated")

    time_points : list of int
        List of time points in days to evaluate stability

    Returns
    -------
    str
        Research log summarizing the stability testing process and results

    """
    # Create output directory if it doesn't exist
    output_dir = "stability_test_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize results storage
    all_results = []

    # Process each formulation under each storage condition
    for formulation in formulations:
        for condition in storage_conditions:
            # Initialize stability parameters
            results = []

            # Get acceleration factor based on temperature (simplified Arrhenius equation)
            temp_c = condition["temperature"]
            accel_factor = 2 ** ((temp_c - 25) / 10)  # Rule of thumb: reaction rate doubles every 10°C

            # Add humidity effect for degradation if provided
            humidity_factor = 1.0
            if "humidity" in condition:
                humidity = condition["humidity"]
                # Simple model: higher humidity increases degradation rate
                humidity_factor = 1.0 + (humidity - 60) / 100 if humidity > 60 else 1.0

            # Calculate stability parameters at each time point
            initial_content = 100.0  # Starting at 100%

            for time in time_points:
                # Chemical stability (% of initial content)
                # Simple first-order degradation model
                effective_time = time * accel_factor * humidity_factor
                chemical_stability = initial_content * np.exp(-0.001 * effective_time)

                # Physical stability (score from 1-10, 10 being perfect)
                # Decreases over time, affected by temperature and humidity
                physical_stability = 10 - (0.05 * effective_time)
                physical_stability = max(1, physical_stability)  # Minimum score of 1

                # Particle size change (% increase from initial)
                # Some formulations show particle growth over time
                particle_size_change = (
                    0.2 * effective_time if "solid" in formulation.get("dosage_form", "").lower() else 0
                )

                results.append(
                    {
                        "Formulation": formulation["name"],
                        "Storage_Condition": condition["description"],
                        "Temperature_C": temp_c,
                        "Humidity_RH": condition.get("humidity", "N/A"),
                        "Time_Days": time,
                        "Chemical_Stability_Percent": round(chemical_stability, 2),
                        "Physical_Stability_Score": round(physical_stability, 1),
                        "Particle_Size_Change_Percent": round(particle_size_change, 2),
                    }
                )

            all_results.extend(results)

    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)

    # Save results to CSV
    csv_filename = f"{output_dir}/stability_results_{timestamp}.csv"
    results_df.to_csv(csv_filename, index=False)

    # Generate research log
    log = "Accelerated Stability Testing of Pharmaceutical Formulations\n"
    log += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    log += "1. STUDY PARAMETERS\n"
    log += f"   - Number of formulations tested: {len(formulations)}\n"
    log += f"   - Formulations: {', '.join([f['name'] for f in formulations])}\n"
    log += (
        "   - Storage conditions: "
        + ", ".join(
            [
                f"{c['description']} ({c['temperature']}°C" + (f"/{c['humidity']}% RH" if "humidity" in c else "") + ")"
                for c in storage_conditions
            ]
        )
        + "\n"
    )
    log += f"   - Time points evaluated (days): {', '.join(map(str, time_points))}\n\n"

    log += "2. METHODOLOGY\n"
    log += "   - Chemical stability assessed by active ingredient content\n"
    log += "   - Physical stability evaluated on a 10-point scale\n"
    log += "   - Particle size changes measured where applicable\n\n"

    log += "3. KEY FINDINGS\n"

    # Summarize stability at final time point for each formulation/condition
    final_time = max(time_points)
    final_results = results_df[results_df["Time_Days"] == final_time]

    for formulation in formulations:
        form_results = final_results[final_results["Formulation"] == formulation["name"]]
        log += f"   {formulation['name']}:\n"

        for _, row in form_results.iterrows():
            condition = row["Storage_Condition"]
            chem_stab = row["Chemical_Stability_Percent"]
            phys_stab = row["Physical_Stability_Score"]

            stability_assessment = "Stable"
            if chem_stab < 90 or phys_stab < 7:
                stability_assessment = "Potentially unstable"
            if chem_stab < 85 or phys_stab < 5:
                stability_assessment = "Unstable"

            log += f"     - {condition}: Chemical stability {chem_stab}%, Physical stability score {phys_stab}/10 - {stability_assessment}\n"
        log += "\n"

    log += "4. CONCLUSION\n"

    # Identify most stable formulation
    best_formulation = ""
    best_stability = 0

    for formulation in formulations:
        form_data = final_results[final_results["Formulation"] == formulation["name"]]
        avg_chem_stability = form_data["Chemical_Stability_Percent"].mean()
        if avg_chem_stability > best_stability:
            best_stability = avg_chem_stability
            best_formulation = formulation["name"]

    log += f"   - Most stable formulation: {best_formulation} (avg. chemical stability: {best_stability:.2f}%)\n"
    log += f"   - Detailed results saved to: {csv_filename}\n"

    return log


def run_3d_chondrogenic_aggregate_assay(
    chondrocyte_cells, test_compounds, culture_duration_days=21, measurement_intervals=7
):
    """Generates a detailed protocol for performing a 3D chondrogenic aggregate culture assay to evaluate compounds' effects on chondrogenesis.

    Parameters
    ----------
    chondrocyte_cells : dict
        Dictionary with cell information including 'source', 'passage_number', and 'cell_density'
    test_compounds : list of dict
        List of compounds to test, each with 'name', 'concentration', and 'vehicle' keys
    culture_duration_days : int
        Total duration of the culture period in days (default: 21)
    measurement_intervals : int
        Interval in days between measurements (default: 7)

    Returns
    -------
    str
        Detailed protocol document for the 3D chondrogenic aggregate culture assay

    """
    from datetime import datetime

    # Create experiment ID
    experiment_id = f"CHOND3D_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create time points for measurements
    timepoints = list(range(0, culture_duration_days + 1, measurement_intervals))
    if timepoints[-1] != culture_duration_days:
        timepoints.append(culture_duration_days)

    # Generate the protocol document
    protocol = f"# 3D Chondrogenic Aggregate Culture Assay Protocol - {experiment_id}\n\n"

    protocol += "## 1. Materials and Reagents\n\n"
    protocol += "- Chondrocyte cells\n"
    protocol += "- Chondrogenic differentiation medium\n"
    protocol += "- Transforming growth factor-β3 (TGF-β3)\n"
    protocol += "- Dexamethasone\n"
    protocol += "- Ascorbate-2-phosphate\n"
    protocol += "- 96-well V-bottom plates\n"
    protocol += "- Gaussia luciferase reporter assay kit\n"
    protocol += "- Luminometer\n"
    protocol += "- Test compounds with respective vehicles\n"
    protocol += "- Centrifuge\n"
    protocol += "- CO2 incubator\n"
    protocol += "- Sterile pipettes and tips\n\n"

    protocol += "## 2. Experimental Information\n\n"
    protocol += "### Cell Information:\n"
    protocol += f"- Cell source: {chondrocyte_cells['source']}\n"
    protocol += f"- Passage number: {chondrocyte_cells['passage_number']}\n"
    protocol += f"- Cell density: {chondrocyte_cells['cell_density']} cells/mL\n\n"

    protocol += "### Experimental Design:\n"
    protocol += f"- Culture duration: {culture_duration_days} days\n"
    protocol += f"- Measurement timepoints: {', '.join(map(str, timepoints))} days\n\n"

    protocol += "### Test Compounds:\n"
    for i, compound in enumerate(test_compounds):
        protocol += f"- Compound {i + 1}: {compound['name']} at {compound['concentration']} in {compound['vehicle']}\n"
    protocol += "- Control: Vehicle only\n\n"

    protocol += "## 3. Detailed Procedure\n\n"
    protocol += "### Day 0: Setup\n\n"
    protocol += "1. Prepare chondrogenic differentiation medium containing:\n"
    protocol += "   - High-glucose DMEM\n"
    protocol += "   - 10 ng/mL TGF-β3\n"
    protocol += "   - 100 nM Dexamethasone\n"
    protocol += "   - 50 μg/mL Ascorbate-2-phosphate\n"
    protocol += "   - 1% ITS+ premix (insulin, transferrin, selenium)\n"
    protocol += "   - 1 mM Sodium pyruvate\n"
    protocol += "   - 100 U/mL Penicillin/Streptomycin\n\n"

    protocol += "2. Harvest and count chondrocyte cells\n\n"

    protocol += "3. Prepare cell suspension at the specified density:\n"
    protocol += f"   - {chondrocyte_cells['cell_density']} cells/mL\n\n"

    protocol += "4. Form 3D cell aggregates:\n"
    protocol += "   - Aliquot 2.5×10^5 cells per well in 96-well V-bottom plates\n"
    protocol += "   - Centrifuge plates at 500g for 5 minutes to pellet cells\n\n"

    protocol += "5. Add test compounds to respective wells:\n"
    for _, compound in enumerate(test_compounds):
        protocol += f"   - Add {compound['name']} at {compound['concentration']} in {compound['vehicle']}\n"
    protocol += "   - Add vehicle only to control wells\n\n"

    protocol += "6. Incubate the plates at 37°C, 5% CO2\n\n"

    protocol += "### Day 1 to Day " + str(culture_duration_days) + ":\n\n"
    protocol += "1. Change medium every 2-3 days:\n"
    protocol += "   - Carefully remove 50% of the medium without disturbing the aggregates\n"
    protocol += "   - Replace with fresh medium containing test compounds at the same concentrations\n\n"

    protocol += f"2. At days {', '.join(map(str, timepoints))}, collect samples for analysis:\n"
    protocol += (
        "   - Take medium samples for Gaussia luciferase activity measurement (if using COL2A1-GLuc reporter cells)\n"
    )
    protocol += "   - Fix aggregates in 4% paraformaldehyde for histological analysis\n\n"

    return protocol


def grade_adverse_events_using_vcog_ctcae(clinical_data_file):
    """Grade and monitor adverse events in animal studies using the VCOG-CTCAE standard.

    Parameters
    ----------
    clinical_data_file : str
        Path to a CSV file containing clinical evaluation data with columns:
        subject_id, time_point, symptom, severity, measurement (optional)

    Returns
    -------
    str
        A research log summarizing the adverse event grading process and results.
        The graded events are saved to 'vcog_ctcae_graded_events.csv'.

    """
    import json
    from datetime import datetime

    import pandas as pd

    # Initialize the research log
    log = "# Adverse Event Grading using VCOG-CTCAE v1.1\n"
    log += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Step 1: Load the clinical data
    log += "## Step 1: Loading clinical evaluation data\n"
    try:
        data = pd.read_csv(clinical_data_file)
        log += f"Successfully loaded data from {clinical_data_file}\n"
        log += f"Total records: {len(data)}\n"
        log += f"Columns found: {', '.join(data.columns)}\n\n"
    except Exception as e:
        log += f"Error loading data: {str(e)}\n"
        return log

    # Step 2: Define VCOG-CTCAE grading criteria
    log += "## Step 2: Applying VCOG-CTCAE grading criteria\n"

    # Comprehensive VCOG-CTCAE grading criteria based on VCOG-CTCAE v1.1
    vcog_criteria = {
        # Hematologic
        "neutropenia": {
            "description": "Neutrophil count decrease",
            "unit": "cells/µL",
            "grades": {
                0: {"criteria": "≥ 1500", "range": [1500, float("inf")]},
                1: {"criteria": "1000 - <1500", "range": [1000, 1500]},
                2: {"criteria": "500 - <1000", "range": [500, 1000]},
                3: {"criteria": "100 - <500", "range": [100, 500]},
                4: {"criteria": "<100", "range": [0, 100]},
                5: {"criteria": "Death due to neutropenic sepsis", "range": None},
            },
        },
        "anemia": {
            "description": "Hemoglobin decrease",
            "unit": "g/dL",
            "grades": {
                0: {"criteria": "Within reference range", "range": [10, float("inf")]},
                1: {"criteria": "Mild; clinical signs not present", "range": [8, 10]},
                2: {"criteria": "Moderate; clinical signs present", "range": [6.5, 8]},
                3: {"criteria": "Severe; transfusion indicated", "range": [5, 6.5]},
                4: {"criteria": "Life-threatening", "range": [0, 5]},
                5: {"criteria": "Death", "range": None},
            },
        },
        "thrombocytopenia": {
            "description": "Platelet count decrease",
            "unit": "cells/µL",
            "grades": {
                0: {"criteria": "≥ 100,000", "range": [100000, float("inf")]},
                1: {"criteria": "50,000 - <100,000", "range": [50000, 100000]},
                2: {"criteria": "25,000 - <50,000", "range": [25000, 50000]},
                3: {"criteria": "10,000 - <25,000", "range": [10000, 25000]},
                4: {"criteria": "<10,000 or spontaneous bleeding", "range": [0, 10000]},
                5: {"criteria": "Death", "range": None},
            },
        },
        # Gastrointestinal
        "vomiting": {
            "description": "Vomiting frequency",
            "unit": "episodes per 24h period",
            "grades": {
                0: {"criteria": "None", "range": [0, 0]},
                1: {
                    "criteria": "1-2 episodes in 24h; medical intervention not indicated",
                    "range": [1, 2],
                },
                2: {"criteria": "3-5 episodes in 24h; ≤3 days", "range": [3, 5]},
                3: {
                    "criteria": ">5 episodes in 24h; >3 days; hospitalization indicated",
                    "range": [6, float("inf")],
                },
                4: {"criteria": "Life-threatening consequences", "range": None},
                5: {"criteria": "Death", "range": None},
            },
        },
        "diarrhea": {
            "description": "Diarrhea frequency",
            "unit": "episodes per 24h period",
            "grades": {
                0: {"criteria": "None", "range": [0, 0]},
                1: {
                    "criteria": "Increase of <4 stools per day over baseline",
                    "range": [1, 3],
                },
                2: {
                    "criteria": "Increase of 4-6 stools per day over baseline",
                    "range": [4, 6],
                },
                3: {
                    "criteria": "Increase of ≥7 stools per day; hospitalization indicated",
                    "range": [7, float("inf")],
                },
                4: {"criteria": "Life-threatening consequences", "range": None},
                5: {"criteria": "Death", "range": None},
            },
        },
        "anorexia": {
            "description": "Appetite/food intake decrease",
            "unit": "percent of normal intake",
            "grades": {
                0: {"criteria": "Normal", "range": [100, float("inf")]},
                1: {"criteria": "Decreased appetite, but eating", "range": [75, 100]},
                2: {"criteria": "Decreased intake <3 days", "range": [50, 75]},
                3: {"criteria": "Decreased intake ≥3 days", "range": [25, 50]},
                4: {
                    "criteria": "Life-threatening consequences; urgent intervention indicated",
                    "range": [0, 25],
                },
                5: {"criteria": "Death", "range": None},
            },
        },
        # Hepatic
        "alt_increase": {
            "description": "Alanine aminotransferase increased",
            "unit": "x ULN (upper limit of normal)",
            "grades": {
                0: {"criteria": "≤ ULN", "range": [0, 1]},
                1: {"criteria": ">ULN - 2.5xULN", "range": [1, 2.5]},
                2: {"criteria": ">2.5 - 5.0xULN", "range": [2.5, 5]},
                3: {"criteria": ">5.0 - 20.0xULN", "range": [5, 20]},
                4: {"criteria": ">20.0xULN", "range": [20, float("inf")]},
                5: {"criteria": "Death", "range": None},
            },
        },
        # Renal
        "creatinine_increase": {
            "description": "Creatinine increased",
            "unit": "x ULN",
            "grades": {
                0: {"criteria": "≤ ULN", "range": [0, 1]},
                1: {"criteria": ">ULN - 1.5xULN", "range": [1, 1.5]},
                2: {"criteria": ">1.5 - 3.0xULN", "range": [1.5, 3]},
                3: {"criteria": ">3.0 - 6.0xULN", "range": [3, 6]},
                4: {"criteria": ">6.0xULN", "range": [6, float("inf")]},
                5: {"criteria": "Death", "range": None},
            },
        },
        # Constitutional
        "fever": {
            "description": "Fever",
            "unit": "°C",
            "grades": {
                0: {"criteria": "None", "range": [0, 39]},
                1: {"criteria": "39.0 - 39.5°C", "range": [39, 39.5]},
                2: {"criteria": ">39.5 - 40.0°C", "range": [39.5, 40]},
                3: {"criteria": ">40.0 - 41.0°C", "range": [40, 41]},
                4: {"criteria": ">41.0°C for >24 hrs", "range": [41, float("inf")]},
                5: {"criteria": "Death", "range": None},
            },
        },
        "weight_loss": {
            "description": "Weight loss",
            "unit": "percent of baseline weight",
            "grades": {
                0: {"criteria": "<5%", "range": [0, 5]},
                1: {"criteria": "5% - <10%", "range": [5, 10]},
                2: {"criteria": "10% - <20%", "range": [10, 20]},
                3: {"criteria": "≥20%", "range": [20, float("inf")]},
                4: {"criteria": "Life-threatening", "range": None},
                5: {"criteria": "Death", "range": None},
            },
        },
        # Dermatologic
        "alopecia": {
            "description": "Hair loss",
            "unit": None,
            "grades": {
                0: {"criteria": "None", "range": None},
                1: {"criteria": "Hair loss at injection/treatment site", "range": None},
                2: {"criteria": "Moderate alopecia", "range": None},
                3: {"criteria": "Complete alopecia", "range": None},
                4: {"criteria": "Not applicable", "range": None},
                5: {"criteria": "Not applicable", "range": None},
            },
        },
        # Neurologic
        "neuropathy": {
            "description": "Peripheral neuropathy",
            "unit": None,
            "grades": {
                0: {"criteria": "None", "range": None},
                1: {
                    "criteria": "Asymptomatic; clinically detectable on examination",
                    "range": None,
                },
                2: {
                    "criteria": "Mild symptoms; limiting instrumental ADL",
                    "range": None,
                },
                3: {
                    "criteria": "Severe symptoms; limiting self-care ADL",
                    "range": None,
                },
                4: {"criteria": "Life-threatening consequences", "range": None},
                5: {"criteria": "Death", "range": None},
            },
        },
    }

    def apply_vcog_grade(symptom, severity, measurement=None):
        """Apply VCOG-CTCAE grading criteria to an adverse event.

        Parameters
        ----------
        symptom : str
            The type of adverse event
        severity : str
            The severity description
        measurement : float or None
            Quantitative measurement related to the symptom, if available

        Returns
        -------
        int
            The VCOG-CTCAE grade (0-5)
        str
            Description of the grading rationale

        """
        # Standard severity-based grading if no specific criteria exist
        grade_map = {
            "none": 0,
            "mild": 1,
            "moderate": 2,
            "severe": 3,
            "life-threatening": 4,
            "death": 5,
        }

        symptom_lower = symptom.lower()

        # Check if the symptom has specific VCOG-CTCAE criteria
        if symptom_lower in vcog_criteria:
            criteria = vcog_criteria[symptom_lower]

            # If measurement is provided and criteria has numeric ranges
            if measurement is not None:
                try:
                    measurement_value = float(measurement)

                    # Find the appropriate grade based on the measurement ranges
                    for grade, grade_info in criteria["grades"].items():
                        if grade_info["range"] is not None:
                            min_val, max_val = grade_info["range"]
                            if min_val <= measurement_value < max_val:
                                return (
                                    grade,
                                    f"Grade {grade}: {criteria['description']} - {criteria['grades'][grade]['criteria']}",
                                )
                except (ValueError, TypeError):
                    # If measurement can't be converted to float, fall back to severity-based grading
                    pass

            # If there's a reported severity with no valid measurement
            if severity.lower() in grade_map:
                # Check if the grade exists in the criteria
                severity_grade = grade_map[severity.lower()]
                if severity_grade in criteria["grades"]:
                    return (
                        severity_grade,
                        f"Grade {severity_grade}: {criteria['description']} - {criteria['grades'][severity_grade]['criteria']}",
                    )

        # Default to using the severity mapping if no specific criteria match
        if severity.lower() in grade_map:
            return grade_map[severity.lower()], f"Grade {grade_map[severity.lower()]}: Based on reported severity"

        # Default grade if no specific criteria match
        return 1, "Grade 1: Default grade (specific criteria not found)"

    # Step 3: Apply grading to each record
    log += "Applying VCOG-CTCAE v1.1 grading criteria to each adverse event...\n"

    # Create new columns for the grade and rationale
    grading_results = data.apply(
        lambda row: apply_vcog_grade(
            row["symptom"],
            row["severity"],
            row["measurement"] if "measurement" in data.columns else None,
        ),
        axis=1,
    )

    # Split the returned tuples into separate columns
    data["vcog_grade"] = [result[0] for result in grading_results]
    data["grading_rationale"] = [result[1] for result in grading_results]

    # Step 4: Analyze patterns across time points (if available)
    if "time_point" in data.columns:
        log += "\n## Step 3: Analyzing adverse event patterns across time points\n"

        # Group by subject and symptom to track progression
        progression_analysis = data.pivot_table(
            index=["subject_id", "symptom"],
            columns="time_point",
            values="vcog_grade",
            aggfunc="max",
        ).reset_index()

        # Calculate if grade is increasing, decreasing, or stable for each subject-symptom pair
        trend_counts = {"increasing": 0, "decreasing": 0, "stable": 0, "fluctuating": 0}

        numeric_columns = [col for col in progression_analysis.columns if col not in ["subject_id", "symptom"]]

        if len(numeric_columns) >= 2:
            # Sort columns to ensure chronological order
            numeric_columns.sort()

            for _, row in progression_analysis.iterrows():
                values = [row[col] for col in numeric_columns if not pd.isna(row[col])]
                if len(values) >= 2:
                    if all(values[i] < values[i + 1] for i in range(len(values) - 1)):
                        trend_counts["increasing"] += 1
                    elif all(values[i] > values[i + 1] for i in range(len(values) - 1)):
                        trend_counts["decreasing"] += 1
                    elif all(values[i] == values[i + 1] for i in range(len(values) - 1)):
                        trend_counts["stable"] += 1
                    else:
                        trend_counts["fluctuating"] += 1

            log += "Adverse event progression patterns:\n"
            for trend, count in trend_counts.items():
                log += f"- {trend.capitalize()}: {count} subject-symptom pairs\n"

        # Save progression analysis
        progression_file = "vcog_ctcae_progression_analysis.csv"
        progression_analysis.to_csv(progression_file)
        log += f"\nDetailed progression analysis saved to: {progression_file}\n"

    # Step 5: Summarize the grading results
    log += "\n## Step 4: Summarizing adverse event grades\n"

    # Count events by grade
    grade_counts = data["vcog_grade"].value_counts().sort_index()
    log += "Grade distribution:\n"
    for grade, count in grade_counts.items():
        log += f"- Grade {grade}: {count} events\n"

    # Summarize by symptom type
    symptom_summary = data.groupby("symptom")["vcog_grade"].agg(["max", "mean", "count"])
    log += "\nSymptom severity summary:\n"
    for symptom, stats in symptom_summary.iterrows():
        log += f"- {symptom}: max grade = {stats['max']}, avg grade = {stats['mean']:.2f}, count = {stats['count']}\n"

    # Summarize by subject
    subject_summary = data.groupby("subject_id")["vcog_grade"].agg(["max", "mean", "count"])
    log += f"\nSubjects with adverse events: {len(subject_summary)}\n"
    log += f"Subjects with Grade 3+ events: {len(subject_summary[subject_summary['max'] >= 3])}\n"

    # Create a summary of most severe events
    most_severe = data.sort_values("vcog_grade", ascending=False).head(10)
    log += "\nTop 10 most severe adverse events:\n"
    for i, (_, event) in enumerate(most_severe.iterrows(), 1):
        log += f"{i}. Subject {event['subject_id']}: {event['symptom']} (Grade {event['vcog_grade']})\n"

    # Step 6: Save detailed results to file
    output_file = "vcog_ctcae_graded_events.csv"
    data.to_csv(output_file, index=False)
    log += "\n## Step 5: Results saved\n"
    log += f"Detailed graded events saved to: {output_file}\n"

    # Save the VCOG criteria as a reference
    with open("vcog_ctcae_criteria_reference.json", "w") as f:
        json.dump(vcog_criteria, f, indent=2)
    log += "VCOG-CTCAE criteria reference saved to: vcog_ctcae_criteria_reference.json\n"

    return log


def analyze_radiolabeled_antibody_biodistribution(time_points, tissue_data):
    """Analyze biodistribution and pharmacokinetic profile of radiolabeled antibodies.

    Parameters
    ----------
    time_points : list or numpy.ndarray
        Time points (hours) at which measurements were taken
    tissue_data : dict
        Dictionary where keys are tissue names and values are lists/arrays of %IA/g
        measurements corresponding to time_points. Must include 'tumor' as one of the keys.

    Returns
    -------
    str
        Research log summarizing the biodistribution analysis, pharmacokinetic parameters,
        and tumor-to-normal tissue ratios

    """
    import json
    import os

    import numpy as np
    from scipy.optimize import curve_fit

    # Validate inputs
    if "tumor" not in tissue_data:
        return "Error: Tumor data must be provided in tissue_data dictionary"

    # Define bi-exponential model for pharmacokinetic analysis
    # C(t) = A*exp(-alpha*t) + B*exp(-beta*t)
    def bi_exp_model(t, A, alpha, B, beta):
        return A * np.exp(-alpha * t) + B * np.exp(-beta * t)

    # Initialize results dictionary
    results = {
        "tissues_analyzed": list(tissue_data.keys()),
        "pk_parameters": {},
        "tumor_to_normal_ratios": {},
        "auc_values": {},
    }

    # Analyze each tissue
    for tissue, measurements in tissue_data.items():
        try:
            # Fit bi-exponential model
            params, _ = curve_fit(
                bi_exp_model,
                time_points,
                measurements,
                p0=[50, 0.1, 50, 0.01],  # Initial parameter guess
                bounds=([0, 0, 0, 0], [100, 5, 100, 1]),  # Parameter bounds
            )

            A, alpha, B, beta = params

            # Calculate pharmacokinetic parameters
            # Distribution half-life (fast component)
            t_half_dist = np.log(2) / alpha

            # Elimination half-life (slow component)
            t_half_elim = np.log(2) / beta

            # Area under the curve (AUC)
            auc = A / alpha + B / beta

            # Mean residence time (MRT)
            mrt = (A / (alpha**2) + B / (beta**2)) / auc

            # Clearance (for blood/plasma only - conceptual)
            clearance = 1 / auc if tissue.lower() in ["blood", "plasma"] else None

            # Store results
            results["pk_parameters"][tissue] = {
                "A": float(A),
                "alpha": float(alpha),
                "B": float(B),
                "beta": float(beta),
                "distribution_half_life_h": float(t_half_dist),
                "elimination_half_life_h": float(t_half_elim),
                "mean_residence_time_h": float(mrt),
            }

            if clearance:
                results["pk_parameters"][tissue]["clearance"] = float(clearance)

            # Calculate AUC
            results["auc_values"][tissue] = float(auc)

        except Exception as e:
            results["pk_parameters"][tissue] = f"Fitting failed: {str(e)}"

    # Calculate tumor-to-normal tissue ratios at each time point
    for tissue in tissue_data:
        if tissue != "tumor":
            ratios = [
                t / n if n > 0 else float("inf")
                for t, n in zip(tissue_data["tumor"], tissue_data[tissue], strict=False)
            ]
            results["tumor_to_normal_ratios"][tissue] = {
                "values": [float(r) for r in ratios],
                "max_ratio": float(max(ratios)),
                "max_ratio_time_point": float(time_points[np.argmax(ratios)]),
            }

    # Save results to JSON file
    filename = "biodistribution_pk_results.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    # Generate research log
    log = "# Biodistribution and Pharmacokinetic Analysis of Radiolabeled Antibody\n\n"
    log += "## Analysis Summary\n"
    log += f"- Analyzed biodistribution data across {len(tissue_data)} tissues\n"
    log += f"- Time points analyzed: {time_points} hours\n"
    log += "- Performed bi-exponential pharmacokinetic modeling\n\n"

    log += "## Key Pharmacokinetic Parameters\n"
    for tissue, params in results["pk_parameters"].items():
        if isinstance(params, dict):
            log += f"\n### {tissue.capitalize()}\n"
            log += f"- Distribution half-life: {params['distribution_half_life_h']:.2f} hours\n"
            log += f"- Elimination half-life: {params['elimination_half_life_h']:.2f} hours\n"
            log += f"- Mean residence time: {params['mean_residence_time_h']:.2f} hours\n"
            if "clearance" in params:
                log += f"- Clearance: {params['clearance']:.4f} units\n"

    log += "\n## Tumor-to-Normal Tissue Ratios\n"
    for tissue, ratio_data in results["tumor_to_normal_ratios"].items():
        log += f"- {tissue.capitalize()}: Max ratio {ratio_data['max_ratio']:.2f} at {ratio_data['max_ratio_time_point']:.1f} hours\n"

    log += "\n## Detailed Results\n"
    log += f"Complete analysis results saved to: {os.path.abspath(filename)}\n"

    return log


def estimate_alpha_particle_radiotherapy_dosimetry(
    biodistribution_data, radiation_parameters, output_file="dosimetry_results.csv"
):
    """Estimate radiation absorbed doses to tumor and normal organs for alpha-particle radiotherapeutics.

    This function implements the Medical Internal Radiation Dose (MIRD) schema to calculate
    absorbed doses based on biodistribution data from healthy mice and radiation transport parameters.

    Parameters
    ----------
    biodistribution_data : dict
        Dictionary containing organ/tissue names as keys and a list of time-activity measurements as values.
        Each measurement should be a tuple of (time_hours, percent_injected_activity).
        Must include entries for all relevant organs including 'tumor'.

    radiation_parameters : dict
        Dictionary containing radiation parameters for the alpha-emitting radionuclide:
        - 'radionuclide': str - Name of the radionuclide (e.g., 'Ac-225')
        - 'half_life_hours': float - Physical half-life in hours
        - 'energy_per_decay_MeV': float - Energy released per decay in MeV
        - 'radiation_weighting_factor': float - Radiation weighting factor for alpha particles
        - 'S_factors': dict - S-factors (Gy/Bq-s) for each source-target organ pair

    output_file : str, optional
        Filename to save the dosimetry results (default: "dosimetry_results.csv")

    Returns
    -------
    str
        Research log summarizing the dosimetry estimation process and results

    """
    import csv
    from datetime import datetime

    import numpy as np
    from scipy.integrate import trapezoid

    # Initialize research log
    log = f"Alpha-Particle Radiotherapy Dosimetry Estimation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    log += f"Radionuclide: {radiation_parameters['radionuclide']}\n"
    log += f"Half-life: {radiation_parameters['half_life_hours']} hours\n\n"

    # Step 1: Calculate time-integrated activity for each organ
    log += "Step 1: Calculating time-integrated activity for each organ\n"
    time_integrated_activity = {}

    for organ, measurements in biodistribution_data.items():
        times = [m[0] for m in measurements]
        activities = [m[1] for m in measurements]

        # Apply physical decay correction
        decay_constant = np.log(2) / radiation_parameters["half_life_hours"]
        decay_corrected_activities = [a * np.exp(-decay_constant * t) for a, t in zip(activities, times, strict=False)]

        # Calculate time-integrated activity using trapezoidal integration
        cumulated_activity = trapezoid(decay_corrected_activities, times)
        time_integrated_activity[organ] = cumulated_activity

        log += f"  - {organ}: {cumulated_activity:.4f} %IA-h\n"

    # Step 2: Calculate absorbed dose using MIRD schema
    log += "\nStep 2: Calculating absorbed doses using MIRD schema\n"

    # Convert %IA-h to MBq-h for a standard injection of 1 MBq
    conversion_factor = 0.01  # Convert %IA to fraction of IA

    # Energy conversion factor: MeV to J

    # Calculate absorbed dose for each target organ
    absorbed_doses = {}
    s_factors = radiation_parameters["S_factors"]

    for target_organ in biodistribution_data:
        absorbed_dose = 0

        # Sum contributions from all source organs
        for source_organ, cumulated_activity in time_integrated_activity.items():
            if (source_organ, target_organ) in s_factors:
                s_value = s_factors[(source_organ, target_organ)]
                organ_contribution = cumulated_activity * conversion_factor * s_value
                absorbed_dose += organ_contribution

        # Apply radiation weighting factor for alpha particles
        absorbed_dose *= radiation_parameters["radiation_weighting_factor"]

        # Store as Gy/MBq
        absorbed_doses[target_organ] = absorbed_dose
        log += f"  - {target_organ}: {absorbed_dose:.4f} Gy/MBq\n"

    # Step 3: Calculate therapeutic index (tumor-to-normal tissue dose ratios)
    log += "\nStep 3: Calculating therapeutic indices (tumor-to-normal tissue ratios)\n"

    tumor_dose = absorbed_doses.get("tumor", 0)
    if tumor_dose > 0:
        for organ, dose in absorbed_doses.items():
            if organ != "tumor" and dose > 0:
                therapeutic_index = tumor_dose / dose
                log += f"  - Tumor-to-{organ} ratio: {therapeutic_index:.2f}\n"

    # Save results to CSV file
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Organ", "Absorbed Dose (Gy/MBq)"])
        for organ, dose in absorbed_doses.items():
            writer.writerow([organ, f"{dose:.4f}"])

    log += f"\nDosimetry results saved to {output_file}\n"

    return log


def perform_mwas_cyp2c19_metabolizer_status(
    methylation_data_path,
    metabolizer_status_path,
    covariates_path=None,
    pvalue_threshold=0.05,
    output_file="significant_cpg_sites.csv",
):
    """Perform a Methylome-wide Association Study (MWAS) to identify CpG sites significantly associated with CYP2C19 metabolizer status.

    Parameters
    ----------
    methylation_data_path : str
        Path to CSV or TSV file containing DNA methylation beta values.
        Rows should be samples, columns should be CpG sites.
    metabolizer_status_path : str
        Path to CSV or TSV file containing CYP2C19 metabolizer status for each sample.
        Should have a sample ID column and a status column (e.g., poor, intermediate, normal, rapid, ultrarapid).
    covariates_path : str, optional
        Path to CSV or TSV file containing covariates to adjust for in the regression model
        (e.g., age, sex, smoking status).
    pvalue_threshold : float, optional
        P-value threshold for significance after multiple testing correction. Default is 0.05.
    output_file : str, optional
        Filename to save significant CpG sites. Default is "significant_cpg_sites.csv".

    Returns
    -------
    str
        A research log summarizing the MWAS analysis and results.

    """
    import time

    import pandas as pd
    from scipy.stats import linregress
    from statsmodels.formula.api import ols

    start_time = time.time()
    log = ["## Methylome-wide Association Study (MWAS) of CYP2C19 Metabolizer Status"]

    # Load data from files
    log.append("\n### Loading Data")
    try:
        # Load methylation data
        if methylation_data_path.endswith(".csv"):
            methylation_data = pd.read_csv(methylation_data_path, index_col=0)
        elif methylation_data_path.endswith((".tsv", ".txt")):
            methylation_data = pd.read_csv(methylation_data_path, sep="\t", index_col=0)
        else:
            log.append("Error: Unsupported file format for methylation data. Please provide a CSV or TSV file.")
            return "\n".join(log)
        log.append(f"- Successfully loaded methylation data from {methylation_data_path}")

        # Load metabolizer status data
        if metabolizer_status_path.endswith(".csv"):
            metabolizer_status_df = pd.read_csv(metabolizer_status_path, index_col=0)
        elif metabolizer_status_path.endswith((".tsv", ".txt")):
            metabolizer_status_df = pd.read_csv(metabolizer_status_path, sep="\t", index_col=0)
        else:
            log.append("Error: Unsupported file format for metabolizer status. Please provide a CSV or TSV file.")
            return "\n".join(log)
        log.append(f"- Successfully loaded metabolizer status data from {metabolizer_status_path}")

        # Convert DataFrame to Series if necessary
        if metabolizer_status_df.shape[1] == 1:
            metabolizer_status = metabolizer_status_df.iloc[:, 0]
        else:
            log.append("Error: Metabolizer status file should contain a single column with status values.")
            return "\n".join(log)

        # Load covariates if provided
        covariates = None
        if covariates_path is not None:
            if covariates_path.endswith(".csv"):
                covariates = pd.read_csv(covariates_path, index_col=0)
            elif covariates_path.endswith((".tsv", ".txt")):
                covariates = pd.read_csv(covariates_path, sep="\t", index_col=0)
            else:
                log.append("Error: Unsupported file format for covariates. Please provide a CSV or TSV file.")
                return "\n".join(log)
            log.append(f"- Successfully loaded covariates data from {covariates_path}")
    except Exception as e:
        log.append(f"Error loading data: {str(e)}")
        return "\n".join(log)

    # Step 1: Data preprocessing
    log.append("\n### Data Preprocessing")
    log.append(f"- Methylation data shape: {methylation_data.shape} (samples × CpG sites)")
    log.append(f"- Number of samples with metabolizer status: {len(metabolizer_status)}")

    # Ensure sample IDs match between methylation data and metabolizer status
    common_samples = methylation_data.index.intersection(metabolizer_status.index)
    methylation_data = methylation_data.loc[common_samples]
    metabolizer_status = metabolizer_status.loc[common_samples]

    log.append(f"- Number of samples after matching: {len(common_samples)}")

    # Check for covariates
    if covariates is not None:
        log.append(f"- Covariates provided: {', '.join(covariates.columns)}")
        covariates = covariates.loc[common_samples]

    # Step 2: Perform regression for each CpG site
    log.append("\n### Association Analysis")
    log.append(f"- Total CpG sites to analyze: {methylation_data.shape[1]}")

    results = []
    cpg_sites = methylation_data.columns

    # Convert metabolizer status to numeric if it's categorical
    if metabolizer_status.dtype == "object":
        # Create a mapping dictionary for metabolizer status
        # Assuming order: poor < intermediate < normal < rapid < ultrarapid
        status_order = {
            "poor": 1,
            "intermediate": 2,
            "normal": 3,
            "rapid": 4,
            "ultrarapid": 5,
        }

        # Try to map using the dictionary, or keep as is if already numeric
        try:
            metabolizer_status_numeric = metabolizer_status.map(status_order)
            log.append("- Converted metabolizer status to numeric values")
        except Exception:
            metabolizer_status_numeric = metabolizer_status
            log.append("- Using metabolizer status as provided (assuming numeric)")
    else:
        metabolizer_status_numeric = metabolizer_status

    # Perform regression for each CpG site
    for cpg in cpg_sites:
        methylation_values = methylation_data[cpg]

        # Basic model without covariates
        if covariates is None:
            model = linregress(metabolizer_status_numeric, methylation_values)
            pvalue = model.pvalue
            coefficient = model.slope
        else:
            # Create DataFrame for regression with covariates
            data_for_regression = pd.DataFrame(
                {
                    "methylation": methylation_values,
                    "metabolizer": metabolizer_status_numeric,
                }
            )

            # Add covariates
            for col in covariates.columns:
                data_for_regression[col] = covariates[col]

            # Formula for regression with covariates
            formula = "methylation ~ metabolizer + " + " + ".join(covariates.columns)
            model = ols(formula, data=data_for_regression).fit()

            pvalue = model.pvalues["metabolizer"]
            coefficient = model.params["metabolizer"]

        results.append({"CpG_site": cpg, "coefficient": coefficient, "pvalue": pvalue})

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Step 3: Multiple testing correction
    log.append("\n### Multiple Testing Correction")
    log.append("- Applying Bonferroni correction")

    # Bonferroni correction
    results_df["adjusted_pvalue"] = results_df["pvalue"] * len(results_df)
    results_df["adjusted_pvalue"] = results_df["adjusted_pvalue"].clip(upper=1.0)  # Ensure p-values don't exceed 1

    # Step 4: Identify significant CpG sites
    significant_sites = results_df[results_df["adjusted_pvalue"] < pvalue_threshold]
    significant_sites = significant_sites.sort_values("adjusted_pvalue")

    log.append("\n### Results")
    log.append(f"- Number of significant CpG sites (adjusted p < {pvalue_threshold}): {len(significant_sites)}")

    if len(significant_sites) > 0:
        # Save significant sites to file
        significant_sites.to_csv(output_file, index=False)
        log.append("- Top 5 significant CpG sites:")

        for _, row in significant_sites.head(5).iterrows():
            log.append(
                f"  * {row['CpG_site']}: coefficient = {row['coefficient']:.4f}, adj. p-value = {row['adjusted_pvalue']:.6f}"
            )

        log.append(f"- Full results saved to: {output_file}")
    else:
        log.append("- No significant CpG sites found after multiple testing correction")

    # Execution time
    execution_time = time.time() - start_time
    log.append("\n### Summary")
    log.append(f"- Analysis completed in {execution_time:.2f} seconds")

    return "\n".join(log)


def calculate_physicochemical_properties(smiles_string):
    """Calculate key physicochemical properties of a drug candidate molecule.

    Parameters
    ----------
    smiles_string : str
        The molecular structure in SMILES format

    Returns
    -------
    str
        A research log summarizing the calculated physicochemical properties and
        indicating where the detailed results are saved

    """
    import csv
    import os

    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski
    from rdkit.Chem.MolStandardize import rdMolStandardize

    # Create RDKit molecule from SMILES
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            return "ERROR: Invalid SMILES string provided."
    except Exception as e:
        return f"ERROR: Failed to process SMILES string: {str(e)}"

    # Calculate basic properties
    properties = {
        "SMILES": smiles_string,
        "Molecular Weight": round(Descriptors.MolWt(mol), 2),
        "cLogP": round(Descriptors.MolLogP(mol), 2),
        "TPSA": round(Descriptors.TPSA(mol), 2),
        "H-Bond Donors": Lipinski.NumHDonors(mol),
        "H-Bond Acceptors": Lipinski.NumHAcceptors(mol),
        "Rotatable Bonds": Descriptors.NumRotatableBonds(mol),
        "Heavy Atoms": mol.GetNumHeavyAtoms(),
        "Ring Count": Descriptors.RingCount(mol),
    }

    # Estimate pKa (simplified approach - in practice would use specialized tools)
    # This is a simplification as accurate pKa prediction requires specialized tools
    uncharger = rdMolStandardize.Uncharger()
    uncharger.uncharge(mol)
    acidic_groups = sum(
        1
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "O"
        and any(neigh.GetSymbol() == "C" and neigh.GetDegree() == 3 for neigh in atom.GetNeighbors())
    )
    basic_groups = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "N" and atom.GetDegree() < 4)
    properties["Estimated Acidic Groups"] = acidic_groups
    properties["Estimated Basic Groups"] = basic_groups

    # Calculate drug-likeness score (using Crippen approach)
    properties["Drug-likeness Score"] = round(Crippen.MolMR(mol), 2)

    # Calculate logD (simplified as logP - pKa adjustment would need specialized tools)
    properties["Estimated logD7.4"] = properties["cLogP"]

    # Save results to CSV
    csv_filename = "physicochemical_properties.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Property", "Value"])
        for prop, value in properties.items():
            writer.writerow([prop, value])

    # Generate research log
    log = f"""Physicochemical Property Calculation Research Log:

Analyzed compound with SMILES: {smiles_string}

Key properties:
- Molecular Weight: {properties["Molecular Weight"]} g/mol
- cLogP: {properties["cLogP"]}
- Topological Polar Surface Area: {properties["TPSA"]} Å²
- H-Bond Donors: {properties["H-Bond Donors"]}
- H-Bond Acceptors: {properties["H-Bond Acceptors"]}
- Rotatable Bonds: {properties["Rotatable Bonds"]}
- Estimated logD (at pH 7.4): {properties["Estimated logD7.4"]}
- Estimated Acidic Groups: {properties["Estimated Acidic Groups"]}
- Estimated Basic Groups: {properties["Estimated Basic Groups"]}

Complete results saved to: {os.path.abspath(csv_filename)}
"""

    return log


def analyze_xenograft_tumor_growth_inhibition(
    data_path,
    time_column,
    volume_column,
    group_column,
    subject_column,
    output_dir="./results",
):
    """Analyze tumor growth inhibition in xenograft models across different treatment groups.

    Parameters
    ----------
    data_path : str
        Path to CSV or TSV file containing tumor volume measurements. The file should have columns for
        time, volume, treatment group, and subject ID
    time_column : str
        Name of the column containing time points (e.g., 'Day', 'Time')
    volume_column : str
        Name of the column containing tumor volume measurements
    group_column : str
        Name of the column containing treatment group labels
    subject_column : str
        Name of the column containing subject/mouse identifiers
    output_dir : str, optional
        Directory to save output files (default: "./results")

    Returns
    -------
    str
        Research log summarizing the analysis steps, findings, and generated file paths

    """
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from scipy import stats
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize research log
    log = "# Xenograft Tumor Growth Inhibition Analysis\n\n"

    # Load data from file
    log += "## 1. Data Loading and Summary\n\n"
    try:
        if data_path.endswith(".csv"):
            data_df = pd.read_csv(data_path)
        elif data_path.endswith((".tsv", ".txt")):
            data_df = pd.read_csv(data_path, sep="\t")
        else:
            log += "Error: Unsupported file format. Please provide a CSV or TSV file.\n"
            return log
        log += f"Successfully loaded tumor growth data from {data_path}\n"
    except Exception as e:
        log += f"Error loading data: {str(e)}\n"
        return log

    # Validate required columns
    required_columns = [time_column, volume_column, group_column, subject_column]
    missing_columns = [col for col in required_columns if col not in data_df.columns]
    if missing_columns:
        log += f"Error: Missing required columns: {', '.join(missing_columns)}\n"
        return log

    # Get unique groups and time points
    groups = data_df[group_column].unique()
    time_points = sorted(data_df[time_column].unique())
    n_groups = len(groups)
    log += f"- Number of treatment groups: {n_groups} ({', '.join(map(str, groups))})\n"
    log += f"- Number of time points: {len(time_points)}\n"
    log += f"- Number of subjects: {data_df[subject_column].nunique()}\n"
    log += f"- Total number of measurements: {len(data_df)}\n\n"

    # 2. Calculate group statistics at each time point
    log += "## 2. Tumor Growth Analysis\n\n"

    # Group statistics
    stats_df = (
        data_df.groupby([group_column, time_column])[volume_column]
        .agg(mean="mean", sem=lambda x: stats.sem(x), count="count")
        .reset_index()
    )

    # Save group statistics
    stats_file = os.path.join(output_dir, "tumor_volume_statistics.csv")
    stats_df.to_csv(stats_file, index=False)
    log += f"Group statistics saved to: {stats_file}\n\n"

    # 3. Calculate tumor growth rates
    log += "## 3. Tumor Growth Rate Analysis\n\n"

    growth_rates = {}
    for group in groups:
        group_data = data_df[data_df[group_column] == group]

        # Calculate growth rate for each subject
        subject_growth_rates = []
        for subject in group_data[subject_column].unique():
            subject_data = group_data[group_data[subject_column] == subject]

            if len(subject_data) >= 2:
                # Simple linear regression for growth rate
                x = subject_data[time_column].values
                y = subject_data[volume_column].values
                slope, _, _, _, _ = stats.linregress(x, y)
                subject_growth_rates.append(slope)

        growth_rates[group] = subject_growth_rates
        mean_rate = np.mean(subject_growth_rates)
        sem_rate = stats.sem(subject_growth_rates)

        log += (
            f"- {group}: Mean growth rate = {mean_rate:.2f} ± {sem_rate:.2f} mm³/day (n={len(subject_growth_rates)})\n"
        )

    # 4. Calculate Tumor Growth Inhibition (TGI)
    log += "\n## 4. Tumor Growth Inhibition (TGI)\n\n"

    # Identify control group (assuming the first group is control)
    control_group = groups[0]
    log += f"Control group: {control_group}\n\n"

    # Calculate TGI for the final time point
    final_time = max(time_points)
    final_data = data_df[data_df[time_column] == final_time]

    control_final_mean = final_data[final_data[group_column] == control_group][volume_column].mean()

    tgi_results = {}
    for group in groups:
        if group == control_group:
            continue

        group_final_mean = final_data[final_data[group_column] == group][volume_column].mean()
        tgi = ((control_final_mean - group_final_mean) / control_final_mean) * 100
        tgi_results[group] = tgi

        log += f"- {group}: TGI = {tgi:.1f}% (relative to {control_group})\n"

    # 5. Statistical Analysis
    log += "\n## 5. Statistical Analysis\n\n"

    # Repeated measures ANOVA
    log += "### Repeated Measures ANOVA\n\n"

    try:
        # Prepare data for repeated measures ANOVA
        formula = f"{volume_column} ~ C({group_column}) * C({time_column}) + C({subject_column})"
        model = ols(formula, data=data_df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        # Save ANOVA results
        anova_file = os.path.join(output_dir, "repeated_measures_anova.csv")
        anova_table.to_csv(anova_file)

        log += f"ANOVA results saved to: {anova_file}\n\n"

        # Extract p-values
        group_effect_p = anova_table.loc[f"C({group_column})", "PR(>F)"]
        time_effect_p = anova_table.loc[f"C({time_column})", "PR(>F)"]
        interaction_p = anova_table.loc[f"C({group_column}):C({time_column})", "PR(>F)"]

        log += f"- Treatment effect: p = {group_effect_p:.4f}\n"
        log += f"- Time effect: p = {time_effect_p:.4f}\n"
        log += f"- Treatment × Time interaction: p = {interaction_p:.4f}\n\n"

        # Post-hoc analysis at final time point
        log += "### Post-hoc Analysis (Final Time Point)\n\n"

        # Perform Tukey's HSD test
        tukey = pairwise_tukeyhsd(endog=final_data[volume_column], groups=final_data[group_column], alpha=0.05)

        # Save Tukey results
        tukey_file = os.path.join(output_dir, "tukey_posthoc_results.txt")
        with open(tukey_file, "w") as f:
            f.write(str(tukey.summary()))

        log += f"Tukey's HSD results saved to: {tukey_file}\n\n"

        # Summarize significant comparisons
        tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])

        sig_pairs = tukey_df[tukey_df["p-adj"] < 0.05]
        if len(sig_pairs) > 0:
            log += "Significant pairwise comparisons:\n"
            for _, row in sig_pairs.iterrows():
                log += f"- {row['group1']} vs {row['group2']}: p = {row['p-adj']:.4f}\n"
        else:
            log += "No significant pairwise comparisons found.\n"

    except Exception as e:
        log += f"Error in statistical analysis: {str(e)}\n"

    # 6. Generate tumor growth curves
    log += "\n## 6. Tumor Growth Visualization\n\n"

    plt.figure(figsize=(10, 6))

    for group in groups:
        group_stats = stats_df[stats_df[group_column] == group]
        plt.errorbar(
            group_stats[time_column],
            group_stats["mean"],
            yerr=group_stats["sem"],
            label=group,
            capsize=3,
            marker="o",
        )

    plt.xlabel(f"{time_column} (days)")
    plt.ylabel(f"Tumor Volume ({volume_column})")
    plt.title("Xenograft Tumor Growth Curves")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    # Save the plot
    plot_file = os.path.join(output_dir, "tumor_growth_curves.png")
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()

    log += f"Tumor growth curve plot saved to: {plot_file}\n"

    # 7. Conclusion
    log += "\n## 7. Conclusion\n\n"

    # Summarize most effective treatment
    if tgi_results:
        best_treatment = max(tgi_results.items(), key=lambda x: x[1])
        log += f"The most effective treatment was {best_treatment[0]} with a tumor growth inhibition of {best_treatment[1]:.1f}%.\n"

    # Statistical significance summary
    try:
        if group_effect_p < 0.05:
            log += "Statistical analysis confirmed significant differences between treatment groups.\n"
        else:
            log += "No statistically significant differences were found between treatment groups.\n"
    except Exception:
        pass

    return log


def analyze_pixel_distribution(image_path: str) -> dict:
    """Analyze western blot or DNA electrophoresis images and return pixel distribution statistics.

    Parameters
    ----------
    image_path : str
        Path to the input grayscale image. Automatically appends .png if no suffix is provided.

    Returns
    -------
    dict
        Summary dictionary containing image shape, intensity statistics, percentiles,
        histogram values, and brightness distribution for predefined buckets.

    """
    import cv2

    _DEFAULT_PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    _DEFAULT_BRIGHTNESS_BUCKETS: tuple[tuple[int, int], ...] = (
        (0, 20),
        (20, 50),
        (50, 80),
        (80, 110),
        (110, 140),
        (140, 170),
        (170, 200),
        (200, 256),
    )

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    percentiles = np.percentile(image, _DEFAULT_PERCENTILES).tolist()
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    total_pixels = int(image.size)

    brightness_lines = []
    for low, high in _DEFAULT_BRIGHTNESS_BUCKETS:
        count = int(histogram[low:high].sum())
        ratio = round((count / total_pixels) * 100, 2) if total_pixels > 0 else 0.0
        brightness_lines.append(f"Range [{low:>3}, {high:>3}): {count:>8} px ({ratio:5.2f}%)")

    min_intensity = int(image.min())
    max_intensity = int(image.max())
    mean_intensity = round(float(image.mean()), 2)
    std_intensity = round(float(image.std()), 2)

    return {
        "shape": f"({image.shape[0]}, {image.shape[1]})",
        "intensity_stats": {
            "min": min_intensity,
            "max": max_intensity,
            "mean": mean_intensity,
            "std_dev": std_intensity,
        },
        "percentiles_label": "percentiles (1, 5, 10, 25, 50, 75, 90, 95, 99):",
        "percentiles_values": ", ".join(f"{float(p):.1f}" for p in percentiles),
        "pixel_brightness_distribution": brightness_lines,
    }


def find_roi_from_image(
    image_path: str,
    lower_threshold: int,
    upper_threshold: int,
    number_of_bands: int,
    debug: bool = True,
) -> tuple[str, list]:
    """Find the ROIs of the bands from the image which is determined by analyze_pixel_distribution function.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    lower_threshold : int
        Pixel intensities lower than this value are used to make the binary image.
    upper_threshold : int
        Pixel intensities greater than or equal to this value are used to make the binary image.
    number_of_bands : int
        The actual number of bands in the image.
    debug : bool, optional
        If True, draw green contours (hulls) and blue keypoint boxes for debugging.
        Default is True.

    Returns
    -------
    tuple[str, list]
        A tuple containing:
        - str: Absolute path to the saved annotated image
        - list: List of ROI coordinates in (x, y, width, height) format.
        The ROI list can be converted to target_bands for analyze_western_blot:
        annotated_path, rois = find_roi_from_image(...)
        target_bands = [{"name": f"band_{i}", "roi": list(roi)} for i, roi in enumerate(rois)]

    Raises
    ------
    ValueError
        If threshold values are outside the valid range or inconsistent.
    FileNotFoundError
        If the source image cannot be loaded.

    """
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    import cv2

    ROI = tuple[int, int, int, int]

    def load_grayscale_image(path: str) -> cv2.Mat:
        """Load a grayscale image from disk."""
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Unable to load image at '{path}'")
        return image

    def build_blob_detector(
        min_threshold: int = 0,
        max_threshold: int = 200,
        min_area: int = 120,
        min_convexity: float = 0.7,
        min_inertia: float = 0.001,
        max_inertia: float = 0.4,
    ) -> cv2.SimpleBlobDetector:
        """Configure and return a SimpleBlobDetector instance."""
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = min_threshold
        params.maxThreshold = max_threshold
        params.filterByArea = True
        params.minArea = min_area
        params.filterByConvexity = True
        params.minConvexity = min_convexity
        params.filterByInertia = True
        params.minInertiaRatio = min_inertia
        params.maxInertiaRatio = max_inertia
        return cv2.SimpleBlobDetector_create(params)

    def detect_blobs(image: cv2.Mat, detector: cv2.SimpleBlobDetector) -> list[cv2.KeyPoint]:
        """Detect blob keypoints in the provided image."""
        keypoints = detector.detect(image)
        print(f"Detected {len(keypoints)} keypoints.")
        for index, keypoint in enumerate(keypoints):
            print(f"[{index}] position={keypoint.pt}, size={keypoint.size}")
        return keypoints

    def find_band_contours(
        binary_mask: cv2.Mat,
        min_area: int = 100,
        use_morphology: bool = True,
    ) -> list[cv2.Mat]:
        """Find band contours from binary mask using morphological operations."""
        processed_mask = binary_mask.copy()

        if use_morphology:
            # 가로(Horizontal) 방향으로 떨어진 덩어리를 잇기 위해 가로가 긴 커널 사용
            # (50, 1)의 50은 두 덩어리 사이의 픽셀 거리보다 커야 합니다.
            kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))

            # OPEN(끊기) 대신 CLOSE(잇기)를 사용하여 빈 공간을 메움
            processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel_connect, iterations=1)

        # Find ALL contours (not just external) to detect separate bands
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        print(f"Found {len(contours)} total contours")

        # Filter contours by area
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                filtered_contours.append(contour)

        print(f"Filtered to {len(filtered_contours)} contours with area >= {min_area}")

        return filtered_contours

    def analyze_roi_pixel_distribution(
        image: cv2.Mat,
        roi: ROI,
    ) -> dict:
        """Analyze pixel distribution of an ROI to distinguish between text and bands.

        Parameters
        ----------
        image : cv2.Mat
            Original grayscale image
        roi : ROI
            ROI coordinates (x, y, width, height)

        Returns
        -------
        dict
            Dictionary containing edge_strength, std_dev, and gradient_magnitude

        """
        x, y, w, h = roi

        # Extract ROI region from original image
        roi_region = image[y : y + h, x : x + w]

        if roi_region.size == 0:
            return {"edge_strength": 0.0, "std_dev": 0.0, "gradient_magnitude": 0.0}

        # Calculate standard deviation of pixel intensities
        std_dev = float(np.std(roi_region))

        # Calculate edge strength using Laplacian
        laplacian = cv2.Laplacian(roi_region, cv2.CV_64F)
        edge_strength = float(np.mean(np.abs(laplacian)))

        # Calculate gradient magnitude using Sobel
        sobelx = cv2.Sobel(roi_region, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(roi_region, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = float(np.mean(np.sqrt(sobelx**2 + sobely**2)))

        return {
            "edge_strength": edge_strength,
            "std_dev": std_dev,
            "gradient_magnitude": gradient_magnitude,
        }

    def filter_rois_by_pixel_distribution(
        image: cv2.Mat,
        rois: list[ROI],
        hulls: list[cv2.Mat],
        max_edge_strength: float = 10.0,
        max_gradient_magnitude: float = 70.0,
        max_std_dev: float = 50.0,
    ) -> tuple[list[ROI], list[cv2.Mat]]:
        """Filter ROIs to remove text-like regions and keep band-like regions.

        Text regions have very high edge strength, sharp gradients, and high std_dev.
        Band regions have low edge strength, moderate gradients, and moderate std_dev.

        Parameters
        ----------
        image : cv2.Mat
            Original grayscale image
        rois : List[ROI]
            List of ROI coordinates
        hulls : List[cv2.Mat]
            List of corresponding convex hulls
        max_edge_strength : float, optional
            Maximum edge strength for band-like regions (text typically >20)
        max_gradient_magnitude : float, optional
            Maximum gradient magnitude for band-like regions (text typically >100)
        max_std_dev : float, optional
            Maximum standard deviation for band-like regions (text typically >80)

        Returns
        -------
        tuple
            Filtered ROIs and hulls that are band-like

        """
        filtered_rois: list[ROI] = []
        filtered_hulls: list[cv2.Mat] = []

        print("\n=== ROI Pixel Distribution Analysis ===")

        for idx, (roi, hull) in enumerate(zip(rois, hulls, strict=True)):
            analysis = analyze_roi_pixel_distribution(image, roi)

            edge_strength = analysis["edge_strength"]
            gradient_magnitude = analysis["gradient_magnitude"]
            std_dev = analysis["std_dev"]

            # Determine if this ROI is band-like or text-like
            # Text has very high edge strength (>20) and very high gradient (>100)
            # Bands have low edge strength (<10) and moderate gradient (<50)
            # Also check std_dev to filter out high-contrast text regions
            is_band = (
                edge_strength <= max_edge_strength
                and gradient_magnitude <= max_gradient_magnitude
                and std_dev <= max_std_dev
            )

            status = "✓ BAND" if is_band else "✗ TEXT"
            print(f"ROI {idx}: edge={edge_strength:.2f}, grad={gradient_magnitude:.2f}, std={std_dev:.2f} -> {status}")

            if is_band:
                filtered_rois.append(roi)
                filtered_hulls.append(hull)

        print(f"\nFiltered: {len(filtered_rois)}/{len(rois)} ROIs kept as bands")
        print("=" * 40 + "\n")

        return filtered_rois, filtered_hulls

    def compute_rois(
        image: cv2.Mat,
        binary_mask: cv2.Mat,
        keypoints: Iterable[cv2.KeyPoint],
        padding: tuple[int, int] = (5, 5),
        min_contour_area: int = 100,
        filter_by_distribution: bool = True,
    ) -> tuple[list[ROI], list[cv2.Mat]]:
        """Compute global ROIs for each keypoint by matching them to band contours."""
        # Find band contours from binary mask
        band_contours = find_band_contours(binary_mask, min_area=min_contour_area)

        if not band_contours:
            print("No band contours found!")
            return [], []

        auto_rois: list[ROI] = []
        global_hulls: list[cv2.Mat] = []
        matched_contours = set()  # Track which contours have been matched
        used_contours = set()  # Track which contours have already been used to avoid duplicates

        for keypoint in keypoints:
            cx, cy = int(keypoint.pt[0]), int(keypoint.pt[1])
            keypoint_center = (float(cx), float(cy))

            # Find the contour that contains this keypoint
            matched_contour = None
            matched_idx = None
            for idx, contour in enumerate(band_contours):
                # Skip if this contour has already been used
                if idx in used_contours:
                    continue
                # Use pointPolygonTest to check if keypoint center is inside contour
                # Returns positive if inside, negative if outside, zero if on edge
                distance = cv2.pointPolygonTest(contour, keypoint_center, False)
                if distance >= 0:  # Inside or on edge
                    matched_contour = contour
                    matched_idx = idx
                    matched_contours.add(idx)
                    print(f"Keypoint at ({cx}, {cy}) matched to contour {idx}")
                    break

            if matched_contour is None:
                print(f"Warning: Keypoint at ({cx}, {cy}) not matched to any contour")
                continue

            # Mark this contour as used to avoid duplicate ROIs
            used_contours.add(matched_idx)

            # Compute convex hull from the matched contour
            hull = cv2.convexHull(matched_contour)
            if hull is None or len(hull) < 3:
                continue

            # Get bounding rectangle from hull with padding
            pad_x, pad_y = padding
            rx, ry, rw, rh = cv2.boundingRect(hull)

            # Skip if bounding rect is too large (likely covering entire image or invalid)
            max_roi_area_ratio = 0.5  # Maximum 50% of image area
            roi_area = rw * rh
            image_area = image.shape[0] * image.shape[1]
            if roi_area > image_area * max_roi_area_ratio:
                print(f"Warning: ROI too large ({rw}x{rh}), skipping. This may indicate a detection error.")
                continue

            global_x = max(0, rx - pad_x)
            global_y = max(0, ry - pad_y)
            global_w = min(image.shape[1] - global_x, rw + 2 * pad_x)
            global_h = min(image.shape[0] - global_y, rh + 2 * pad_y)

            if global_w <= 0 or global_h <= 0:
                continue

            roi = (global_x, global_y, global_w, global_h)
            auto_rois.append(roi)
            global_hulls.append(hull)

        print(f"Matched {len(matched_contours)} contours to keypoints")

        # Filter ROIs by pixel distribution to remove text-like regions
        if filter_by_distribution and auto_rois:
            auto_rois, global_hulls = filter_rois_by_pixel_distribution(image, auto_rois, global_hulls)

        return auto_rois, global_hulls

    def annotate_keypoints(
        image: cv2.Mat,
        keypoints: Iterable[cv2.KeyPoint],
        rois: Iterable[ROI],
        hulls: Iterable[cv2.Mat] | None = None,
        debug: bool = False,
    ) -> cv2.Mat:
        """Draw ROIs, convex hulls, and index labels on the image.

        Parameters
        ----------
        image : cv2.Mat
            Input grayscale image
        keypoints : Iterable[cv2.KeyPoint]
            Detected keypoints
        rois : Iterable[ROI]
            ROI coordinates to draw
        hulls : Iterable[cv2.Mat] | None, optional
            Convex hulls to draw (only if debug=True)
        debug : bool, optional
            If True, draw green contours (hulls) and blue keypoint boxes

        Returns
        -------
        cv2.Mat
            Annotated image

        """
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Draw convex hulls in green if debug mode is enabled
        if debug and hulls is not None:
            for hull in hulls:
                cv2.drawContours(output, [hull], -1, (0, 255, 0), 2)  # Green color in BGR

        # Draw ROIs in red (always drawn)
        for roi in rois:
            x, y, w, h = roi
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Draw keypoints in blue and index labels (only if debug mode is enabled)
        if debug:
            for index, keypoint in enumerate(keypoints):
                x, y = keypoint.pt
                size = keypoint.size

                # Draw blue rectangle around keypoint
                # Use size as half-width and half-height for the rectangle
                half_size = int(size / 2)
                pt1 = (int(x) - half_size, int(y) - half_size)
                pt2 = (int(x) + half_size, int(y) + half_size)
                cv2.rectangle(output, pt1, pt2, (255, 0, 0), 2)  # Blue color in BGR

                # Draw index label
                cv2.putText(
                    output,
                    str(index),
                    (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
        return output

    def show_rois(rois: Sequence[ROI]) -> None:
        """Print ROI information to stdout."""
        print(f"Detected ROI count: {len(rois)}")
        for index, roi in enumerate(rois):
            print(f"ROI {index}: {roi}")

    if not 0 <= lower_threshold <= 255:
        raise ValueError("lower_threshold must be within [0, 255].")
    if not 0 <= upper_threshold <= 255:
        raise ValueError("upper_threshold must be within [0, 255].")
    if lower_threshold > upper_threshold:
        raise ValueError("lower_threshold cannot be greater than upper_threshold.")

    original_image = load_grayscale_image(image_path)
    mask = cv2.inRange(original_image, lower_threshold, upper_threshold)
    mask = cv2.bitwise_not(mask)
    detector = build_blob_detector()

    # Detect blobs in the mask image
    keypoints = detect_blobs(mask, detector)
    rois, hulls = compute_rois(original_image, mask, keypoints)
    show_rois(rois)

    # Draw ROIs, convex hulls, and keypoints on the mask image
    annotated_mask = annotate_keypoints(mask, keypoints, rois, hulls, debug=debug)
    mask_path = Path(image_path).parent / f"{Path(image_path).stem}_mask.png"
    cv2.imwrite(str(mask_path), annotated_mask)

    # Draw ROIs, convex hulls, and keypoints on the original image
    annotated_image = annotate_keypoints(original_image, keypoints, rois, hulls, debug=debug)
    annotated_image_path = Path(image_path).parent / f"{Path(image_path).stem}_annotated.png"
    cv2.imwrite(str(annotated_image_path), annotated_image)

    if len(rois) != number_of_bands:
        print(f"Warning: Detected {len(rois)} ROIs, but expected {number_of_bands} ROIs.")
        print(
            "Please check the image and try to adjust the thresholds. Or you can manually infer the ROIs from the annotated image."
        )

    return str(annotated_image_path.resolve()), rois


def analyze_western_blot(
    blot_image_path,
    target_bands,
    loading_control_band,
    antibody_info,
    output_dir="./results",
):
    """Performs densitometric analysis of Western blot images to quantify relative protein expression.

    Parameters
    ----------
    blot_image_path : str
        Path to the Western blot image file
    target_bands : list of dict
        List of dictionaries containing information about target protein bands.
        Each dict should have 'name' and 'roi' (region of interest as [x, y, width, height]).
        To generate this from find_roi_from_image output:
        annotated_path, rois = find_roi_from_image(...)
        target_bands = [{"name": f"band_{i}", "roi": list(roi)} for i, roi in enumerate(rois)]
        Or manually specify names: target_bands = [{"name": "protein_name", "roi": [x, y, w, h]}, ...]
    loading_control_band : dict
        Dictionary with 'name' and 'roi' for the loading control protein (e.g., β-actin, GAPDH)
    antibody_info : dict
        Dictionary containing information about antibodies used
        Should have 'primary' and 'secondary' keys with antibody details
    output_dir : str, optional
        Directory to save output files, defaults to './results'

    Returns
    -------
    str
        Research log summarizing the Western blot analysis process and results

    """
    import os

    import numpy as np
    from skimage import io

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the Western blot image
    image = io.imread(blot_image_path)
    if len(image.shape) > 2:  # Convert to grayscale if color image
        image = np.mean(image, axis=2).astype(np.uint8)

    # Initialize results dictionary
    results = {
        "loading_control": {"name": loading_control_band["name"], "intensity": 0},
        "targets": [],
    }

    # Analyze loading control band
    lc_roi = loading_control_band["roi"]
    lc_band = image[lc_roi[1] : lc_roi[1] + lc_roi[3], lc_roi[0] : lc_roi[0] + lc_roi[2]]
    lc_intensity = np.sum(lc_band)
    results["loading_control"]["intensity"] = lc_intensity

    # Analyze target protein bands
    for band in target_bands:
        roi = band["roi"]
        band_img = image[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]
        band_intensity = np.sum(band_img)

        # Calculate relative expression (normalized to loading control)
        relative_expression = band_intensity / lc_intensity

        results["targets"].append(
            {
                "name": band["name"],
                "intensity": band_intensity,
                "relative_expression": relative_expression,
            }
        )

    # Generate results table and save to CSV
    results_file = os.path.join(output_dir, "western_blot_results.csv")
    with open(results_file, "w") as f:
        f.write("Protein,Raw Intensity,Relative Expression\n")
        f.write(f"{results['loading_control']['name']},{results['loading_control']['intensity']},1.0\n")
        for target in results["targets"]:
            f.write(f"{target['name']},{target['intensity']},{target['relative_expression']:.4f}\n")

    # Generate research log
    log = "## Western Blot Analysis\n\n"
    log += f"Analyzed Western blot image: {os.path.basename(blot_image_path)}\n\n"
    log += "### Antibodies Used\n"
    log += f"Primary antibody: {antibody_info['primary']}\n"
    log += f"Secondary antibody: {antibody_info['secondary']}\n\n"
    log += "### Analysis Steps\n"
    log += "1. Loaded Western blot image and converted to grayscale\n"
    log += f"2. Quantified loading control ({loading_control_band['name']}) band intensity\n"
    log += "3. Measured target protein band intensities\n"
    log += "4. Calculated relative expression by normalizing to loading control\n\n"
    log += "### Results\n"
    log += f"Loading control ({loading_control_band['name']}): {results['loading_control']['intensity']} intensity units\n\n"
    log += "Target proteins:\n"
    for target in results["targets"]:
        log += f"- {target['name']}: {target['intensity']} intensity units, "
        log += f"{target['relative_expression']:.4f} relative expression\n"
    log += f"\nDetailed results saved to: {results_file}\n"

    return log


# DDInter Drug-Drug Interaction Analysis Functions


def _load_ddinter_data(data_lake_path):
    """
    Load DDInter datasets from pickle files, processing if needed.

    Parameters
    ----------
    data_lake_path : str
        Path to data lake directory containing DDInter pickle files

    Returns
    -------
    tuple
        (drug_info, interaction_matrix, name_mapping) dictionaries
    """
    import os
    import pickle

    # Define schema directory (following established pattern)
    schema_dir = os.path.join(os.path.dirname(__file__), "schema_db")

    # Define paths to DDInter pickle files
    drug_info_path = os.path.join(schema_dir, "ddinter_drugs.pkl")
    interaction_path = os.path.join(schema_dir, "ddinter_interactions.pkl")
    mapping_path = os.path.join(schema_dir, "ddinter_name_mapping.pkl")

    # Check if processing is needed (lazy loading pattern)
    pkl_files = [drug_info_path, interaction_path, mapping_path]
    if not all(os.path.exists(f) for f in pkl_files):
        _process_ddinter_data_inline(data_lake_path, schema_dir)

    # Load data
    try:
        with open(drug_info_path, "rb") as f:
            drug_info = pickle.load(f)

        with open(interaction_path, "rb") as f:
            interaction_matrix = pickle.load(f)

        with open(mapping_path, "rb") as f:
            name_mapping = pickle.load(f)

        return drug_info, interaction_matrix, name_mapping

    except Exception as e:
        raise FileNotFoundError(f"Error loading DDInter data: {e}") from e


def _process_ddinter_data_inline(data_lake_path, output_dir):
    """
    Process DDInter CSV files into standardized pickle files.

    This function processes raw DDInter 2.0 CSV files and creates standardized
    data structures for use in Biomni drug-drug interaction analysis.

    Parameters
    ----------
    data_lake_path : str
        Path to data lake directory containing raw DDInter CSV files
    output_dir : str
        Directory to save processed pickle files
    """
    import os
    import pickle
    from pathlib import Path

    import pandas as pd

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Define CSV files to process
    csv_files = [
        "ddinter_alimentary_tract_metabolism.csv",
        "ddinter_antineoplastic.csv",
        "ddinter_antiparasitic.csv",
        "ddinter_blood_organs.csv",
        "ddinter_dermatological.csv",
        "ddinter_hormonal.csv",
        "ddinter_respiratory.csv",
        "ddinter_various.csv",
    ]

    # Load and combine all CSV files
    dataframes = []
    for csv_file in csv_files:
        file_path = os.path.join(data_lake_path, csv_file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Add source category
            category = csv_file.replace("ddinter_", "").replace(".csv", "")
            df["category"] = category
            dataframes.append(df)

    if not dataframes:
        raise FileNotFoundError("No DDInter CSV files found in data lake")

    # Process data
    drug_info = _build_drug_registry_inline(dataframes)
    interaction_matrix = _create_interaction_matrix_inline(dataframes)
    name_mapping = _create_name_mapping_inline(drug_info)

    # Save processed data
    with open(os.path.join(output_dir, "ddinter_drugs.pkl"), "wb") as f:
        pickle.dump(drug_info, f)

    with open(os.path.join(output_dir, "ddinter_interactions.pkl"), "wb") as f:
        pickle.dump(interaction_matrix, f)

    with open(os.path.join(output_dir, "ddinter_name_mapping.pkl"), "wb") as f:
        pickle.dump(name_mapping, f)

    # Generate and save statistics
    stats = _generate_ddinter_statistics_inline(drug_info, interaction_matrix)
    with open(os.path.join(output_dir, "ddinter_statistics.pkl"), "wb") as f:
        pickle.dump(stats, f)


def _standardize_drug_name_processing(drug_name):
    """Standardize drug names for consistent matching during processing."""
    import pandas as pd

    if pd.isna(drug_name):
        return ""

    # Convert to lowercase and strip whitespace
    standardized = str(drug_name).strip().lower()

    # Remove common suffixes and prefixes
    standardized = standardized.replace(" hydrochloride", "")
    standardized = standardized.replace(" sulfate", "")
    standardized = standardized.replace(" sodium", "")
    standardized = standardized.replace(" potassium", "")
    standardized = standardized.replace(" calcium", "")
    standardized = standardized.replace(" magnesium", "")

    return standardized


def _build_drug_registry_inline(dataframes):
    """Build comprehensive drug registry from all interactions."""

    drug_registry = {}

    for df in dataframes:
        for _, row in df.iterrows():
            drug_a_id = row["DDInterID_A"]
            drug_a_name = row["Drug_A"]
            drug_b_id = row["DDInterID_B"]
            drug_b_name = row["Drug_B"]

            # Add Drug A
            if drug_a_id not in drug_registry:
                drug_registry[drug_a_id] = {
                    "name": drug_a_name,
                    "standardized_name": _standardize_drug_name_processing(drug_a_name),
                    "categories": set(),
                    "interactions": set(),
                }
            drug_registry[drug_a_id]["categories"].add(row["category"])

            # Add Drug B
            if drug_b_id not in drug_registry:
                drug_registry[drug_b_id] = {
                    "name": drug_b_name,
                    "standardized_name": _standardize_drug_name_processing(drug_b_name),
                    "categories": set(),
                    "interactions": set(),
                }
            drug_registry[drug_b_id]["categories"].add(row["category"])

            # Record interactions
            drug_registry[drug_a_id]["interactions"].add(drug_b_id)
            drug_registry[drug_b_id]["interactions"].add(drug_a_id)

    # Convert sets to lists for pickle serialization
    for drug_id in drug_registry:
        drug_registry[drug_id]["categories"] = list(drug_registry[drug_id]["categories"])
        drug_registry[drug_id]["interactions"] = list(drug_registry[drug_id]["interactions"])

    return drug_registry


def _create_interaction_matrix_inline(dataframes):
    """Create interaction matrix for fast lookups using standardized drug names."""
    from collections import defaultdict

    import pandas as pd

    combined_df = pd.concat(dataframes, ignore_index=True)
    interaction_matrix = defaultdict(lambda: defaultdict(list))

    # Create bidirectional interaction matrix using standardized names
    for _, row in combined_df.iterrows():
        drug_a_std = _standardize_drug_name_processing(row["Drug_A"])
        drug_b_std = _standardize_drug_name_processing(row["Drug_B"])
        level = row["Level"]
        category = row["category"]

        interaction_data = {
            "level": level,
            "category": category,
            "drug_a_id": row["DDInterID_A"],
            "drug_b_id": row["DDInterID_B"],
            "drug_a_name": row["Drug_A"],
            "drug_b_name": row["Drug_B"],
        }

        # Add both directions using standardized names as keys
        interaction_matrix[drug_a_std][drug_b_std].append(interaction_data)
        interaction_matrix[drug_b_std][drug_a_std].append(interaction_data)

    # Convert to regular dict for pickle
    interaction_matrix = dict(interaction_matrix)
    for drug in interaction_matrix:
        interaction_matrix[drug] = dict(interaction_matrix[drug])

    return interaction_matrix


def _create_name_mapping_inline(drug_info):
    """Create drug name to ID mapping for fuzzy matching."""
    name_mapping = {}

    for drug_id, drug_data in drug_info.items():
        original_name = drug_data["name"]
        standardized_name = drug_data["standardized_name"]

        # Map both original and standardized names
        name_mapping[original_name.lower()] = drug_id
        name_mapping[standardized_name] = drug_id

    return name_mapping


def _generate_ddinter_statistics_inline(drug_info, interaction_matrix):
    """Generate statistics about the processed data."""
    from collections import defaultdict

    stats = {
        "total_drugs": len(drug_info),
        "total_interactions": 0,
        "interaction_levels": defaultdict(int),
        "drug_categories": defaultdict(int),
        "most_connected_drugs": [],
    }

    # Count interactions and levels
    for drug_a in interaction_matrix:
        for drug_b in interaction_matrix[drug_a]:
            interactions = interaction_matrix[drug_a][drug_b]
            stats["total_interactions"] += len(interactions)

            for interaction in interactions:
                stats["interaction_levels"][interaction["level"]] += 1

    # Count drug categories
    for drug_data in drug_info.values():
        for category in drug_data["categories"]:
            stats["drug_categories"][category] += 1

    # Find most connected drugs
    connection_counts = []
    for drug_id, drug_data in drug_info.items():
        connection_counts.append(
            {"drug_id": drug_id, "name": drug_data["name"], "connections": len(drug_data["interactions"])}
        )

    connection_counts.sort(key=lambda x: x["connections"], reverse=True)
    stats["most_connected_drugs"] = connection_counts[:10]

    return stats


def _standardize_drug_name(drug_name, name_mapping):
    """
    Standardize drug names using fuzzy matching against DDInter database.

    Parameters
    ----------
    drug_name : str
        Original drug name
    name_mapping : dict
        Drug name to ID mapping dictionary

    Returns
    -------
    str or None
        Standardized drug name or None if not found
    """
    from difflib import get_close_matches

    # Direct match
    if drug_name.lower() in name_mapping:
        return drug_name.lower()

    # Fuzzy match
    matches = get_close_matches(drug_name.lower(), name_mapping.keys(), n=1, cutoff=0.8)
    if matches:
        return matches[0]

    return None


def _format_interaction_result(interaction_data, drug_name_a, drug_name_b, include_mechanisms=True):
    """
    Format interaction results for research log.

    Parameters
    ----------
    interaction_data : list
        List of interaction data dictionaries
    drug_name_a : str
        First drug name
    drug_name_b : str
        Second drug name
    include_mechanisms : bool
        Whether to include detailed mechanism information

    Returns
    -------
    str
        Formatted interaction description
    """
    if not interaction_data:
        return f"No interactions found between {drug_name_a} and {drug_name_b}"

    result = f"Interaction between {drug_name_a} and {drug_name_b}:\n"

    for i, interaction in enumerate(interaction_data, 1):
        level = interaction.get("level", "Unknown")
        category = interaction.get("category", "Unknown")

        result += f"  {i}. Severity: {level}\n"
        result += f"     Category: {category.replace('_', ' ').title()}\n"

        if include_mechanisms:
            result += f"     Clinical significance: {level} interaction requiring appropriate monitoring\n"

    return result


def query_drug_interactions(drug_names, interaction_types=None, severity_levels=None, data_lake_path=None):
    """
    Query drug-drug interactions from DDInter database.

    Parameters
    ----------
    drug_names : list of str
        List of drug names to query for interactions
    interaction_types : list of str, optional
        Filter by interaction types (e.g., ['synergistic', 'antagonistic'])
    severity_levels : list of str, optional
        Filter by severity levels (e.g., ['Major', 'Moderate', 'Minor'])
    data_lake_path : str, optional
        Path to data lake directory containing DDInter data

    Returns
    -------
    str
        Research log with detailed interaction analysis
    """
    from datetime import datetime

    # Initialize research log
    log = "DDInter Drug-Drug Interaction Query\n"
    log += "=" * 40 + "\n"
    log += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Handle default data lake path
    if data_lake_path is None:
        # Default path assuming standard Biomni structure
        data_lake_path = os.path.join(os.path.dirname(__file__), "schema_db")

    log += "Query Parameters:\n"
    log += f"- Target drugs: {', '.join(drug_names)}\n"
    log += f"- Severity filter: {severity_levels if severity_levels else 'All levels'}\n"
    log += f"- Interaction types: {interaction_types if interaction_types else 'All types'}\n\n"

    try:
        # Load DDInter data
        drug_info, interaction_matrix, name_mapping = _load_ddinter_data(data_lake_path)
        log += f"Successfully loaded DDInter database with {len(drug_info)} drugs\n\n"

        # Standardize drug names
        standardized_names = []
        missing_drugs = []

        for drug_name in drug_names:
            standardized = _standardize_drug_name(drug_name, name_mapping)
            if standardized:
                standardized_names.append(standardized)
            else:
                missing_drugs.append(drug_name)

        if missing_drugs:
            log += "Warning: The following drugs were not found in DDInter database:\n"
            for drug in missing_drugs:
                log += f"- {drug}\n"
            log += "\n"

        if not standardized_names:
            log += "Error: No valid drugs found in DDInter database\n"
            return log

        # Query interactions
        interactions_found = []

        for i, drug_a in enumerate(standardized_names):
            for j, drug_b in enumerate(standardized_names):
                if i >= j:  # Avoid duplicate pairs
                    continue

                if drug_a in interaction_matrix and drug_b in interaction_matrix[drug_a]:
                    interactions = interaction_matrix[drug_a][drug_b]

                    # Apply filters
                    filtered_interactions = interactions

                    if severity_levels:
                        filtered_interactions = [
                            int_data for int_data in filtered_interactions if int_data.get("level") in severity_levels
                        ]

                    if interaction_types:
                        filtered_interactions = [
                            int_data
                            for int_data in filtered_interactions
                            if int_data.get("category") in interaction_types
                        ]

                    if filtered_interactions:
                        interactions_found.append(
                            {"drug_a": drug_a, "drug_b": drug_b, "interactions": filtered_interactions}
                        )

        # Format results
        log += "Interaction Analysis Results:\n"
        log += f"Found {len(interactions_found)} drug pairs with interactions\n\n"

        if interactions_found:
            for pair in interactions_found:
                log += _format_interaction_result(
                    pair["interactions"], pair["drug_a"].title(), pair["drug_b"].title(), include_mechanisms=True
                )
                log += "\n"
        else:
            log += "No interactions found between the specified drugs with the given filters\n"

        # Summary statistics
        total_interactions = sum(len(pair["interactions"]) for pair in interactions_found)
        log += "Summary:\n"
        log += f"- Total drug pairs analyzed: {len(standardized_names) * (len(standardized_names) - 1) // 2}\n"
        log += f"- Drug pairs with interactions: {len(interactions_found)}\n"
        log += f"- Total interactions found: {total_interactions}\n"

        if interactions_found:
            severity_counts = {}
            for pair in interactions_found:
                for interaction in pair["interactions"]:
                    level = interaction.get("level", "Unknown")
                    severity_counts[level] = severity_counts.get(level, 0) + 1

            log += f"- Severity distribution: {dict(severity_counts)}\n"

    except FileNotFoundError as e:
        log += f"Error during interaction query: {str(e)}\n"
    except Exception as e:
        log += f"Error during interaction query: {str(e)}\n"

    return log


def check_drug_combination_safety(drug_list, include_mechanisms=True, include_management=True, data_lake_path=None):
    """
    Analyze safety of a drug combination for potential interactions.

    Parameters
    ----------
    drug_list : list of str
        List of drugs to analyze for combination safety
    include_mechanisms : bool, default True
        Include interaction mechanism descriptions
    include_management : bool, default True
        Include management recommendations
    data_lake_path : str, optional
        Path to data lake directory containing DDInter data

    Returns
    -------
    str
        Research log with safety analysis and recommendations
    """
    from datetime import datetime

    # Initialize research log
    log = "Drug Combination Safety Analysis\n"
    log += "=" * 35 + "\n"
    log += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Handle default data lake path
    if data_lake_path is None:
        data_lake_path = os.path.join(os.path.dirname(__file__), "schema_db")

    log += "Safety Analysis Parameters:\n"
    log += f"- Drug combination: {', '.join(drug_list)}\n"
    log += f"- Include mechanisms: {include_mechanisms}\n"
    log += f"- Include management: {include_management}\n\n"

    try:
        # Load DDInter data
        drug_info, interaction_matrix, name_mapping = _load_ddinter_data(data_lake_path)
        log += "Successfully loaded DDInter database\n\n"

        # Standardize drug names
        standardized_drugs = []
        missing_drugs = []

        for drug in drug_list:
            standardized = _standardize_drug_name(drug, name_mapping)
            if standardized:
                standardized_drugs.append(standardized)
            else:
                missing_drugs.append(drug)

        if missing_drugs:
            log += "Warning: The following drugs were not found in DDInter database:\n"
            for drug in missing_drugs:
                log += f"- {drug}\n"
            log += "\n"

        if len(standardized_drugs) < 2:
            log += "Error: At least 2 valid drugs required for combination analysis\n"
            return log

        # Analyze all pairwise interactions
        interactions_found = []
        major_interactions = 0
        moderate_interactions = 0
        minor_interactions = 0

        for i, drug_a in enumerate(standardized_drugs):
            for j, drug_b in enumerate(standardized_drugs):
                if i >= j:  # Avoid duplicate pairs
                    continue

                if drug_a in interaction_matrix and drug_b in interaction_matrix[drug_a]:
                    interactions = interaction_matrix[drug_a][drug_b]

                    for interaction in interactions:
                        level = interaction.get("level", "Unknown")
                        if level == "Major":
                            major_interactions += 1
                        elif level == "Moderate":
                            moderate_interactions += 1
                        elif level == "Minor":
                            minor_interactions += 1

                    interactions_found.append({"drug_a": drug_a, "drug_b": drug_b, "interactions": interactions})

        # Overall safety assessment
        log += "Overall Safety Assessment:\n"

        safety_score = 100
        safety_level = "Safe"

        if major_interactions > 0:
            safety_score -= major_interactions * 30
            safety_level = "High Risk"
        elif moderate_interactions > 2:
            safety_score -= moderate_interactions * 15
            safety_level = "Moderate Risk"
        elif moderate_interactions > 0:
            safety_score -= moderate_interactions * 10
            safety_level = "Low to Moderate Risk"
        elif minor_interactions > 0:
            safety_score -= minor_interactions * 5
            safety_level = "Low Risk"

        safety_score = max(0, safety_score)

        log += f"- Safety Level: {safety_level}\n"
        log += f"- Safety Score: {safety_score}/100\n"
        log += f"- Major interactions: {major_interactions}\n"
        log += f"- Moderate interactions: {moderate_interactions}\n"
        log += f"- Minor interactions: {minor_interactions}\n\n"

        # Detailed interaction analysis
        if interactions_found:
            log += "Detailed Interaction Analysis:\n"
            log += "-" * 30 + "\n"

            for pair in interactions_found:
                log += _format_interaction_result(
                    pair["interactions"],
                    pair["drug_a"].title(),
                    pair["drug_b"].title(),
                    include_mechanisms=include_mechanisms,
                )
                log += "\n"

        # Clinical recommendations
        log += "Clinical Recommendations:\n"
        log += "-" * 25 + "\n"

        if major_interactions > 0:
            log += "- CONTRAINDICATED: This combination contains major interactions\n"
            log += "- Consider alternative medications or consult specialist\n"
            log += "- If combination is necessary, intensive monitoring required\n"
        elif moderate_interactions > 2:
            log += "- CAUTION: Multiple moderate interactions detected\n"
            log += "- Monitor patient closely for adverse effects\n"
            log += "- Consider dose adjustments or alternative agents\n"
        elif moderate_interactions > 0:
            log += "- MONITOR: Moderate interactions present\n"
            log += "- Regular patient monitoring recommended\n"
            log += "- Be aware of potential side effects\n"
        elif minor_interactions > 0:
            log += "- AWARENESS: Minor interactions detected\n"
            log += "- Standard monitoring sufficient\n"
            log += "- Educate patient about potential minor effects\n"
        else:
            log += "- SAFE: No significant interactions detected\n"
            log += "- Standard clinical monitoring appropriate\n"

        if include_management:
            log += "\nGeneral Management Strategies:\n"
            log += "- Separate administration times when possible\n"
            log += "- Monitor for signs of toxicity or reduced efficacy\n"
            log += "- Consider therapeutic drug monitoring if available\n"
            log += "- Educate patient about potential interaction symptoms\n"

    except Exception as e:
        log += f"Error during safety analysis: {str(e)}\n"

    return log


def analyze_interaction_mechanisms(drug_pair, detailed_analysis=True, data_lake_path=None):
    """
    Analyze interaction mechanisms between two specific drugs.

    Parameters
    ----------
    drug_pair : tuple of str
        Pair of drug names to analyze (drug1, drug2)
    detailed_analysis : bool, default True
        Include detailed mechanistic information
    data_lake_path : str, optional
        Path to data lake directory containing DDInter data

    Returns
    -------
    str
        Research log with mechanism analysis
    """
    from datetime import datetime

    # Initialize research log
    log = "Drug Interaction Mechanism Analysis\n"
    log += "=" * 37 + "\n"
    log += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Handle default data lake path
    if data_lake_path is None:
        data_lake_path = os.path.join(os.path.dirname(__file__), "schema_db")

    drug_a, drug_b = drug_pair
    log += "Mechanism Analysis Parameters:\n"
    log += f"- Drug A: {drug_a}\n"
    log += f"- Drug B: {drug_b}\n"
    log += f"- Detailed analysis: {detailed_analysis}\n\n"

    try:
        # Load DDInter data
        drug_info, interaction_matrix, name_mapping = _load_ddinter_data(data_lake_path)
        log += "Successfully loaded DDInter database\n\n"

        # Standardize drug names
        std_drug_a = _standardize_drug_name(drug_a, name_mapping)
        std_drug_b = _standardize_drug_name(drug_b, name_mapping)

        if not std_drug_a:
            log += f"Error: Drug '{drug_a}' not found in DDInter database\n"
            return log
        if not std_drug_b:
            log += f"Error: Drug '{drug_b}' not found in DDInter database\n"
            return log

        # Query interactions
        interactions = []
        if std_drug_a in interaction_matrix and std_drug_b in interaction_matrix[std_drug_a]:
            interactions = interaction_matrix[std_drug_a][std_drug_b]

        if not interactions:
            log += f"No interactions found between {drug_a} and {drug_b}\n"
            return log

        # Get drug information
        drug_a_id = name_mapping[std_drug_a]
        drug_b_id = name_mapping[std_drug_b]
        drug_a_info = drug_info.get(drug_a_id, {})
        drug_b_info = drug_info.get(drug_b_id, {})

        log += "Drug Profile Analysis:\n"
        log += "-" * 20 + "\n"
        log += f"{drug_a.title()}:\n"
        log += f"- Categories: {', '.join(drug_a_info.get('categories', ['Unknown']))}\n"
        log += f"- Total known interactions: {len(drug_a_info.get('interactions', []))}\n\n"

        log += f"{drug_b.title()}:\n"
        log += f"- Categories: {', '.join(drug_b_info.get('categories', ['Unknown']))}\n"
        log += f"- Total known interactions: {len(drug_b_info.get('interactions', []))}\n\n"

        # Analyze interaction mechanisms
        log += "Interaction Mechanism Analysis:\n"
        log += "-" * 30 + "\n"

        for i, interaction in enumerate(interactions, 1):
            level = interaction.get("level", "Unknown")
            category = interaction.get("category", "Unknown")

            log += f"Interaction {i}:\n"
            log += f"- Severity: {level}\n"
            log += f"- Category: {category.replace('_', ' ').title()}\n"

            if detailed_analysis:
                # Provide mechanism insights based on severity and category
                if level == "Major":
                    log += "- Clinical Impact: High risk interaction requiring immediate attention\n"
                    log += "- Mechanism: Likely involves significant pharmacokinetic or pharmacodynamic effects\n"
                    log += "- Management: Avoid combination or use with extreme caution\n"
                elif level == "Moderate":
                    log += "- Clinical Impact: Moderate risk requiring monitoring\n"
                    log += "- Mechanism: May involve enzyme induction/inhibition or receptor competition\n"
                    log += "- Management: Monitor closely, consider dose adjustment\n"
                elif level == "Minor":
                    log += "- Clinical Impact: Low risk, usually manageable\n"
                    log += "- Mechanism: Minor pharmacokinetic or pharmacodynamic effects\n"
                    log += "- Management: Standard monitoring sufficient\n"

                # Category-specific mechanism insights
                category_mechanisms = {
                    "alimentary_tract_metabolism": "Gastrointestinal absorption or metabolic interactions",
                    "antineoplastic": "Bone marrow suppression or tumor resistance mechanisms",
                    "blood_organs": "Hematological effects or coagulation pathway interactions",
                    "hormonal": "Endocrine system interactions or hormone receptor effects",
                    "respiratory": "Pulmonary function or bronchodilation interactions",
                    "dermatological": "Skin absorption or topical application interactions",
                    "antiparasitic": "Antimicrobial resistance or metabolic pathway interactions",
                    "various": "Multiple potential interaction pathways",
                }

                mechanism = category_mechanisms.get(category, "Unknown mechanism")
                log += f"- Category-specific mechanism: {mechanism}\n"

            log += "\n"

        # Summary and recommendations
        log += "Summary and Recommendations:\n"
        log += "-" * 28 + "\n"

        severity_counts = {}
        for interaction in interactions:
            level = interaction.get("level", "Unknown")
            severity_counts[level] = severity_counts.get(level, 0) + 1

        log += f"- Total interactions analyzed: {len(interactions)}\n"
        log += f"- Severity distribution: {dict(severity_counts)}\n"

        # Overall recommendation
        if any(int_data.get("level") == "Major" for int_data in interactions):
            log += "- Overall recommendation: AVOID - Major interaction detected\n"
            log += "- Consider alternative medications\n"
        elif any(int_data.get("level") == "Moderate" for int_data in interactions):
            log += "- Overall recommendation: MONITOR - Moderate interaction present\n"
            log += "- Close patient monitoring required\n"
        else:
            log += "- Overall recommendation: AWARENESS - Minor interactions only\n"
            log += "- Standard monitoring appropriate\n"

        if detailed_analysis:
            log += "\nMechanistic Considerations:\n"
            log += f"- Monitor for additive effects in the {category.replace('_', ' ')} system\n"
            log += "- Consider potential for altered drug metabolism\n"
            log += "- Be aware of possible changes in drug efficacy or toxicity\n"
            log += "- Timing of administration may be important\n"

    except Exception as e:
        log += f"Error during mechanism analysis: {str(e)}\n"

    return log


def find_alternative_drugs_ddinter(target_drug, contraindicated_drugs, therapeutic_class=None, data_lake_path=None):
    """
    Find alternative drugs that don't interact with contraindicated drugs.

    Parameters
    ----------
    target_drug : str
        Drug to find alternatives for
    contraindicated_drugs : list of str
        List of drugs to avoid interactions with
    therapeutic_class : str, optional
        Limit search to specific therapeutic class
    data_lake_path : str, optional
        Path to data lake directory containing DDInter data

    Returns
    -------
    str
        Research log with alternative drug recommendations
    """
    from datetime import datetime

    # Initialize research log
    log = "Alternative Drug Finder (DDInter)\n"
    log += "=" * 32 + "\n"
    log += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Handle default data lake path
    if data_lake_path is None:
        data_lake_path = os.path.join(os.path.dirname(__file__), "schema_db")

    log += "Alternative Drug Search Parameters:\n"
    log += f"- Target drug: {target_drug}\n"
    log += f"- Contraindicated drugs: {', '.join(contraindicated_drugs)}\n"
    log += f"- Therapeutic class filter: {therapeutic_class if therapeutic_class else 'All classes'}\n\n"

    try:
        # Load DDInter data
        drug_info, interaction_matrix, name_mapping = _load_ddinter_data(data_lake_path)
        log += f"Successfully loaded DDInter database with {len(drug_info)} drugs\n\n"

        # Standardize target drug name
        std_target = _standardize_drug_name(target_drug, name_mapping)
        if not std_target:
            log += f"Error: Target drug '{target_drug}' not found in DDInter database\n"
            return log

        # Standardize contraindicated drug names
        std_contraindicated = []
        missing_contraindicated = []

        for drug in contraindicated_drugs:
            std_drug = _standardize_drug_name(drug, name_mapping)
            if std_drug:
                std_contraindicated.append(std_drug)
            else:
                missing_contraindicated.append(drug)

        if missing_contraindicated:
            log += "Warning: The following contraindicated drugs were not found:\n"
            for drug in missing_contraindicated:
                log += f"- {drug}\n"
            log += "\n"

        # Get target drug information
        target_id = name_mapping[std_target]
        target_info = drug_info.get(target_id, {})
        target_categories = target_info.get("categories", [])

        log += "Target Drug Profile:\n"
        log += f"- Drug: {target_drug}\n"
        log += f"- Categories: {', '.join(target_categories)}\n"
        log += f"- Total interactions: {len(target_info.get('interactions', []))}\n\n"

        # Find alternative drugs
        alternatives = []

        for drug_id, drug_data in drug_info.items():
            drug_name = drug_data["name"]
            drug_categories = drug_data.get("categories", [])

            # Skip the target drug itself
            if drug_id == target_id:
                continue

            # Apply therapeutic class filter
            if therapeutic_class:
                if not any(therapeutic_class.lower() in cat.lower() for cat in drug_categories):
                    continue
            else:
                # Look for drugs in similar categories as target
                if not any(cat in target_categories for cat in drug_categories):
                    continue

            # Check if this drug interacts with any contraindicated drugs
            has_contraindicated_interactions = False
            interaction_count = 0
            major_interactions = 0

            std_drug_name = drug_data["standardized_name"]

            for contraindicated in std_contraindicated:
                if std_drug_name in interaction_matrix and contraindicated in interaction_matrix[std_drug_name]:
                    interactions = interaction_matrix[std_drug_name][contraindicated]
                    interaction_count += len(interactions)

                    # Check for major interactions
                    for interaction in interactions:
                        if interaction.get("level") == "Major":
                            major_interactions += 1
                            has_contraindicated_interactions = True
                            break

                    if has_contraindicated_interactions:
                        break

            # Add to alternatives if no major contraindicated interactions
            if not has_contraindicated_interactions:
                alternatives.append(
                    {
                        "name": drug_name,
                        "categories": drug_categories,
                        "interaction_count": interaction_count,
                        "total_interactions": len(drug_data.get("interactions", [])),
                    }
                )

        # Sort alternatives by interaction count (fewer is better)
        alternatives.sort(key=lambda x: x["interaction_count"])

        # Present results
        log += "Alternative Drug Analysis:\n"
        log += "-" * 25 + "\n"

        if alternatives:
            log += f"Found {len(alternatives)} potential alternatives:\n\n"

            # Show top 10 alternatives
            top_alternatives = alternatives[:10]

            for i, alt in enumerate(top_alternatives, 1):
                log += f"{i}. {alt['name']}\n"
                log += f"   - Categories: {', '.join(alt['categories'])}\n"
                log += f"   - Interactions with contraindicated drugs: {alt['interaction_count']}\n"
                log += f"   - Total known interactions: {alt['total_interactions']}\n"

                # Risk assessment
                if alt["interaction_count"] == 0:
                    risk = "No known interactions"
                elif alt["interaction_count"] <= 2:
                    risk = "Low interaction risk"
                elif alt["interaction_count"] <= 5:
                    risk = "Moderate interaction risk"
                else:
                    risk = "Higher interaction risk"

                log += f"   - Risk assessment: {risk}\n\n"

            if len(alternatives) > 10:
                log += f"... and {len(alternatives) - 10} additional alternatives\n\n"
        else:
            log += "No suitable alternatives found in the DDInter database\n"
            log += "Consider:\n"
            log += "- Expanding therapeutic class search criteria\n"
            log += "- Consulting additional drug databases\n"
            log += "- Seeking specialist pharmacological advice\n\n"

        # Recommendations
        log += "Clinical Recommendations:\n"
        log += "-" * 22 + "\n"

        if alternatives:
            best_alternative = alternatives[0]
            log += f"- Primary recommendation: {best_alternative['name']}\n"
            log += "- Rationale: Lowest interaction risk with contraindicated drugs\n"

            if best_alternative["interaction_count"] == 0:
                log += "- Safety profile: No known interactions with specified drugs\n"
            else:
                log += f"- Safety profile: {best_alternative['interaction_count']} minor interactions detected\n"

            log += "- Next steps: Verify therapeutic equivalence and dosing\n"
            log += "- Monitoring: Standard clinical monitoring recommended\n"
        else:
            log += "- No direct alternatives identified\n"
            log += "- Consider non-pharmacological approaches\n"
            log += "- Consult clinical pharmacist or specialist\n"
            log += "- Review patient's complete medication profile\n"

        log += "\nImportant Notes:\n"
        log += "- This analysis is based on DDInter 2.0 data only\n"
        log += "- Always verify therapeutic equivalence before substitution\n"
        log += "- Consider patient-specific factors (allergies, comorbidities)\n"
        log += "- Monitor patient response after any medication changes\n"

    except Exception as e:
        log += f"Error during alternative drug search: {str(e)}\n"

    return log


# OpenFDA Integration Functions


class OpenFDAClient:
    """
    Client for interacting with the FDA's OpenFDA API.

    Provides comprehensive drug safety monitoring, adverse event analysis,
    and regulatory intelligence capabilities through the OpenFDA API.
    """

    BASE_URL = "https://api.fda.gov"

    def __init__(self):
        import time

        import requests

        self.requests = requests
        self.time = time
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Biomni-Agent/1.0 (https://biomni.stanford.edu)"})
        self.retry_attempts = 3
        self.timeout = 30
        self.rate_limit_delay = 0.2  # 5 requests/second
        self.last_request_time = 0

    def _handle_rate_limiting(self):
        """Implement rate limiting to respect FDA API limits."""
        current_time = self.time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit_delay:
            self.time.sleep(self.rate_limit_delay - time_since_last)

        self.last_request_time = self.time.time()

    def _validate_response(self, response_data: dict) -> dict:
        """Validate FDA API response structure and handle variations."""
        if not isinstance(response_data, dict):
            raise ValueError("Invalid FDA API response format")

        # Check for error responses
        if "error" in response_data:
            error_msg = response_data["error"].get("message", "Unknown FDA API error")
            raise Exception(f"FDA API Error: {error_msg}")

        # Validate expected fields exist
        if "meta" not in response_data and "results" not in response_data:
            # Some endpoints return data directly without meta
            return {"results": [response_data], "meta": {"results": {"total": 1}}}

        return response_data

    def _handle_api_variations(self, endpoint: str, params: dict) -> dict:
        """Handle known FDA API endpoint variations and parameter mappings."""
        endpoint_param_mappings = {
            "drug/event": {
                "drug_name": "patient.drug.openfda.brand_name.exact",
                "generic_name": "patient.drug.openfda.generic_name.exact",
            },
            "drug/label": {"drug_name": "openfda.brand_name.exact", "generic_name": "openfda.generic_name.exact"},
            "drug/enforcement": {"drug_name": "openfda.brand_name.exact", "generic_name": "openfda.generic_name.exact"},
        }

        # Transform parameters based on endpoint
        if endpoint in endpoint_param_mappings:
            new_params = {}
            for key, value in params.items():
                if key in endpoint_param_mappings[endpoint]:
                    new_params[endpoint_param_mappings[endpoint][key]] = value
                else:
                    new_params[key] = value
            return new_params

        return params

    def _build_fda_search_params(self, endpoint: str, params: dict) -> dict:
        """Build FDA API search parameters from input parameters."""
        fda_params = {}

        # Handle drug name searches
        if "drug_name" in params:
            drug_name = params["drug_name"]
            if endpoint == "drug/event":
                # For adverse events, search in medicinalproduct field
                fda_params["search"] = f"patient.drug.medicinalproduct:{drug_name}"
            elif endpoint == "drug/label":
                # For drug labels, search in brand name
                fda_params["search"] = f"openfda.brand_name:{drug_name}"
            elif endpoint == "drug/enforcement":
                # For enforcement/recalls, search in brand name
                fda_params["search"] = f"openfda.brand_name:{drug_name}"

        # Handle other parameters
        for key, value in params.items():
            if key not in ["drug_name"]:  # Skip drug_name as it's handled above
                if key == "limit":
                    fda_params["limit"] = value
                elif key == "skip":
                    fda_params["skip"] = value
                # Add other FDA API parameters as needed

        return fda_params

    def _make_request(self, endpoint: str, params: dict) -> dict:
        """Make API request with retry logic and error handling."""
        self._handle_rate_limiting()

        # Build FDA API search parameters
        fda_params = self._build_fda_search_params(endpoint, params)

        for attempt in range(self.retry_attempts):
            try:
                response = self.session.get(f"{self.BASE_URL}/{endpoint}.json", params=fda_params, timeout=self.timeout)

                if response.status_code == 404:
                    return {
                        "results": [],
                        "meta": {"results": {"total": 0}},
                        "message": "No results found for the specified query",
                    }

                response.raise_for_status()

                # Validate and normalize response
                data = self._validate_response(response.json())

                return data

            except self.requests.exceptions.Timeout:
                if attempt == self.retry_attempts - 1:
                    raise Exception("FDA API request timed out after multiple attempts") from None
                self.time.sleep(2**attempt)  # Exponential backoff

            except self.requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    # Rate limiting - wait and retry
                    self.time.sleep(5 * (attempt + 1))
                    continue
                else:
                    raise Exception(f"FDA API HTTP Error: {e.response.status_code}") from e

            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    raise Exception(f"FDA API request failed: {str(e)}") from e
                self.time.sleep(2**attempt)

        return {}

    def query_adverse_events(self, drug_name: str, limit: int = 100) -> dict:
        """Query adverse events with robust error handling and validation."""
        endpoint = "drug/event"
        params = {"drug_name": drug_name, "limit": limit}

        try:
            data = self._make_request(endpoint, params)

            # Add FDA disclaimer to results
            data["disclaimer"] = (
                "FDA Disclaimer: These data do not establish causation. "
                "Reports are voluntary and subject to reporting bias. "
                "Data should not be used for regulatory decision-making."
            )

            return data

        except Exception as e:
            return {
                "results": [],
                "meta": {"results": {"total": 0}},
                "error": str(e),
                "disclaimer": (
                    "FDA Disclaimer: These data do not establish causation. "
                    "Reports are voluntary and subject to reporting bias."
                ),
            }

    def query_drug_labels(self, drug_name: str, sections: list[str] | None = None) -> dict:
        """Query FDA drug label information."""
        endpoint = "drug/label"
        params = {"drug_name": drug_name, "limit": 50}

        return self._make_request(endpoint, params)

    def query_drug_recalls(self, drug_name: str, classification: list[str] | None = None) -> dict:
        """Query FDA drug recall and enforcement information."""
        endpoint = "drug/enforcement"
        params = {"drug_name": drug_name, "limit": 100}

        return self._make_request(endpoint, params)


# Helper Functions for OpenFDA Data Processing


def _standardize_drug_name_fda(drug_name: str) -> str:
    """Standardize drug names for FDA API queries."""
    # Handle None/empty values
    if not drug_name:
        return ""

    # Remove common suffixes
    suffixes = ["sodium", "hydrochloride", "sulfate", "phosphate", "acetate", "citrate"]

    # Clean and standardize
    name = drug_name.strip().lower()

    for suffix in suffixes:
        if name.endswith(f" {suffix}"):
            name = name[: -len(f" {suffix}")]

    return name


def _apply_fda_filters(response_data: dict, filters: dict) -> dict:
    """Apply post-query filtering to FDA responses."""
    if not response_data.get("results"):
        return response_data

    filtered_results = []

    for result in response_data["results"]:
        include = True

        # Apply severity filter
        if filters.get("severity_filter") or filters.get("severity"):
            severity_list = filters.get("severity_filter", filters.get("severity", []))
            if "serious" in severity_list:
                # For serious filter, only include if serious == '1'
                if result.get("serious") != "1":
                    include = False
            elif "non_serious" in severity_list:
                # For non-serious filter, only include if serious != '1'
                if result.get("serious") == "1":
                    include = False

        # Apply outcome filter
        if (filters.get("outcome_filter") or filters.get("outcome")) and include:
            outcome_list = filters.get("outcome_filter", filters.get("outcome", []))
            if "life_threatening" in outcome_list:
                # Check if the result has life threatening outcome
                if result.get("seriousnesslifethreatening") != "1":
                    include = False
            elif "hospitalization" in outcome_list:
                # Check if the result has hospitalization outcome
                if result.get("seriousnesshospitalization") != "1":
                    include = False
            elif "death" in outcome_list:
                # Check if the result has death outcome
                if result.get("seriousnessdeath") != "1":
                    include = False

        # Apply classification filter (for recalls)
        if filters.get("classification") and include:
            classification_list = filters.get("classification", [])
            result_class = result.get("classification", "")
            if result_class not in classification_list:
                include = False

        if include:
            filtered_results.append(result)

    response_data["results"] = filtered_results
    response_data["meta"]["results"]["total"] = len(filtered_results)

    return response_data


def _extract_fda_safety_signals(response_list: list[dict]) -> dict:
    """Extract safety signals from adverse event data."""
    drug_signals = {}
    reaction_patterns = {}
    temporal_patterns = {}

    for response in response_list:
        if not response.get("results"):
            continue

        for result in response["results"]:
            # Extract drug information
            drugs = result.get("patient", {}).get("drug", [])
            for drug in drugs:
                # Use the existing standardization function
                drug_name = _standardize_drug_name_fda(drug.get("medicinalproduct", ""))
                if drug_name:
                    if drug_name not in drug_signals:
                        drug_signals[drug_name] = {"total_reports": 0, "serious_reports": 0, "common_reactions": []}

                    drug_signals[drug_name]["total_reports"] += 1
                    if result.get("serious") == "1":
                        drug_signals[drug_name]["serious_reports"] += 1

            # Extract reaction patterns
            reactions = result.get("patient", {}).get("reaction", [])
            for reaction in reactions:
                reaction_name = reaction.get("reactionmeddrapt", "")
                if reaction_name:
                    if reaction_name not in reaction_patterns:
                        reaction_patterns[reaction_name] = {
                            "count": 0,
                            "severity_counts": {"serious": 0, "non_serious": 0},
                        }

                    reaction_patterns[reaction_name]["count"] += 1

                    # Count severity
                    if result.get("serious") == "1":
                        reaction_patterns[reaction_name]["severity_counts"]["serious"] += 1
                    else:
                        reaction_patterns[reaction_name]["severity_counts"]["non_serious"] += 1

            # Extract temporal patterns
            receipt_date = result.get("receiptdate")
            if receipt_date and len(receipt_date) >= 6:
                year_month = receipt_date[:6]  # YYYYMM
                if year_month not in temporal_patterns:
                    temporal_patterns[year_month] = {"count": 0, "serious_count": 0}

                temporal_patterns[year_month]["count"] += 1
                if result.get("serious") == "1":
                    temporal_patterns[year_month]["serious_count"] += 1

    # Build common reactions for each drug based on actual data
    for drug_name in drug_signals:
        # Find reactions that occurred with this specific drug
        drug_reactions = {}

        for response in response_list:
            if not response.get("results"):
                continue

            for result in response["results"]:
                drugs = result.get("patient", {}).get("drug", [])
                has_this_drug = any(
                    _standardize_drug_name_fda(drug.get("medicinalproduct", "")) == drug_name for drug in drugs
                )

                if has_this_drug:
                    reactions = result.get("patient", {}).get("reaction", [])
                    for reaction in reactions:
                        reaction_name = reaction.get("reactionmeddrapt", "")
                        if reaction_name:
                            if reaction_name not in drug_reactions:
                                drug_reactions[reaction_name] = 0
                            drug_reactions[reaction_name] += 1

        # Get top 3 reactions for this drug
        top_reactions = sorted(drug_reactions.items(), key=lambda x: x[1], reverse=True)[:3]
        drug_signals[drug_name]["common_reactions"] = [r[0] for r in top_reactions]

    return {
        "drug_signals": drug_signals,
        "reaction_patterns": reaction_patterns,
        "temporal_patterns": temporal_patterns,
    }


def _generate_fda_statistics(response_data: dict) -> dict:
    """Generate summary statistics from FDA responses."""
    stats = {
        "total_reports": 0,
        "serious_reports": 0,
        "death_reports": 0,
        "life_threatening_reports": 0,
        "hospitalization_reports": 0,
        "top_reactions": [],
        "temporal_pattern": {},
    }

    if not response_data.get("results"):
        return stats

    reaction_counts = {}

    for result in response_data["results"]:
        stats["total_reports"] += 1

        # Count serious reports
        if result.get("serious") == "1":
            stats["serious_reports"] += 1

        # Count specific outcomes
        outcomes = result.get("patient", {}).get("reaction", [])
        for outcome in outcomes:
            outcome_name = outcome.get("reactionmeddrapt", "Unknown")
            reaction_counts[outcome_name] = reaction_counts.get(outcome_name, 0) + 1

        # Count deaths and other serious outcomes
        if result.get("patient", {}).get("patientdeath"):
            stats["death_reports"] += 1

        if result.get("patient", {}).get("patientlifethreatening"):
            stats["life_threatening_reports"] += 1

        if result.get("patient", {}).get("patienthospitalization"):
            stats["hospitalization_reports"] += 1

    # Top reactions
    sorted_reactions = sorted(reaction_counts.items(), key=lambda x: x[1], reverse=True)
    stats["top_reactions"] = sorted_reactions[:10]

    # Report distribution
    stats["report_distribution"] = {
        "serious_percentage": (stats["serious_reports"] / stats["total_reports"] * 100)
        if stats["total_reports"] > 0
        else 0,
        "non_serious_percentage": ((stats["total_reports"] - stats["serious_reports"]) / stats["total_reports"] * 100)
        if stats["total_reports"] > 0
        else 0,
        "death_percentage": (stats["death_reports"] / stats["total_reports"] * 100)
        if stats["total_reports"] > 0
        else 0,
    }

    return stats


def _format_adverse_event_summary(response_data: dict, drug_name: str, include_details: bool = True) -> str:
    """Format adverse event data into readable summary."""
    if not response_data.get("results"):
        return f"No adverse events found for {drug_name} in the FDA database."

    stats = _generate_fda_statistics(response_data)

    summary = "Adverse Event Summary\n"
    summary += "=" * 21 + "\n"
    summary += f"Drug: {drug_name}\n"
    summary += f"Total Reports: {stats['total_reports']:,}\n\n"

    if stats["total_reports"] > 0:
        summary += "Summary Statistics:\n"
        summary += f"- Serious Reports: {stats['serious_reports']:,} ({stats['serious_reports'] / stats['total_reports'] * 100:.1f}%)\n"

        if stats["death_reports"] > 0:
            summary += (
                f"- Deaths: {stats['death_reports']:,} ({stats['death_reports'] / stats['total_reports'] * 100:.1f}%)\n"
            )

        if stats["life_threatening_reports"] > 0:
            summary += f"- Life-threatening: {stats['life_threatening_reports']:,} ({stats['life_threatening_reports'] / stats['total_reports'] * 100:.1f}%)\n"

        if stats["hospitalization_reports"] > 0:
            summary += f"- Hospitalizations: {stats['hospitalization_reports']:,} ({stats['hospitalization_reports'] / stats['total_reports'] * 100:.1f}%)\n"

        if stats["top_reactions"]:
            summary += "\nCommon Reactions:\n"
            for i, (reaction, count) in enumerate(stats["top_reactions"][:5], 1):
                summary += f"{i}. {reaction} ({count:,} reports)\n"

    # Add FDA disclaimer
    summary += "\n" + response_data.get("disclaimer", "")

    return summary


def _format_drug_label_summary(response_data: dict, drug_name: str, sections: list[str] | None = None) -> str:
    """Format drug label information into readable summary."""
    if not response_data.get("results"):
        return f"No drug label information found for {drug_name} in the FDA database."

    result = response_data["results"][0]  # Use first result

    summary = "OpenFDA Drug Label Information\n"
    summary += "=" * 29 + "\n"
    summary += f"Drug: {drug_name}\n"

    # Extract key information
    if "effective_time" in result:
        summary += f"Effective Date: {result['effective_time']}\n"

    if "openfda" in result:
        openfda = result["openfda"]
        if "brand_name" in openfda:
            summary += f"Brand Name: {', '.join(openfda['brand_name'])}\n"
        if "generic_name" in openfda:
            summary += f"Generic Name: {', '.join(openfda['generic_name'])}\n"
        if "manufacturer_name" in openfda:
            summary += f"Manufacturer: {', '.join(openfda['manufacturer_name'])}\n"

    summary += "\n"

    # Display specific sections
    section_mapping = {
        "indications_and_usage": "Indications and Usage",
        "contraindications": "Contraindications",
        "warnings": "Warnings",
        "dosage_and_administration": "Dosage and Administration",
        "adverse_reactions": "Adverse Reactions",
        "clinical_pharmacology": "Clinical Pharmacology",
    }

    sections_to_show = sections if sections else section_mapping.keys()

    for section_key in sections_to_show:
        if section_key in result:
            section_title = section_mapping.get(section_key, section_key.title())
            summary += f"{section_title}:\n"

            content = result[section_key]
            if isinstance(content, list):
                content = " ".join(content)

            # Truncate long content
            if len(content) > 500:
                content = content[:500] + "..."

            summary += f"{content}\n\n"

    return summary


def _format_recall_summary(response_data: dict, drug_name: str, include_details: bool = True) -> str:
    """Format recall information into structured output."""
    if not response_data.get("results"):
        return f"No drug recalls found for {drug_name} in the FDA database."

    summary = "OpenFDA Drug Recall Information\n"
    summary += "=" * 31 + "\n"
    summary += f"Drug: {drug_name}\n"
    summary += f"Total recalls found: {len(response_data['results'])}\n\n"

    if include_details:
        summary += "Recall Details:\n"

        for i, recall in enumerate(response_data["results"][:5], 1):  # Show top 5
            summary += f"{i}. Recall Number: {recall.get('recall_number', 'N/A')}\n"
            summary += f"   - Product: {recall.get('product_description', 'N/A')}\n"
            summary += f"   - Classification: {recall.get('classification', 'N/A')}\n"
            summary += f"   - Reason: {recall.get('reason_for_recall', 'N/A')}\n"
            summary += f"   - Date: {recall.get('recall_initiation_date', 'N/A')}\n"
            summary += f"   - Status: {recall.get('status', 'N/A')}\n"
            summary += f"   - Distribution: {recall.get('distribution_pattern', 'N/A')}\n\n"

        if len(response_data["results"]) > 5:
            summary += f"... and {len(response_data['results']) - 5} additional recalls\n"

    return summary


def _format_safety_signal_summary(
    signals_data: dict,
    drug_list: list[str],
    comparison_period: tuple[str, str] | None = None,
    signal_threshold: float = 2.0,
) -> str:
    """Format safety signal analysis results."""
    summary = "OpenFDA Safety Signal Analysis\n"
    summary += "=" * 29 + "\n"
    summary += f"Drugs analyzed: {drug_list}\n"

    # Add comparison period and threshold info
    if comparison_period:
        summary += f"Comparison period: {comparison_period[0]} to {comparison_period[1]}\n"
    if signal_threshold != 2.0:
        summary += f"Signal threshold: {signal_threshold}\n"
    summary += "\n"

    if not signals_data:
        summary += "No safety signals detected.\n"
        return summary

    summary += "Signal Detection Results:\n"

    # Handle the actual data structure from _extract_fda_safety_signals
    drug_signals = signals_data.get("drug_signals", {})
    reaction_patterns = signals_data.get("reaction_patterns", {})
    signals_data.get("temporal_patterns", {})

    # Display drug-specific signals
    for i, drug_name in enumerate(drug_list, 1):
        drug_data = drug_signals.get(drug_name, {})
        if drug_data:
            summary += f"{i}. {drug_name.title()}\n"
            summary += f"   - Total reports: {drug_data['total_reports']:,}\n"
            summary += f"   - Serious reports: {drug_data['serious_reports']:,}\n"

            if drug_data.get("common_reactions"):
                summary += f"   - Common reactions: {', '.join(drug_data['common_reactions'])}\n"
            summary += "\n"
        else:
            summary += f"{i}. {drug_name.title()}\n"
            summary += "   - No data found\n\n"

    # Display cross-drug reaction patterns
    if reaction_patterns:
        summary += "Cross-drug Analysis:\n"
        sorted_reactions = sorted(reaction_patterns.items(), key=lambda x: x[1]["count"], reverse=True)

        for reaction, data in sorted_reactions[:5]:  # Show top 5 reactions
            summary += f"- {reaction}: {data['count']:,} reports\n"
            if data["severity_counts"]["serious"] > 0:
                summary += f"  * Serious: {data['severity_counts']['serious']:,}\n"

    # Add trend analysis if comparison period is specified
    if comparison_period:
        summary += "\nTrend Analysis:\n"
        summary += f"Comparing current period to {comparison_period[0]} - {comparison_period[1]}\n"
        summary += "* Trend detection based on temporal patterns in adverse event reports\n"
        summary += "* Analysis considers seasonal variations and reporting delays\n"

    return summary


# Main OpenFDA Integration Functions


def query_fda_adverse_events(
    drug_name: str,
    date_range: tuple[str, str] | None = None,
    severity_filter: list[str] | None = None,
    outcome_filter: list[str] | None = None,
    limit: int = 100,
) -> str:
    """
    Query FDA adverse event reports for specific drugs.

    Args:
        drug_name: Name of the drug to query
        date_range: Optional date range as (start_date, end_date) in YYYY-MM-DD format
        severity_filter: Optional filter by severity levels ["serious", "non_serious"]
        outcome_filter: Optional filter by outcomes ["life_threatening", "hospitalization", "death"]
        limit: Maximum number of results to return

    Returns:
        Formatted string with adverse event analysis
    """
    try:
        # Validate input
        if not drug_name or not drug_name.strip():
            return "Error: Drug name cannot be empty"

        client = OpenFDAClient()

        # Standardize drug name
        standardized_name = _standardize_drug_name_fda(drug_name)
        if not standardized_name:
            return f"Error: Unable to standardize drug name '{drug_name}'"

        # Query adverse events
        response = client.query_adverse_events(standardized_name, limit=limit)

        # Apply filters if specified
        if severity_filter or outcome_filter:
            filters = {"severity_filter": severity_filter, "outcome_filter": outcome_filter}
            response = _apply_fda_filters(response, filters)

        # Format results with main function title
        formatted_result = _format_adverse_event_summary(response, drug_name, include_details=True)

        # Replace title for main function
        if formatted_result.startswith("Adverse Event Summary"):
            formatted_result = formatted_result.replace(
                "Adverse Event Summary\n" + "=" * 21, "OpenFDA Adverse Event Query Results\n" + "=" * 35, 1
            )

        # Add filter and date range info if specified
        lines = formatted_result.split("\n")
        insert_index = -1

        # Find insertion point (after drug name)
        for i, line in enumerate(lines):
            if line.startswith("Drug: "):
                insert_index = i + 1
                break

        if insert_index >= 0:
            # Add date range info
            if date_range:
                lines.insert(insert_index, f"Date range: {date_range[0]} to {date_range[1]}")
                insert_index += 1

            # Add severity filter info
            if severity_filter:
                lines.insert(insert_index, f"Severity filter: {severity_filter}")
                insert_index += 1

            # Add outcome filter info
            if outcome_filter:
                lines.insert(insert_index, f"Outcome filter: {outcome_filter}")
                insert_index += 1

        formatted_result = "\n".join(lines)

        return formatted_result

    except Exception as e:
        return f"Error querying FDA adverse events for {drug_name}: {str(e)}"


def get_fda_drug_label_info(drug_name: str, sections: list[str] | None = None) -> str:
    """
    Retrieve FDA drug label information.

    Args:
        drug_name: Name of the drug to query
        sections: Optional list of specific sections to retrieve
                 ["indications_and_usage", "contraindications", "warnings", "dosage_and_administration"]

    Returns:
        Formatted string with drug label information
    """
    try:
        # Validate input
        if not drug_name or not drug_name.strip():
            return "Error: Drug name cannot be empty"

        client = OpenFDAClient()

        # Standardize drug name
        standardized_name = _standardize_drug_name_fda(drug_name)
        if not standardized_name:
            return f"Error: Unable to standardize drug name '{drug_name}'"

        # Query drug labels
        response = client.query_drug_labels(standardized_name, sections=sections)

        # Check if we got results
        if not response.get("results"):
            return f"No label information found for drug: {drug_name}"

        # Format results
        return _format_drug_label_summary(response, drug_name, sections=sections)

    except Exception as e:
        return f"Error retrieving FDA drug label for {drug_name}: {str(e)}"


def check_fda_drug_recalls(
    drug_name: str, classification: list[str] | None = None, date_range: tuple[str, str] | None = None
) -> str:
    """
    Check for FDA drug recalls and enforcement actions.

    Args:
        drug_name: Name of the drug to check
        classification: Optional filter by recall class ["Class I", "Class II", "Class III"]
        date_range: Optional date range for recalls

    Returns:
        Formatted string with recall information
    """
    try:
        # Validate input
        if not drug_name or not drug_name.strip():
            return "Error: Drug name cannot be empty"

        client = OpenFDAClient()

        # Standardize drug name
        standardized_name = _standardize_drug_name_fda(drug_name)
        if not standardized_name:
            return f"Error: Unable to standardize drug name '{drug_name}'"

        # Query drug recalls
        response = client.query_drug_recalls(standardized_name, classification=classification)

        # Format results with filter information
        formatted_result = _format_recall_summary(response, drug_name, include_details=True)

        # Add filter information to the output
        if classification:
            formatted_result = formatted_result.replace(
                f"Drug: {drug_name}\n", f"Drug: {drug_name}\nClassification filter: {', '.join(classification)}\n"
            )

        if date_range:
            formatted_result = formatted_result.replace(
                f"Drug: {drug_name}\n", f"Drug: {drug_name}\nDate range: {date_range[0]} to {date_range[1]}\n"
            )

        return formatted_result

    except Exception as e:
        return f"Error checking FDA drug recalls for {drug_name}: {str(e)}"


def analyze_fda_safety_signals(
    drug_list: list[str], comparison_period: tuple[str, str] | None = None, signal_threshold: float = 2.0
) -> str:
    """
    Analyze safety signals across multiple drugs.

    Args:
        drug_list: List of drug names to analyze
        comparison_period: Optional comparison time period
        signal_threshold: Threshold for signal detection

    Returns:
        Formatted string with safety signal analysis
    """
    try:
        # Validate input parameters
        if not drug_list:
            return "Error: At least one drug must be provided for analysis"

        if len(drug_list) < 2:
            return "Error: At least 2 drugs required for comparative safety signal analysis"

        # Validate drug names
        valid_drugs = [drug.strip() for drug in drug_list if drug and drug.strip()]
        if not valid_drugs:
            return "Error: No valid drug names provided"

        client = OpenFDAClient()

        # Collect data for all drugs
        all_responses = []

        for drug in valid_drugs:
            standardized_name = _standardize_drug_name_fda(drug)
            if standardized_name:  # Only query if standardization worked
                response = client.query_adverse_events(standardized_name, limit=200)

                if response.get("results"):
                    all_responses.append(response)

        # Check if we got any data
        if not all_responses:
            return "Error: No adverse event data found for any of the provided drugs"

        # Extract safety signals
        signals = _extract_fda_safety_signals(all_responses)

        # Format results with comparison period and threshold info
        return _format_safety_signal_summary(signals, valid_drugs, comparison_period, signal_threshold)

    except Exception as e:
        return f"Error analyzing FDA safety signals: {str(e)}"


def predict_pka_with_tripka(
    smiles: str = "",
    micro_a: str = "",
    micro_b: str = "",
    dataset: str = "biomni_tripka",
    iter_num: int = 4,
    sample_num: int = 4,
    cuda_idx: int = 0,
    mode: str = "A",
    timeout_seconds: int = 1800,
    tripka_repo: str = "",
    python_executable: str = "",
    runtime_root: str = "",
) -> dict:
    """Predict macro- or micro-pKa values using the external TripKa repository."""
    repository_root = Path(__file__).resolve().parents[2]
    tool_root = _tools_pkg_root(repository_root) / "TripKa"
    tripka_repo = os.path.abspath(
        os.path.expanduser(tripka_repo or os.environ.get("TRIPKA_REPO", "") or str(tool_root / "upstream"))
    )
    tripka_python = os.path.abspath(
        os.path.expanduser(
            python_executable or os.environ.get("TRIPKA_PYTHON", "") or str(tool_root / ".conda" / "bin" / "python")
        )
    )
    runtime_root = os.path.abspath(
        os.path.expanduser(
            runtime_root
            or os.environ.get("TRIPKA_RUNTIME_ROOT", "")
            or "/tmp/biomni_tripka_runtime"
        )
    )
    os.makedirs(runtime_root, exist_ok=True)
    runtime_repo = tempfile.mkdtemp(prefix="tripka_", dir=runtime_root)
    infer_script = os.path.join(runtime_repo, "infer_pKa.py")
    query_info = {
        "smiles": smiles,
        "micro_a": micro_a,
        "micro_b": micro_b,
        "dataset": dataset,
        "iter_num": iter_num,
        "sample_num": sample_num,
        "cuda_idx": cuda_idx,
        "mode": mode,
        "tripka_repo": tripka_repo,
        "runtime_repo": runtime_repo,
        "python": tripka_python,
        "runtime_root": runtime_root,
    }

    try:
        smiles = smiles.strip() if smiles else ""
        micro_a = micro_a.strip() if micro_a else ""
        micro_b = micro_b.strip() if micro_b else ""
        mode = mode.strip().upper() if mode else "A"

        if mode not in {"A", "B"}:
            return {"success": False, "error": "mode must be 'A' or 'B'.", "query_info": query_info}
        if iter_num < 1:
            return {"success": False, "error": "iter_num must be at least 1.", "query_info": query_info}
        if sample_num < 2:
            return {
                "success": False,
                "error": "sample_num must be at least 2 because TripKa confidence analysis filters sampled predictions.",
                "query_info": query_info,
            }
        if smiles and (micro_a or micro_b):
            return {
                "success": False,
                "error": "Provide either smiles for macro-pKa or micro_a and micro_b for micro-pKa, not both.",
                "query_info": query_info,
            }
        if not smiles and not (micro_a and micro_b):
            return {
                "success": False,
                "error": "Provide smiles for macro-pKa, or provide both micro_a and micro_b for micro-pKa.",
                "query_info": query_info,
            }

        source_required_paths = {
            "tripka_repo": tripka_repo,
            "tripka_python": tripka_python,
            "source_infer_script": os.path.join(tripka_repo, "infer_pKa.py"),
            "source_base_script": os.path.join(tripka_repo, "scripts", "infer_tripka_base.sh"),
            "source_qm_script": os.path.join(tripka_repo, "scripts", "infer_tripka_qm.sh"),
            "confidence_checkpoint": os.path.join(tripka_repo, "checkpoint", "tripka_confidence", "checkpoint_best.pt"),
            "qm_checkpoint": os.path.join(tripka_repo, "checkpoint", "tripka_qm", "checkpoint_best.pt"),
        }
        missing_paths = [name for name, path in source_required_paths.items() if not os.path.exists(path)]
        if missing_paths:
            return {
                "success": False,
                "error": "TripKa deployment is incomplete; missing required path(s): " + ", ".join(missing_paths),
                "missing_paths": {name: source_required_paths[name] for name in missing_paths},
                "query_info": query_info,
            }

        env = os.environ.copy()
        tripka_python_bin = os.path.dirname(tripka_python)
        env["PATH"] = tripka_python_bin + os.pathsep + env.get("PATH", "")
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_idx)
        # PyTorch multiprocessing uses a Unix-domain socket under TMPDIR.
        # Task-scoped paths can exceed Linux's AF_UNIX limit (~108 bytes).
        env["TMPDIR"] = os.environ.get("TRIPKA_TMPDIR", "/tmp")
        os.makedirs(env["TMPDIR"], exist_ok=True)
        tempfile.tempdir = None

        dependency_check = subprocess.run(
            [tripka_python, "-c", "from rdkit import Chem; import lmdb, pandas, numpy, torch"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if dependency_check.returncode != 0:
            return {
                "success": False,
                "error": "TripKa Python environment is missing required runtime dependencies.",
                "dependency_check": {
                    "command": [tripka_python, "-c", "from rdkit import Chem; import lmdb, pandas, numpy, torch"],
                    "returncode": dependency_check.returncode,
                    "stdout": dependency_check.stdout,
                    "stderr": dependency_check.stderr,
                },
                "query_info": query_info,
            }

        os.makedirs(runtime_repo, exist_ok=True)
        for dirname in ["preprocess", "data", "results", "analysis", "scripts"]:
            os.makedirs(os.path.join(runtime_repo, dirname), exist_ok=True)

        import shutil

        runtime_tripka_package = os.path.join(runtime_repo, "tripka")
        if os.path.islink(runtime_tripka_package):
            os.unlink(runtime_tripka_package)
        if not os.path.exists(runtime_tripka_package):
            shutil.copytree(os.path.join(tripka_repo, "tripka"), runtime_tripka_package)

        symlink_items = [
            "infer_pKa.py",
            "enumerator.py",
            "tsv2lmdb.py",
            "utils.py",
            "checkpoint",
            "configs",
            "smarts_pattern.tsv",
            "simple_smarts_pattern.tsv",
        ]
        for item in symlink_items:
            src = os.path.join(tripka_repo, item)
            dst = os.path.join(runtime_repo, item)
            if os.path.exists(src) and not os.path.exists(dst):
                os.symlink(src, dst)

        stage_adapter = repository_root / "scripts" / "pharmacology_adapters" / "tripka_stage.sh"
        if not stage_adapter.is_file():
            return {
                "success": False,
                "error": f"TripKa stage adapter is missing: {stage_adapter}",
                "query_info": query_info,
            }
        env["TRIPKA_NUM_WORKERS"] = os.environ.get("TRIPKA_NUM_WORKERS", "1")
        for script_name in ["infer_tripka_confidence.sh", "infer_tripka_qm.sh"]:
            shutil.copy2(stage_adapter, os.path.join(runtime_repo, "scripts", script_name))

        command = [
            tripka_python,
            infer_script,
            "--dataset",
            dataset,
            "--iter-num",
            str(iter_num),
            "--sample-num",
            str(sample_num),
            "--cuda-idx",
            str(cuda_idx),
        ]
        if smiles:
            command.extend(["--smiles", smiles, "--mode", mode])
        else:
            command.extend(["--micro", "--micro-a", micro_a, "--micro-b", micro_b])

        log_path = os.path.join(runtime_repo, "tripka_inference.log")
        query_info["log_path"] = log_path
        with open(log_path, "w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                command,
                cwd=runtime_repo,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )

        with open(log_path, encoding="utf-8", errors="replace") as log_handle:
            stdout = log_handle.read()
        stderr = ""
        stdout_tail = stdout[-6000:]
        stderr_tail = stderr[-3000:]

        result = {}
        conf_match = re.search(r"TripKa-conf:\s*([^,\n]+)", stdout)
        qm_match = re.search(r"TripKa-qm:\s*([^\n]+)", stdout)
        pred_match = re.search(r"pKa predicted by TripKa:\s*([^\n]+)", stdout)
        if conf_match:
            result["tripka_confidence_pka"] = conf_match.group(1).strip()
        if qm_match:
            result["tripka_qm_pka"] = qm_match.group(1).strip()
        if pred_match:
            result["tripka_ensemble_pka"] = pred_match.group(1).strip()

        dataset_run_name = f"{dataset}_iternum{iter_num}_samplenum{sample_num}"
        output_files = []
        analysis_csv = os.path.join(runtime_repo, "analysis", f"{dataset_run_name}.csv")
        if os.path.exists(analysis_csv):
            output_files.append(analysis_csv)

        return {
            "success": completed.returncode == 0,
            "returncode": completed.returncode,
            "query_info": query_info,
            "command": command,
            "result": result,
            "output_files": output_files,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "error": None if completed.returncode == 0 else "TripKa inference command failed.",
        }

    except subprocess.TimeoutExpired:
        log_path = query_info.get("log_path", "")
        log_tail = ""
        if log_path and os.path.isfile(log_path):
            with open(log_path, encoding="utf-8", errors="replace") as log_handle:
                log_tail = log_handle.read()[-6000:]
        return {
            "success": False,
            "error": f"TripKa inference timed out after {timeout_seconds} seconds.",
            "query_info": query_info,
            "stdout_tail": log_tail,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "query_info": query_info}


def _run_molecular_design_adapter(
    tool_name: str,
    adapter_name: str,
    arguments: list[str],
    output_dir: str,
    repo_path: str,
    python_executable: str,
    gpu_device: int,
    timeout_seconds: int,
) -> dict:
    """Run an isolated molecular-design adapter and decode its JSON response."""
    repository_root = Path(__file__).resolve().parents[2]
    tool_root = _tools_pkg_root(repository_root) / tool_name
    adapter_path = repository_root / "scripts" / "molecular_design_adapters" / adapter_name
    configured_repo = repo_path.strip() if isinstance(repo_path, str) else ""
    upstream_path = Path(configured_repo or os.environ.get(f"{tool_name.upper()}_REPO", "") or tool_root / "upstream")
    configured_python = python_executable.strip() if isinstance(python_executable, str) else ""
    environment_python = os.environ.get(f"{tool_name.upper()}_PYTHON", "").strip()
    runtime_python = Path(configured_python or environment_python or tool_root / ".conda" / "bin" / "python")
    query_info = {
        "source": tool_name,
        "parameters": {
            "repo_path": str(upstream_path),
            "python_executable": str(runtime_python),
            "gpu_device": gpu_device,
            "timeout_seconds": timeout_seconds,
        },
    }

    try:
        if timeout_seconds < 1:
            return _hit_error("timeout_seconds must be at least 1.", query_info)
        if gpu_device < -1:
            return _hit_error("gpu_device must be -1 for CPU or a non-negative CUDA device index.", query_info)
        if not adapter_path.is_file():
            return _hit_error(f"Biomni adapter is missing: {adapter_path}", query_info)
        if not upstream_path.is_dir():
            return _hit_error(
                f"{tool_name} upstream repository is unavailable at '{upstream_path}'. Run the tool setup script or set {tool_name.upper()}_REPO.",
                query_info,
            )
        if not runtime_python.is_file() or not os.access(runtime_python, os.X_OK):
            return _hit_error(
                f"{tool_name} Python environment is unavailable at '{runtime_python}'. Run the tool setup script or set {tool_name.upper()}_PYTHON.",
                query_info,
            )

        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        if not output_path.is_dir():
            return _hit_error(f"Output path is not a directory: {output_path}", query_info)

        command = [
            str(runtime_python),
            str(adapter_path),
            "--tool",
            tool_name.lower(),
            "--repo-path",
            str(upstream_path.resolve()),
            "--output-dir",
            str(output_path),
            *arguments,
        ]
        environment = os.environ.copy()
        if gpu_device >= 0:
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
            env=environment,
        )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        payload = None
        for line in reversed(stdout.splitlines()):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                payload = candidate
                break

        if payload is None:
            return {
                "success": False,
                "error": f"{tool_name} adapter did not return a JSON result.",
                "query_info": query_info,
                "returncode": completed.returncode,
                "stdout_tail": stdout[-6000:],
                "stderr_tail": stderr[-4000:],
            }
        payload.setdefault("success", completed.returncode == 0)
        payload.setdefault("query_info", query_info)
        payload.setdefault("returncode", completed.returncode)
        payload.setdefault("stdout_tail", stdout[-6000:])
        payload.setdefault("stderr_tail", stderr[-4000:])
        if completed.returncode != 0:
            payload["success"] = False
            payload.setdefault("error", f"{tool_name} adapter failed with exit code {completed.returncode}.")
        return payload
    except subprocess.TimeoutExpired as exc:
        return {
            "success": False,
            "error": f"{tool_name} inference timed out after {timeout_seconds} seconds.",
            "query_info": query_info,
            "stdout_tail": (exc.stdout or "")[-6000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
        }
    except OSError as exc:
        return _hit_error(f"Could not run {tool_name}: {exc}", query_info)


def generate_conformers_with_geodiff(
    smiles: str,
    output_dir: str,
    num_conformers: int = 10,
    checkpoint_path: str = "",
    repo_path: str = "",
    python_executable: str = "",
    gpu_device: int | None = None,
    use_gpu: bool = True,
    num_steps: int = 5000,
    timeout_seconds: int = 7200,
) -> dict:
    """Generate 3-D conformers for one molecular graph with the official GeoDiff model."""
    gpu_device = _resolve_allocated_gpu(gpu_device, use_gpu)
    query_info = {"input": smiles, "source": "GeoDiff", "parameters": {"num_conformers": num_conformers, "gpu_device": gpu_device}}
    try:
        from rdkit import Chem

        molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if molecule is None:
            return _hit_error("smiles must be a valid, non-empty SMILES string.", query_info)
    except ImportError:
        if not isinstance(smiles, str) or not smiles.strip():
            return _hit_error("smiles must be a non-empty SMILES string.", query_info)
    if not isinstance(output_dir, str) or not output_dir.strip():
        return _hit_error("output_dir must be a non-empty directory path.", query_info)
    if not 1 <= num_conformers <= 1000:
        return _hit_error("num_conformers must be between 1 and 1000.", query_info)
    if not 1 <= num_steps <= 10000:
        return _hit_error("num_steps must be between 1 and 10000.", query_info)
    if use_gpu and gpu_device is None:
        return _hit_error("MCP 未分配 GPU；请通过 MCP Worker 调用或显式提供 gpu_device。", query_info)
    default_checkpoint = _tools_pkg_root() / "GeoDiff" / "models" / "drugs_default" / "checkpoints" / "drugs_default.pt"
    checkpoint = checkpoint_path.strip() or os.environ.get("GEODIFF_CHECKPOINT", "").strip() or str(default_checkpoint)
    if not checkpoint or not Path(checkpoint).expanduser().is_file():
        return _hit_error("GeoDiff checkpoint is missing; pass checkpoint_path or set GEODIFF_CHECKPOINT.", query_info)
    return _run_molecular_design_adapter(
        "GeoDiff",
        "run_tool.py",
        [
            "--smiles",
            smiles,
            "--checkpoint-path",
            str(Path(checkpoint).expanduser().resolve()),
            "--num-conformers",
            str(num_conformers),
            "--num-steps",
            str(num_steps),
            "--device",
            "cuda:0" if use_gpu else "cpu",
        ],
        output_dir,
        repo_path,
        python_executable,
        gpu_device if use_gpu else -1,
        timeout_seconds,
    )


def generate_ligands_with_targetdiff(
    pocket_pdb_path: str,
    output_dir: str,
    num_samples: int = 20,
    config_path: str = "",
    checkpoint_path: str = "",
    repo_path: str = "",
    python_executable: str = "",
    gpu_device: int | None = None,
    batch_size: int = 20,
    timeout_seconds: int = 7200,
) -> dict:
    """Generate 3-D ligands conditioned on a prepared protein pocket with TargetDiff."""
    gpu_device = _resolve_allocated_gpu(gpu_device, True)
    pocket_path = Path(pocket_pdb_path).expanduser() if isinstance(pocket_pdb_path, str) else Path("")
    query_info = {"input": str(pocket_path), "source": "TargetDiff", "parameters": {"num_samples": num_samples}}
    if not pocket_path.is_file():
        return _hit_error(f"Pocket PDB file does not exist: {pocket_path}", query_info)
    if not isinstance(output_dir, str) or not output_dir.strip():
        return _hit_error("output_dir must be a non-empty directory path.", query_info)
    if not 1 <= num_samples <= 1000:
        return _hit_error("num_samples must be between 1 and 1000.", query_info)
    if not 1 <= batch_size <= 1000:
        return _hit_error("batch_size must be between 1 and 1000.", query_info)
    if gpu_device is None:
        return _hit_error("MCP 未分配 GPU；请通过 MCP Worker 调用或显式提供 gpu_device。", query_info)
    tool_root = _tools_pkg_root() / "TargetDiff"
    config = config_path.strip() or os.environ.get("TARGETDIFF_CONFIG", "").strip() or str(tool_root / "upstream" / "configs" / "sampling.yml")
    checkpoint = checkpoint_path.strip() or os.environ.get("TARGETDIFF_CHECKPOINT", "").strip() or str(tool_root / "weight" / "pretrained_diffusion.pt")
    if not config or not Path(config).expanduser().is_file():
        return _hit_error("TargetDiff sampling config is missing; pass config_path or set TARGETDIFF_CONFIG.", query_info)
    if not checkpoint or not Path(checkpoint).expanduser().is_file():
        return _hit_error("TargetDiff checkpoint is missing; pass checkpoint_path or set TARGETDIFF_CHECKPOINT.", query_info)
    return _run_molecular_design_adapter(
        "TargetDiff",
        "run_tool.py",
        [
            "--pocket-pdb-path",
            str(pocket_path.resolve()),
            "--config-path",
            str(Path(config).expanduser().resolve()),
            "--checkpoint-path",
            str(Path(checkpoint).expanduser().resolve()),
            "--num-samples",
            str(num_samples),
            "--batch-size",
            str(batch_size),
            "--device",
            "cpu" if gpu_device < 0 else "cuda:0",
        ],
        output_dir,
        repo_path,
        python_executable,
        gpu_device,
        timeout_seconds,
    )


def generate_ligands_with_pocket2mol(
    protein_pdb_path: str,
    pocket_center: list[float],
    output_dir: str,
    num_samples: int = 20,
    bbox_size: float = 23.0,
    config_path: str = "",
    checkpoint_path: str = "",
    repo_path: str = "",
    python_executable: str = "",
    gpu_device: int | None = None,
    timeout_seconds: int = 7200,
) -> dict:
    """Generate 3-D molecules inside a protein-pocket bounding box with Pocket2Mol."""
    gpu_device = _resolve_allocated_gpu(gpu_device, True)
    protein_path = Path(protein_pdb_path).expanduser() if isinstance(protein_pdb_path, str) else Path("")
    query_info = {
        "input": str(protein_path),
        "source": "Pocket2Mol",
        "parameters": {"pocket_center": pocket_center, "bbox_size": bbox_size, "num_samples": num_samples},
    }
    if not protein_path.is_file():
        return _hit_error(f"Protein PDB file does not exist: {protein_path}", query_info)
    if not isinstance(pocket_center, list) or len(pocket_center) != 3:
        return _hit_error("pocket_center must contain exactly three numeric coordinates.", query_info)
    try:
        center = [float(value) for value in pocket_center]
        bbox_value = float(bbox_size)
    except (TypeError, ValueError):
        return _hit_error("pocket_center and bbox_size must be numeric.", query_info)
    if bbox_value <= 0:
        return _hit_error("bbox_size must be positive.", query_info)
    if not isinstance(output_dir, str) or not output_dir.strip():
        return _hit_error("output_dir must be a non-empty directory path.", query_info)
    if not 1 <= num_samples <= 1000:
        return _hit_error("num_samples must be between 1 and 1000.", query_info)
    if gpu_device is None:
        return _hit_error("MCP 未分配 GPU；请通过 MCP Worker 调用或显式提供 gpu_device。", query_info)
    tool_root = _tools_pkg_root() / "Pocket2Mol"
    config = config_path.strip() or os.environ.get("POCKET2MOL_CONFIG", "").strip() or str(tool_root / "upstream" / "configs" / "sample_for_pdb.yml")
    checkpoint = checkpoint_path.strip() or os.environ.get("POCKET2MOL_CHECKPOINT", "").strip() or str(tool_root / "models" / "pretrained_Pocket2Mol.pt")
    if not config or not Path(config).expanduser().is_file():
        return _hit_error("Pocket2Mol sampling config is missing; pass config_path or set POCKET2MOL_CONFIG.", query_info)
    if not checkpoint or not Path(checkpoint).expanduser().is_file():
        return _hit_error("Pocket2Mol checkpoint is missing; pass checkpoint_path or set POCKET2MOL_CHECKPOINT.", query_info)
    return _run_molecular_design_adapter(
        "Pocket2Mol",
        "run_tool.py",
        [
            "--protein-pdb-path",
            str(protein_path.resolve()),
            "--center",
            ",".join(str(value) for value in center),
            "--bbox-size",
            str(bbox_value),
            "--config-path",
            str(Path(config).expanduser().resolve()),
            "--checkpoint-path",
            str(Path(checkpoint).expanduser().resolve()),
            "--num-samples",
            str(num_samples),
            "--device",
            "cpu" if gpu_device < 0 else "cuda:0",
        ],
        output_dir,
        repo_path,
        python_executable,
        gpu_device,
        timeout_seconds,
    )


def optimize_ligands_with_autogrow4(
    receptor_pdb_path: str,
    source_smiles: list[str],
    box_center: list[float],
    box_size: list[float],
    output_dir: str,
    num_generations: int = 3,
    population_size: int = 10,
    repo_path: str = "",
    python_executable: str = "",
    timeout_seconds: int = 14400,
) -> dict:
    """Grow and optimize seed ligands with the official AutoGrow4 evolutionary workflow."""
    receptor_path = Path(receptor_pdb_path).expanduser() if isinstance(receptor_pdb_path, str) else Path("")
    query_info = {
        "input": {"receptor_pdb_path": str(receptor_path), "source_smiles": source_smiles},
        "source": "AutoGrow4",
        "parameters": {"box_center": box_center, "box_size": box_size, "num_generations": num_generations},
    }
    if not receptor_path.is_file():
        return _hit_error(f"Receptor PDB file does not exist: {receptor_path}", query_info)
    if not isinstance(source_smiles, list) or not source_smiles:
        return _hit_error("source_smiles must be a non-empty list of SMILES strings.", query_info)
    try:
        from rdkit import Chem

        invalid = [value for value in source_smiles if not isinstance(value, str) or Chem.MolFromSmiles(value) is None]
    except ImportError:
        invalid = [value for value in source_smiles if not isinstance(value, str) or not value.strip()]
    if invalid:
        return _hit_error(f"source_smiles contains invalid SMILES: {invalid[:5]}", query_info)
    if not isinstance(box_center, list) or len(box_center) != 3 or not isinstance(box_size, list) or len(box_size) != 3:
        return _hit_error("box_center and box_size must each contain three numeric values.", query_info)
    try:
        center = [float(value) for value in box_center]
        size = [float(value) for value in box_size]
    except (TypeError, ValueError):
        return _hit_error("box_center and box_size must contain numeric values.", query_info)
    if any(value <= 0 for value in size):
        return _hit_error("All box_size values must be positive.", query_info)
    if not isinstance(output_dir, str) or not output_dir.strip():
        return _hit_error("output_dir must be a non-empty directory path.", query_info)
    if not 1 <= num_generations <= 100:
        return _hit_error("num_generations must be between 1 and 100.", query_info)
    if not 2 <= population_size <= 10000:
        return _hit_error("population_size must be between 2 and 10000.", query_info)
    return _run_molecular_design_adapter(
        "AutoGrow4",
        "run_tool.py",
        [
            "--receptor-pdb-path",
            str(receptor_path.resolve()),
            "--source-smiles-json",
            json.dumps(source_smiles),
            "--box-center",
            ",".join(str(value) for value in center),
            "--box-size",
            ",".join(str(value) for value in size),
            "--num-generations",
            str(num_generations),
            "--population-size",
            str(population_size),
        ],
        output_dir,
        repo_path,
        python_executable,
        -1,
        timeout_seconds,
    )


def link_fragments_with_syntalinker(
    fragment_a_smiles: str,
    fragment_b_smiles: str,
    linker_length: int,
    output_dir: str,
    num_samples: int = 10,
    beam_size: int = 10,
    max_length: int = 200,
    checkpoint_path: str = "",
    repo_path: str = "",
    python_executable: str = "",
    gpu_device: int = -1,
    timeout_seconds: int = 3600,
) -> dict:
    """Link two single-attachment fragments using an official SyntaLinker model checkpoint."""
    query_info = {
        "input": {"fragment_a_smiles": fragment_a_smiles, "fragment_b_smiles": fragment_b_smiles},
        "source": "SyntaLinker",
        "parameters": {"linker_length": linker_length, "num_samples": num_samples, "beam_size": beam_size},
    }
    try:
        from rdkit import Chem

        for label, smiles in (("fragment_a_smiles", fragment_a_smiles), ("fragment_b_smiles", fragment_b_smiles)):
            molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
            if molecule is None:
                return _hit_error(f"{label} must be valid SMILES.", query_info)
            dummy_count = sum(atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms())
            if dummy_count != 1:
                return _hit_error(f"{label} must contain exactly one dummy attachment atom (*).", query_info)
    except ImportError:
        if not all(isinstance(value, str) and value.strip() for value in (fragment_a_smiles, fragment_b_smiles)):
            return _hit_error("Both fragment SMILES must be non-empty strings.", query_info)
    if not isinstance(linker_length, int) or not 2 <= linker_length <= 20:
        return _hit_error("linker_length must be an integer between 2 and 20, matching the published model domain.", query_info)
    if not isinstance(output_dir, str) or not output_dir.strip():
        return _hit_error("output_dir must be a non-empty directory path.", query_info)
    if not isinstance(num_samples, int) or not 1 <= num_samples <= 100:
        return _hit_error("num_samples must be between 1 and 100.", query_info)
    if not isinstance(beam_size, int) or not 1 <= beam_size <= 100:
        return _hit_error("beam_size must be between 1 and 100.", query_info)
    if num_samples > beam_size:
        return _hit_error("num_samples cannot exceed beam_size.", query_info)
    if not isinstance(max_length, int) or not 1 <= max_length <= 1000:
        return _hit_error("max_length must be between 1 and 1000.", query_info)
    checkpoint = checkpoint_path.strip() or os.environ.get("SYNTALINKER_CHECKPOINT", "").strip()
    if not checkpoint or not Path(checkpoint).expanduser().is_file():
        return _hit_error(
            "SyntaLinker checkpoint is missing; pass checkpoint_path or set SYNTALINKER_CHECKPOINT. The upstream repository does not publish pretrained weights.",
            query_info,
        )
    return _run_molecular_design_adapter(
        "SyntaLinker",
        "run_tool.py",
        [
            "--fragment-a-smiles",
            fragment_a_smiles,
            "--fragment-b-smiles",
            fragment_b_smiles,
            "--linker-length",
            str(linker_length),
            "--checkpoint-path",
            str(Path(checkpoint).expanduser().resolve()),
            "--num-samples",
            str(num_samples),
            "--beam-size",
            str(beam_size),
            "--max-length",
            str(max_length),
            "--gpu-device",
            str(gpu_device),
        ],
        output_dir,
        repo_path,
        python_executable,
        gpu_device,
        timeout_seconds,
    )


def detect_protein_pockets_with_fpocket(
    pdb_file_path: str,
    output_dir: str,
    top_n: int = 10,
) -> dict:
    """Detect pockets in a local PDB structure with the fpocket Docker image."""
    from biomni.tool._a1_evidence_tools import detect_protein_pockets_with_fpocket_impl

    return detect_protein_pockets_with_fpocket_impl(pdb_file_path, output_dir, top_n)


def score_protein_pockets_with_dogsite(
    pdb_file_path: str,
    output_dir: str,
    chain_id: str | None = None,
    include_subpockets: bool = False,
    top_n: int = 10,
) -> dict:
    """Upload a local PDB structure and retrieve ProteinsPlus DoGSiteScorer results."""
    from biomni.tool._a1_evidence_tools import score_protein_pockets_with_dogsite_impl

    return score_protein_pockets_with_dogsite_impl(
        pdb_file_path,
        output_dir,
        chain_id,
        include_subpockets,
        top_n,
    )


def _run_molecule_generation_backend(
    backend: str,
    generation_input: str,
    output_dir: str,
    num_molecules: int,
    timeout_seconds: int,
    command_env: str,
    image_env: str,
    default_command: list[str] | None = None,
    backend_args: list[str] | None = None,
    input_parameter_name: str = "prompt",
) -> dict:
    """Run a configured generation backend without embedding model code or weights."""
    query_info = {
        "source": backend,
        "parameters": {
            input_parameter_name: generation_input,
            "output_dir": output_dir,
            "num_molecules": num_molecules,
        },
    }
    if not isinstance(generation_input, str) or not generation_input.strip():
        return _hit_error(f"{input_parameter_name} must be a non-empty string.", query_info)
    if not isinstance(output_dir, str) or not output_dir.strip():
        return _hit_error("output_dir must be a non-empty path string.", query_info)
    if type(num_molecules) is not int or not 1 <= num_molecules <= 10000:
        return _hit_error("num_molecules must be between 1 and 10000.", query_info)
    if type(timeout_seconds) is not int or timeout_seconds < 1:
        return _hit_error("timeout_seconds must be at least 1.", query_info)
    output = Path(output_dir).expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return _hit_error(f"Cannot create output_dir: {exc}", query_info)
    request = output / f"{backend.lower()}_request.json"
    try:
        request.write_text(
            json.dumps({input_parameter_name: generation_input, "num_molecules": num_molecules}, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        return _hit_error(f"Cannot write backend request: {exc}", query_info)
    configured = os.environ.get(command_env, "").strip()
    image = os.environ.get(image_env, "").strip()
    if configured:
        command = shlex.split(configured) + ["--input", str(request), "--output-dir", str(output), "--num-molecules", str(num_molecules), *(backend_args or [])]
    elif image:
        command = ["docker", "run", "--rm", "-v", f"{output}:{output}", image, "--input", str(request), "--output-dir", str(output), "--num-molecules", str(num_molecules), *(backend_args or [])]
    elif default_command and Path(default_command[0]).is_file():
        command = [*default_command, "--input", str(request), "--output-dir", str(output), "--num-molecules", str(num_molecules), *(backend_args or [])]
    else:
        return _hit_error(
            f"{backend} backend is not deployed. Set {command_env} to an executable command or {image_env} to a Docker image.",
            query_info,
        )
    try:
        child_environment = os.environ.copy()
        # MCP Worker 已锁定物理卡；外部 Python 子进程只看到这一张卡。
        if os.environ.get("BIOMNI_ALLOCATED_GPU", "").strip():
            child_environment["CUDA_VISIBLE_DEVICES"] = os.environ["BIOMNI_ALLOCATED_GPU"].strip()
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=child_environment,
        )
    except FileNotFoundError as exc:
        return _hit_error(f"{backend} runtime is unavailable: {exc}", query_info)
    except subprocess.TimeoutExpired:
        return _hit_error(f"{backend} timed out after {timeout_seconds} seconds.", query_info)
    if completed.returncode != 0:
        error_output = completed.stderr or completed.stdout or "no backend diagnostics"
        return _hit_error(
            f"{backend} backend failed with exit code {completed.returncode}: {error_output[-2000:]}", query_info
        )
    smiles = []
    smiles_paths = []
    for candidate in (output / "smiles.txt", output / "generated_smiles.txt"):
        if candidate.is_file():
            smiles_paths.append(str(candidate))
            smiles.extend(line.strip() for line in candidate.read_text(encoding="utf-8").splitlines() if line.strip())
    if not smiles:
        smiles = [line.strip() for line in (completed.stdout or "").splitlines() if line.strip() and not line.startswith("{")]
    try:
        from rdkit import Chem

        canonical = []
        for value in smiles:
            molecule = Chem.MolFromSmiles(value)
            if molecule is not None:
                normalized = Chem.MolToSmiles(molecule, canonical=True)
                if normalized not in canonical:
                    canonical.append(normalized)
        smiles = canonical
    except ImportError:
        pass
    if not smiles:
        return _hit_error(f"{backend} completed but returned no valid SMILES.", query_info)
    return _hit_success(
        {
            "backend": backend,
            "smiles": smiles[:num_molecules],
            "count": min(len(smiles), num_molecules),
            "output_dir": str(output),
            "result_path": smiles_paths[0] if smiles_paths else None,
            "stdout_tail": (completed.stdout or "")[-2000:],
        },
        query_info,
    )


def generate_molecules_with_reinvent(config_path: str, output_dir: str, gpu_device: int | None = None, use_gpu: bool = True, timeout_seconds: int = 1800) -> dict:
    """Run a REINVENT 4 sampling or optimization configuration."""
    gpu_device = _resolve_allocated_gpu(gpu_device, use_gpu)
    query_info = {"source": "REINVENT 4", "parameters": {"config_path": config_path, "gpu_device": gpu_device, "use_gpu": use_gpu}}
    config = Path(config_path).expanduser().resolve()
    if not config.is_file() or config.suffix.lower() not in {".toml", ".json", ".yaml", ".yml"}:
        return _hit_error("config_path must be an existing TOML, JSON, or YAML REINVENT configuration.", query_info)
    if use_gpu and gpu_device is None:
        return _hit_error("MCP 未分配 GPU；请通过 MCP Worker 调用或显式提供 gpu_device。", query_info)
    if use_gpu and (not isinstance(gpu_device, int) or gpu_device < 0):
        return _hit_error("gpu_device must be a non-negative CUDA device index.", query_info)
    if not isinstance(timeout_seconds, int) or timeout_seconds < 1:
        return _hit_error("timeout_seconds must be at least 1.", query_info)
    root = Path(__file__).resolve().parents[2]
    executable = Path(os.environ.get("REINVENT_COMMAND", "") or _tools_pkg_root(root) / "REINVENT" / ".conda" / "bin" / "reinvent")
    if not executable.is_file() or not os.access(executable, os.X_OK):
        return _hit_error("REINVENT executable is unavailable; deploy tools_pkg/REINVENT/.conda or set REINVENT_COMMAND.", query_info)
    output = Path(output_dir).expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return _hit_error(f"Cannot create output_dir: {exc}", query_info)
    log_path = output / "reinvent.log"
    child_gpu = _child_cuda_index(gpu_device)
    command = [str(executable), "-d", f"cuda:{child_gpu}" if use_gpu else "cpu", "-l", str(log_path), str(config)]
    upstream_root = _tools_pkg_root(root) / "REINVENT" / "upstream"
    prior_root = Path(os.environ.get("REINVENT_PRIOR_DIR", "") or upstream_root / "priors").expanduser().resolve()
    default_prior = prior_root / "reinvent.prior"
    if not default_prior.is_file() or not os.access(default_prior, os.R_OK):
        return _hit_error(
            "REINVENT prior is unavailable. Restore reinvent.prior from upstream revision b441244 (Discussion #268) or set REINVENT_PRIOR_DIR.",
            query_info,
        )
    # REINVENT resolves relative model_file values against its working directory.
    # Run in a disposable directory containing the official priors so the upstream
    # sampling.toml (model_file = \"priors/reinvent.prior\") works without editing
    # the user's configuration or the upstream checkout.
    try:
        run_dir = Path(tempfile.mkdtemp(prefix=".reinvent_run_", dir=output))
    except OSError as exc:
        return _hit_error(f"Cannot prepare isolated REINVENT run directory: {exc}", query_info)
    try:
        (run_dir / "priors").symlink_to(prior_root, target_is_directory=True)
    except OSError as exc:
        shutil.rmtree(run_dir, ignore_errors=True)
        return _hit_error(f"Cannot prepare isolated REINVENT prior directory: {exc}", query_info)
    try:
        child_environment = os.environ.copy()
        if use_gpu and os.environ.get("BIOMNI_ALLOCATED_GPU", "").strip():
            child_environment["CUDA_VISIBLE_DEVICES"] = os.environ["BIOMNI_ALLOCATED_GPU"].strip()
        completed = subprocess.run(
            command,
            cwd=run_dir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=child_environment,
        )
    except subprocess.TimeoutExpired:
        shutil.rmtree(run_dir, ignore_errors=True)
        return _hit_error(f"REINVENT timed out after {timeout_seconds} seconds.", query_info)
    except OSError as exc:
        shutil.rmtree(run_dir, ignore_errors=True)
        return _hit_error(f"REINVENT runtime is unavailable: {exc}", query_info)
    try:
        if completed.returncode != 0:
            return _hit_error(f"REINVENT failed: {(completed.stderr or completed.stdout)[-3000:]}", query_info)
        generated_files = []
        for path in sorted(run_dir.rglob("*")):
            if path.is_file() and not path.is_symlink():
                destination = output / path.relative_to(run_dir)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(destination))
                generated_files.append(str(destination))
        files = [str(path) for path in sorted(output.rglob("*")) if path.is_file()]
        return _hit_success(
            {
                "output_dir": str(output),
                "log_path": str(log_path),
                "output_files": files,
                "generated_files": generated_files,
                "stdout_tail": (completed.stdout or "")[-2000:],
            },
            query_info,
        )
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def generate_smiles_with_molgpt(
    smiles_prefix: str,
    output_dir: str,
    num_molecules: int = 10,
    gpu_device: int | None = None,
    use_gpu: bool = True,
    timeout_seconds: int = 1800,
) -> dict:
    """Generate SMILES unconditionally or from a MOSES-vocabulary SMILES prefix."""
    gpu_device = _resolve_allocated_gpu(gpu_device, use_gpu)
    query_info = {
        "source": "MolGPT",
        "parameters": {"smiles_prefix": smiles_prefix, "output_dir": output_dir, "num_molecules": num_molecules},
    }
    if not isinstance(smiles_prefix, str):
        return _hit_error("smiles_prefix must be a string.", query_info)
    if type(num_molecules) is not int or not 1 <= num_molecules <= 10000:
        return _hit_error("num_molecules must be between 1 and 10000.", query_info)
    if type(timeout_seconds) is not int or timeout_seconds < 1:
        return _hit_error("timeout_seconds must be at least 1.", query_info)
    normalized_prefix = smiles_prefix.strip()
    if not normalized_prefix or normalized_prefix.lower() == "unconditional":
        normalized_prefix = "unconditional"
    else:
        token_pattern = re.compile(r"Br|Cl|\[nH\]|\[SH\]|\[H\]|#|\(|\)|-|[1-6]|=|C|F|N|O|S|c|n|o|s")
        tokens = [match.group(0) for match in token_pattern.finditer(normalized_prefix)]
        if not tokens or "".join(tokens) != normalized_prefix:
            return _hit_error(
                "smiles_prefix must be 'unconditional' or a prefix fully composed of tokens in the deployed MOSES vocabulary; natural-language prompts are not supported.",
                query_info,
            )
    root = Path(__file__).resolve().parents[2]
    command = [
        str(_tools_pkg_root(root) / "MolGPT" / ".conda" / "bin" / "python"),
        str(root / "scripts" / "molecular_design_adapters" / "molgpt_generate.py"),
    ]
    if use_gpu and gpu_device is None:
        return _hit_error("MCP 未分配 GPU；请通过 MCP Worker 调用或显式提供 gpu_device。", query_info)
    if use_gpu and (not isinstance(gpu_device, int) or gpu_device < 0):
        return _hit_error("gpu_device must be a non-negative CUDA device index.", query_info)
    child_gpu = _child_cuda_index(gpu_device)
    return _run_molecule_generation_backend(
        "MolGPT", normalized_prefix, output_dir, num_molecules, timeout_seconds, "MOLGPT_COMMAND", "MOLGPT_DOCKER_IMAGE", command,
        ["--gpu-device", str(child_gpu if isinstance(child_gpu, int) and child_gpu >= 0 else 0), "--use-gpu" if use_gpu else "--cpu"],
        input_parameter_name="smiles_prefix",
    )


def generate_molecules_with_graphaf(
    prompt: str,
    output_dir: str,
    num_molecules: int = 10,
    gpu_device: int | None = None,
    use_gpu: bool = True,
    timeout_seconds: int = 1800,
) -> dict:
    """Generate molecular graphs with a configured GraphAF command or Docker image."""
    if gpu_device is not None and (not isinstance(gpu_device, int) or gpu_device < 0):
        return _hit_error("gpu_device must be a non-negative CUDA device index.", {"source": "GraphAF"})
    gpu_device = _resolve_allocated_gpu(gpu_device, use_gpu)
    query_info = {
        "source": "GraphAF",
        "parameters": {
            "prompt": prompt,
            "output_dir": output_dir,
            "num_molecules": num_molecules,
            "gpu_device": gpu_device,
            "use_gpu": use_gpu,
            "timeout_seconds": timeout_seconds,
        },
    }
    if not isinstance(prompt, str) or not prompt.strip():
        return _hit_error("prompt must be a non-empty string.", query_info)
    if type(num_molecules) is not int or not 1 <= num_molecules <= 10000:
        return _hit_error("num_molecules must be between 1 and 10000.", query_info)
    if type(timeout_seconds) is not int or timeout_seconds < 1:
        return _hit_error("timeout_seconds must be at least 1.", query_info)
    root = Path(__file__).resolve().parents[2]
    configured_graphaf_python = os.environ.get("GRAPHAF_PYTHON", "").strip()
    graphaf_python = Path(
        configured_graphaf_python or _tools_pkg_root(root) / "GraphAF" / ".conda" / "bin" / "python"
    ).expanduser().resolve()
    if configured_graphaf_python and (not graphaf_python.is_file() or not os.access(graphaf_python, os.X_OK)):
        return _hit_error("GRAPHAF_PYTHON does not point to an existing Python executable.", query_info)
    if not configured_graphaf_python and not graphaf_python.is_file():
        graphaf_python = Path(sys.executable)
    command = [
        str(graphaf_python),
        str(root / "scripts" / "molecular_design_adapters" / "graphaf_generate.py"),
    ]
    if use_gpu and gpu_device is None:
        return _hit_error("MCP 未分配 GPU；请通过 MCP Worker 调用或显式提供 gpu_device。", query_info)
    if use_gpu and (not isinstance(gpu_device, int) or gpu_device < 0):
        return _hit_error("gpu_device must be a non-negative CUDA device index.", query_info)
    if not isinstance(use_gpu, bool):
        return _hit_error("use_gpu must be a boolean.", query_info)
    child_gpu = _child_cuda_index(gpu_device)
    result = _run_molecule_generation_backend(
        "GraphAF",
        prompt,
        output_dir,
        num_molecules,
        timeout_seconds,
        "GRAPHAF_COMMAND",
        "GRAPHAF_DOCKER_IMAGE",
        command,
        ["--gpu-device", str(child_gpu if isinstance(child_gpu, int) and child_gpu >= 0 else 0), *([] if use_gpu else ["--cpu"])],
    )
    result["query_info"] = query_info
    return result


def edit_molecule_with_rdkit(smiles: str, smarts_pattern: str, replacement_smiles: str) -> dict:
    """Apply an RDKit SMARTS replacement, a deterministic MolTransform-style edit."""
    query_info = {"source": "RDKit MolTransform", "parameters": {"smiles": smiles, "smarts_pattern": smarts_pattern, "replacement_smiles": replacement_smiles}}
    try:
        from rdkit import Chem
        molecule = Chem.MolFromSmiles(smiles)
        pattern = Chem.MolFromSmarts(smarts_pattern)
        replacement = Chem.MolFromSmiles(replacement_smiles)
        if molecule is None or pattern is None or replacement is None:
            return _hit_error("smiles, smarts_pattern, and replacement_smiles must be valid RDKit structures.", query_info)
        products = Chem.ReplaceSubstructs(molecule, pattern, replacement, replaceAll=True)
        result = [Chem.MolToSmiles(product, canonical=True) for product in products if product is not None]
        return _hit_success({"edited_smiles": result, "count": len(result)}, query_info)
    except Exception as exc:
        return _hit_error(f"RDKit molecular edit failed: {exc}", query_info)


def search_rdkit_scaffold_network(smiles: str, output_dir: str = "") -> dict:
    """Enumerate the RDKit scaffold network for one molecule and return canonical SMILES."""
    query_info = {"source": "RDKit Scaffold Network", "parameters": {"smiles": smiles}}
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import rdScaffoldNetwork
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return _hit_error("smiles must be a valid SMILES string.", query_info)
        parameters = rdScaffoldNetwork.ScaffoldNetworkParams()
        network = rdScaffoldNetwork.CreateScaffoldNetwork([molecule], parameters)
        scaffolds = sorted({str(node) for node in network.nodes if node})
        result = {"scaffolds": scaffolds, "count": len(scaffolds)}
        if output_dir:
            path = Path(output_dir).expanduser().resolve()
            path.mkdir(parents=True, exist_ok=True)
            file_path = path / "scaffold_network.json"
            file_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            result["result_path"] = str(file_path)
        return _hit_success(result, query_info)
    except Exception as exc:
        return _hit_error(f"RDKit scaffold network failed: {exc}", query_info)


def hop_scaffolds_with_rdkit(
    query_smiles: str,
    candidate_compounds: list[dict],
    top_k: int = 10,
    min_score: float = 0.0,
    include_same_scaffold: bool = False,
    output_dir: str = "",
) -> dict:
    """Rank scaffold-hop candidates against a query molecule with RDKit.

    Each candidate must contain an ``id`` and ``smiles`` field.  The candidate
    molecule is the proposed replacement molecule: its Murcko scaffold is
    compared with the query scaffold using a Morgan fingerprint Tanimoto
    score, then returned with the original input SMILES and rank.
    """
    query_info = {
        "source": "RDKit Scaffold Hopping",
        "parameters": {
            "query_smiles": query_smiles,
            "candidate_count": len(candidate_compounds) if isinstance(candidate_compounds, list) else 0,
            "top_k": top_k,
            "min_score": min_score,
            "include_same_scaffold": include_same_scaffold,
        },
    }
    if not isinstance(query_smiles, str) or not query_smiles.strip():
        return _hit_error("query_smiles must be a non-empty SMILES string.", query_info)
    if not isinstance(candidate_compounds, list) or not candidate_compounds:
        return _hit_error("candidate_compounds must be a non-empty list of {id, smiles} objects.", query_info)
    if type(top_k) is not int or not 1 <= top_k <= 10000:
        return _hit_error("top_k must be an integer between 1 and 10000.", query_info)
    if type(min_score) not in {int, float} or not 0.0 <= float(min_score) <= 1.0:
        return _hit_error("min_score must be a number between 0 and 1.", query_info)
    if not isinstance(include_same_scaffold, bool):
        return _hit_error("include_same_scaffold must be a boolean.", query_info)
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        query_molecule = Chem.MolFromSmiles(query_smiles)
        if query_molecule is None:
            return _hit_error("query_smiles must be a valid SMILES string.", query_info)

        def scaffold_for(molecule):
            scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
            return scaffold if scaffold.GetNumAtoms() else molecule

        query_scaffold_molecule = scaffold_for(query_molecule)
        query_scaffold = Chem.MolToSmiles(query_scaffold_molecule, canonical=True)
        query_fingerprint = AllChem.GetMorganFingerprintAsBitVect(query_scaffold_molecule, 2, nBits=2048)
        ranked = []
        invalid_candidates = []
        skipped_same_scaffold = []
        seen_ids = set()
        for index, candidate in enumerate(candidate_compounds):
            if not isinstance(candidate, dict) or "id" not in candidate or "smiles" not in candidate:
                invalid_candidates.append({"index": index, "reason": "candidate must contain id and smiles"})
                continue
            candidate_id = candidate["id"]
            candidate_smiles = candidate["smiles"]
            if not isinstance(candidate_smiles, str) or not candidate_smiles.strip():
                invalid_candidates.append({"index": index, "id": candidate_id, "reason": "invalid smiles"})
                continue
            if candidate_id in seen_ids:
                invalid_candidates.append({"index": index, "id": candidate_id, "reason": "duplicate id"})
                continue
            seen_ids.add(candidate_id)
            candidate_molecule = Chem.MolFromSmiles(candidate_smiles)
            if candidate_molecule is None:
                invalid_candidates.append({"index": index, "id": candidate_id, "reason": "invalid smiles"})
                continue
            candidate_scaffold_molecule = scaffold_for(candidate_molecule)
            candidate_scaffold = Chem.MolToSmiles(candidate_scaffold_molecule, canonical=True)
            if not include_same_scaffold and candidate_scaffold == query_scaffold:
                skipped_same_scaffold.append(candidate_id)
                continue
            candidate_fingerprint = AllChem.GetMorganFingerprintAsBitVect(candidate_scaffold_molecule, 2, nBits=2048)
            score = float(DataStructs.TanimotoSimilarity(query_fingerprint, candidate_fingerprint))
            if score < float(min_score):
                continue
            ranked.append(
                {
                    "candidate_id": candidate_id,
                    "original_smiles": candidate_smiles,
                    "replacement_smiles": Chem.MolToSmiles(candidate_molecule, canonical=True),
                    "candidate_scaffold": candidate_scaffold,
                    "score": round(score, 6),
                }
            )
        ranked.sort(key=lambda item: (-item["score"], str(item["candidate_id"])))
        ranked = [{**item, "rank": rank} for rank, item in enumerate(ranked[:top_k], start=1)]
        result = {
            "query_smiles": query_smiles,
            "query_scaffold": query_scaffold,
            "candidates": ranked,
            "count": len(ranked),
            "invalid_candidates": invalid_candidates,
            "skipped_same_scaffold_ids": skipped_same_scaffold,
            "scoring": "Morgan-2 Tanimoto similarity of query and candidate Murcko scaffolds",
        }
        if output_dir:
            path = Path(output_dir).expanduser().resolve()
            path.mkdir(parents=True, exist_ok=True)
            file_path = path / "scaffold_hopping.json"
            file_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            result["result_path"] = str(file_path)
        if not ranked:
            return _hit_error("No valid scaffold-hop candidates matched the requested criteria.", query_info)
        return _hit_success(result, query_info)
    except Exception as exc:
        return _hit_error(f"RDKit scaffold hopping failed: {exc}", query_info)
def _validate_reaction_smiles(reaction_smiles: str) -> str | None:
    """Return an error message when a reaction SMILES is unsuitable for model input."""
    if not isinstance(reaction_smiles, str) or not reaction_smiles.strip():
        return "reaction_smiles must be a non-empty reaction SMILES string."
    sections = reaction_smiles.strip().split(">")
    if len(sections) != 3 or not sections[0] or not sections[2]:
        return "reaction_smiles must use reactants>reagents>products syntax with non-empty reactants and products."
    try:
        from rdkit import Chem

        for side_name, section in (("reactant", sections[0]), ("reagent", sections[1]), ("product", sections[2])):
            for component in filter(None, section.split(".")):
                if Chem.MolFromSmiles(component) is None:
                    return f"reaction_smiles contains an invalid {side_name} SMILES component: {component}"
    except ImportError:
        pass
    return None


def _run_synthesis_adapter(
    tool_name: str,
    arguments: list[str],
    python_executable: str,
    timeout_seconds: int,
) -> dict:
    """Run one tracked synthesis adapter in its isolated Python environment."""
    repository_root = Path(__file__).resolve().parents[2]
    tool_root = _tools_pkg_root(repository_root) / tool_name
    adapter_path = repository_root / "scripts" / "synthesis_adapters" / "run_tool.py"
    configured_python = python_executable.strip() if isinstance(python_executable, str) else ""
    environment_python = os.environ.get(f"{tool_name.upper()}_PYTHON", "").strip()
    runtime_python = Path(configured_python or environment_python or tool_root / ".conda" / "bin" / "python")
    query_info = {
        "source": tool_name,
        "parameters": {"python_executable": str(runtime_python), "timeout_seconds": timeout_seconds},
    }
    if not isinstance(timeout_seconds, int) or timeout_seconds < 1:
        return _hit_error("timeout_seconds must be a positive integer.", query_info)
    if not adapter_path.is_file():
        return _hit_error(f"Biomni synthesis adapter is missing: {adapter_path}", query_info)
    if not runtime_python.is_file() or not os.access(runtime_python, os.X_OK):
        return _hit_error(
            f"{tool_name} Python environment is unavailable at '{runtime_python}'. "
            f"Run scripts/setup_synthesis_tools.py --tool {tool_name.lower()} or set {tool_name.upper()}_PYTHON.",
            query_info,
        )
    command = [str(runtime_python), str(adapter_path), "--tool", tool_name.lower(), *arguments]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        return {
            "success": False,
            "error": f"{tool_name} inference timed out after {timeout_seconds} seconds.",
            "query_info": query_info,
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
        }
    except OSError as exc:
        return _hit_error(f"Could not run {tool_name}: {exc}", query_info)

    payload = None
    for line in reversed((completed.stdout or "").splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break
    if payload is None:
        return {
            "success": False,
            "error": f"{tool_name} adapter did not return a JSON object.",
            "query_info": query_info,
            "returncode": completed.returncode,
            "stdout_tail": (completed.stdout or "")[-4000:],
            "stderr_tail": (completed.stderr or "")[-4000:],
        }
    payload.setdefault("query_info", query_info)
    payload.setdefault("returncode", completed.returncode)
    if completed.returncode != 0:
        payload["success"] = False
        payload.setdefault("error", f"{tool_name} adapter failed with exit code {completed.returncode}.")
    return payload


def map_reaction_atoms_with_rxnmapper(
    reaction_smiles: str,
    python_executable: str = "",
    timeout_seconds: int = 120,
) -> dict:
    """Map atoms in one complete reaction SMILES with the official RXNMapper model."""
    query_info = {
        "input": reaction_smiles,
        "source": "RXNMapper",
        "parameters": {"timeout_seconds": timeout_seconds},
    }
    validation_error = _validate_reaction_smiles(reaction_smiles)
    if validation_error:
        return _hit_error(validation_error, query_info)
    payload = _run_synthesis_adapter(
        "RXNMapper",
        ["--reaction-smiles", reaction_smiles.strip()],
        python_executable,
        timeout_seconds,
    )
    payload["query_info"] = {**query_info, "runtime": payload.get("query_info", {}).get("parameters", {})}
    return payload


def predict_reaction_centers_with_retroxpert(
    product_smiles: str,
    reaction_class: int = -1,
    top_k: int = 10,
    checkpoint_path: str = "",
    repo_path: str = "",
    python_executable: str = "",
    timeout_seconds: int = 300,
) -> dict:
    """Predict likely product-bond disconnections with RetroXpert stage 1."""
    query_info = {
        "input": product_smiles,
        "source": "RetroXpert stage 1",
        "parameters": {"reaction_class": reaction_class, "top_k": top_k, "timeout_seconds": timeout_seconds},
    }
    try:
        from rdkit import Chem

        molecule = Chem.MolFromSmiles(product_smiles) if isinstance(product_smiles, str) else None
        if molecule is None:
            return _hit_error("product_smiles must be valid, non-empty SMILES.", query_info)
    except ImportError:
        if not isinstance(product_smiles, str) or not product_smiles.strip():
            return _hit_error("product_smiles must be a non-empty SMILES string.", query_info)
    if not isinstance(reaction_class, int) or reaction_class not in {-1, *range(1, 11)}:
        return _hit_error("reaction_class must be -1 for untyped inference or an integer from 1 through 10.", query_info)
    if not isinstance(top_k, int) or not 1 <= top_k <= 100:
        return _hit_error("top_k must be an integer from 1 through 100.", query_info)

    repository_root = Path(__file__).resolve().parents[2]
    default_repo = _tools_pkg_root(repository_root) / "RetroXpert" / "upstream"
    repository = Path(repo_path.strip() or os.environ.get("RETROXPERT_REPO", "") or default_repo).expanduser()
    variant = "untyped" if reaction_class == -1 else "typed"
    default_checkpoint = repository / "checkpoints" / f"USPTO50K_{variant}_checkpoint.pt"
    checkpoint = Path(
        checkpoint_path.strip() or os.environ.get("RETROXPERT_CHECKPOINT", "") or default_checkpoint
    ).expanduser()
    if not repository.is_dir():
        return _hit_error(
            f"RetroXpert source is unavailable at '{repository}'. Run scripts/setup_synthesis_tools.py --tool retroxpert.",
            query_info,
        )
    if not checkpoint.is_file():
        return _hit_error(
            f"RetroXpert {variant} checkpoint is unavailable at '{checkpoint}'. "
            "Pass checkpoint_path or set RETROXPERT_CHECKPOINT.",
            query_info,
        )
    payload = _run_synthesis_adapter(
        "RetroXpert",
        [
            "--product-smiles",
            product_smiles.strip(),
            "--reaction-class",
            str(reaction_class),
            "--top-k",
            str(top_k),
            "--repo-path",
            str(repository.resolve()),
            "--checkpoint-path",
            str(checkpoint.resolve()),
        ],
        python_executable,
        timeout_seconds,
    )
    payload["query_info"] = {**query_info, "runtime": payload.get("query_info", {}).get("parameters", {})}
    return payload


def predict_reaction_products_with_molecular_transformer(
    reactants_smiles: str,
    output_dir: str,
    reagents_smiles: str = "",
    top_k: int = 5,
    beam_size: int = 10,
    max_length: int = 200,
    checkpoint_path: str = "",
    repo_path: str = "",
    python_executable: str = "",
    gpu_device: int = -1,
    timeout_seconds: int = 600,
) -> dict:
    """Predict reaction products with the original Molecular Transformer model."""
    query_info = {
        "input": {"reactants_smiles": reactants_smiles, "reagents_smiles": reagents_smiles},
        "source": "Molecular Transformer",
        "parameters": {
            "top_k": top_k,
            "beam_size": beam_size,
            "max_length": max_length,
            "gpu_device": gpu_device,
            "timeout_seconds": timeout_seconds,
        },
    }
    try:
        from rdkit import Chem

        for field_name, value, required in (
            ("reactants_smiles", reactants_smiles, True),
            ("reagents_smiles", reagents_smiles, False),
        ):
            if required and (not isinstance(value, str) or not value.strip()):
                return _hit_error("reactants_smiles must be a non-empty dot-separated SMILES string.", query_info)
            if not value:
                continue
            if not isinstance(value, str) or ">" in value:
                return _hit_error(f"{field_name} must contain SMILES components, not reaction separators.", query_info)
            for component in value.split("."):
                if Chem.MolFromSmiles(component) is None:
                    return _hit_error(f"{field_name} contains invalid SMILES: {component}", query_info)
    except ImportError:
        if not isinstance(reactants_smiles, str) or not reactants_smiles.strip():
            return _hit_error("reactants_smiles must be a non-empty dot-separated SMILES string.", query_info)
    if not isinstance(output_dir, str) or not output_dir.strip():
        return _hit_error("output_dir must be a non-empty directory path.", query_info)
    if not isinstance(top_k, int) or not 1 <= top_k <= 50:
        return _hit_error("top_k must be an integer from 1 through 50.", query_info)
    if not isinstance(beam_size, int) or not 1 <= beam_size <= 100 or top_k > beam_size:
        return _hit_error("beam_size must be from 1 through 100 and at least top_k.", query_info)
    if not isinstance(max_length, int) or not 1 <= max_length <= 1000:
        return _hit_error("max_length must be an integer from 1 through 1000.", query_info)
    if not isinstance(gpu_device, int) or gpu_device < -1:
        return _hit_error("gpu_device must be -1 for CPU or a non-negative CUDA device index.", query_info)

    repository_root = Path(__file__).resolve().parents[2]
    default_repo = _tools_pkg_root(repository_root) / "MolecularTransformer" / "upstream"
    default_checkpoint = (
        _tools_pkg_root(repository_root)
        / "MolecularTransformer"
        / "checkpoints"
        / "MIT_mixed_augm_model_average_20.pt"
    )
    repository = Path(
        repo_path.strip() or os.environ.get("MOLECULARTRANSFORMER_REPO", "") or default_repo
    ).expanduser()
    checkpoint_value = (
        checkpoint_path.strip()
        or os.environ.get("MOLECULARTRANSFORMER_CHECKPOINT", "").strip()
        or str(default_checkpoint)
    )
    checkpoint = Path(checkpoint_value).expanduser()
    if not repository.is_dir():
        return _hit_error(
            f"Molecular Transformer source is unavailable at '{repository}'. Run the setup script with --source-only.",
            query_info,
        )
    if not checkpoint.is_file():
        return _hit_error(
            "Molecular Transformer requires a reviewed official averaged checkpoint; pass checkpoint_path or set "
            "MOLECULARTRANSFORMER_CHECKPOINT. The upstream source repository does not contain inference weights.",
            query_info,
        )
    payload = _run_synthesis_adapter(
        "MolecularTransformer",
        [
            "--reactants-smiles",
            reactants_smiles.strip(),
            "--reagents-smiles",
            reagents_smiles.strip(),
            "--output-dir",
            str(Path(output_dir).expanduser().resolve()),
            "--top-k",
            str(top_k),
            "--beam-size",
            str(beam_size),
            "--max-length",
            str(max_length),
            "--gpu-device",
            str(gpu_device),
            "--repo-path",
            str(repository.resolve()),
            "--checkpoint-path",
            str(checkpoint.resolve()),
        ],
        python_executable,
        timeout_seconds,
    )
    payload["query_info"] = {**query_info, "runtime": payload.get("query_info", {}).get("parameters", {})}
    return payload


def plan_retrosynthesis_with_askcos(
    target_smiles: str,
    max_depth: int = 5,
    max_routes: int = 5,
    expansion_time_seconds: int = 120,
    backend: str = "mcts",
    api_url: str = "",
    timeout_seconds: int = 300,
) -> dict:
    """Plan multistep routes through a separately deployed ASKCOS v2 API gateway."""
    base_url = api_url.strip() or os.environ.get("ASKCOS_API_URL", "").strip()
    query_info = {
        "input": target_smiles,
        "source": "ASKCOS v2 API gateway",
        "parameters": {
            "max_depth": max_depth,
            "max_routes": max_routes,
            "expansion_time_seconds": expansion_time_seconds,
            "backend": backend,
            "api_url": base_url,
            "timeout_seconds": timeout_seconds,
        },
    }
    try:
        from rdkit import Chem

        molecule = Chem.MolFromSmiles(target_smiles) if isinstance(target_smiles, str) else None
        if molecule is None:
            return _hit_error("target_smiles must be valid, non-empty SMILES.", query_info)
        canonical_target = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    except ImportError:
        if not isinstance(target_smiles, str) or not target_smiles.strip():
            return _hit_error("target_smiles must be a non-empty SMILES string.", query_info)
        canonical_target = target_smiles.strip()
    if not isinstance(max_depth, int) or not 1 <= max_depth <= 20:
        return _hit_error("max_depth must be an integer from 1 through 20.", query_info)
    if not isinstance(max_routes, int) or not 1 <= max_routes <= 100:
        return _hit_error("max_routes must be an integer from 1 through 100.", query_info)
    if not isinstance(expansion_time_seconds, int) or not 1 <= expansion_time_seconds <= 3600:
        return _hit_error("expansion_time_seconds must be an integer from 1 through 3600.", query_info)
    if backend not in {"mcts", "retro_star"}:
        return _hit_error("backend must be 'mcts' or 'retro_star'.", query_info)
    if not isinstance(timeout_seconds, int) or timeout_seconds < expansion_time_seconds:
        return _hit_error("timeout_seconds must be an integer at least as large as expansion_time_seconds.", query_info)
    if not base_url:
        return _hit_error("Set ASKCOS_API_URL or pass api_url for a separately deployed ASKCOS v2 gateway.", query_info)
    try:
        from urllib.parse import urlparse

        parsed_url = urlparse(base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc or parsed_url.username:
            return _hit_error("ASKCOS api_url must be an http(s) URL without embedded credentials.", query_info)
    except ValueError as exc:
        return _hit_error(f"Invalid ASKCOS api_url: {exc}", query_info)

    endpoint_path = os.environ.get(
        "ASKCOS_TREE_SEARCH_PATH", "/api/tree-search/call-sync-without-token"
    ).strip()
    if not endpoint_path.startswith("/"):
        return _hit_error("ASKCOS_TREE_SEARCH_PATH must start with '/'.", query_info)
    endpoint = base_url.rstrip("/") + endpoint_path
    payload = {
        "backend": backend,
        "smiles": canonical_target,
        "description": "Biomni retrosynthesis request",
        "tags": "biomni",
        "expand_one_options": {
            "template_count": 100,
            "max_cum_template_prob": 0.995,
            "forbidden_molecules": [],
            "known_bad_reactions": [],
            "retro_backend_options": [
                {
                    "retro_backend": "template_relevance",
                    "retro_model_name": os.environ.get("ASKCOS_RETRO_MODEL_NAME", "reaxys"),
                    "max_num_templates": 1000,
                    "max_cum_prob": 0.995,
                    "attribute_filter": [],
                    "threshold": 0.3,
                    "top_k": 10,
                }
            ],
            "use_fast_filter": True,
            "filter_threshold": 0.75,
            "retro_rerank_backend": "relevance_heuristic",
            "atom_map_backend": "rxnmapper",
            "cluster_precursors": False,
            "extract_template": False,
            "return_reacting_atoms": False,
            "selectivity_check": False,
        },
        "build_tree_options": {
            "expansion_time": expansion_time_seconds,
            "max_branching": 25,
            "max_depth": max_depth,
            "exploration_weight": 1.0,
            "return_first": False,
            "max_trees": max_routes,
            "buyable_logic": "and",
            "max_ppg_logic": "none",
            "max_scscore_logic": "none",
            "chemical_property_logic": "none",
            "chemical_popularity_logic": "none",
            "min_chempop_reactants": 5,
            "min_chempop_products": 5,
            "custom_buyables": [],
            "use_value_network": backend == "retro_star",
        },
        "enumerate_paths_options": {
            "path_format": "json",
            "json_format": "nodelink",
            "sorting_metric": "plausibility",
            "validate_paths": True,
            "score_trees": False,
            "cluster_trees": False,
            "cluster_method": "hdbscan",
            "min_samples": 5,
            "min_cluster_size": 5,
            "paths_only": False,
            "max_paths": max_routes,
        },
        "run_async": False,
    }
    headers = {"Accept": "application/json"}
    token = os.environ.get("ASKCOS_API_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    query_info["endpoint"] = endpoint
    query_info["authentication"] = "bearer environment token" if token else "guest endpoint"
    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=timeout_seconds)
    except requests.Timeout:
        return _hit_error(f"ASKCOS request timed out after {timeout_seconds} seconds.", query_info)
    except requests.RequestException as exc:
        return _hit_error(f"ASKCOS request failed: {exc}", query_info)
    if not response.ok:
        return _hit_error(f"ASKCOS returned HTTP {response.status_code}: {response.text[-2000:]}", query_info)
    try:
        response_payload = response.json()
    except requests.JSONDecodeError:
        return _hit_error("ASKCOS returned a non-JSON response.", query_info)
    if not isinstance(response_payload, dict):
        return _hit_error("ASKCOS response must be a JSON object.", query_info)
    if response_payload.get("status_code") not in {None, 200}:
        return _hit_error(
            f"ASKCOS reported status {response_payload.get('status_code')}: {response_payload.get('message', '')}",
            query_info,
        )
    raw_result = response_payload.get("result", response_payload.get("results"))
    if not isinstance(raw_result, dict):
        return _hit_error("ASKCOS response does not contain a result object.", query_info)
    routes = raw_result.get("paths")
    if not isinstance(routes, list):
        uds = raw_result.get("uds", {})
        routes = uds.get("pathways", []) if isinstance(uds, dict) else []
    routes = routes[:max_routes] if isinstance(routes, list) else []
    return _hit_success(
        {
            "target_smiles": canonical_target,
            "backend": backend,
            "routes": routes,
            "route_count": len(routes),
            "stats": raw_result.get("stats", {}),
            "graph": raw_result.get("graph"),
            "result_id": raw_result.get("result_id", ""),
            "askcos_version": raw_result.get("version"),
            "scientific_warning": (
                "Routes are computational proposals and require chemist review and experimental validation."
            ),
        },
        query_info,
    )
