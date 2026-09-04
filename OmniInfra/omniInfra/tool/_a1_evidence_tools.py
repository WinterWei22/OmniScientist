"""Private implementations for the A1 target-evidence tool integrations."""

import csv
import re
import statistics
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

PHAROS_GRAPHQL_URL = "https://pharos-api.ncats.io/graphql"
OPENTARGETS_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
PROTEINSPLUS_UPLOAD_URL = "https://proteins.plus/api/pdb_files_rest"
PROTEINSPLUS_DOGSITE_URL = "https://proteins.plus/api/dogsite_rest"
MAGECK_IMAGE = "quay.io/biocontainers/mageck:0.5.9.5--py310hc52dbad_9"
FPOCKET_IMAGE = "quay.io/biocontainers/fpocket:4.0.0"

GTEX_FILENAME = "gtex_tissue_gene_tpm.parquet"
DEPMAP_DEPENDENCY_FILENAME = "DepMap_CRISPRGeneDependency.csv"
DEPMAP_EFFECT_FILENAME = "DepMap_CRISPRGeneEffect.csv"
DEPMAP_MODEL_FILENAME = "DepMap_Model.csv"
DEPMAP_EXPRESSION_FILENAME = "DepMap_OmicsExpressionProteinCodingGenesTPMLogp1.csv"


def _failure(tool: str, query_info: dict, error: str, source: dict | None = None) -> dict:
    result = {"success": False, "tool": tool, "query_info": query_info, "error": error}
    if source is not None:
        result["source"] = source
    return result


def _success(tool: str, source: dict, query_info: dict, result: dict) -> dict:
    return {"success": True, "tool": tool, "source": source, "query_info": query_info, "result": result}


def _validate_gene_query(tool: str, gene_symbol: str, max_results: int, **extra: object) -> dict | None:
    query_info = {"gene_symbol": gene_symbol, **extra}
    if not isinstance(gene_symbol, str) or not gene_symbol.strip():
        return _failure(tool, query_info, "gene_symbol must be a non-empty HGNC gene symbol")
    if not isinstance(max_results, int) or isinstance(max_results, bool) or max_results < 1:
        return _failure(tool, query_info, "result limit must be a positive integer")
    return None


def _graphql_request(url: str, query: str, variables: dict, timeout: int = 60) -> tuple[dict | None, str | None]:
    try:
        response = requests.post(
            url,
            json={"query": query, "variables": variables},
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return None, f"GraphQL request failed: {exc}"
    if not response.ok:
        return None, f"GraphQL service returned HTTP {response.status_code}: {response.text[:500]}"
    try:
        payload = response.json()
    except ValueError:
        return None, "GraphQL service returned a non-JSON response"
    if payload.get("errors"):
        messages = "; ".join(str(item.get("message", item)) for item in payload["errors"])
        return None, f"GraphQL errors: {messages}"
    return payload.get("data", {}), None


_PHAROS_QUERY = """
query OmniInfraPharosTarget($symbol: String!, $diseaseTop: Int!, $ligandTop: Int!) {
  target(q: {sym: $symbol}) {
    tcrdid
    sym
    name
    tdl
    fam
    description
    diseases(top: $diseaseTop) {
      name
      mondoID
      associationCount
      directAssociationCount
      associations(top: 100) {
        did
        drug
        type
        name
        source
        zscore
        evidence
        conf
        reference
        log2foldchange
        pvalue
        score
      }
    }
    ligands(top: $ligandTop) {
      ligid
      name
      description
      isdrug
      smiles
      actcnt
      targetCount
      activities(all: false) {
        type
        moa
        value
        reference
      }
    }
  }
}
"""


def query_pharos_target_impl(
    gene_symbol: str,
    disease_name: str | None = None,
    max_results: int = 10,
) -> dict:
    tool = "query_pharos_target"
    query_info = {"gene_symbol": gene_symbol, "disease_name": disease_name, "max_results": max_results}
    invalid = _validate_gene_query(tool, gene_symbol, max_results, disease_name=disease_name)
    if invalid:
        return invalid
    data, error = _graphql_request(
        PHAROS_GRAPHQL_URL,
        _PHAROS_QUERY,
        {
            "symbol": gene_symbol.strip().upper(),
            "diseaseTop": 100 if disease_name else max_results,
            "ligandTop": max_results,
        },
    )
    source = {"name": "Pharos", "endpoint": PHAROS_GRAPHQL_URL}
    if error:
        return _failure(tool, query_info, error, source)
    target = (data or {}).get("target")
    if not target:
        return _failure(tool, query_info, f"Pharos did not find target {gene_symbol.strip().upper()}", source)

    diseases = target.pop("diseases", []) or []
    ligands = target.pop("ligands", []) or []
    if disease_name:
        term = disease_name.casefold()
        diseases = [row for row in diseases if term in str(row.get("name", "")).casefold()]
    diseases = diseases[:max_results]
    ligands = ligands[:max_results]
    return _success(
        tool,
        source,
        query_info,
        {
            "target": target,
            "disease_associations": diseases,
            "ligand_and_drug_evidence": ligands,
        },
    )


_OPENTARGETS_SEARCH_QUERY = """
query OmniInfraResolveTarget($queryString: String!) {
  search(queryString: $queryString, entityNames: ["target"], page: {index: 0, size: 10}) {
    hits { id entity name }
  }
}
"""


_OPENTARGETS_GENETICS_QUERY = """
query OmniInfraTargetGenetics($ensemblId: String!, $page: Pagination!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    approvedName
    credibleSets(page: $page) {
      count
      rows {
        studyLocusId
        studyId
        studyType
        chromosome
        position
        region
        finemappingMethod
        credibleSetIndex
        credibleSetlog10BF
        purityMinR2
        purityMeanR2
        pValueMantissa
        pValueExponent
        beta
        standardError
        variant { id rsIds }
        study {
          id
          studyType
          traitFromSource
          projectId
          publicationTitle
          pubmedId
          nSamples
          diseases { id name }
        }
        l2GPredictions(page: $page) {
          count
          rows {
            studyLocusId
            score
            shapBaseValue
            target { id approvedSymbol }
            features { name value shapValue }
          }
        }
        colocalisation(page: $page) {
          count
          rows {
            studyLocusId
            chromosome
            colocalisationMethod
            rightStudyType
            numberColocalisingVariants
            h3
            h4
            clpp
            betaRatioSignAverage
            otherStudyLocus {
              studyLocusId
              studyId
              studyType
              qtlGeneId
              study { id studyType traitFromSource projectId }
            }
          }
        }
      }
    }
  }
}
"""


def _resolve_opentargets_gene(gene_symbol: str) -> tuple[dict | None, str | None]:
    data, error = _graphql_request(
        OPENTARGETS_GRAPHQL_URL,
        _OPENTARGETS_SEARCH_QUERY,
        {"queryString": gene_symbol},
    )
    if error:
        return None, error
    hits = ((data or {}).get("search") or {}).get("hits", [])
    symbol = gene_symbol.casefold()
    exact = [hit for hit in hits if hit.get("entity") == "target" and str(hit.get("name", "")).casefold() == symbol]
    if not exact:
        return None, f"Open Targets did not resolve HGNC symbol {gene_symbol}"
    return exact[0], None


def query_opentargets_genetic_evidence_impl(
    gene_symbol: str,
    disease_name: str | None = None,
    max_results: int = 10,
) -> dict:
    tool = "query_opentargets_genetic_evidence"
    query_info = {"gene_symbol": gene_symbol, "disease_name": disease_name, "max_results": max_results}
    invalid = _validate_gene_query(tool, gene_symbol, max_results, disease_name=disease_name)
    if invalid:
        return invalid
    resolved, error = _resolve_opentargets_gene(gene_symbol.strip().upper())
    source = {"name": "Open Targets Platform", "endpoint": OPENTARGETS_GRAPHQL_URL, "api_version": "v4"}
    if error:
        return _failure(tool, query_info, error, source)
    data, error = _graphql_request(
        OPENTARGETS_GRAPHQL_URL,
        _OPENTARGETS_GENETICS_QUERY,
        {"ensemblId": resolved["id"], "page": {"index": 0, "size": max_results}},
        timeout=90,
    )
    if error:
        return _failure(tool, query_info, error, source)
    target = (data or {}).get("target")
    if not target:
        return _failure(tool, query_info, f"Open Targets returned no target for {resolved['id']}", source)

    rows = (target.get("credibleSets") or {}).get("rows", [])
    if disease_name:
        term = disease_name.casefold()
        rows = [
            row
            for row in rows
            if term in str((row.get("study") or {}).get("traitFromSource", "")).casefold()
            or any(
                term in str(disease.get("name", "")).casefold()
                for disease in (row.get("study") or {}).get("diseases", [])
            )
        ]
    rows = rows[:max_results]
    studies: list[dict] = []
    credible_sets: list[dict] = []
    locus_to_gene: list[dict] = []
    colocalisations: list[dict] = []
    seen_studies: set[str] = set()
    for row in rows:
        study = row.get("study") or {}
        if study.get("id") and study["id"] not in seen_studies:
            studies.append(study)
            seen_studies.add(study["id"])
        credible_sets.append(
            {key: value for key, value in row.items() if key not in {"study", "l2GPredictions", "colocalisation"}}
        )
        for prediction in (row.get("l2GPredictions") or {}).get("rows", []):
            if (prediction.get("target") or {}).get("id") == target.get("id"):
                locus_to_gene.append(prediction)
        colocalisations.extend((row.get("colocalisation") or {}).get("rows", []))
    return _success(
        tool,
        source,
        query_info,
        {
            "gene": {
                "input_symbol": gene_symbol,
                "ensembl_id": target.get("id"),
                "approved_symbol": target.get("approvedSymbol"),
                "approved_name": target.get("approvedName"),
            },
            "studies": studies[:max_results],
            "credible_sets": credible_sets[:max_results],
            "locus_to_gene": locus_to_gene[:max_results],
            "colocalisations": colocalisations[:max_results],
        },
    )


def _find_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    normalized = {re.sub(r"[^a-z0-9]", "", str(column).casefold()): str(column) for column in frame.columns}
    for alias in aliases:
        match = normalized.get(re.sub(r"[^a-z0-9]", "", alias.casefold()))
        if match:
            return match
    return None


def _json_value(value: object) -> object:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _records(frame: pd.DataFrame) -> list[dict]:
    return [{str(key): _json_value(value) for key, value in row.items()} for row in frame.to_dict(orient="records")]


def query_gtex_expression_impl(gene_symbol: str, data_lake_path: str, top_n: int = 10) -> dict:
    tool = "query_gtex_expression"
    query_info = {"gene_symbol": gene_symbol, "data_lake_path": data_lake_path, "top_n": top_n}
    invalid = _validate_gene_query(tool, gene_symbol, top_n, data_lake_path=data_lake_path)
    if invalid:
        return invalid
    path = Path(data_lake_path).expanduser() / GTEX_FILENAME
    source = {"name": "GTEx data lake snapshot", "file": str(path.resolve())}
    if not path.is_file():
        return _failure(tool, query_info, f"GTEx snapshot not found: {path}", source)
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        return _failure(tool, query_info, f"Could not read GTEx Parquet snapshot: {exc}", source)
    gene_col = _find_column(frame, ("gene_symbol", "gene_name", "symbol", "hgnc_symbol", "gene"))
    tissue_col = _find_column(frame, ("tissue", "tissue_name", "tissue_site_detail", "smtsd"))
    value_col = _find_column(frame, ("tpm", "median_tpm", "expression_tpm", "mean_tpm", "expression"))
    if not gene_col or not tissue_col or not value_col:
        return _failure(tool, query_info, "GTEx snapshot must contain gene symbol, tissue, and TPM columns", source)
    matched = frame[frame[gene_col].astype(str).str.casefold() == gene_symbol.strip().casefold()].copy()
    if matched.empty:
        return _failure(
            tool, query_info, f"Gene {gene_symbol.strip().upper()} is not present in the GTEx snapshot", source
        )
    matched[value_col] = pd.to_numeric(matched[value_col], errors="coerce")
    matched = matched.sort_values(value_col, ascending=False, na_position="last")
    identifier_columns = [
        column for column in ("gene_id", "gencode_id", "ensembl_id", "Description") if column in matched.columns
    ]
    metadata_columns = [
        column for column in ("release", "dataset_release", "unit", "statistic") if column in matched.columns
    ]
    gene = {"input_symbol": gene_symbol, "matched_symbol": str(matched.iloc[0][gene_col])}
    gene.update({column: _json_value(matched.iloc[0][column]) for column in identifier_columns})
    metadata = {column: _json_value(matched.iloc[0][column]) for column in metadata_columns}
    metadata.update({"file_name": path.name, "rows_for_gene": int(len(matched)), "expression_column": value_col})
    return _success(
        tool,
        source,
        query_info,
        {
            "gene": gene,
            "expression_by_tissue": _records(matched),
            "top_expressed_tissues": _records(matched.head(top_n)),
            "dataset_metadata": metadata,
        },
    )


def _depmap_gene_column(frame: pd.DataFrame, gene_symbol: str) -> str | None:
    symbol = gene_symbol.casefold()
    for column in frame.columns:
        text = str(column).strip()
        if text.casefold() == symbol or text.split(" (", 1)[0].casefold() == symbol:
            return str(column)
    return None


def _describe(values: pd.Series) -> dict:
    numeric = pd.to_numeric(values, errors="coerce").dropna().tolist()
    if not numeric:
        return {"available_values": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {
        "available_values": len(numeric),
        "mean": float(statistics.fmean(numeric)),
        "median": float(statistics.median(numeric)),
        "minimum": float(min(numeric)),
        "maximum": float(max(numeric)),
    }


def query_depmap_gene_dependency_impl(
    gene_symbol: str,
    data_lake_path: str,
    lineage: str | None = None,
    include_expression: bool = False,
    top_n: int = 20,
) -> dict:
    tool = "query_depmap_gene_dependency"
    query_info = {
        "gene_symbol": gene_symbol,
        "data_lake_path": data_lake_path,
        "lineage": lineage,
        "include_expression": include_expression,
        "top_n": top_n,
    }
    invalid = _validate_gene_query(
        tool, gene_symbol, top_n, data_lake_path=data_lake_path, lineage=lineage, include_expression=include_expression
    )
    if invalid:
        return invalid
    root = Path(data_lake_path).expanduser()
    required = {
        "dependency": root / DEPMAP_DEPENDENCY_FILENAME,
        "effect": root / DEPMAP_EFFECT_FILENAME,
        "model": root / DEPMAP_MODEL_FILENAME,
    }
    source = {
        "name": "DepMap data lake snapshot",
        "files": {name: str(path.resolve()) for name, path in required.items()},
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        return _failure(tool, query_info, f"Required DepMap snapshot files not found: {', '.join(missing)}", source)
    try:
        dependency = pd.read_csv(required["dependency"])
        effect = pd.read_csv(required["effect"])
        models = pd.read_csv(required["model"])
    except Exception as exc:
        return _failure(tool, query_info, f"Could not read DepMap snapshot: {exc}", source)
    dependency_gene = _depmap_gene_column(dependency, gene_symbol)
    effect_gene = _depmap_gene_column(effect, gene_symbol)
    if not dependency_gene or not effect_gene:
        return _failure(
            tool,
            query_info,
            f"Gene {gene_symbol.strip().upper()} is not present in both DepMap CRISPR matrices",
            source,
        )
    dep_id = _find_column(dependency, ("ModelID", "DepMap_ID", "Unnamed: 0"))
    effect_id = _find_column(effect, ("ModelID", "DepMap_ID", "Unnamed: 0"))
    model_id = _find_column(models, ("ModelID", "DepMap_ID"))
    if not dep_id or not effect_id or not model_id:
        return _failure(tool, query_info, "DepMap files must contain a ModelID or DepMap_ID column", source)
    dep = dependency[[dep_id, dependency_gene]].rename(
        columns={dep_id: "model_id", dependency_gene: "dependency_probability"}
    )
    eff = effect[[effect_id, effect_gene]].rename(columns={effect_id: "model_id", effect_gene: "gene_effect"})
    model_rows = models.rename(columns={model_id: "model_id"})
    merged = model_rows.merge(eff, on="model_id", how="inner").merge(dep, on="model_id", how="inner")
    lineage_col = _find_column(merged, ("OncotreeLineage", "lineage", "Lineage"))
    name_col = _find_column(merged, ("ModelCondition", "CellLineName", "model_name", "StrippedCellLineName"))
    if lineage:
        if not lineage_col:
            return _failure(tool, query_info, "DepMap model metadata does not contain a lineage column", source)
        merged = merged[merged[lineage_col].astype(str).str.casefold() == lineage.casefold()]
        if merged.empty:
            return _failure(tool, query_info, f"No DepMap models exactly match lineage {lineage}", source)
    merged["gene_effect"] = pd.to_numeric(merged["gene_effect"], errors="coerce")
    merged["dependency_probability"] = pd.to_numeric(merged["dependency_probability"], errors="coerce")
    merged = merged.sort_values(["gene_effect", "dependency_probability"], ascending=[True, False], na_position="last")
    output = pd.DataFrame(
        {
            "model_id": merged["model_id"],
            "model_name": merged[name_col] if name_col else None,
            "lineage": merged[lineage_col] if lineage_col else None,
            "gene_effect": merged["gene_effect"],
            "dependency_probability": merged["dependency_probability"],
        }
    )
    lineage_summary: list[dict] = []
    if lineage_col:
        for lineage_value, group in merged.groupby(lineage_col, dropna=False):
            lineage_summary.append(
                {
                    "lineage": _json_value(lineage_value),
                    "model_count": int(len(group)),
                    "gene_effect": _describe(group["gene_effect"]),
                    "dependency_probability": _describe(group["dependency_probability"]),
                }
            )
    result = {
        "gene": {"input_symbol": gene_symbol, "dependency_column": dependency_gene, "gene_effect_column": effect_gene},
        "model_dependencies": _records(output.head(top_n)),
        "lineage_summary": lineage_summary,
        "dataset_metadata": {"model_count": int(len(merged)), "files": [path.name for path in required.values()]},
    }
    if include_expression:
        expression_path = root / DEPMAP_EXPRESSION_FILENAME
        source["files"]["expression"] = str(expression_path.resolve())
        if not expression_path.is_file():
            return _failure(tool, query_info, f"DepMap expression snapshot not found: {expression_path}", source)
        try:
            expression = pd.read_csv(expression_path)
        except Exception as exc:
            return _failure(tool, query_info, f"Could not read DepMap expression snapshot: {exc}", source)
        expression_gene = _depmap_gene_column(expression, gene_symbol)
        expression_id = _find_column(expression, ("ModelID", "DepMap_ID", "Unnamed: 0"))
        if not expression_gene or not expression_id:
            return _failure(
                tool, query_info, f"DepMap expression file does not contain {gene_symbol} and a model ID", source
            )
        expression_rows = expression[[expression_id, expression_gene]].rename(
            columns={expression_id: "model_id", expression_gene: "expression_log2_tpm_plus_1"}
        )
        expression_rows = output[["model_id", "model_name", "lineage"]].merge(
            expression_rows, on="model_id", how="left"
        )
        result["expression_context"] = _records(expression_rows.head(top_n))
        result["gene"]["expression_column"] = expression_gene
        result["dataset_metadata"]["files"].append(expression_path.name)
    return _success(tool, source, query_info, result)


def _validate_paths(
    tool: str, input_path: str, output_dir: str, input_name: str
) -> tuple[Path | None, Path | None, dict | None]:
    query_info = {input_name: input_path, "output_dir": output_dir}
    source = Path(input_path).expanduser()
    if not source.is_file():
        return None, None, _failure(tool, query_info, f"Input file not found: {source}")
    target = Path(output_dir).expanduser()
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return None, None, _failure(tool, query_info, f"Could not create output directory: {exc}")
    return source.resolve(), target.resolve(), None


def _run_command(command: list[str], timeout: int = 900) -> tuple[subprocess.CompletedProcess[str] | None, str | None]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError:
        return None, "Docker executable was not found"
    except subprocess.TimeoutExpired:
        return None, f"Docker command exceeded {timeout} seconds"
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        return completed, f"Container exited with code {completed.returncode}: {detail[:2000]}"
    return completed, None


def _read_tsv(path: Path) -> list[dict]:
    frame = pd.read_csv(path, sep="\t")
    return _records(frame)


def analyze_crispr_screen_with_mageck_impl(
    count_table_path: str,
    treatment_samples: list[str],
    control_samples: list[str],
    output_dir: str,
    normalization_method: str = "median",
    top_n: int = 20,
) -> dict:
    tool = "analyze_crispr_screen_with_mageck"
    query_info = {
        "count_table_path": count_table_path,
        "treatment_samples": treatment_samples,
        "control_samples": control_samples,
        "output_dir": output_dir,
        "normalization_method": normalization_method,
        "top_n": top_n,
    }
    if (
        not treatment_samples
        or not control_samples
        or not all(isinstance(value, str) and value for value in treatment_samples + control_samples)
    ):
        return _failure(
            tool, query_info, "treatment_samples and control_samples must be non-empty lists of sample column names"
        )
    if set(treatment_samples) & set(control_samples):
        return _failure(tool, query_info, "Treatment and control sample lists must not overlap")
    if normalization_method not in {"median", "total", "control", "none"}:
        return _failure(tool, query_info, "normalization_method must be one of: median, total, control, none")
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        return _failure(tool, query_info, "top_n must be a positive integer")
    count_path, output_path, error_result = _validate_paths(tool, count_table_path, output_dir, "count_table_path")
    if error_result:
        return error_result
    try:
        with count_path.open(newline="") as handle:
            header = next(csv.reader(handle, delimiter="\t"))
    except (OSError, StopIteration) as exc:
        return _failure(tool, query_info, f"Could not read count table header: {exc}")
    missing = [name for name in treatment_samples + control_samples if name not in header]
    if missing:
        return _failure(tool, query_info, f"Sample columns not found in count table: {', '.join(missing)}")
    prefix = output_path / "mageck"
    gene_summary = output_path / "mageck.gene_summary.txt"
    sgrna_summary = output_path / "mageck.sgrna_summary.txt"
    if gene_summary.exists() or sgrna_summary.exists():
        return _failure(tool, query_info, "MAGeCK output files already exist in output_dir")
    command = [
        "docker",
        "run",
        "--rm",
        "--mount",
        f"type=bind,source={count_path.parent},target=/input,readonly",
        "--mount",
        f"type=bind,source={output_path},target=/output",
        MAGECK_IMAGE,
        "mageck",
        "test",
        "-k",
        f"/input/{count_path.name}",
        "-t",
        ",".join(treatment_samples),
        "-c",
        ",".join(control_samples),
        "-n",
        "/output/mageck",
        "--norm-method",
        normalization_method,
    ]
    completed, error = _run_command(command)
    source = {"name": "MAGeCK", "container_image": MAGECK_IMAGE}
    if error:
        return _failure(tool, query_info, error, source)
    if not gene_summary.is_file():
        return _failure(tool, query_info, f"MAGeCK did not create expected result file: {gene_summary}", source)
    try:
        frame = pd.read_csv(gene_summary, sep="\t")
    except Exception as exc:
        return _failure(tool, query_info, f"Could not parse MAGeCK gene summary: {exc}", source)
    negative_sort = next((name for name in ("neg|fdr", "neg|p-value", "neg|score") if name in frame.columns), None)
    positive_sort = next((name for name in ("pos|fdr", "pos|p-value", "pos|score") if name in frame.columns), None)
    negative = frame.sort_values(negative_sort).head(top_n) if negative_sort else frame.head(top_n)
    positive = frame.sort_values(positive_sort).head(top_n) if positive_sort else frame.head(top_n)
    version_run, _ = _run_command(["docker", "run", "--rm", MAGECK_IMAGE, "mageck", "--version"], timeout=120)
    version = (version_run.stdout or version_run.stderr).strip() if version_run else None
    return _success(
        tool,
        source,
        query_info,
        {
            "comparison": {
                "treatment_samples": treatment_samples,
                "control_samples": control_samples,
                "normalization_method": normalization_method,
            },
            "gene_results": {"negative_selection": _records(negative), "positive_selection": _records(positive)},
            "output_files": {
                "gene_summary": str(gene_summary),
                "sgrna_summary": str(sgrna_summary) if sgrna_summary.exists() else None,
            },
            "run_metadata": {
                "container_image": MAGECK_IMAGE,
                "mageck_version": version,
                "exit_code": completed.returncode,
                "output_prefix": str(prefix),
            },
        },
    )


def _snake_case(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _parse_fpocket_info(path: Path) -> list[dict]:
    pockets: list[dict] = []
    current: dict | None = None
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        pocket_match = re.match(r"Pocket\s+(\d+)\s*:?", line, flags=re.IGNORECASE)
        if pocket_match:
            current = {"pocket_id": int(pocket_match.group(1))}
            pockets.append(current)
            continue
        if current is None or ":" not in line:
            continue
        key, value = (item.strip() for item in line.split(":", 1))
        try:
            parsed: object = float(value)
        except ValueError:
            parsed = value
        current[_snake_case(key)] = parsed
    score_key = "score"
    return sorted(pockets, key=lambda row: float(row.get(score_key, float("-inf"))), reverse=True)


def detect_protein_pockets_with_fpocket_impl(pdb_file_path: str, output_dir: str, top_n: int = 10) -> dict:
    tool = "detect_protein_pockets_with_fpocket"
    query_info = {"pdb_file_path": pdb_file_path, "output_dir": output_dir, "top_n": top_n}
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        return _failure(tool, query_info, "top_n must be a positive integer")
    pdb_path, output_path, error_result = _validate_paths(tool, pdb_file_path, output_dir, "pdb_file_path")
    if error_result:
        return error_result
    if pdb_path.suffix.casefold() != ".pdb":
        return _failure(tool, query_info, "pdb_file_path must point to a .pdb file")
    container_name = "omniInfra_input.pdb"
    result_dir = output_path / "omniInfra_input_out"
    if result_dir.exists():
        return _failure(tool, query_info, f"fpocket output already exists: {result_dir}")
    command = [
        "docker",
        "run",
        "--rm",
        "--mount",
        f"type=bind,source={output_path},target=/work",
        "--mount",
        f"type=bind,source={pdb_path},target=/work/{container_name},readonly",
        "--workdir",
        "/work",
        FPOCKET_IMAGE,
        "fpocket",
        "-f",
        f"/work/{container_name}",
    ]
    completed, error = _run_command(command)
    source = {"name": "fpocket", "container_image": FPOCKET_IMAGE}
    if error:
        return _failure(tool, query_info, error, source)
    info_file = result_dir / "omniInfra_input_info.txt"
    if not info_file.is_file():
        return _failure(tool, query_info, f"fpocket did not create expected info file: {info_file}", source)
    try:
        pockets = _parse_fpocket_info(info_file)
    except OSError as exc:
        return _failure(tool, query_info, f"Could not parse fpocket result: {exc}", source)
    version_text = completed.stderr or completed.stdout
    version_match = re.search(r"fpocket\s+([0-9.]+)", version_text, flags=re.IGNORECASE)
    return _success(
        tool,
        source,
        query_info,
        {
            "structure": {"input_file": str(pdb_path)},
            "pockets": pockets[:top_n],
            "output_files": {"result_directory": str(result_dir), "info_file": str(info_file)},
            "run_metadata": {
                "container_image": FPOCKET_IMAGE,
                "fpocket_version": version_match.group(1) if version_match else "4.0",
                "exit_code": completed.returncode,
            },
        },
    )


def _request_json(response: requests.Response, action: str) -> tuple[dict | None, str | None]:
    if not response.ok:
        return None, f"{action} returned HTTP {response.status_code}: {response.text[:500]}"
    try:
        return response.json(), None
    except ValueError:
        return None, f"{action} returned a non-JSON response"


def _download_result(url: str, output_dir: Path, label: str) -> tuple[Path | None, str | None]:
    try:
        response = requests.get(url, timeout=120)
    except requests.RequestException as exc:
        return None, f"Could not download {label}: {exc}"
    if not response.ok:
        return None, f"Could not download {label}: HTTP {response.status_code}"
    name = Path(urlparse(url).path).name or label
    path = output_dir / name
    if path.exists():
        return None, f"Output file already exists: {path}"
    try:
        path.write_bytes(response.content)
    except OSError as exc:
        return None, f"Could not write {label}: {exc}"
    return path, None


def _parse_dogsite_table(path: Path) -> list[dict]:
    frame = pd.read_csv(path, sep="\t")
    if len(frame.columns) == 1:
        frame = pd.read_csv(path, sep=r"\s+", engine="python")
    records = _records(frame)
    drug_key = next((key for key in frame.columns if _snake_case(str(key)) in {"drug_score", "drugscore"}), None)
    if drug_key:
        records.sort(key=lambda row: float(row.get(str(drug_key)) or float("-inf")), reverse=True)
    return records


def score_protein_pockets_with_dogsite_impl(
    pdb_file_path: str,
    output_dir: str,
    chain_id: str | None = None,
    include_subpockets: bool = False,
    top_n: int = 10,
) -> dict:
    tool = "score_protein_pockets_with_dogsite"
    query_info = {
        "pdb_file_path": pdb_file_path,
        "output_dir": output_dir,
        "chain_id": chain_id,
        "include_subpockets": include_subpockets,
        "top_n": top_n,
    }
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        return _failure(tool, query_info, "top_n must be a positive integer")
    pdb_path, output_path, error_result = _validate_paths(tool, pdb_file_path, output_dir, "pdb_file_path")
    if error_result:
        return error_result
    if pdb_path.suffix.casefold() != ".pdb":
        return _failure(tool, query_info, "pdb_file_path must point to a .pdb file")
    source = {
        "name": "ProteinsPlus DoGSiteScorer",
        "upload_endpoint": PROTEINSPLUS_UPLOAD_URL,
        "job_endpoint": PROTEINSPLUS_DOGSITE_URL,
    }
    try:
        with pdb_path.open("rb") as handle:
            upload_response = requests.post(
                PROTEINSPLUS_UPLOAD_URL,
                files={"pdb_file[pathvar]": (pdb_path.name, handle, "chemical/x-pdb")},
                headers={"Accept": "application/json"},
                timeout=120,
            )
    except requests.RequestException as exc:
        return _failure(tool, query_info, f"ProteinsPlus PDB upload failed: {exc}", source)
    upload, error = _request_json(upload_response, "ProteinsPlus PDB upload")
    if error:
        return _failure(tool, query_info, error, source)
    uploaded_id = upload.get("id")
    upload_location = upload.get("location")
    upload_deadline = time.monotonic() + 120
    while not uploaded_id and upload_location and time.monotonic() < upload_deadline:
        try:
            upload_status_response = requests.get(upload_location, headers={"Accept": "application/json"}, timeout=60)
        except requests.RequestException as exc:
            return _failure(tool, query_info, f"ProteinsPlus PDB loading status failed: {exc}", source)
        upload_status, error = _request_json(upload_status_response, "ProteinsPlus PDB loading status")
        if error:
            return _failure(tool, query_info, error, source)
        uploaded_id = upload_status.get("id")
        if not uploaded_id:
            time.sleep(2)
    if not uploaded_id:
        return _failure(tool, query_info, "ProteinsPlus upload response did not contain a structure ID", source)
    job_request = {
        "dogsite": {
            "pdbCode": str(uploaded_id),
            "analysisDetail": "1" if include_subpockets else "0",
            "bindingSitePredictionGranularity": "1",
            "ligand": "",
            "chain": chain_id or "",
        }
    }
    try:
        job_response = requests.post(
            PROTEINSPLUS_DOGSITE_URL,
            json=job_request,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            timeout=120,
        )
    except requests.RequestException as exc:
        return _failure(tool, query_info, f"DoGSiteScorer job submission failed: {exc}", source)
    job, error = _request_json(job_response, "DoGSiteScorer job submission")
    if error:
        return _failure(tool, query_info, error, source)
    location = job.get("location")
    if not location:
        return _failure(tool, query_info, "DoGSiteScorer response did not contain a job location", source)
    job_id = Path(urlparse(location).path).name
    deadline = time.monotonic() + 600
    final: dict | None = None
    while time.monotonic() < deadline:
        try:
            poll_response = requests.get(location, headers={"Accept": "application/json"}, timeout=60)
        except requests.RequestException as exc:
            return _failure(tool, query_info, f"DoGSiteScorer status request failed: {exc}", source)
        payload, error = _request_json(poll_response, "DoGSiteScorer status request")
        if error:
            return _failure(tool, query_info, error, source)
        if payload.get("status_code") == 200 and payload.get("result_table"):
            final = payload
            break
        if payload.get("status_code") != 202:
            return _failure(tool, query_info, f"DoGSiteScorer job failed: {payload.get('message', payload)}", source)
        time.sleep(5)
    if final is None:
        return _failure(tool, query_info, "DoGSiteScorer job did not finish within 600 seconds", source)
    output_files: dict[str, object] = {}
    table_path, error = _download_result(final["result_table"], output_path, "result_table")
    if error:
        return _failure(tool, query_info, error, source)
    output_files["result_table"] = str(table_path)
    for key in ("descriptor_explanation", "parameters"):
        if isinstance(final.get(key), str) and str(final[key]).startswith("http"):
            downloaded, error = _download_result(final[key], output_path, key)
            if error:
                return _failure(tool, query_info, error, source)
            output_files[key] = str(downloaded)
    for key in ("residues", "pockets"):
        paths: list[str] = []
        for index, url in enumerate(final.get(key, []), start=1):
            downloaded, error = _download_result(url, output_path, f"{key}_{index}")
            if error:
                return _failure(tool, query_info, error, source)
            paths.append(str(downloaded))
        output_files[key] = paths
    try:
        pockets = _parse_dogsite_table(table_path)
    except Exception as exc:
        return _failure(tool, query_info, f"Could not parse DoGSiteScorer result table: {exc}", source)
    return _success(
        tool,
        source,
        query_info,
        {
            "structure": {"input_file": str(pdb_path), "uploaded_structure_id": uploaded_id},
            "job": {"job_id": job_id, "status": "completed"},
            "pockets": pockets[:top_n],
            "output_files": output_files,
            "service_metadata": {
                "service": "ProteinsPlus DoGSiteScorer",
                "endpoint": PROTEINSPLUS_DOGSITE_URL,
                "request_parameters": job_request["dogsite"],
                "returned_parameters": final.get("parameters"),
            },
        },
    )
