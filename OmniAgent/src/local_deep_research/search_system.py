import asyncio
import json
import logging
import os
import re
import textwrap
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path
from .config import (
    settings,
    get_claude_openai,
    get_deepseek_r1,
    get_deepseek_v3,
    get_gpt4_1,
    get_gpt4_1_mini,
    get_qwen_siliconflow,
)

from .connect_mcp import OmniagentMCPToolClient, mcp_servers

# Import new data structures and parsers
from .data_structures import ResearchContext
from .tool_parsers import parse_tool_result
from .evidence_parser import ToolParseAgent, BibliographyRegistry, BibliographyEntry
from .trace_logger import ResearchTraceLogger

# Import utilities from the new support module
from .search_system_support import (
    compress_all_llm,
    extract_and_convert_list,
    parse_single,
    safe_json_from_text,
    SourcesReference,
)
from .tool_executor import ToolExecutor
from .tool_selector import ToolSelector
from .tools.template.templateagent import (
    retrieve_small_template,
    retrieve_large_template,
)
from .utilties.search_utilities import (
    invoke_with_timeout_and_retry,
    remove_think_tags,
    write_log_process_safe,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  ##  DEBUG


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


log_dir = os.path.join(ROOT_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)

log_file_path = os.path.join(
    log_dir, f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
print(f"log save in {log_file_path}")


def add_hard_breaks_to_references(response_content: str) -> str:
    """ """
    pattern = r"(##\s*References\s*\n)(.*?)(\n##\s|\Z)"  #
    match = re.search(pattern, response_content, re.DOTALL | re.IGNORECASE)

    if not match:
        return response_content  #

    header = match.group(1)
    refs_block = match.group(2)
    tail = match.group(3)

    updated_refs = []
    for line in refs_block.splitlines():
        if re.match(r"\s*\[\^\^\d+\]", line):  #
            updated_line = line.rstrip() + "\n\n"  #
        else:
            updated_line = line
        updated_refs.append(updated_line)

    new_refs_block = "\n".join(updated_refs)
    new_content = (
        response_content[: match.start()]
        + header
        + new_refs_block
        + tail
        + response_content[match.end() :]
    )

    return new_content


class ReferencePool:
    """
    the reference pool for the search system
    - only receive the new literature explicitly declared by LLM (extracted_papers or NewRef)
    - do not make academic judgment on the URL anymore
    - the pool is only added, not deleted, and the number is stable
    """

    def __init__(self) -> None:
        self.pool: List[SourcesReference] = []
        self.link2idx: dict[str, int] = {}
        self.dirty: bool = False  # mark if the right column needs to be refreshed

    def add(self, title: str, citation: str, link: str) -> int:
        if not link:
            return -1
        if link in self.link2idx:  # already exists
            return self.link2idx[link]

        idx = len(self.pool) + 1  # 1-based index
        self.link2idx[link] = idx
        self.pool.append(
            SourcesReference(title=title or link, subtitle=citation or "", link=link)
        )
        self.dirty = True
        return idx


logger = logging.getLogger(__name__)


class AdvancedSearchSystem:
    def __init__(
        self,
        verbose=True,
        mid_path="./middleresult.txt",
        report_path="./report.txt",
        max_iterations=2,
        questions_per_iteration=2,
        is_report=False,
        chosen_tools: list[str] = None,  ## None for using all tools
        error_log_path: str = "",
        using_model="gpt4_1",
        session_log_dir: str = None,  # NEW: Session log directory
        validation_batch_size: int = 12,
        candidate_pool_max: int = 250,
        enable_llm_prescreen: bool = True,
    ):
        self._action_blocks: dict[int, dict] = {}
        self.ref_pool = ReferencePool()
        self.all_links_of_system: list[str] = []
        self.chosen_tools = chosen_tools
        self.is_report = is_report
        self.validation_batch_size = max(0, int(validation_batch_size))
        self.candidate_pool_max = max(50, int(candidate_pool_max))
        self.enable_llm_prescreen = bool(enable_llm_prescreen)
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.questions_per_iteration = questions_per_iteration
        self.block_callback = None
        self.mid_path = mid_path
        self.report_path = report_path
        self.knowledge_chunks = []
        self.session_log_dir = session_log_dir

        # Setup logging directory first
        log_dir = os.path.join(ROOT_DIR, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        if error_log_path == "":
            error_log_path = os.path.join(
                log_dir, f"error_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
        self.error_log_path = error_log_path

        # Initialize new components
        self.research_context = ResearchContext(current_knowledge="", iteration_count=0)
        self.bibliography = BibliographyRegistry(
            Path(
                os.path.join(
                    log_dir,
                    f"bibliography_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                )
            )
        )

        self.all_links_of_system = []
        self.questions_by_iteration = {}

        ### Model independent, you can use any model you like
        self.using_model = using_model
        if self.using_model == "claude":
            self.model = get_claude_openai()
            self.reasoning_model = get_claude_openai()
            self.tool_planning_model = get_deepseek_v3()
            self.fast_model = get_deepseek_v3()
            self.report_model = get_gpt4_1()

        if self.using_model == "deepseek":
            self.model = get_deepseek_r1()
            self.reasoning_model = get_deepseek_r1()
            self.tool_planning_model = get_deepseek_v3()
            self.fast_model = get_deepseek_v3()
            self.report_model = get_gpt4_1()

        if self.using_model == "gpt4_1":
            self.model = get_gpt4_1()
            self.reasoning_model = get_gpt4_1()
            self.tool_planning_model = get_gpt4_1()
            self.fast_model = get_gpt4_1_mini()
            self.report_model = get_gpt4_1()

        if self.using_model == "gpt4_1_mini":
            self.model = get_gpt4_1_mini()
            self.reasoning_model = get_gpt4_1_mini()
            self.tool_planning_model = get_gpt4_1_mini()
            self.fast_model = get_gpt4_1_mini()
            self.tool_query_model = get_gpt4_1_mini()
            self.report_model = get_gpt4_1()

        if self.using_model == "qwen":
            self.model = get_qwen_siliconflow()
            self.reasoning_model = get_qwen_siliconflow()
            self.tool_planning_model = get_qwen_siliconflow()
            self.fast_model = get_qwen_siliconflow()
            self.tool_query_model = get_qwen_siliconflow()
            self.report_model = get_qwen_siliconflow()

        # Initialize ToolParseAgent (using fast model)
        self.tool_parse_agent = ToolParseAgent(self.fast_model)

        # Report-only: candidate pool cache (HGNC-like symbols) to support long-tail target discovery
        self._candidate_symbols: set[str] = set()
        self._validated_symbols: set[str] = set()
        # Report-only: "pinned" symbols that must not be evicted by candidate_pool_max truncation.
        # This prevents long-tail candidates from disappearing after they were selected/validated once.
        self._pinned_symbols: set[str] = set()
        # Report-only: simple stats to rank candidates (avoid alphabetical bias)
        self._candidate_stats: dict[str, dict] = {}
        # Report-only: avoid spamming pool profiling every iteration
        self._last_pool_profile_iteration: int = -1

    def _frequent_symbol_prefix_hints(
        self, symbols: list[str], *, min_count: int = 4, max_hints: int = 8
    ) -> list[str]:
        """
        Report-only helper: infer recurring symbol-prefix pattern hints from the current pool itself.
        This avoids hard-coding any domain categories. Hints are used to guide LLM expansion toward
        "items similar to what already appears" (by naming patterns) without leaking specific targets.
        """
        if not symbols:
            return []
        counts: dict[str, int] = {}
        # Use short alphabetic prefixes (3-5 chars) as weak naming-pattern signals (e.g., ABC, CXCR, etc.).
        for s in symbols:
            if not isinstance(s, str):
                continue
            t = s.strip().upper()
            if not t:
                continue
            for k in (3, 4, 5):
                if len(t) >= k and t[:k].isalpha():
                    p = t[:k]
                    counts[p] = counts.get(p, 0) + 1

        # Prefer longer prefixes when they occur (more specific), but still require frequency.
        hints = [p for p, c in counts.items() if c >= int(min_count)]
        hints.sort(key=lambda p: (-len(p), -counts.get(p, 0), p))

        # De-dup: if a longer prefix is present, drop its shorter prefix to avoid redundancy.
        out: list[str] = []
        for p in hints:
            if any(
                p.startswith(existing) or existing.startswith(p) for existing in out
            ):
                # Keep only the more specific one (longer)
                if any(existing.startswith(p) for existing in out):
                    continue
            out.append(p)
            if len(out) >= int(max_hints):
                break
        return out

    async def _profile_candidate_pool_llm(self, query: str, symbols: list[str]) -> dict:

        if not symbols:
            return {}

        # Bound prompt size: provide a representative prefix (pool_list is already ranked)
        seed = symbols[: min(len(symbols), 180)]

        prompt = (
            "You are analyzing a candidate pool (HGNC gene symbols) for target discovery.\n"
            "Goal: summarize the pool composition and propose gap-filling exploration queries.\n"
            "Rules:\n"
            "1) Output MUST be valid JSON (an object).\n"
            "2) Do NOT mention any specific gene symbols in the output.\n"
            "3) Keep it conservative and intent-aligned with the user query.\n"
            "4) Proposed queries must be self-contained and executable as search/database queries.\n"
            "5) At least 2 proposed queries MUST explicitly request a machine-extractable list/table of gene symbols.\n"
            "6) Do NOT use subjective words like 'novel/emerging/promising' as the primary search anchor.\n"
            "JSON schema:\n"
            "{\n"
            '  "bucket_counts": {"surface_receptor":0,"surface_enzyme":0,"adhesion_ecm":0,"transporter":0,"secreted":0,'
            '"immune_lineage":0,"intracellular_signaling":0,"nuclear_tf":0,"cell_cycle":0,"metabolic":0,"other_unknown":0},\n'
            '  "bias_notes": ["..."],\n'
            '  "gap_fill_queries": ["...", "...", "..."]\n'
            "}\n"
            f"User query: {query}\n"
            f"Candidate symbols (subset, ranked): {seed}\n"
        )

        try:
            resp = await invoke_with_timeout_and_retry(
                self.fast_model,
                prompt,
                timeout=60.0,
                max_retries=2,
                retry_delay=10.0,
            )
            data = safe_json_from_text(resp.content)
            if isinstance(data, dict):
                return data
        except Exception as e:
            logger.warning(f"Report-only candidate pool profiling failed: {e}")

        return {}

    def _hard_filter_candidate_symbols(self, symbols: list[str]) -> list[str]:
        """
        Report-mode normalization filter for candidate gene symbols.

        The goal is to remove clearly ambiguous or incomplete tokens before downstream
        validation, while keeping the rule set intentionally small to preserve recall.
        """
        if not symbols:
            return []

        # Ambiguous aliases or family-level shorthands that are often not stable HGNC symbols.
        blacklist = {
            "ER",  # ambiguous; often means ESR1/ESR2 but not HGNC itself
            "CYP",  # family prefix, not a symbol
            "AKT",  # family, not a symbol (AKT1/2/3)
            "JNK",  # family, not a symbol
            "JNK1",  # common alias for MAPK8 (not HGNC)
            "CRAF",  # common alias for RAF1 (not HGNC)
            "53BP1",  # common alias for TP53BP1 (not HGNC)
        }

        # Prefixes that are typically incomplete unless followed by a numeric suffix.
        incomplete_prefixes = ("AKR1C",)

        out = []
        for s in symbols:
            if not isinstance(s, str):
                continue
            ss = s.strip().upper()
            if not ss:
                continue
            # Remove incomplete prefix-only forms (keep explicit numbered symbols).
            if ss.startswith(incomplete_prefixes) and not ss[-1].isdigit():
                continue
            # Drop short ambiguous tokens; keep only explicit, high-confidence exceptions.
            if len(ss) <= 2 and ss not in {"AR"}:
                continue
            out.append(ss)

        return sorted(set(out))

    def _update_candidate_stats(
        self, symbol: str, tool_name: str, iteration: int
    ) -> None:
        s = symbol.upper()
        st = self._candidate_stats.get(s)
        if not st:
            st = {"count": 0, "last_seen": -1, "sources": set()}
            self._candidate_stats[s] = st
        st["count"] += 1
        st["last_seen"] = max(int(st.get("last_seen", -1)), int(iteration))
        try:
            st["sources"].add(tool_name)
        except Exception:
            pass

    def _get_ranked_candidate_pool(self) -> list[str]:
        """
        Deterministic, non-alphabetical ordering to avoid systematic bias.
        IMPORTANT: keep this generic (no family-prefix heuristics) to avoid leaking specific target families.
        """
        pool = list(self._candidate_symbols)
        pool = self._hard_filter_candidate_symbols(pool)

        def key_fn(s: str):
            st = self._candidate_stats.get(s, {})
            return (
                -int(st.get("last_seen", -1)),
                -int(st.get("count", 0)),
                -len(st.get("sources", []))
                if isinstance(st.get("sources", None), set)
                else 0,
                s,
            )

        pool.sort(key=key_fn)
        # Report-only: keep pinned symbols within the truncated pool so they remain visible in later iterations/logs.
        if self.is_report and getattr(self, "_pinned_symbols", None):
            pinned = [s for s in pool if s in self._pinned_symbols]
            rest = [s for s in pool if s not in self._pinned_symbols]
            pool = pinned + rest

        return pool[: self.candidate_pool_max]

    def _extract_candidate_symbols_regex(self, texts: list[str]) -> list[str]:
        """
        Broad, conservative-ish regex harvest of HGNC-like symbols from raw tool text.
        This is intentionally permissive; we'll optionally clean with LLM afterward (report-mode only).
        """
        if not texts:
            return []

        # Very common non-gene tokens to drop early (LLM will further clean).
        stop = {
            "DNA",
            "RNA",
            "ATP",
            "GTP",
            "ADMET",
            "DMSO",
            "COVID",
            "SARS",
            "GO",
            "KEGG",
            "PDB",
            "FDA",
            "IC50",
            "EC50",
            "KD",
            "KI",
            "MHC",
            "TCR",
        }

        # HGNC symbols are often 2-10 chars, uppercase letters/digits.
        pattern = re.compile(r"\b[A-Z0-9]{2,10}\b")
        found: set[str] = set()
        for t in texts:
            if not t:
                continue
            for m in pattern.findall(t):
                if m in stop:
                    continue
                # Avoid pure numbers
                if m.isdigit():
                    continue
                found.add(m)
        # Keep deterministic order
        return sorted(found)

    async def _clean_candidate_symbols_llm(self, symbols: list[str]) -> list[str]:
        """
        LLM-based cleaning pass: keep only plausible HGNC gene symbols.
        This does NOT try to discover new symbols; it only filters/normalizes.
        """
        if not symbols:
            return []

        # Hard cap to control prompt size
        symbols = symbols[: max(200, min(len(symbols), self.candidate_pool_max))]

        prompt = (
            "You are cleaning a noisy list of tokens extracted from biomedical text.\n"
            "Task: return ONLY plausible human HGNC gene symbols.\n"
            "Rules:\n"
            "1) Output MUST be a JSON array of strings.\n"
            "2) Keep tokens that look like real gene symbols (e.g., FGFR4, ERBB2, SLC7A11).\n"
            "3) Remove generic abbreviations (e.g., DNA, RNA, ATP, GO, KEGG, IC50), units, and non-gene words.\n"
            "4) Do NOT invent new symbols. Only filter from the provided list.\n"
            "5) Keep uppercase letters/digits; if unsure, keep it.\n"
            f"Input tokens ({len(symbols)}): {symbols}\n"
        )

        try:
            resp = await invoke_with_timeout_and_retry(
                self.fast_model,
                prompt,
                timeout=60.0,
                max_retries=2,
                retry_delay=10.0,
            )
            cleaned = extract_and_convert_list(resp.content)
            if isinstance(cleaned, list):
                # Normalize
                out = []
                for s in cleaned:
                    if not isinstance(s, str):
                        continue
                    ss = s.strip().upper()
                    if ss:
                        out.append(ss)
                # Dedup & cap
                out = sorted(set(out))[: self.candidate_pool_max]
                return out
        except Exception as e:
            logger.warning(f"LLM cleaning for candidate symbols failed: {e}")

        # Fallback: regex-only list
        return sorted(set(symbols))[: self.candidate_pool_max]

    async def _prescreen_symbols_for_validation_llm(
        self, query: str, symbols: list[str]
    ) -> dict:
        """
        Report-only: LLM prescreen to prioritize which symbols to validate next.
        This is allowed to be more assertive in deprioritizing candidates for validation, while remaining
        conservative about permanently dropping symbols from the global pool.
        """
        if not symbols:
            return {"selected": [], "deprioritized": [], "drop": []}

        # Keep prompt small and stable
        symbols = symbols[: self.candidate_pool_max]
        limit = min(self.validation_batch_size, len(symbols))
        if limit <= 0:
            return {"selected": [], "deprioritized": [], "drop": []}

        prompt = (
            "You are prioritizing which candidate gene symbols to validate next with structured target tools.\n"
            "You must follow the user's constraints from the query.\n"
            "Task:\n"
            f"- Select up to {limit} symbols that are MOST worth validating next.\n"
            "- Identify symbols that should be deprioritized for validation in this iteration because they are unlikely to match the user's constraints.\n"
            "- You MAY propose a drop list ONLY for (a) obvious non-gene artifacts, OR (b) symbols that do not match the user's query.\n"
            "Rules:\n"
            "1) Output MUST be JSON with keys: selected (array of strings), deprioritized (array of strings), drop (array of strings).\n"
            "2) Do NOT invent new symbols.\n"
            "3) If unsure, keep the symbol (do not put it into drop).\n"
            f"User query: {query}\n"
            f"Symbols: {symbols}\n"
        )

        try:
            resp = await invoke_with_timeout_and_retry(
                self.fast_model,
                prompt,
                timeout=60.0,
                max_retries=2,
                retry_delay=10.0,
            )
            data = safe_json_from_text(resp.content)
            if isinstance(data, dict):
                sel = data.get("selected", [])
                dep = data.get("deprioritized", data.get("excluded", []))
                drp = data.get("drop", [])
                if isinstance(sel, list) and isinstance(dep, list):
                    sel2 = []
                    for s in sel:
                        if isinstance(s, str) and s.strip():
                            sel2.append(s.strip().upper())
                    dep2 = []
                    for s in dep:
                        if isinstance(s, str) and s.strip():
                            dep2.append(s.strip().upper())
                    drp2 = []
                    if isinstance(drp, list):
                        for s in drp:
                            if isinstance(s, str) and s.strip():
                                drp2.append(s.strip().upper())
                    # Dedup + cap
                    sel2 = [s for s in sel2 if s in set(symbols)]
                    sel2 = list(dict.fromkeys(sel2))[:limit]
                    dep2 = [s for s in dep2 if s in set(symbols) and s not in set(sel2)]
                    dep2 = list(dict.fromkeys(dep2))
                    # drop must be a subset of deprioritized/excluded in spirit; enforce conservatively
                    # Also: only allow dropping symbols explicitly mentioned in the user query (case-insensitive),
                    # unless they are obvious non-gene artifacts (handled by hard filtering elsewhere).
                    q_up = (query or "").upper()
                    dep2_set = set(dep2)
                    # If the model proposes drop items, treat them as deprioritized too.
                    for s in drp2:
                        if s in set(symbols) and s not in set(sel2):
                            dep2_set.add(s)
                    dep2 = list(
                        dict.fromkeys(
                            [s for s in dep2 if s in dep2_set]
                            + [s for s in drp2 if s in dep2_set]
                        )
                    )
                    drp2 = [s for s in drp2 if s in set(symbols) and (s in q_up)]
                    drp2 = list(dict.fromkeys(drp2))
                    return {"selected": sel2, "deprioritized": dep2, "drop": drp2}
        except Exception as e:
            logger.warning(f"LLM prescreen for validation failed: {e}")

        # Fallback: just take the first N symbols (deterministic)
        return {"selected": symbols[:limit], "deprioritized": [], "drop": []}

    async def _propose_additional_candidates_llm(
        self, query: str, symbols: list[str], k: int
    ) -> list[str]:
        """
        Report-only: small "divergence" step.
        Propose additional plausible HGNC-like gene symbols that may be missing from the pool,
        inspired by the current pool and the user intent. These are hypotheses and must be validated later.
        Constraints: keep generic, no target leakage, no hard-coded examples.
        """
        if k <= 0:
            return []
        if not symbols:
            return []

        # Keep prompt size bounded; provide a representative subset
        seed = symbols[: max(60, min(len(symbols), 120))]
        prefix_hints = self._frequent_symbol_prefix_hints(
            symbols, min_count=4, max_hints=8
        )

        prompt = (
            "You are helping expand a candidate pool for target discovery.\n"
            "Task: propose up to N additional plausible human HGNC gene symbols that might be missing.\n"
            "Constraints:\n"
            "- Do NOT repeat any symbol already in the pool.\n"
            "- Do NOT invent obviously invalid tokens; propose only symbols you believe are real gene symbols.\n"
            "- Keep proposals aligned with the user's constraints in the query.\n"
            "- Use the pool itself to infer recurring naming/symbol patterns; propose a few plausible missing symbols following similar patterns if they fit the user intent.\n"
            "- Output MUST be a JSON array of strings.\n"
            f"N={int(k)}\n"
            f"User query: {query}\n"
            f"Current pool (subset): {seed}\n"
            f"Prefix-pattern hints inferred from the pool (optional guidance): {prefix_hints}\n"
        )

        try:
            resp = await invoke_with_timeout_and_retry(
                self.fast_model,
                prompt,
                timeout=60.0,
                max_retries=2,
                retry_delay=10.0,
            )
            proposed = extract_and_convert_list(resp.content)
            if isinstance(proposed, list):
                out: list[str] = []
                existing = set(s.upper() for s in symbols if isinstance(s, str))
                for s in proposed:
                    if not isinstance(s, str):
                        continue
                    ss = s.strip().upper()
                    if not ss:
                        continue
                    if ss in existing:
                        continue
                    # Must look HGNC-like
                    if not re.fullmatch(r"[A-Z0-9]{2,10}", ss):
                        continue
                    out.append(ss)
                out = self._hard_filter_candidate_symbols(out)
                return out[: int(k)]
        except Exception as e:
            logger.warning(f"LLM candidate expansion failed: {e}")

        return []

    def _mcp_payload_to_json(self, payload):
        """
        MCP tools often return: [{'type':'text','text':'{...json...}', ...}, ...]
        Return parsed JSON when possible; otherwise return raw payload.
        """
        try:
            if isinstance(payload, list) and payload:
                first = payload[0]
                if isinstance(first, dict) and isinstance(first.get("text"), str):
                    txt = first["text"]
                    return json.loads(txt)
        except Exception:
            pass
        return payload

    async def _validate_candidate_symbols(self, symbols: list[str]) -> list[dict]:
        """
        Lightweight validation for a small batch of symbols (report-only):
        - GO CC terms (membrane/extracellular)
        - target classes
        - associated diseases (top few)
        """
        if not symbols:
            return []

        # Limit to configured batch size (caller may already prescreen/order)
        batch = symbols[: self.validation_batch_size]
        sem = asyncio.Semaphore(5)

        async def one(sym: str) -> dict:
            async with sem:
                out = {"symbol": sym, "go_cc": [], "classes": [], "diseases": []}
                try:
                    go_raw = await self.mcp_tool_client.call_tool(
                        "get_target_gene_ontology_by_name", {"target_name": sym}
                    )
                    go = self._mcp_payload_to_json(go_raw)
                    # OpenTargets shape
                    terms = (
                        go.get("data", {}).get("target", {}).get("geneOntology", [])
                        if isinstance(go, dict)
                        else []
                    )
                    out["go_cc"] = [
                        t.get("term", {}).get("name")
                        for t in terms
                        if isinstance(t, dict) and t.get("aspect") == "C"
                    ][:8]
                except Exception:
                    pass
                try:
                    cls_raw = await self.mcp_tool_client.call_tool(
                        "get_target_classes_by_name", {"target_name": sym}
                    )
                    cls = self._mcp_payload_to_json(cls_raw)
                    rows = (
                        cls.get("data", {}).get("target", {}).get("targetClass", [])
                        if isinstance(cls, dict)
                        else []
                    )
                    out["classes"] = [
                        r.get("label")
                        for r in rows
                        if isinstance(r, dict) and r.get("label")
                    ][:6]
                except Exception:
                    pass
                try:
                    dis_raw = await self.mcp_tool_client.call_tool(
                        "get_associated_diseases_phenotypes_by_target_name",
                        {"target_name": sym},
                    )
                    dis = self._mcp_payload_to_json(dis_raw)
                    rows = (
                        dis.get("data", {})
                        .get("target", {})
                        .get("associatedDiseases", {})
                        .get("rows", [])
                        if isinstance(dis, dict)
                        else []
                    )
                    out["diseases"] = [
                        r.get("disease", {}).get("name")
                        for r in rows
                        if isinstance(r, dict) and r.get("disease")
                    ][:5]
                except Exception:
                    pass
                return out

        results = await asyncio.gather(*[one(s) for s in batch])
        return [r for r in results if r]

    async def _get_follow_up_questions(
        self, current_knowledge: str, query: str
    ) -> List[str]:
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d")

        logger.info("Questions by iteration: %s", str(self.questions_by_iteration))

        chance = 3
        current_try = 0
        while current_try < chance:
            # Report-mode only: add generic, non-leaky constraints to make sub-queries executable
            # and list-producing (helps preserve long-tail candidates without naming any target/family).
            report_mode_constraints = ""
            if self.is_report:
                report_mode_constraints = """

                ## Report-mode constraints (generic; do not add special-case keywords or examples)
                - Every sub_query must be self-contained and executable without referring to previous sub-queries or their outputs.
                - Do NOT use subjective evaluation words (e.g., "novel", "emerging", "promising") as the primary search anchor.
                - At least 2 sub-queries MUST explicitly request a machine-extractable list/table of concrete candidate identifiers
                  (i.e., extractable gene-symbol lists/signatures/tables suitable for downstream validation), rather than narrative-only answers.
                """
            if self.questions_by_iteration:
                prompt = f"""
                # Biomedical Research Intelligent Search Assistant
                
                ## Context
                - Main research question: "{query}"
                - Current date: {current_time}
                - Iteration: {len(self.questions_by_iteration)} completed rounds
                - Past sub-questions: {str(self.questions_by_iteration)}
                - Current accumulated knowledge: {current_knowledge}
                               
                ## Your Task
                1. Analyze the main query to identify its core components, key terms, and primary questions.
                2. Break down the main query into no more than {self.questions_per_iteration} sub-queries that collectively address the main query.
                
                ## Guidelines
                1. Identify key information gaps in current knowledge
                2. Formulate each sub-query as a clear, open-ended question that avoids bias, assumptions, or leading language (e.g., use neutral phrasing instead of affirmative or suggestive statements).
                3. Include only sub-queries that are essential and directly relevant to addressing the main query.
                4. Break down complex concepts or relationships into distinct, independently answerable questions.
                5. Keep sub-queries concise, specific, and straightforward (e.g., inquire about definitions, roles, or relationships, such as "What is X?" or "How does X relate to Y?").
                6. Avoid repeating past searches
                
                ## Output Format
                You must provide your response in the following structured JSON format:
                {{
                    "thoughts": "Brief analysis of the problem and what needs investigation",
                    "strategy": ["1", "2", "3", "4", "5"],
                    "sub_queries": ["Search_Query_1", "Search_Query_2", "Search_Query_3", "Search_Query_4", "Search_Query_5"]
                }}
                
                ## Important Notes
                - Use terminology CONSISTENT with the main query, preserving specialized terms without arbitrary modifications.
                - Each query should target different aspects of the research question
                - Queries should be specific and actionable
                {report_mode_constraints}
                """
            else:
                prompt = f"""
                # Intelligent Search Assistant
                
                ## Context
                - Main query to answer: "{query}"
                - Current date: {current_time}
                - This is the first search iteration
                
                 ## Your Task
                1. Analyze the main query to identify its core components, key terms, and primary questions.
                2. Break down the main query into no more than {self.questions_per_iteration} sub-queries that collectively address the main query.
                
                ## Guidelines
                1. Identify key information gaps in current knowledge
                2. Formulate each sub-query as a clear, open-ended question that avoids bias, assumptions, or leading language (e.g., use neutral phrasing instead of affirmative or suggestive statements).
                3. Include only sub-queries that are essential and directly relevant to addressing the main query.
                4. Break down complex concepts or relationships into distinct, independently answerable questions.
                5. Keep sub-queries concise, specific, and straightforward (e.g., inquire about definitions, roles, or relationships, such as "What is X?" or "How does X relate to Y?").
                6. Avoid repeating past searches
                
                ## Output Format
                You must provide your response in the following structured JSON format:
                {{
                    "thoughts": "Brief analysis of the problem and what needs investigation",
                    "strategy": ["1", "2", "3", "4", "5"],
                    "sub_queries": ["Search_Query_1", "Search_Query_2", "Search_Query_3", "Search_Query_4", "Search_Query_5"]
                }}
                
                ## Important Notes
                - Use terminology CONSISTENT with the main query, preserving specialized terms without arbitrary modifications.
                - Each query should target different aspects of the research question
                - Queries should be specific and actionable
                {report_mode_constraints}
                """

            try:
                response = await invoke_with_timeout_and_retry(
                    self.reasoning_model,
                    prompt,
                    timeout=90.0,
                    max_retries=3,
                    retry_delay=30.0,
                )
            except Exception as e:
                logger.warning(f"Failed to generate follow-up questions: {e}")
                # Write error info to local log file in a process-safe way
                log_msg = f"[{datetime.now().isoformat()}] Failed to generate follow-up questions: {e}\n"
                write_log_process_safe(self.error_log_path, log_msg)
                current_try += 1
                retry_msg = f"⚠️  Retry {current_try}/{chance}: Failed to generate follow-up questions, retrying..."
                logger.warning(retry_msg)
                log_msg_retry = f"[{datetime.now().isoformat()}] {retry_msg}\n"
                write_log_process_safe(self.error_log_path, log_msg_retry)
                await asyncio.sleep(5)
                continue

            # try to parse the response in JSON format
            try:
                response_text = response.content

                logger.info(
                    f"important info stream in for process 0: break query in subquery"
                )
                logger.info(f"important info stream in for process 0: prompt: {prompt}")
                logger.info(
                    f"important info stream in for process 0: planning Agent {response_text}"
                )

                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    parsed_response = safe_json_from_text(response_text)
                    if not parsed_response:
                        raise ValueError("No valid JSON found")

                    # extract the tool queries
                    questions = parsed_response.get("sub_queries", [])

                    if questions and len(questions) > 0:
                        questions = questions[: self.questions_per_iteration]
                        logger.info(
                            f"✅ Successfully generated {len(questions)} questions via JSON parsing"
                        )

                        # extract the thoughts and strategy content
                        thoughts_content = parsed_response.get("thoughts", "")
                        strategy_content = "\n".join(
                            [
                                f"{i + 1}. {step}"
                                for i, step in enumerate(
                                    parsed_response.get("strategy", [])
                                )
                            ]
                        )

                        # Log Planning Agent Activity
                        if hasattr(self, "trace_logger"):
                            iteration_display = len(self.questions_by_iteration)
                            self.trace_logger.log_phase(
                                "Planning", iteration=iteration_display
                            )
                            self.trace_logger.log_agent_activity(
                                "Planner",
                                prompt,
                                {
                                    "thoughts": thoughts_content,
                                    "strategy": parsed_response.get("strategy", []),
                                    "sub_queries": questions,
                                },
                                thoughts=thoughts_content,
                            )
                            self.trace_logger.log_sub_queries(questions)

                        break
                    else:
                        raise ValueError("No valid tool queries in JSON response")

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(
                    f"JSON parsing failed: {e}, falling back to legacy parsing"
                )
                # fall back to the original list extraction method
                questions = extract_and_convert_list(response.content)
                if questions is None or len(questions) == 0:
                    current_try += 1
                    logger.warning(
                        f"⚠️  Retry {current_try}/{chance}: Failed to extract questions, retrying..."
                    )
                    await asyncio.sleep(10)  # use async sleep
                    continue
                else:
                    questions = questions[: self.questions_per_iteration]
                    logger.info(
                        f"✅ Successfully generated {len(questions)} questions via legacy parsing"
                    )

                    # try to extract the thoughts and strategy content (fallback method)
                    thoughts_content, strategy_content = (
                        self._extract_thoughts_and_strategy(response.content)
                    )
                    logger.info(
                        "Agent : _extract_thoughts_and_strategy: thoughts_content",
                        thoughts_content,
                    )
                    logger.info(
                        "Agent : _extract_thoughts_and_strategy: strategy_content",
                        strategy_content,
                    )
                    break

        # Handle case where questions is still None after all retries
        if "questions" not in locals() or questions is None:
            questions = []
            logger.error(
                "⚠️  Warning: Failed to generate follow-up questions after all retries. Using empty list."
            )
        return questions

    def _extract_thoughts_and_strategy(self, response_content: str) -> tuple[str, str]:
        """Extract Thoughts and Strategy sections from the response content"""
        thoughts_content = ""
        strategy_content = ""

        try:
            # Ensure response_content is not None
            if response_content is None:
                response_content = ""

            lines = response_content.split("\n")
            current_section = None

            for line in lines:
                line_stripped = line.strip()
                if line_stripped.startswith("Thoughts:"):
                    current_section = "thoughts"
                    # Safe extraction after "Thoughts:"
                    if len(line_stripped) > 9:
                        extracted_part = line_stripped[9:].strip()
                        if extracted_part is not None:
                            thoughts_content += extracted_part + "\n"
                elif line_stripped.startswith("Strategy:"):
                    current_section = "strategy"
                    # Safe extraction after "Strategy:"
                    if len(line_stripped) > 9:
                        extracted_part = line_stripped[9:].strip()
                        if extracted_part is not None:
                            strategy_content += extracted_part + "\n"
                elif (
                    current_section == "thoughts"
                    and line_stripped
                    and not line_stripped.startswith("[")
                ):
                    thoughts_content += line + "\n"
                elif current_section == "strategy" and line_stripped:
                    strategy_content += line + "\n"
                elif line_stripped.startswith("[") and current_section:
                    # Stop when we hit the tool list again
                    break

        except Exception as e:
            print(f"Warning: Failed to extract thoughts and strategy: {e}")
            thoughts_content = "Unable to extract analysis from response"
            strategy_content = "Unable to extract strategy from response"

        # Ensure we never return None values
        thoughts_result = (
            thoughts_content.strip() if thoughts_content is not None else ""
        )
        strategy_result = (
            strategy_content.strip() if strategy_content is not None else ""
        )

        return thoughts_result, strategy_result

    async def process_multiple_knowledge_chunks(
        self, query: str, current_key_info: str
    ) -> str:
        """ """

        if not hasattr(self, "knowledge_chunks") or not self.knowledge_chunks:
            return current_key_info.strip()

        try:
            lines = []
            for chunk in self.knowledge_chunks:
                key_info = chunk.get("key_info", "").strip()
                if key_info:
                    lines.append(key_info)

            knowledge_raw_md = "\n\n".join(lines)

            prompt = f"""
    You are assisting in organizing multi-round research findings for the main question: "{query}".

    Instructions:
    - Consolidate the following markdown facts into a clean, concise format.
    - DO NOT add new information or modify the meaning.
    - DO NOT remove or reformat <URL> references – they are placeholders for citations.
    - Avoid repeating similar facts or elaborating unnecessarily.
    - Use natural Markdown formatting (paragraphs or bullet points).
    - Try to keep the result under 2000 words.

    ### Findings:
    {knowledge_raw_md}
            """.strip()

        except Exception as e:
            logger.warning(
                f"process_multiple_knowledge_chunks (preprocessing) failed: {e}"
            )
            return current_key_info.strip()

        try:
            response = await invoke_with_timeout_and_retry(
                self.fast_model,
                prompt=prompt,
                timeout=60.0,
                max_retries=2,
                retry_delay=30.0,
            )
            current_knowledge = response.content.strip()

            if "<" not in current_knowledge or ">" not in current_knowledge:
                logger.warning("LLM output lacks <URL> pattern, fallback to raw.")
                return knowledge_raw_md

            return current_knowledge

        except Exception as e:
            logger.warning(f"process_multiple_knowledge_chunks (LLM) failed: {e}")
            return knowledge_raw_md

    async def _get_follow_up_questions_with_templates(
        self, current_knowledge: str, query: str
    ) -> List[str]:
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d")

        try:
            logger.info(f"Retrieving template for query: {query}")
            template = retrieve_large_template(query)
            # Log the template
            if hasattr(self, "trace_logger"):
                self.trace_logger.log_template(template)

            tuple_examples = []
            for i in range(self.questions_per_iteration):
                tuple_examples.append(
                    f"(Tool_Index_{i + 1}, Tool_Name_{i + 1}, Search_Query_{i + 1})"
                )
            output_format_example = f"[{', '.join(tuple_examples)}, ...]"
        except Exception as e:
            logger.warning(f"Failed to retrieve template: {e}")
            template = "No template available for this query."
            output_format_example = (
                "[('Tool_Index_1', 'Tool_Name_1', 'Search_Query_1'), ...]"
            )

        chance = 3
        current_try = 0
        while current_try < chance:
            if self.questions_by_iteration:
                prompt = f"""
                # Intelligent Search Assistant
                
                ## Context
                - Main query: "{query}"
                - Current date: {current_time}
                - Already searched {len(self.questions_by_iteration)} iterations
                
                ## Current Status
                - Past questions: {str(self.questions_by_iteration)}
                - Gathered knowledge: {current_knowledge}
                
                ## Your Task
                Analyze what's missing to fully answer the main query. Generate exactly {self.questions_per_iteration} focused sub-questions for downstream tool selection.
                
                ## Reference Example
                The following is a detailed example of how to decompose a similar research query into multiple search objectives and tools:
                <thinking_template only for task:{query}, do not use it directly, start from here>
                            {template} for in user query: {query}
                <thinking_template only for task:{query}, do not use it directly, end from here>
                - The thinking template is only a reference supplement to help resolve implicit preferences when user requirements are unclear.
                - If the user provides a specific, concrete action strategy or clear preferences/constraints, strictly prioritize the user's plan and ignore any template preferences that do not strongly connect to the main query.
                - If the thinking template is weakly related or irrelevant to the main query, use it selectively or ignore it to avoid drifting.
                - IMPORTANT: the template may mention tools. Your `sub_queries` MUST NOT include any tool names, tool tags, brackets, or tool-routing hints.
                
                ## Guidelines
                1. Identify key information gaps in current knowledge
                2. Formulate each sub-query as a clear, open-ended question that avoids bias, assumptions, or leading language (e.g., use neutral phrasing instead of affirmative or suggestive statements).
                3. Include only sub-queries that are essential and directly relevant to addressing the main query.
                4. Break down complex concepts or relationships into distinct, independently answerable questions.
                5. Keep sub-queries concise, specific, and straightforward (e.g., inquire about definitions, roles, or relationships, such as "What is X?" or "How does X relate to Y?").
                6. Avoid repeating past searches
                7. Each `sub_queries[i]` MUST be a normal sentence ending with a question mark ("?") and MUST NOT contain tool names or any bracketed tool syntax.
                
                ## Output Format
                You must provide your response in the following structured JSON format:
                {{
                    "thoughts": "Brief analysis of the problem and what needs investigation",
                    "strategy": ["..."],
                    "sub_queries": ["Question_1", "Question_2", "..."]
                }}
                
                ## Important Notes
                - Use terminology CONSISTENT with the main query, preserving specialized terms without arbitrary modifications.
                - Each query should target different aspects of the research question
                - Queries should be specific and actionable
                """
            else:
                prompt = f"""
                # Intelligent Search Assistant
                
                ## Context
                - Main query to answer: "{query}"
                - Current date: {current_time}
                - This is the first search iteration
                
                ## Your Task
                Analyze what's missing to fully answer the main query. Generate exactly {self.questions_per_iteration} focused sub-questions for downstream tool selection.
                
                ## Reference Example
                The following is a detailed example of how to decompose a similar research query into multiple search objectives and tools:
                <thinking_template only for task:{query}, do not use it directly, start from here>
                            {template} for in user query: {query}
                <thinking_template only for task:{query}, do not use it directly, end from here>
                - The thinking template is only a reference supplement to help resolve implicit preferences when user requirements are unclear.
                - If the user provides a specific, concrete action strategy or clear preferences/constraints, strictly prioritize the user's plan and ignore any template preferences that do not strongly connect to the main query.
                - If the thinking template is weakly related or irrelevant to the main query, use it selectively or ignore it to avoid drifting.
                - IMPORTANT: the template may mention tools. Your `sub_queries` MUST NOT include any tool names, tool tags, brackets, or tool-routing hints.
                
                ## Guidelines
                1. Identify key information gaps in current knowledge
                2. Formulate each sub-query as a clear, open-ended question that avoids bias, assumptions, or leading language (e.g., use neutral phrasing instead of affirmative or suggestive statements).
                3. Include only sub-queries that are essential and directly relevant to addressing the main query.
                4. Break down complex concepts or relationships into distinct, independently answerable questions.
                5. Keep sub-queries concise, specific, and straightforward (e.g., inquire about definitions, roles, or relationships, such as "What is X?" or "How does X relate to Y?").
                6. Avoid repeating past searches
                7. Each `sub_queries[i]` MUST be a normal sentence ending with a question mark ("?") and MUST NOT contain tool names or any bracketed tool syntax.
                
                ## Output Format
                You must provide your response in the following structured JSON format:
                {{
                    "thoughts": "Brief analysis of the problem and what needs investigation",
                    "strategy": ["..."],
                    "sub_queries": ["Question_1", "Question_2", "..."]
                }}
                
                ## Important Notes
                - Use terminology CONSISTENT with the main query, preserving specialized terms without arbitrary modifications.
                - Each query should target different aspects of the research question
                - Queries should be specific and actionable
                """
            try:
                response = await invoke_with_timeout_and_retry(
                    self.reasoning_model,
                    prompt,
                    timeout=90.0,
                    max_retries=3,
                    retry_delay=60.0,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to generate follow-up questions with template: {e}"
                )
                # Write error info to local log file in a process-safe way
                log_msg = f"[{datetime.now().isoformat()}] Failed to generate follow-up questions with template: {e}\n"
                write_log_process_safe(self.error_log_path, log_msg)
                current_try += 1
                retry_msg = f"⚠️  Retry {current_try}/{chance}: Failed to generate follow-up questions with template, retrying..."
                logger.warning(retry_msg)
                log_msg_retry = f"[{datetime.now().isoformat()}] {retry_msg}\n"
                write_log_process_safe(self.error_log_path, log_msg_retry)
                await asyncio.sleep(5)
                continue

            try:
                response_text = response.content
                logger.info(
                    f"important info stream in for process 0: break query in subquery"
                )
                logger.info(f"important info stream in for process 0: prompt: {prompt}")
                logger.info(
                    f"important info stream in for process 0: planning Agent {response_text}"
                )

                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    parsed_response = safe_json_from_text(response_text)
                    if not parsed_response:
                        raise ValueError("No valid JSON found")

                    # extract the tool queries
                    questions = parsed_response.get("sub_queries", [])

                    if questions and len(questions) > 0:
                        questions = questions[: self.questions_per_iteration]
                        logger.info(
                            f"✅ Successfully generated {len(questions)} questions via JSON parsing"
                        )

                        # extract the thoughts and strategy content
                        thoughts_content = parsed_response.get("thoughts", "")
                        strategy_content = "\n".join(
                            [
                                f"{i + 1}. {step}"
                                for i, step in enumerate(
                                    parsed_response.get("strategy", [])
                                )
                            ]
                        )

                        # Trace: record the Planner output (actual generated sub-queries)
                        if hasattr(self, "trace_logger"):
                            iteration_display = len(self.questions_by_iteration)
                            self.trace_logger.log_phase(
                                "Planning", iteration=iteration_display
                            )
                            self.trace_logger.log_agent_activity(
                                "Planner",
                                prompt,
                                {
                                    "parse_mode": "json",
                                    "thoughts": thoughts_content,
                                    "strategy": parsed_response.get("strategy", []),
                                    "sub_queries": questions,
                                },
                                thoughts=thoughts_content,
                            )
                            self.trace_logger.log_sub_queries(questions)
                        break
                    else:
                        raise ValueError("No valid tool queries in JSON response")

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(
                    f"JSON parsing failed: {e}, falling back to legacy parsing"
                )
                # fall back to the original list extraction method
                questions = extract_and_convert_list(response.content)
                if questions is None or len(questions) == 0:
                    current_try += 1
                    logger.warning(
                        f"⚠️  Retry {current_try}/{chance}: Failed to extract questions, retrying..."
                    )
                    await asyncio.sleep(10)  # use async sleep
                    continue
                else:
                    questions = questions[: self.questions_per_iteration]
                    logger.info(
                        f"✅ Successfully generated {len(questions)} questions via legacy parsing"
                    )

                    # try to extract the thoughts and strategy content (fallback method)
                    thoughts_content, strategy_content = (
                        self._extract_thoughts_and_strategy(response.content)
                    )

                    # Trace: record the Planner output (actual generated sub-queries)
                    if hasattr(self, "trace_logger"):
                        iteration_display = len(self.questions_by_iteration)
                        self.trace_logger.log_phase(
                            "Planning", iteration=iteration_display
                        )
                        self.trace_logger.log_agent_activity(
                            "Planner",
                            prompt,
                            {
                                "parse_mode": "legacy",
                                "thoughts": thoughts_content,
                                "strategy": strategy_content,
                                "sub_queries": questions,
                            },
                            thoughts=thoughts_content,
                        )
                        self.trace_logger.log_sub_queries(questions)
                    break

        # Handle case where questions is still None after all retries
        if "questions" not in locals() or questions is None:
            questions = []
            logger.error(
                "⚠️  Warning: Failed to generate follow-up questions after all retries. Using empty list."
            )
        return questions

    async def _compress_knowledge(self, current_knowledge: str, query: str) -> str:
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d")
        prompt = f"""You are a research synthesis agent working on the query: {query}

        You've collected multiple sets of keypoints from different subqueries. Below are all the accumulated keypoints with their citations:
        current_time: {current_time}
        {current_knowledge}

        Please synthesize and compress these keypoints into a concise, non-redundant list that:
        1. Eliminates duplicated information
        2. Merges related points
        3. Preserves the most important insights
        4. Retains all necessary citations and urls for each point
        5. Maintains the markdown list format

        Your goal is to create a comprehensive yet streamlined summary that captures all essential information while reducing redundancy. When multiple sources support the same point, include all relevant citations.
        Generate the compressed markdown keypoints list. Then follows the sources information.
        """
        try:
            response = await invoke_with_timeout_and_retry(
                self.model, prompt, timeout=60.0, max_retries=3, retry_delay=60.0
            )
        except Exception as e:
            logger.warning(f"Failed to compress knowledge: {e}")
            log_msg = (
                f"[{datetime.now().isoformat()}] Failed to compress knowledge: {e}\n"
            )
            write_log_process_safe(self.error_log_path, log_msg)
            raise e

        response_content = remove_think_tags(response.content)
        response_content = str(response_content)

        return response_content

    async def _extract_knowledge(
        self, facts_md: str, refs_in_round: List[Dict]
    ) -> Tuple[str, List[Dict]]:
        """
        Extract key information from the provided facts and references.
        """

        existing_pool = [
            {"idx": idx, "url": ref.link, "description": ref.title}
            for idx, ref in enumerate(self.ref_pool.pool, 1)
        ]

        prompt = textwrap.dedent(f"""
        You are an expert research assistant.

        ### Task
        1. Read the **candidate facts** and **candidate references** below.
        2. Remove URLs that are broken, extremely obviously unrelated to the query, or duplicate with the same information; but we need some references, add URLs at first time and when adding new refernece in pool that are useful for the current query if possible, but do not invent and never repeat.
        3. For every remaining URL, refine (or keep) its *description* so it is concise and copied verbatim from the source title/metadata.  
        • If the link is a scholaly article (DOI / PubMed / arXiv / journal), also fill an APA-style citation in **apa_citation**.  
        • Otherwise leave **apa_citation** empty.
        4. Produce a **Key Information** markdown block that integrates the facts or important details and cites sources *only via the raw URL string* (no numbering).  
        – If a fact has no URL, include it normally.  
        – When citing, wrap the URL in angle brackets like `<https://example.com/...>` (this is a placeholder that will be replaced by the system).
        5. Output JSON with two keys only:
        "key_information" and "cleaned_refs".

        ### Existing reference pool (keep these, don't modify)
        ```json
        {json.dumps(existing_pool, ensure_ascii=False)}
        ```

        ### Candidate facts (markdown)
        ```
        {facts_md}
        ```

        ### Candidate references (raw)
        ```json
        {json.dumps(refs_in_round, ensure_ascii=False)}
        ```

        ### Output JSON example
        ```json
        {{
        "key_information": "- **Point 1** … (<https://doi.org/...>)\\n- **Point 2** …",
        "cleaned_refs": [
            {{"url": "https://doi.org/…", "description": "Paper title …", "apa_citation": "Author A. (2024)…"}},
            {{"url": "https://news.com/…", "description": "News headline …", "apa_citation": ""}}
        ]
        }}
        ```

        Strictly follow the schema; do not add other keys.
        """)
        try:
            resp = await invoke_with_timeout_and_retry(
                self.model, prompt, timeout=60.0, max_retries=3, retry_delay=30.0
            )
            logger.info(
                f"important info stream in for process 5: extract_knowledge (Knowledge extraction Agent)"
            )
            logger.info(f"important info stream in for process 5: prompt: {prompt}")
            logger.info(
                f"important info stream in for process 5: _extract_knowledge (Knowledge extraction Agent): {resp.content}"
            )

        except Exception as e:
            logger.warning(f"Failed to extract knowledge: {e}")
            log_msg = (
                f"[{datetime.now().isoformat()}] Failed to extract knowledge: {e}\n"
            )
            write_log_process_safe(self.error_log_path, log_msg)
            raise e
        data = safe_json_from_text(resp.content) or {}
        key_info = data.get("key_information", "").strip()
        cleaned_refs = data.get("cleaned_refs", [])

        if not isinstance(key_info, str):
            key_info = ""
        if not isinstance(cleaned_refs, list):
            cleaned_refs = []

        key_info = re.sub(r"<(https?://[^>]+)>", r"\1", key_info)

        return key_info, cleaned_refs

    async def _answer_query(
        self,
        current_knowledge: str,
        query: str,
        current_iteration: int,
        max_iterations: int,
    ) -> str:
        is_final = current_iteration >= max_iterations
        title = "Closing Thoughts" if is_final else "Reflection"
        print("start answer query")
        print(query)
        existing_refs = [
            f"[{idx}] {ref.link} — {ref.title}"
            for idx, ref in enumerate(self.ref_pool.pool, 1)
        ]
        refs_block = "\n".join(existing_refs) or "*None yet*"
        if is_final:
            prompt_body = f""" 
            ## Original Query ##
            {query}

            ## Accumulated Knowledge ##
            {current_knowledge}

            ## Available SOURCES ##
            {refs_block}

            ### Instructions

            You are tasked with generating a **precise, well-reasoned, and evidence-aligned final answer** to the query above.

            Follow these strict guidelines:

            - **Focus only on information that is directly relevant, scientifically sound, and consistent.**
            - If the provided sources include **irrelevant, weakly related, or clearly flawed data**, **disregard them** in your reasoning.
            - Do **not invent** new references or URLs — cite sources only using the `[^^n]` format.
            - If certain key facts are **not explicitly stated in the sources**, you may rely on **logical inference, background knowledge, and cautious reasoning** to form a credible answer.
            - When evidence is incomplete or ambiguous, you are encouraged to **prioritize plausibility** and **choose the most reasonable and justifiable conclusion** based on available information.

            ---

            ### Output Markdown Template

            ## Conclusion

            Provide a clear, concise, and fully supported response to the query.  
            Avoid discussing unrelated details or speculative content unless essential to your reasoning.  
            In the face of conflicting or insufficient data, select the explanation that is **best supported by reasoning and contextual clues**.

            ## Thoughts

            Briefly describe your reasoning process:  
            - How did you assess the quality and reliability of the evidence?  
            - Which sources did you rely on, and which did you discard (and why)?  
            - If part of your answer was based on deduction rather than direct evidence, explain your logic clearly.

            ## Key Findings

            [A short summary (100–150 words) of the most important mechanistic or conceptual insight, written in a formal tone using markdown.]

            ## References

            [List all relevant references in this format — each must be on a new line:]

            [^^1] APA-style citation or description <https://example.com/source1>
            [^^2] APA-style citation or description <https://example.com/source2> 
            ...
            """
        else:
            critic_report_constraints = ""
            if self.is_report:
                critic_report_constraints = """
            ### Report-mode requirement (generic; no special-case keywords or examples)
            - Your reflection must be anchored to concrete missing information.
            - In Thoughts, include a short checklist of the most important missing evidence items (each item must name a specific missing attribute or missing evidence type).
            - In Strategy, every numbered item must end with a single-line "NextQuery:" that is an executable query string for the next iteration.
            """

            prompt_body = f"""
            ## Query ##
            {query}

            ## Current Iteration ##
            {current_iteration} / {max_iterations}

            ## Knowledge So Far ##
            {current_knowledge}

            ## Available SOURCES ##
            {refs_block}

            ### Instructions
            1. Critically reflect on progress and propose strategy for next round.
            2. When you mention evidence, cite with existing `[^^n]` numbers only.
            {critic_report_constraints}


            ### Output Markdown Template
            Your response MUST follow this exact format in markdown,The classification of large and small titles should comply with the markdown specification:

            ## Thoughts
            Analyze,this is just a recommended title content range, but you don’t have to write a fixed title; you can answer each sub-heading in points: 
            ### What has been discovered so far, 
            ### What gaps or limitations exist in current knowledge, 
            ### What aspects of the original query remain unanswered, 
            ### How to improve research quality and coverage]

            ## Strategy 
            Provide numbered strategy points for the next research iteration, similar to initial strategy format,Do NOT include subheadings**, bullet points, or further nested structure under each point. Each point should be a **self-contained strategy**, ideally 2–3 sentences long, combining direction and rationale:
            1. Specific research direction with reasoning
            2. Another research direction with reasoning
            3. etc.
            ...

            ## References
            [Provide a list of references that you think are relevant to the query, with each reference on a new line(consider markdown's new line rules), such as:
            [^^1] academic format citation(if actually have else only show url)<https://example.com/source1> \\n (this \\n must include for new line)
            [^^2] academic format citation(if actually have else only show url)<https://example.com/source2> \\n (this \\n must include for new line)
            ...

            """

        prompt = textwrap.dedent(f"""
        You are an expert research assistant.

        {prompt_body}

        ### Strict Citation Rules
        * Use **numbered square-bracket citations** based on the **index in Available SOURCES** (e.g., [^^1], [^^2], ...).
        * Do not fabricate, skip, or renumber citations arbitrarily.
        * Each number must consistently refer to the **same URL or source** throughout the document.
        * If a source lacks a formal citation title or description, directly use the URL as a clickable markdown link.
        * Using markdown format, note that the title structure follows the common format of markdown.
        """)

        try:
            response = await invoke_with_timeout_and_retry(
                self.model, prompt, timeout=100.0, max_retries=3, retry_delay=30.0
            )
        except Exception as e:
            logger.warning(f"Failed to answer query: {e}")
            log_msg = f"[{datetime.now().isoformat()}] Failed to answer query: {e}\n"
            write_log_process_safe(self.error_log_path, log_msg)
            raise e

        response_content = remove_think_tags(response.content)
        response_content = add_hard_breaks_to_references(response_content)

        logger.info(f"important info stream in for process 6: answer query")
        logger.info(f"important info stream in for process 6: prompt: {prompt}")
        logger.info(
            f"important info stream in for process 6: response_content: (Critic or answer query Agent) {response_content}"
        )

        return response_content

    async def _generate_detailed_report(
        self, current_knowledge: str, findings: List[Dict], query: str, iteration: int
    ):
        """
        Generate a publication-style report, with in-text [^^n] citations
        and a final reference list strictly come from self.ref_pool。
        """
        pool_objs = []
        for idx, ref in enumerate(self.ref_pool.pool, 1):
            pool_objs.append(
                {
                    "idx": idx,
                    "url": ref.link,
                    "apa": ref.subtitle or "",
                    "desc": ref.title,
                }
            )
        pool_json = json.dumps(pool_objs, ensure_ascii=False, indent=2)

        prompt = textwrap.dedent(f"""
        You are a senior biomedical research analyst.  Write a professional report that synthesises all evidence.

        ### Research Context
        * Original Query: {query}
        * Iterations Completed: {iteration}

        ### Knowledge Base
        {current_knowledge}

        ### Citation Pool  (MUST cite only these!)
        ```json
        {pool_json}
        ```
        ## Task 
        Generate a professional research report that would be suitable for academic or clinical publication. The report should be comprehensive, well-structured, and scientifically rigorous.

        
        ### Strict Citation Rules
        1. In the narrative, cite sources only as `[^^n]`, where n equals "idx".
        2. Do **NOT** invent new URLs or citations.
        3. Every `[^^n]` used must appear in the final **References** list.
        4. If an entry's `"apa"` field is non-empty, write:
            `[^^n] {{"apa"}} <{{"url"}}>`
        else write:
            `[^^n] <{{"url"}}>`


        **IMPORTANT REQUIREMENTS:**
        1. The detailed report need to be 2000-3000 words in length.
        2. Write in a scholarly, narrative style similar to academic publications
        3. Provide in-depth analysis and synthesis of the research findings
        4. Include specific details, mechanisms, clinical implications, and evidence

        Your response MUST follow this exact format in markdown:

        Detailed
        ## Key Findings
        [200-300 word paragraph to directly and fully answer all parts of the instruction with specificity, providing some insight but in concise language,It will be better if some important statements be supported by references from the provided research.]

        ## Ideas
        [Paragraph to suggest potential research directions or new ideas based on the findings. You are allowed to reason and think more actively here and make inferences and assumptions based on the information you receive, but do not invent new references or URLs.]
        
        ## Detailed Analysis(s)
        [Main body: 1000-2000 words organized into logical sections you determine based on each aspect of the topic. Each section should contribute to answering the original instruction. Include comprehensive analysis, specific evidence, methodologies, and limitations where relevant. Evidence should be provided from both broader perspectives and concrete examples. It will be better if some statements be supported by references from the provided research.]

        ## Conclusion
        [200-300 words summarizing the evidences, discuss practical implications, reflect on agent's gathered data quality, and propose next-step recommendations, It will be better if some statements be supported by references from the provided research.]
        
        ## References
        [Provide a list of references that you think are relevant to the query, with each reference on a new line (consider markdown's new line rules), such as:
        [^^1] academic format citation(if actually have else only show url)<https://example.com/source1>
        [^^2] academic format citation(if actually have else only show url)<https://example.com/source2>
        ...
        ]
        """)

        logger.info("Generating detailed report with prompt:")
        logger.info(prompt)
        try:
            response = await invoke_with_timeout_and_retry(
                self.report_model,
                prompt,
                timeout=180.0,
                max_retries=3,
                retry_delay=60.0,
            )
        except Exception as e:
            logger.warning(f"Failed to generate detailed report: {e}")
            log_msg = f"[{datetime.now().isoformat()}] Failed to generate detailed report: {e}\n"
            write_log_process_safe(self.error_log_path, log_msg)
            raise e
        logger.info("Detailed report response first received:")
        logger.info(response.content)
        response_content = remove_think_tags(response.content)
        response_content = str(response_content)
        logger.info("Detailed report response second received:")
        logger.info(response_content)

        summary_content, detailed_content = self._extract_report_sections(
            response_content
        )

        logger.info(f"✅ Generated detailed research report for: {query}")
        logger.info(f"important info stream in for process 7: report")
        logger.info(f"prompt: {prompt}")
        logger.info(f"response_content: {response_content}")

        return response_content

    def _extract_report_sections(self, response_content: str):
        """
        extract the Key Findings from the markdown format report output by the model, and keep the full report content.

        parameters:
            response_content (str): the full markdown format content output by the model

        return:
            summary_content (str): the extracted Key Findings part (summary)
            detailed_content (str): the full report content
        """
        # use regex to extract the ## Key Findings part
        key_findings_match = re.search(
            r"## Key Findings\s+(.*?)(?=\n##\s+Detailed Analysis|\n##\s+Conclusion|\Z)",
            response_content,
            re.DOTALL | re.IGNORECASE,
        )

        if key_findings_match:
            summary_content = key_findings_match.group(1)  # .strip()
        else:
            summary_content = ""

        detailed_content = response_content  # .strip()

        return summary_content, detailed_content

    async def initialize(self):
        """Initialize async components"""

        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        tool_type_info_path = os.path.join(CURRENT_DIR, "cache_data", "tool_info.xlsx")
        embedding_cache = os.path.join(
            CURRENT_DIR, "cache_data", "tool_desc_embedding.pkl"
        )

        embedding_api_key = settings.embedding_api_key

        try:
            self.mcp_tool_client = OmniagentMCPToolClient(mcp_servers, self.chosen_tools)
            await self.mcp_tool_client.initialize()
        except Exception as e:
            logger.warning(f"Failed to initialize MCP tool client in initialize: {e}")
            log_msg = f"[{datetime.now().isoformat()}] Failed to initialize MCP tool client in initialize: {e}\n"
            write_log_process_safe(self.error_log_path, log_msg)
            raise e

        try:
            self.mcp_tool_dict = self.mcp_tool_client.tool2source
        except Exception as e:
            logger.warning(f"Failed to get MCP tool dict in initialize: {e}")
            log_msg = f"[{datetime.now().isoformat()}] Failed to get MCP tool dict in initialize: {e}\n"
            write_log_process_safe(self.error_log_path, log_msg)
            raise e

        try:
            self.tool_selector = ToolSelector(
                self.tool_planning_model,
                self.reasoning_model,
                self.mcp_tool_client,
                tool_type_info_path,
                embedding_api_key,
                embedding_cache,
                self.chosen_tools,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize tool selector in initialize: {e}")
            log_msg = f"[{datetime.now().isoformat()}] Failed to initialize tool selector in initialize: {e}\n"
            write_log_process_safe(self.error_log_path, log_msg)
            raise e

        try:
            self.tool_executor = ToolExecutor(
                self.mcp_tool_client, self.error_log_path, self.fast_model
            )
        except Exception as e:
            logger.warning(f"Failed to initialize tool executor in initialize: {e}")
            log_msg = f"[{datetime.now().isoformat()}] Failed to initialize tool executor in initialize: {e}\n"
            write_log_process_safe(self.error_log_path, log_msg)
            raise e

        num_tools = len(self.mcp_tool_client.mcp_tools)
        num_general_tools = len(self.tool_selector.general_tool_selector.general_tools)
        num_expert_tools = num_tools - num_general_tools
        logger.info("🔧 NUMBER OF AVAILABLE TOOLS IN POOL: %d", num_tools)
        logger.info("🔧      GENERAL TOOLS: %d", num_general_tools)
        logger.info("🔧      EXPERT TOOLS: %d", num_expert_tools)

    async def analyze_topic(
        self, query: str, question_id: str | int = "unknown"
    ) -> Dict:
        # Initialize Trace Logger
        log_dir = (
            self.session_log_dir
            if self.session_log_dir
            else os.path.join(ROOT_DIR, "logs")
        )
        self.trace_logger = ResearchTraceLogger(log_dir, query, question_id=question_id)

        logger.info("Starting research on topic: %s", query)
        logger.info("%s", "\n" + "=" * 80)
        logger.info("RESEARCH CONFIGURATION:")
        logger.info("%s", "=" * 80)
        logger.info("Query: %s", query)
        logger.info("Max Iterations: %d", self.max_iterations)
        logger.info("Questions per Iteration: %d", self.questions_per_iteration)
        logger.info(
            "Available Tools: %d",
            len(self.mcp_tool_client.mcp_tools)
            if self.mcp_tool_client.mcp_tools
            else 0,
        )
        logger.info("%s", "=" * 80 + "\n")

        findings = []
        current_knowledge = ""
        iteration = 0

        if self.verbose:
            with open(self.mid_path, "a", encoding="utf-8") as f:
                f.write(f"User Qeury : {query} \n")

        # Check if search engine is available
        if (
            len(self.mcp_tool_client.mcp_tools) == 0
            or self.mcp_tool_client.mcp_tools is None
        ):
            error_msg = "Error: No mcp_tool_client.mcp_tools available. Please check your configuration."

            return {
                "findings": [],
                "iterations": 0,
                "questions": {},
                "current_knowledge": "",
                "error": error_msg,
            }

        while iteration < self.max_iterations:
            if self.verbose:
                with open(self.mid_path, "a", encoding="utf-8") as f:
                    f.write(f"Iteration : {iteration} \n")

            questions = await self._get_follow_up_questions_with_templates(
                current_knowledge, query
            )

            logger.info(
                f"important info stream in iteration {iteration} for process 1: query and subquery"
            )
            logger.info(
                f"important info stream in iteration {iteration} for process 1; query: {query}"
            )
            logger.info(
                f"important info stream in iteration {iteration} for process 1; questions: {questions}"
            )

            question_texts = questions
            self.questions_by_iteration[iteration] = question_texts

            fullquery_tool_results = []
            for q_idx, question in enumerate(question_texts, 1):
                tool_and_input_list = await self.tool_selector.run(question) or []
                logger.debug("debug: tool_and_input_list")
                logger.debug("Tool and input list: %s", str(tool_and_input_list))

                if self.verbose:
                    with open(self.mid_path, "a", encoding="utf-8") as f:
                        f.write(f"\n Tool and tool query in subquery : {question} : \n")
                        f.write(f"\n {tool_and_input_list} \n")

                logger.info(
                    f"important info stream in iteration {iteration} for process 2: Tool and input list for subquery (Tool calling agents)"
                )
                logger.info(
                    f"important info stream in iteration {iteration} for process 2: subquery: {question}"
                )
                logger.info(
                    f"important info stream in iteration {iteration} for process 2: tool_and_input_list: {tool_and_input_list}"
                )

                try:
                    tool_calling_results = (
                        await self.tool_executor.run(tool_and_input_list) or []
                    )
                except Exception as e:
                    logger.warning("Error during tool execution: %s", str(e))
                    tool_calling_results = []

                logger.debug("debug: tool_calling_results")
                logger.debug("Tool calling results: %s", str(tool_calling_results))

                logger.info(
                    f"important info stream in iteration {iteration} for process 3: Tool result for subquery"
                )
                logger.info(
                    f"important info stream in iteration {iteration} for process 3: subquery: {question}"
                )
                logger.info(
                    f"important info stream in iteration {iteration} for process 3: tool_calling_results: {tool_calling_results}"
                )

                try:
                    logger.info(
                        "Starting enhanced tool parsing and knowledge extraction..."
                    )

                    # 1. Convert raw results to BasicToolResult
                    basic_results = []
                    for i, res in enumerate(tool_calling_results):
                        sub_q = (
                            tool_and_input_list[i]["item"]
                            if i < len(tool_and_input_list)
                            else question
                        )
                        basic = parse_tool_result(res, sub_q, iteration)
                        if basic:
                            basic_results.append(basic)

                    # 2. Deep Extraction with ToolParseAgent
                    parsed_results = await self.tool_parse_agent.parse_results(
                        query, basic_results
                    )

                    # Log Execution
                    if hasattr(self, "trace_logger"):
                        self.trace_logger.log_phase(
                            "Execution & Knowledge Extraction", iteration=iteration
                        )
                        for parsed in parsed_results:
                            self.trace_logger.log_tool_execution(
                                parsed.basic.tool_name,
                                parsed.basic.raw_payload
                                if hasattr(parsed.basic, "raw_payload")
                                else {}, 
                                parsed.basic.raw_text,
                                parsed.basic.success,
                            )
                            # Log extracted knowledge
                            if parsed.summary:
                                self.trace_logger.log_knowledge_update(
                                    f"**Summary from {parsed.basic.tool_name}**:\n{parsed.summary}"
                                )
                            if parsed.evidence_entries:
                                details = "\n".join(
                                    [
                                        f"- {e.detailed_findings}"
                                        for e in parsed.evidence_entries
                                        if e.detailed_findings
                                    ]
                                )
                                if details:
                                    self.trace_logger.log_knowledge_update(
                                        f"**Detailed Findings**:\n{details}", priority=2
                                    )

                    # 3. Update Bibliography & Context
                    for parsed in parsed_results:
                        # Add to bibliography
                        for entry in parsed.evidence_entries:
                            bib_entry = BibliographyEntry(
                                key=entry.hash_value,
                                title=entry.title or "Untitled",
                                url=entry.url,
                                doi=entry.doi,
                                authors=entry.authors,
                                year=entry.year,
                                fetched_at=datetime.now().isoformat(),
                                first_seen_round=iteration,
                            )
                            self.bibliography.add_or_update(bib_entry)

                            # Add to reference pool (for backward compatibility and final report)
                            if entry.url:
                                self.ref_pool.add(entry.title, "", entry.url)
                                self.all_links_of_system.append(entry.url)

                        # Add to ResearchContext (Memory Compression)
                        # Priority 1: The summary generated by the agent
                        if parsed.summary:
                            self.research_context.add_knowledge(
                                f"Source: {parsed.basic.tool_name}\nSummary: {parsed.summary}",
                                priority=1,
                            )

                        # Priority 2: Detailed findings and key sentences
                        for entry in parsed.evidence_entries:
                            content_block = []
                            if entry.detailed_findings:
                                content_block.append(
                                    f"Detailed Findings: {entry.detailed_findings}"
                                )
                            if entry.key_sentences:
                                content_block.append(
                                    f"Key Points: {'; '.join(entry.key_sentences)}"
                                )

                            if content_block:
                                self.research_context.add_knowledge(
                                    f"Evidence from {entry.title}:\n"
                                    + "\n".join(content_block),
                                    priority=2,
                                )

                    # ------------------------------------------------------------------
                    # Report-only: build a persistent candidate pool (HGNC-like symbols),
                    # then validate a small batch with structured target tools.
                    # This supports long-tail "signal rescue" without affecting benchmarks.
                    # ------------------------------------------------------------------
                    if self.is_report:
                        try:
                            # Harvest raw text from selected tool outputs (high recall)
                            harvest_texts: list[str] = []
                            for parsed in parsed_results:
                                if parsed.basic.tool_name in {
                                    "paper_search",
                                    "tavily_search",
                                    "search_assay",
                                    "search_activity",
                                }:
                                    harvest_texts.append(parsed.basic.raw_text or "")
                                    if parsed.summary:
                                        harvest_texts.append(parsed.summary)
                                    for e in parsed.evidence_entries[:10]:
                                        if e.detailed_findings:
                                            harvest_texts.append(e.detailed_findings)
                                        if e.key_sentences:
                                            harvest_texts.extend(e.key_sentences[:5])

                            regex_syms = self._extract_candidate_symbols_regex(
                                harvest_texts
                            )
                            regex_syms = self._hard_filter_candidate_symbols(regex_syms)
                            cleaned_syms = await self._clean_candidate_symbols_llm(
                                regex_syms
                            )
                            cleaned_syms = self._hard_filter_candidate_symbols(
                                cleaned_syms
                            )

                            # Update persistent pool
                            for s in cleaned_syms:
                                self._candidate_symbols.add(s)
                                # Update stats (per iteration, per source tool) to rank candidates later
                                self._update_candidate_stats(
                                    s, parsed.basic.tool_name, iteration
                                )

                            pool_list = self._get_ranked_candidate_pool()
                            expansion_added: list[str] = []

                            # Report-only: LLM divergence step (optional).
                            # This expands the pool with hypothesis candidates inspired by the current pool and user intent.
                            # It does not affect benchmarks and does not guarantee correctness; validation loop will handle it.
                            if self.enable_llm_prescreen:
                                try:
                                    proposed = (
                                        await self._propose_additional_candidates_llm(
                                            query=query,
                                            symbols=pool_list,
                                            k=min(
                                                8, max(0, self.candidate_pool_max // 50)
                                            ),
                                        )
                                    )
                                    if proposed:
                                        expansion_added = list(proposed)
                                        for s in proposed:
                                            self._candidate_symbols.add(s)
                                            self._update_candidate_stats(
                                                s, "llm_expansion", iteration
                                            )
                                        pool_list = self._get_ranked_candidate_pool()
                                except Exception as e:
                                    logger.warning(
                                        f"Report-only LLM expansion step failed: {e}"
                                    )

                            # Report-only: pool profiling (classification + gap-fill queries) to guide next planning.
                            pool_profile: dict = {}
                            if (
                                self.enable_llm_prescreen
                                and len(pool_list) >= 60
                                and self._last_pool_profile_iteration != iteration
                            ):
                                pool_profile = await self._profile_candidate_pool_llm(
                                    query=query, symbols=pool_list
                                )
                                self._last_pool_profile_iteration = iteration
                                if pool_profile:
                                    self.research_context.add_knowledge(
                                        "[Candidate Pool Profile]\n"
                                        + json.dumps(pool_profile, ensure_ascii=False),
                                        priority=1,
                                    )
                                    if hasattr(self, "trace_logger"):
                                        # Keep trace concise: counts + number of suggested queries
                                        bc = pool_profile.get("bucket_counts", {})
                                        gq = pool_profile.get("gap_fill_queries", [])
                                        self.trace_logger.log_knowledge_update(
                                            f"**[Candidate Pool Profile]** buckets={bc} gap_fill_queries={len(gq) if isinstance(gq, list) else 0}",
                                            priority=1,
                                        )

                            # Optional: conservative LLM prescreen to choose which symbols to validate next.
                            # This does NOT delete anything from the global pool; it only prioritizes validation.
                            if (
                                self.enable_llm_prescreen
                                and self.validation_batch_size > 0
                            ):
                                # Build a deterministic, intent-aligned short-list for prescreening:
                                # prioritize unvalidated symbols by recency/frequency (generic; no family-prefix heuristics).
                                shortlist = [
                                    s
                                    for s in pool_list
                                    if s not in self._validated_symbols
                                ]
                                shortlist.sort(
                                    key=lambda s: (
                                        -int(
                                            self._candidate_stats.get(s, {}).get(
                                                "last_seen", -1
                                            )
                                        ),
                                        -int(
                                            self._candidate_stats.get(s, {}).get(
                                                "count", 0
                                            )
                                        ),
                                        s,
                                    )
                                )
                                shortlist = shortlist[
                                    : max(
                                        50, min(len(shortlist), self.candidate_pool_max)
                                    )
                                ]

                                prescreen = (
                                    await self._prescreen_symbols_for_validation_llm(
                                        query, shortlist
                                    )
                                )
                                validate_list = (
                                    prescreen.get("selected", [])
                                    or shortlist[: self.validation_batch_size]
                                )
                                excluded = prescreen.get(
                                    "deprioritized", prescreen.get("excluded", [])
                                )
                                drop = prescreen.get("drop", [])
                            else:
                                shortlist = [
                                    s
                                    for s in pool_list
                                    if s not in self._validated_symbols
                                ]
                                shortlist.sort(
                                    key=lambda s: (
                                        -int(
                                            self._candidate_stats.get(s, {}).get(
                                                "last_seen", -1
                                            )
                                        ),
                                        -int(
                                            self._candidate_stats.get(s, {}).get(
                                                "count", 0
                                            )
                                        ),
                                        s,
                                    )
                                )
                                validate_list = shortlist[: self.validation_batch_size]
                                excluded = []
                                drop = []

                            # High-confidence drop (report-only): remove only tokens the LLM is extremely sure are NOT genes.
                            # This is a speed optimization. It should be conservative to avoid losing long-tail candidates.
                            if drop:
                                for s in drop:
                                    self._candidate_symbols.discard(s)
                                    self._validated_symbols.discard(s)
                                    try:
                                        self._pinned_symbols.discard(s)
                                    except Exception:
                                        pass
                                    self._candidate_stats.pop(s, None)
                                pool_list = self._get_ranked_candidate_pool()

                            # Validate only symbols we haven't validated before (cache)
                            validate_list = [
                                s
                                for s in validate_list
                                if s not in self._validated_symbols
                            ]
                            # Pin selected symbols so they won't be evicted by candidate_pool_max truncation later.
                            # This keeps long-tail candidates "alive" in logs and downstream reasoning once they enter validation.
                            if validate_list:
                                try:
                                    self._pinned_symbols.update(
                                        [
                                            s
                                            for s in validate_list
                                            if isinstance(s, str) and s
                                        ]
                                    )
                                except Exception:
                                    pass
                            if validate_list:
                                validations = await self._validate_candidate_symbols(
                                    validate_list
                                )
                                for v in validations:
                                    sym = v.get("symbol")
                                    if isinstance(sym, str) and sym:
                                        self._validated_symbols.add(sym)
                                        try:
                                            self._pinned_symbols.add(sym)
                                        except Exception:
                                            pass
                            else:
                                validations = []

                            # Write a compact, reusable block into Memory Bank
                            self.research_context.add_knowledge(
                                "[Candidate Pool]\n"
                                f"HGNC-like symbols (cap={self.candidate_pool_max}): {pool_list}\n"
                                f"llm_expansion_added={expansion_added}\n"
                                f"[Candidate Pool Prescreen]\nselected_for_validation={validate_list}\nexcluded_obvious={excluded}\ndrop_high_confidence={drop}\n"
                                "[Candidate Pool Validation]\n"
                                + json.dumps(validations, ensure_ascii=False),
                                priority=1,
                            )

                            # Make it visible in trace logs too (so you can see the pool each run)
                            if hasattr(self, "trace_logger"):
                                self.trace_logger.log_knowledge_update(
                                    f"**[Candidate Pool]** size={len(pool_list)} cap={self.candidate_pool_max}\n\n"
                                    f"HGNC-like symbols (cap={self.candidate_pool_max}): {pool_list}\n"
                                    f"Selected_for_validation={validate_list}\n"
                                    f"Excluded_obvious={excluded}\n",
                                    priority=1,
                                )
                        except Exception as e:
                            logger.warning(
                                f"Report-only candidate pool/validation failed: {e}"
                            )

                    logger.info(
                        f"important info stream in iteration{iteration} for process 4: Enhanced parsing completed"
                    )
                    logger.info(
                        f"Current context length: {len(self.research_context.current_knowledge)} chars"
                    )

                except Exception as e:
                    logger.warning(f"Error during enhanced parsing: {e}", exc_info=True)

                logger.info(
                    "Questions by iteration: %s", str(self.questions_by_iteration)
                )

            iteration += 1

            # Update current_knowledge from ResearchContext
            # This replaces the old process_multiple_knowledge_chunks logic
            current_knowledge = self.research_context.current_knowledge

            logger.info(
                f"important info stream in for process new: knowledge memory Agent {current_knowledge}"
            )

            # Send Analysis & Strategy Refinement (only if not the last iteration)
            logger.info("getting final answer")

            final_answer = await self._answer_query(
                current_knowledge, query, iteration, self.max_iterations
            )
            if iteration >= self.max_iterations:
                answer_title = "Final Answer"
            else:
                answer_title = "Critc"

            # Store critic output in ResearchContext with high priority to prevent compression loss
            # This ensures anomalies identified by critic are preserved as "residual connections"
            if not (iteration >= self.max_iterations):
                # Extract Thoughts section (contains anomaly detection)
                thoughts_match = re.search(
                    r"##\s+Thoughts\s+(.*?)(?=\n##\s+Strategy|\n##\s+References|\Z)",
                    final_answer,
                    re.DOTALL | re.IGNORECASE,
                )
                if thoughts_match:
                    thoughts_text = thoughts_match.group(1).strip()
                    # Add as Priority 1 to ensure it survives compression
                    self.research_context.add_knowledge(
                        f"[Critic Reflection - Iteration {iteration}]\n{thoughts_text}",
                        priority=1,
                    )
                    logger.info(
                        f"Added critic thoughts to ResearchContext with Priority 1"
                    )

            # Log Answer/Critic
            if hasattr(self, "trace_logger"):
                self.trace_logger.log_phase(answer_title, iteration=iteration)
                self.trace_logger.log_agent_activity(
                    "Critic/Writer",
                    "Generate answer based on accumulated knowledge...",  # Prompt is too long to repeat here
                    final_answer,
                )

            if self.verbose:
                with open(self.mid_path, "a", encoding="utf-8") as f:
                    f.write(f"\n {answer_title} : \n")
                    f.write(f"\n {final_answer} \n")

            logger.info("Critic or answer: ")
            logger.info(final_answer)

        print(f"self.is_report {self.is_report}")
        final_report = ""
        if self.is_report:
            print("Generating final report")
            logger.info("Generating detailed report after final iteration")
            try:
                final_report = await self._generate_detailed_report(
                    current_knowledge, findings, query, iteration
                )
                if self.verbose:
                    print(f"writing final report in{self.report_path}")
                    with open(self.report_path, "a", encoding="utf-8") as f:
                        f.write(f"Query : {query} : \n")
                        f.write(f"\n {final_report} \n")

            except Exception as e:
                logger.warning(f"Warning: Failed to generate detailed report: {e}")

        # Save Final Case JSON
        if hasattr(self, "trace_logger"):
            if final_report:
                self.trace_logger.case_data["final_report"] = final_report
            self.trace_logger.save_case_json()

        current_knowledge = final_answer
        return {
            "findings": findings,
            "iterations": iteration,
            "questions": self.questions_by_iteration,
            "current_knowledge": current_knowledge,
            "final_report": final_report,
        }
