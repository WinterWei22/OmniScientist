import contextlib
import os
import re
from pathlib import Path

import requests
from langchain_core.messages import HumanMessage

from omniInfra.config import default_config
from omniInfra.llm import get_llm
from omniInfra.model.embedding_cache import QwenEmbeddingCache

QWEN_RERANKER_MODEL_ALIASES = {
    "qwen-rerank": "qwen3-rerank",
    "qwen-vl-rerank": "qwen3-vl-rerank",
}
QWEN_RERANKER_API_DEFAULTS = {
    "qwen3-rerank": {
        "style": "qwen-compatible",
        "base_url": "https://dashscope.aliyuncs.com/compatible-api/v1/reranks",
    },
    "qwen3-vl-rerank": {
        "style": "dashscope-native",
        "base_url": "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
    },
}


class QwenSemanticResourceRanker:
    """Qwen embedding retrieval with API-based Qwen reranking.

    Resource embeddings use the incremental API cache.  Reranking is always
    performed through the configured Qwen/DashScope HTTP endpoint; no local
    reranker weights are loaded in-process.  This keeps A1, MCP,
    and the standalone gateway on the same remotely hosted reranker and also
    allows a future model such as ``qwen3-vl-rerank`` to be selected by an
    environment variable.
    """

    DEFAULT_INSTRUCTION = (
        "Given a multi-step biomedical research task, decide whether this single OmniInfra resource directly supports "
        "at least one step. The resource does not need to solve the whole task. Prefer exact database interfaces and "
        "specialized tools over generic analysis methods, and retain complementary resources needed by different steps."
    )

    def __init__(
        self,
        embedding_model: str | None = None,
        reranker_model: str | None = None,
        *,
        device: str | None = None,
        embedding_batch_size: int | None = None,
        reranker_batch_size: int | None = None,
        max_length: int | None = None,
        local_files_only: bool = True,
        instruction: str = DEFAULT_INSTRUCTION,
        embedding_backend: str | None = None,
        reranker_backend: str | None = None,
        reranker_base_url: str | None = None,
        reranker_api_key: str | None = None,
        reranker_timeout_seconds: float | None = None,
    ):
        self.embedding_backend = (embedding_backend or os.getenv("OMNIINFRA_EMBEDDING_BACKEND", "api")).lower()
        if self.embedding_backend not in {"api", "local"}:
            raise ValueError("OMNIINFRA_EMBEDDING_BACKEND must be 'api' or 'local'")
        self.embedding_model_name = embedding_model or (
            os.getenv("OMNIINFRA_EMBEDDING_MODEL", "qwen3.7-text-embedding")
            if self.embedding_backend == "api"
            else os.getenv("OMNIINFRA_EMBEDDING_MODEL_PATH", "Qwen/Qwen3-Embedding-4B")
        )
        self.reranker_backend = (reranker_backend or os.getenv("OMNIINFRA_RERANKER_BACKEND", "api")).lower()
        if self.reranker_backend != "api":
            raise ValueError("Local Qwen reranking is no longer supported; set OMNIINFRA_RERANKER_BACKEND=api")
        requested_reranker_model = reranker_model or os.getenv(
            "OMNIINFRA_RERANKER_MODEL", os.getenv("OMNIINFRA_RERANKER_API_MODEL", "qwen3-rerank")
        )
        self.reranker_model_name = QWEN_RERANKER_MODEL_ALIASES.get(
            requested_reranker_model.strip().lower(), requested_reranker_model.strip()
        )
        model_defaults = QWEN_RERANKER_API_DEFAULTS.get(self.reranker_model_name.lower(), {})
        self.reranker_api_style = (
            os.getenv("OMNIINFRA_RERANKER_API_STYLE") or model_defaults.get("style") or "dashscope-native"
        ).lower()
        if self.reranker_api_style not in {"qwen-compatible", "dashscope-native"}:
            raise ValueError("OMNIINFRA_RERANKER_API_STYLE must be 'qwen-compatible' or 'dashscope-native'")
        self.reranker_base_url = (
            reranker_base_url
            or os.getenv("OMNIINFRA_RERANKER_BASE_URL")
            or model_defaults.get("base_url")
            or "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
        ).rstrip("/")
        self.reranker_api_key = reranker_api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        self.reranker_timeout_seconds = float(
            reranker_timeout_seconds or os.getenv("OMNIINFRA_RERANKER_TIMEOUT_SECONDS", "60")
        )
        self.device = device or os.getenv("OMNIINFRA_RETRIEVAL_DEVICE", "auto")
        self.embedding_batch_size = max(
            1,
            int(embedding_batch_size or os.getenv("OMNIINFRA_EMBEDDING_BATCH_SIZE", "16")),
        )
        self.reranker_batch_size = max(
            1,
            int(reranker_batch_size or os.getenv("OMNIINFRA_RERANKER_BATCH_SIZE", "16")),
        )
        self.max_length = max(128, int(max_length or os.getenv("OMNIINFRA_RETRIEVAL_MAX_LENGTH", "1024")))
        self.local_files_only = local_files_only
        self.instruction = instruction
        self.embedding_cache = (
            QwenEmbeddingCache(model=self.embedding_model_name) if self.embedding_backend == "api" else None
        )
        self._query_embedding_cache: dict[str, object] = {}
        self._embedding_tokenizer = None
        self._embedding_model = None
        self._reranker_pair_cache: dict[tuple[str, str], float] = {}
        self.last_reranker_stats: dict[str, int | str] = {
            "api_calls": 0,
            "documents": 0,
            "cache_hits": 0,
            "model": self.reranker_model_name,
        }

    @staticmethod
    def _model_load_error(kind: str, model_name: str) -> RuntimeError:
        env_name = "OMNIINFRA_EMBEDDING_MODEL_PATH" if kind == "embedding" else "OMNIINFRA_RERANKER_MODEL"
        return RuntimeError(
            f"Unable to load the Qwen {kind} backend for {model_name!r}. "
            f"Configure {env_name} and the corresponding API credentials."
        )

    def _load_embedding_model(self):
        if self._embedding_model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                self.embedding_model_name,
                padding_side="left",
                local_files_only=self.local_files_only,
            )
            device = "cuda:0" if self.device == "auto" and torch.cuda.is_available() else self.device
            if device == "auto":
                device = "cpu"
            dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else None
            model = AutoModel.from_pretrained(
                self.embedding_model_name,
                torch_dtype=dtype,
                local_files_only=self.local_files_only,
            ).eval()
            model.to(device)
        except Exception as error:
            raise self._model_load_error("embedding", self.embedding_model_name) from error
        self._embedding_tokenizer = tokenizer
        self._embedding_model = model

    def _read_reranker_api_key(self) -> str:
        if self.reranker_api_key:
            return self.reranker_api_key
        credential_paths = (
            Path("qwen_apikey.txt"),
            Path(__file__).resolve().parents[2] / "qwen_apikey.txt",
            Path(__file__).resolve().parents[3] / "qwen_apikey.txt",
        )
        for path in credential_paths:
            if not path.is_file():
                continue
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if line.startswith(("DASHSCOPE_API_KEY=", "QWEN_API_KEY=", "API_KEY=")):
                    return line.split("=", 1)[1].strip()
                if (
                    line
                    and not line.startswith("#")
                    and "=" not in line
                    and not line.startswith(("http://", "https://"))
                ):
                    return line
        raise ValueError("Qwen reranker API key not found; set DASHSCOPE_API_KEY or QWEN_API_KEY")

    def _rerank_api(self, query: str, documents: list[str]) -> list[float]:
        """Call the configured Qwen rerank API and restore scores to input order."""
        if self.reranker_api_style == "qwen-compatible":
            payload = {
                "model": self.reranker_model_name,
                "documents": documents,
                "query": query,
                "top_n": len(documents),
                "instruct": self.instruction,
            }
        else:
            payload = {
                "model": self.reranker_model_name,
                "input": {"query": query, "documents": documents},
                "parameters": {"return_documents": False, "top_n": len(documents)},
            }
        response = requests.post(
            self.reranker_base_url,
            headers={
                "Authorization": f"Bearer {self._read_reranker_api_key()}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.reranker_timeout_seconds,
        )
        try:
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            detail = getattr(response, "text", "")[:500]
            raise RuntimeError(f"Qwen reranker API request failed: {detail}") from exc

        output = body.get("output") if isinstance(body, dict) else None
        results = output.get("results") if isinstance(output, dict) else None
        if results is None and isinstance(body, dict):
            results = body.get("results") or body.get("data")
        if not isinstance(results, list):
            raise RuntimeError("Qwen reranker API returned no results list")
        scores = [None] * len(documents)
        for position, result in enumerate(results):
            if not isinstance(result, dict):
                continue
            index = result.get("index", position)
            score = result.get("relevance_score", result.get("score"))
            if score is None:
                continue
            try:
                index = int(index)
                if 0 <= index < len(scores):
                    scores[index] = float(score)
            except (TypeError, ValueError):
                continue
        if any(score is None for score in scores):
            raise RuntimeError(
                f"Qwen reranker API returned {len(results)} usable scores for {len(documents)} documents"
            )
        return [float(score) for score in scores]

    def _encode(self, texts: list[str], *, is_query: bool):
        import torch
        import torch.nn.functional as functional

        if self.embedding_backend == "api":
            assert self.embedding_cache is not None
            encoded_texts = texts
            if is_query:
                encoded_texts = [f"Instruct: {self.instruction}\nQuery:{text}" for text in texts]
                uncached = [text for text in encoded_texts if text not in self._query_embedding_cache]
                if uncached:
                    vectors = self.embedding_cache.embed_texts(uncached)
                    for text, vector in zip(uncached, vectors, strict=True):
                        self._query_embedding_cache[text] = torch.tensor(vector, dtype=torch.float32)
                return torch.stack([self._query_embedding_cache[text] for text in encoded_texts])
            vectors = self.embedding_cache.ensure_documents(encoded_texts)
            return torch.tensor(vectors, dtype=torch.float32)

        self._load_embedding_model()
        if is_query:
            texts = [f"Instruct: {self.instruction}\nQuery:{text}" for text in texts]
        embeddings = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.embedding_batch_size):
                batch = texts[start : start + self.embedding_batch_size]
                inputs = self._embedding_tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self._embedding_model.device) for key, value in inputs.items()}
                outputs = self._embedding_model(**inputs)
                attention_mask = inputs["attention_mask"]
                if bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item()):
                    pooled = outputs.last_hidden_state[:, -1]
                else:
                    sequence_lengths = attention_mask.sum(dim=1) - 1
                    batch_indices = torch.arange(
                        outputs.last_hidden_state.shape[0],
                        device=outputs.last_hidden_state.device,
                    )
                    pooled = outputs.last_hidden_state[batch_indices, sequence_lengths]
                embeddings.append(functional.normalize(pooled, p=2, dim=1).float().cpu())
        return torch.cat(embeddings, dim=0)

    def embedding_scores(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        query_embedding = self._encode([query], is_query=True)
        document_embeddings = self._encode(documents, is_query=False)
        return (query_embedding @ document_embeddings.T)[0].tolist()

    def semantic_rank(self, query: str, documents: list[str]) -> list[tuple[float, int]]:
        """Return embedding-only scores and original indices for lightweight MCP search."""
        scores = self.embedding_scores(query, documents)
        return sorted(((score, index) for index, score in enumerate(scores)), key=lambda item: (-item[0], item[1]))

    def reranker_scores(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        formatted_query = f"Instruct: {self.instruction}\nQuery: {query}"
        scores: list[float] = [0.0] * len(documents)
        missing_documents: list[str] = []
        missing_indices: list[int] = []
        cache_hits = 0
        for index, document in enumerate(documents):
            cached = self._reranker_pair_cache.get((formatted_query, document))
            if cached is None:
                missing_documents.append(document)
                missing_indices.append(index)
            else:
                scores[index] = cached
                cache_hits += 1
        api_calls = 0
        for start in range(0, len(missing_documents), self.reranker_batch_size):
            batch_documents = missing_documents[start : start + self.reranker_batch_size]
            batch_indices = missing_indices[start : start + self.reranker_batch_size]
            batch_scores = self._rerank_api(formatted_query, batch_documents)
            api_calls += 1
            for index, document, score in zip(batch_indices, batch_documents, batch_scores, strict=True):
                self._reranker_pair_cache[(formatted_query, document)] = score
                scores[index] = score
        self.last_reranker_stats = {
            "api_calls": api_calls,
            "documents": len(documents),
            "cache_hits": cache_hits,
            "model": self.reranker_model_name,
        }
        return scores


class ToolRetriever:
    """Retrieve tools from the tool registry."""

    DEFAULT_CATEGORY_LIMITS = {
        "tools": 16,
        "data_lake": 6,
        "libraries": 6,
        "know_how": 2,
    }
    DEFAULT_SEMANTIC_CANDIDATE_LIMITS = {
        "tools": 48,
        "data_lake": 18,
        "libraries": 18,
        "know_how": 6,
    }
    DEFAULT_CATALOG_CHAR_BUDGET = 16000
    DESCRIPTION_CHAR_LIMIT = 600

    def __init__(
        self,
        category_limits: dict[str, int] | None = None,
        semantic_candidate_limits: dict[str, int] | None = None,
        catalog_char_budget: int = DEFAULT_CATALOG_CHAR_BUDGET,
        semantic_backend=None,
    ):
        self.category_limits = {**self.DEFAULT_CATEGORY_LIMITS, **(category_limits or {})}
        self.semantic_candidate_limits = {
            **self.DEFAULT_SEMANTIC_CANDIDATE_LIMITS,
            **(semantic_candidate_limits or {}),
        }
        self.catalog_char_budget = max(1000, int(catalog_char_budget))
        self.semantic_backend = semantic_backend or QwenSemanticResourceRanker()
        self.last_retrieval_diagnostics: dict = {}

    @staticmethod
    def _resource_name_and_description(resource, index: int) -> tuple[str, str]:
        if isinstance(resource, dict):
            return str(resource.get("name", f"Resource {index}")), str(resource.get("description", ""))
        if isinstance(resource, str):
            return resource, ""
        return str(getattr(resource, "name", resource)), str(getattr(resource, "description", ""))

    @classmethod
    def _resource_document(cls, resource, index: int, category: str) -> str:
        name, description = cls._resource_name_and_description(resource, index)
        readable_name = name.replace("_", " ")
        document = (
            f"Resource category: {category}.\n"
            f"Resource name: {name} ({readable_name}).\n"
            f"Description: {description[: cls.DESCRIPTION_CHAR_LIMIT]}"
        )
        if isinstance(resource, dict):
            parameters = resource.get("required_parameters", []) + resource.get("optional_parameters", [])
            parameter_details = []
            for parameter in parameters:
                if isinstance(parameter, dict):
                    parameter_details.append(f"{parameter.get('name', '')}: {parameter.get('description', '')}")
            if parameter_details:
                document += "\nInputs: " + "; ".join(parameter_details)[: cls.DESCRIPTION_CHAR_LIMIT]
        return document

    @staticmethod
    def _reranker_queries(query: str) -> list[str]:
        """Split explicit multi-step tasks so one tool need only match one step."""
        step_matches = list(re.finditer(r"(?im)^\s*step\s+\d+\s*[.:]", query))
        if len(step_matches) < 2:
            return [query]
        return [
            query[match.start() : step_matches[index + 1].start()].strip()
            if index + 1 < len(step_matches)
            else query[match.start() :].strip()
            for index, match in enumerate(step_matches)
        ]

    def local_retrieval(
        self,
        query: str,
        resources: dict,
        *,
        category_limits: dict[str, int] | None = None,
        catalog_char_budget: int | None = None,
    ) -> dict:
        """Retrieve a bounded shortlist with API embeddings and Qwen reranking."""
        limits = {**self.category_limits, **(category_limits or {})}
        requested_budget = self.catalog_char_budget if catalog_char_budget is None else catalog_char_budget
        remaining_chars = max(0, int(requested_budget))
        ranked: dict[str, list] = {}
        diagnostics = {
            "embedding_model": getattr(
                self.semantic_backend, "embedding_model_name", type(self.semantic_backend).__name__
            ),
            "reranker_model": getattr(
                self.semantic_backend, "reranker_model_name", type(self.semantic_backend).__name__
            ),
            "reranker_backend": getattr(
                self.semantic_backend, "reranker_backend", type(self.semantic_backend).__name__
            ),
            "reranker_api_style": getattr(self.semantic_backend, "reranker_api_style", None),
            "reranker_errors": [],
            "categories": {},
        }

        for category in self.DEFAULT_CATEGORY_LIMITS:
            candidates = resources.get(category, [])
            documents = [
                self._resource_document(resource, index, category) for index, resource in enumerate(candidates)
            ]
            embedding_scores = self.semantic_backend.embedding_scores(query, documents)
            if len(embedding_scores) != len(candidates):
                raise ValueError(
                    f"Embedding backend returned {len(embedding_scores)} scores for {len(candidates)} resources"
                )
            embedding_ranked = sorted(
                zip(embedding_scores, range(len(candidates)), candidates, documents, strict=True),
                key=lambda item: (-item[0], item[1]),
            )[: max(0, int(self.semantic_candidate_limits.get(category, 0)))]
            semantic_documents = [item[3] for item in embedding_ranked]
            try:
                scores_by_query = [
                    self.semantic_backend.reranker_scores(subquery, semantic_documents)
                    for subquery in self._reranker_queries(query)
                ]
            except Exception as error:
                # Keep retrieval usable when the remote reranker is temporarily
                # unavailable.  The API path remains the primary backend; this
                # explicit fallback only reuses already-computed embedding scores.
                diagnostics["reranker_errors"].append({"category": category, "error": str(error)})
                scores_by_query = [[item[0] for item in embedding_ranked]]
            reranker_scores = [max(scores) for scores in zip(*scores_by_query, strict=True)] if scores_by_query else []
            if len(reranker_scores) != len(embedding_ranked):
                raise ValueError(
                    f"Reranker backend returned {len(reranker_scores)} scores for {len(embedding_ranked)} resources"
                )
            raw_reranked = sorted(
                zip(reranker_scores, embedding_ranked, strict=True),
                key=lambda item: (-item[0], -item[1][0], item[1][1]),
            )
            reranker_rank_by_index = {item[1][1]: rank for rank, item in enumerate(raw_reranked, start=1)}
            reranker_score_by_index = {item[1][1]: item[0] for item in raw_reranked}
            embedding_rank_by_index = {item[1]: rank for rank, item in enumerate(embedding_ranked, start=1)}

            # Reciprocal-rank fusion prevents a cross-encoder mistake on one
            # subtask from erasing a very strong embedding hit. Both semantic
            # stages therefore contribute to the final bounded order.
            fusion_k = 60
            fused_ranked = sorted(
                embedding_ranked,
                key=lambda item: (
                    -(
                        1 / (fusion_k + embedding_rank_by_index[item[1]])
                        + 1 / (fusion_k + reranker_rank_by_index[item[1]])
                    ),
                    item[1],
                ),
            )
            final_rank_by_index = {item[1]: rank for rank, item in enumerate(fused_ranked, start=1)}
            ranked[category] = [item[2] for item in fused_ranked[: max(0, int(limits.get(category, 0)))]]
            diagnostics["categories"][category] = [
                {
                    "name": self._resource_name_and_description(resource, original_index)[0],
                    "embedding_score": float(score),
                    "embedding_rank": embedding_rank,
                    "reranker_score": float(reranker_score_by_index[original_index]),
                    "reranker_rank": reranker_rank_by_index[original_index],
                    "final_rank": final_rank_by_index[original_index],
                }
                for embedding_rank, (score, original_index, resource, _document) in enumerate(
                    embedding_ranked,
                    start=1,
                )
            ]
        self.last_retrieval_diagnostics = diagnostics

        selected = {category: [] for category in self.DEFAULT_CATEGORY_LIMITS}
        queues = {category: list(items) for category, items in ranked.items()}
        made_progress = True
        while made_progress and remaining_chars > 0:
            made_progress = False
            for category in self.DEFAULT_CATEGORY_LIMITS:
                if not queues[category]:
                    continue
                resource = queues[category].pop(0)
                rendered = self._format_resources_for_prompt([resource])
                cost = len(rendered) + 1
                if cost <= remaining_chars:
                    selected[category].append(resource)
                    remaining_chars -= cost
                    made_progress = True

        return selected

    def prompt_based_retrieval(
        self,
        query: str,
        resources: dict,
        llm=None,
        *,
        catalog_char_budget: int | None = None,
        max_output_tokens: int = 512,
    ) -> dict:
        """Use a prompt-based approach to retrieve the most relevant resources for a query.

        Args:
            query: The user's query
            resources: A dictionary with keys 'tools', 'data_lake', 'libraries', and 'know_how',
                      each containing a list of available resources
            llm: Optional LLM instance to use for retrieval (if None, will create a new one)

        Returns:
            A dictionary with the same keys, but containing only the most relevant resources

        """
        # Always perform embedding recall and API reranking first. The
        # previous implementation formatted the complete catalog here, so
        # "retrieval" itself was the first context-overflowing model request.
        candidate_resources = self.local_retrieval(
            query,
            resources,
            catalog_char_budget=catalog_char_budget,
        )
        if not any(candidate_resources.values()):
            return candidate_resources

        # Build prompt sections for the bounded candidate resources.
        prompt_sections = []
        prompt_sections.append(f"""
You are an expert biomedical research assistant. Your task is to select the relevant resources to help answer a user's query.

USER QUERY: {query}

Below are the available resources. For each category, select items that are directly or indirectly relevant to answering the query.
Be generous in your selection - include resources that might be useful for the task, even if they're not explicitly mentioned in the query.
It's better to include slightly more resources than to miss potentially useful ones.

AVAILABLE TOOLS:
{self._format_resources_for_prompt(candidate_resources.get("tools", []))}

AVAILABLE DATA LAKE ITEMS:
{self._format_resources_for_prompt(candidate_resources.get("data_lake", []))}

AVAILABLE SOFTWARE LIBRARIES:
{self._format_resources_for_prompt(candidate_resources.get("libraries", []))}""")

        # Add know-how section if available
        if candidate_resources.get("know_how"):
            prompt_sections.append(f"""
AVAILABLE KNOW-HOW DOCUMENTS (Best Practices & Protocols):
{self._format_resources_for_prompt(candidate_resources.get("know_how", []))}""")

        # Build response format based on available categories
        response_format = """
For each category, respond with ONLY the indices of the relevant items in the following format:
TOOLS: [list of indices]
DATA_LAKE: [list of indices]
LIBRARIES: [list of indices]"""

        if candidate_resources.get("know_how"):
            response_format += "\nKNOW_HOW: [list of indices]"

        response_format += """

For example:
TOOLS: [0, 3, 5, 7, 9]
DATA_LAKE: [1, 2, 4]
LIBRARIES: [0, 2, 4, 5, 8]"""

        if candidate_resources.get("know_how"):
            response_format += "\nKNOW_HOW: [0, 1]"

        response_format += """

If a category has no relevant items, use an empty list, e.g., DATA_LAKE: []

IMPORTANT GUIDELINES:
1. Be generous but not excessive - aim to include all potentially relevant resources
2. ALWAYS prioritize database tools for general queries - include as many database tools as possible
3. Include all literature search tools
4. For wet lab sequence type of queries, ALWAYS include molecular biology tools
5. For data lake items, include datasets that could provide useful information
6. For libraries, include those that provide functions needed for analysis
7. For know-how documents, include those that provide relevant protocols, best practices, or troubleshooting guidance
8. Don't exclude resources just because they're not explicitly mentioned in the query
9. When in doubt about a database tool or molecular biology tool, include it rather than exclude it

FINAL RESPONSE CONTRACT:
Return only the category lines below, with no rationale, Markdown, or text before or after them.
Put every resource needed by any explicit step in the corresponding list. Do not stop after
describing the first step. The first line must be the complete TOOLS list.
"""

        prompt = "\n".join(prompt_sections) + response_format

        # Use the provided LLM or create a new one
        if llm is None:
            # Retrieval's optional semantic selector follows the same central
            # tool model as database/genomics helpers; never silently fall back
            # to an unrelated OpenAI model.
            llm = get_llm(model=default_config.tool_llm, source=default_config.tool_source)

        # Invoke the LLM
        if hasattr(llm, "invoke"):
            # For LangChain-style LLMs
            retrieval_llm = llm.bind(max_tokens=max(1, int(max_output_tokens))) if hasattr(llm, "bind") else llm
            response = retrieval_llm.invoke([HumanMessage(content=prompt)])
            response_content = response.content
        else:
            # For other LLM interfaces
            response_content = str(llm(prompt))

        # Parse the response to extract the selected indices
        selected_indices = self._parse_llm_response(response_content)

        # A malformed or empty model response must not trigger a full-catalog
        # fallback. The semantic/reranked shortlist is already bounded.
        if not any(selected_indices.values()):
            return candidate_resources

        # Get the selected resources
        selected_resources = {
            "tools": [
                candidate_resources["tools"][i]
                for i in selected_indices.get("tools", [])
                if 0 <= i < len(candidate_resources.get("tools", []))
            ],
            "data_lake": [
                candidate_resources["data_lake"][i]
                for i in selected_indices.get("data_lake", [])
                if 0 <= i < len(candidate_resources.get("data_lake", []))
            ],
            "libraries": [
                candidate_resources["libraries"][i]
                for i in selected_indices.get("libraries", [])
                if 0 <= i < len(candidate_resources.get("libraries", []))
            ],
        }

        # Add know-how if present
        if candidate_resources.get("know_how"):
            selected_resources["know_how"] = [
                candidate_resources["know_how"][i]
                for i in selected_indices.get("know_how", [])
                if 0 <= i < len(candidate_resources.get("know_how", []))
            ]

        return selected_resources

    def _format_resources_for_prompt(self, resources: list) -> str:
        """Format resources for inclusion in the prompt."""
        formatted = []
        for i, resource in enumerate(resources):
            if isinstance(resource, dict):
                # Handle dictionary format (from tool registry or data lake/libraries with descriptions)
                name = resource.get("name", f"Resource {i}")
                description = str(resource.get("description", ""))[: self.DESCRIPTION_CHAR_LIMIT]
                formatted.append(f"{i}. {name}: {description}")
            elif isinstance(resource, str):
                # Handle string format (simple strings)
                formatted.append(f"{i}. {resource}")
            else:
                # Try to extract name and description from tool objects
                name = getattr(resource, "name", str(resource))
                desc = getattr(resource, "description", "")
                formatted.append(f"{i}. {name}: {desc}")

        return "\n".join(formatted) if formatted else "None available"

    def _parse_llm_response(self, response) -> dict:
        """Parse the LLM response to extract the selected indices.

        Accepts either a plain string or a Responses API-style list of content blocks.
        """
        # Normalize response to string if it's a list of content blocks (Responses API)
        if isinstance(response, list):
            parts = []
            for item in response:
                # LangChain Responses API returns list of dicts like {"type": "text", "text": "..."}
                if isinstance(item, dict):
                    if item.get("type") == "text" and "text" in item:
                        parts.append(str(item.get("text", "")))
                    # If it's a tool_call or other block, ignore for this simple parsing
                elif isinstance(item, str):
                    parts.append(item)
            response = "\n".join([p for p in parts if p])
        elif not isinstance(response, str):
            response = str(response)
        selected_indices = {"tools": [], "data_lake": [], "libraries": [], "know_how": []}

        # Models occasionally emit an initial partial category line and a later corrected
        # line in their rationale. Use the last valid occurrence so a later complete
        # contract is not shadowed by an incomplete first line.
        def parse_last_indices(pattern: str) -> list[int]:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for body in reversed(matches):
                if not body.strip():
                    continue
                with contextlib.suppress(ValueError):
                    return [int(idx.strip()) for idx in body.split(",") if idx.strip()]
            return []

        selected_indices["tools"] = parse_last_indices(r"TOOLS:\s*\[(.*?)\]")
        selected_indices["data_lake"] = parse_last_indices(r"DATA_LAKE:\s*\[(.*?)\]")
        selected_indices["libraries"] = parse_last_indices(r"LIBRARIES:\s*\[(.*?)\]")
        selected_indices["know_how"] = parse_last_indices(r"KNOW[-_]HOW:\s*\[(.*?)\]")

        return selected_indices
