import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ToolResult:

    links: List[str]
    summary: str
    raw: Any
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

        self.metadata["timestamp"] = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "links": self.links,
            "summary": self.summary,
            "raw": self.raw,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }

    @classmethod
    def from_legacy_result(cls, legacy_result: Any) -> "ToolResult":
        if isinstance(legacy_result, dict):
            return cls(
                links=legacy_result.get("documents", []),
                summary=legacy_result.get("content", ""),
                raw=legacy_result,
                success=True,
            )
        elif isinstance(legacy_result, list):
            links = []
            summary_parts = []

            for item in legacy_result:
                if isinstance(item, dict):
                    if "link" in item:
                        links.append(item["link"])
                    if "content" in item or "summary" in item:
                        summary_parts.append(
                            item.get("content", item.get("summary", ""))
                        )
                elif isinstance(item, str):
                    if item.startswith("http"):
                        links.append(item)
                    else:
                        summary_parts.append(item)

            return cls(
                links=links,
                summary="\n".join(summary_parts),
                raw=legacy_result,
                success=True,
            )
        else:
            return cls(
                links=[], summary=str(legacy_result), raw=legacy_result, success=True
            )

    @classmethod
    def error_result(cls, error_message: str, raw_error: Any = None) -> "ToolResult":
        return cls(
            links=[],
            summary=f"Error: {error_message}",
            raw=raw_error,
            success=False,
            error_message=error_message,
        )


@dataclass
class ResearchContext:

    current_knowledge: str
    iteration_count: int
    max_context_length: int = 32000
    priority_keywords: List[str] = None

    def __post_init__(self):
        if self.priority_keywords is None:
            self.priority_keywords = []

    def get_token_count(self) -> int:
        return len(self.current_knowledge) // 4

    def needs_compression(self) -> bool:
        return self.get_token_count() > self.max_context_length * 0.8

    def add_knowledge(self, new_knowledge: str, priority: int = 1):
        self.current_knowledge += f"\n\n[Priority {priority}] {new_knowledge}"

        if self.needs_compression():
            self._compress_knowledge()

    def _compress_knowledge(self):
        paragraphs = self.current_knowledge.split("\n\n")

        scored_paragraphs = []
        for index, para in enumerate(paragraphs):
            score = 0

            if "[Priority 1]" in para:
                score += 10
            elif "[Priority 2]" in para:
                score += 5
            elif "[Priority 3]" in para:
                score += 2

            for keyword in self.priority_keywords:
                score += para.lower().count(keyword.lower()) * 2

            if index > len(paragraphs) * 0.7:
                score += 3

            scored_paragraphs.append((score, para))

        scored_paragraphs.sort(key=lambda x: x[0], reverse=True)
        target_tokens = int(self.max_context_length * 0.6)

        compressed_content = []
        current_tokens = 0

        for score, para in scored_paragraphs:
            paragraph_tokens = max(1, len(para) // 4)
            if current_tokens + paragraph_tokens > target_tokens:
                continue
            compressed_content.append(para)
            current_tokens += paragraph_tokens

        self.current_knowledge = "\n\n".join(compressed_content)
        self.current_knowledge += (
            f"\n\n[COMPRESSED] Content compressed at iteration {self.iteration_count}"
        )


@dataclass
class StructuredOutput:

    thoughts: str
    strategy: List[str]
    key_info: List[str]
    tool_queries: List[Dict[str, Any]]
    confidence: float = 0.0

    @classmethod
    def from_json(cls, json_str: str) -> "StructuredOutput":
        try:
            data = json.loads(json_str)
            return cls(
                thoughts=data.get("thoughts", ""),
                strategy=data.get("strategy", []),
                key_info=data.get("key_info", []),
                tool_queries=data.get("tool_queries", []),
                confidence=data.get("confidence", 0.0),
            )
        except json.JSONDecodeError as e:
            return cls(
                thoughts=f"JSON parsing error: {str(e)}",
                strategy=[],
                key_info=[],
                tool_queries=[],
                confidence=0.0,
            )

    def to_legacy_questions(self) -> List[tuple]:
        questions = []
        for query_data in self.tool_queries:
            questions.append(
                (
                    query_data.get("tool_index", 0),
                    query_data.get("tool_name", "unknown"),
                    query_data.get("query", ""),
                )
            )
        return questions
