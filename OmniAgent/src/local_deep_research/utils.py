"""
tool function
"""

import ast
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from colorlog import ColoredFormatter
from fuzzywuzzy import fuzz


biological_entities = [
    "Small molecule",
    "Drug/Drug class",
    "Protein/Gene",
    "Therapeutic target",
    "RNA",
    "Amino acid",
    "Biomarker",
    "Molecular function",
    "Biological process",
    "Pathway",
    "Mutation",
    "Pharmacology/Toxicology",
    "Cell type",
    "Cell line",
    "Cellular component",
    "Tissue/Organ",
    "Organism/Species",
    "Disease",
    "Phenotype",
    "Assay",
    "Cancer type",
    "Clinical",
    "Others",
]


def extract_and_convert_dict(text):
    """
    A more robust dictionary extraction function that supports multiple formats and error handling

    Args:
        text (str): Text string containing a dictionary

    Returns:
        dict or None: The extracted dictionary, returns None if extraction fails
    """

    json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"

    # Find all possible dictionary matches
    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        # Clean the string
        dict_str = match.strip()
        parsers = [
            ast.literal_eval,  # Safest method
            eval,  # Last resort (Note: eval has security risks)
        ]

        for parser in parsers:
            try:
                if parser == eval:
                    # Add some basic security checks for eval
                    if any(
                        dangerous in dict_str
                        for dangerous in [
                            "import",
                            "__",
                            "exec",
                            "eval",
                            "open",
                            "file",
                        ]
                    ):
                        continue

                result = parser(dict_str)
                if isinstance(result, dict):
                    return result
            except:
                continue

    return None


def exact_match_entity_type(entity_type, entity_list=biological_entities, threshold=60):
    """
    Match the entity type returned by the large model to the most similar category in the predefined biological entity category list.

    Args:
        entity_type (str): The entity type returned by the large model.
        entity_list (list): Predefined list of biological entity categories, defaults to biological_entities.
        threshold (int): Similarity threshold (0-100), returns 'Others' if below threshold, defaults to 80.

    Returns:
        str: The most similar entity category name, returns 'Others' if no match found.
    """
    if entity_type in entity_list:
        return entity_type

    if not entity_type or not isinstance(entity_type, str):
        print(f"Invalid entity type: {entity_type}, returning 'Others'")
        return "Others"

    # Convert to lowercase for case-insensitive matching
    entity_type = entity_type.lower().strip()

    # Initialize highest similarity and corresponding category
    max_similarity = 0
    best_match = "Others"

    # Iterate through predefined category list
    for category in entity_list:
        category_lower = category.lower()
        similarity = fuzz.partial_ratio(entity_type, category_lower)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = category  # Keep original category name (preserve case)
        # If exact match, return immediately
        if entity_type == category_lower:
            return category

    # Check if threshold is met
    if max_similarity >= threshold:
        return best_match
    else:
        print(
            f"Entity type '{entity_type}' not found (highest similarity {max_similarity} < {threshold}), returning 'Others'"
        )
        return "Others"


query = 'Which KRAS G12C inhibitors are currently in clinical development (as of 2024)? {"A": "Sotorasib", "B": "Adagrasib", "C": "Divarasib", "D": "MRTX1133"}'
suitable_tools = [
    ("get_clinical_studies_info_by_drug_name", '{"drug_name": "Sotorasib"}'),
    ("get_clinical_studies_info_by_drug_name", '{"drug_name": "Adagrasib"}'),
    ("get_clinical_studies_info_by_drug_name", '{"drug_name": "Divarasib"}'),
    ("get_clinical_studies_info_by_drug_name", '{"drug_name": "MRTX1133"}'),
    ("get_general_info_by_compound_name", '{"name": "Sotorasib"}'),
    ("get_general_info_by_compound_name", '{"name": "Adagrasib"}'),
    ("get_general_info_by_compound_name", '{"name": "Divarasib"}'),
    ("get_general_info_by_compound_name", '{"name": "MRTX1133"}'),
    ("get_general_info_by_protein_or_gene_name", '{"name": "KRAS"}'),
    ("get_general_info_by_protein_or_gene_name", '{"name": "KRAS"'),
    ("tavily_search", '{"query": "KRAS G12C inhibitors"}'),
]


def generate_tools_descriptions(tool_list):
    all_tool_desc = ""
    for tool in tool_list:
        tool_name = tool.name
        tool_desc = tool.description
        tool_args_schema = tool.args_schema
        description = (
            "#" * 20
            + f"\nTool Name: {tool_name}\nTool Purpose: {tool_desc}\nTool Input Schema: {tool_args_schema}\n\n"
        )
        all_tool_desc += description
    return all_tool_desc


class ResearchLogger:
    """"""

    def __init__(self, name: str = "research", debug_mode: bool = False):
        self.logger = logging.getLogger(name)
        self.debug_mode = debug_mode
        self._setup_logger()

    def _setup_logger(self):
        """"""
        if self.logger.handlers:
            return  


        level = logging.DEBUG if self.debug_mode else logging.INFO
        self.logger.setLevel(level)


        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)


        colored_formatter = ColoredFormatter(
            "%(log_color)s%(asctime)s [%(name)s] %(levelname)s: %(message)s%(reset)s",
            datefmt="%H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
        console_handler.setFormatter(colored_formatter)

        if self.debug_mode:
            log_dir = Path("dist/logs")
            log_dir.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(
                log_dir / f"research_{datetime.now().strftime('%Y%m%d')}.log"
            )
            file_handler.setLevel(logging.DEBUG)

            file_formatter = logging.Formatter(
                "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        self.logger.addHandler(console_handler)

    def debug(self, message: str, **kwargs):

        self.logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs):

        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs):

        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs):

        self.logger.error(message, extra=kwargs)

    def progress(self, message: str, progress: Optional[int] = None, **kwargs):

        progress_text = f" ({progress}%)" if progress is not None else ""
        self.logger.info(f"🔄 {message}{progress_text}", extra=kwargs)

    def result(self, message: str, count: int = 0, **kwargs):

        self.logger.info(f"📊 {message} (count: {count})", extra=kwargs)

    def tool_call(self, tool_name: str, query: str, **kwargs):

        self.logger.info(
            f"🔧 {tool_name}: {query[:50]}{'...' if len(query) > 50 else ''}",
            extra=kwargs,
        )

def detect_content_type(url: str) -> str:
    url_lower = url.lower()

    if any(
        url_lower.endswith(ext)
        for ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp"]
    ):
        return "image"
    elif any(
        url_lower.endswith(ext)
        for ext in [".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm", ".mkv"]
    ):
        return "video"
    elif url_lower.endswith(".pdf"):
        return "pdf"
    else:
        return "iframe"


def clean_text_format(text: str) -> str:
    if not text:
        return ""

    text = str(text).replace("\\n", "\n").replace("\\t", " ")

    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def extract_json_from_response(response_text: str) -> Optional[dict]:
    try:
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1

        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        pass

    return None


research_logger = ResearchLogger("research", debug_mode=False)



def log_debug(message: str, **kwargs):
    research_logger.debug(message, **kwargs)


def log_info(message: str, **kwargs):
    research_logger.info(message, **kwargs)


def log_warning(message: str, **kwargs):
    research_logger.warning(message, **kwargs)


def log_error(message: str, **kwargs):
    research_logger.error(message, **kwargs)


def log_progress(message: str, progress: Optional[int] = None, **kwargs):
    research_logger.progress(message, progress, **kwargs)


def log_tool_call(tool_name: str, query: str, **kwargs):
    research_logger.tool_call(tool_name, query, **kwargs)
