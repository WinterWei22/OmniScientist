import asyncio
import json
import logging
import os
import urllib.parse
from datetime import datetime
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI

from .utils import exact_match_entity_type, extract_and_convert_dict
from .utilties.search_utilities import (
    write_log_process_safe, write_json_log_process_safe
)

logger = logging.getLogger(__name__)


def convert_search_patent_input(tool_input: dict) -> dict:
    try:
        if "field" in tool_input and "value" in tool_input:
            logger.info(f"search_patent parameters have been correctly formatted, no conversion needed: {tool_input}")
            return tool_input

        query = tool_input.get("query", "")
        if not query:
            if "target" in tool_input:
                query = tool_input["target"]
            elif "patent_name" in tool_input:
                query = tool_input["patent_name"]
            elif "related_diseases" in tool_input:
                query = tool_input["related_diseases"]
            elif "moa" in tool_input:
                query = tool_input["moa"]
            else:
                logger.warning("Missing query content for search_patent tool")
                return tool_input

        processed_input = {"field": "target", "value": query}
        logger.info(f"search_patent parameters intelligent conversion completed: {tool_input} -> {processed_input}")
        return processed_input

    except Exception as e:
        logger.error(f"Error occurred while processing search_patent parameters: {e}")
        return tool_input


tool_input_convert_map = {
    "search_patent": convert_search_patent_input,
    # Normalize parameter naming for OpenTargets-like target tools
    # (These MCP tools expect `target_name`, but the selector sometimes emits `name` or `query` or a raw string.)
    "get_target_gene_ontology_by_name": lambda x: {"target_name": x.get("target_name") or x.get("name") or x.get("query") or ""} if isinstance(x, dict) else {"target_name": str(x)},
    "get_target_classes_by_name": lambda x: {"target_name": x.get("target_name") or x.get("name") or x.get("query") or ""} if isinstance(x, dict) else {"target_name": str(x)},
    "get_associated_diseases_phenotypes_by_target_name": lambda x: {"target_name": x.get("target_name") or x.get("name") or x.get("query") or ""} if isinstance(x, dict) else {"target_name": str(x)},
    # Normalize ontology tool input. Avoid wildcard patterns that can trigger 400 in Ensembl REST.
    "get_ontology_name": lambda x: (
        {"name": (x.get("name") or x.get("query") or x.get("term") or "").replace("*", "").strip()}
        if isinstance(x, dict)
        else {"name": str(x).replace("*", "").strip()}
    ),
}


class ToolExecutor:
    """
    Execute the tools according to the tool calling input.
    """

    def __init__(self, mcp_client, error_log_path, llm_light, max_concurrent: int = 6):
        self.mcp_client = mcp_client
        self.tool_map = mcp_client.mcp_tool_map
        self.error_log_path = error_log_path
        self.execution_failed_log_path = os.path.join(os.path.dirname(error_log_path), "execution_failed_tools.json")
        self.llm_light = llm_light
        self.semaphore = asyncio.Semaphore(max_concurrent)

    def _normalize_tool_result(self, result):
        if isinstance(result, list) and len(result) == 1:
            item = result[0]
            text = None
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
            elif hasattr(item, "text"):
                text = item.text
            if isinstance(text, str):
                try:
                    return json.loads(text)
                except Exception:
                    return result
        return result
        
    async def execute_tool_with_timeout(self, tool_invoke_info, timeout=150.0, max_retries=2):
        for attempt in range(max_retries):
            try:
                tool_calling_result = await asyncio.wait_for(
                    self.execute_tool(tool_invoke_info),
                    timeout=timeout
                )
                return tool_calling_result
            except asyncio.TimeoutError:
                print(f"Attempt {attempt + 1}/{max_retries} timed out when executing tool after {timeout}s")
                log_msg = f"[{datetime.now().isoformat()}] Attempt {attempt + 1}/{max_retries} timed out when executing tool after {timeout}s. Tool_invoke_info: {tool_invoke_info}"
                write_log_process_safe(self.error_log_path, log_msg)
                if attempt < max_retries - 1:
                    print(f"Retrying {attempt + 1}/{max_retries}...")
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed when executing tool with error: {e}")
                log_msg = f"[{datetime.now().isoformat()}] Attempt {attempt + 1}/{max_retries} failed when executing tool with error: {e}. Tool_invoke_info: {tool_invoke_info}"
                write_log_process_safe(self.error_log_path, log_msg)
                if attempt < max_retries - 1:
                    print(f"Retrying {attempt + 1}/{max_retries}...")
                else:
                    logging.error(f"Max retries reached when executing tool. Giving up. Tool_invoke_info: {tool_invoke_info}")
                    log_msg = f"[{datetime.now().isoformat()}] Max retries reached when executing tool. Giving up. Tool_invoke_info: {tool_invoke_info}. Error: {e}\n"
                    write_log_process_safe(self.error_log_path, log_msg)
                    tool_calling_result = {
                        "content": f"Error: Failed after {max_retries} attempts when executing tool: {e}. Tool_invoke_info: {tool_invoke_info}",
                        "tool_name": tool_invoke_info.get("tool", "Unknown"),
                        "toolsuite": None,
                        "success": False,
                    }
                    return tool_calling_result
        log_msg = f"[{datetime.now().isoformat()}] Max retries reached when executing tool. Giving up. Tool_invoke_info: {tool_invoke_info}"
        write_log_process_safe(self.error_log_path, log_msg)
        
        tool_calling_result = {
            "content": f"Error: Failed after {max_retries} attempts when executing tool. Tool_invoke_info: {tool_invoke_info}",
            "tool_name": tool_invoke_info.get("tool", "Unknown"),
            "toolsuite": None,
            "success": False,
        }
        
        return tool_calling_result

    async def execute_tool(self, tool_invoke_info):
        """
        Execute single tool.
        """
        tool_name = tool_invoke_info.get("tool", "")
        tool_input = tool_invoke_info.get("tool_input", {})

        # Best-effort normalization for known schema mismatches before calling MCP tools
        converter = tool_input_convert_map.get(tool_name)
        if converter:
            try:
                tool_input = converter(tool_input)
                tool_invoke_info = {**tool_invoke_info, "tool_input": tool_input}
            except Exception as e:
                logging.error(f"Tool input convert failed for {tool_name}: {e}. tool_input={tool_input}")

        # Guardrail: don't call tools with null/empty inputs (common LLM failure mode).
        if tool_input is None:
            raise ValueError(f"Tool input is None for {tool_name}")
        if isinstance(tool_input, str) and tool_input.strip().lower() in {"null", "none", ""}:
            raise ValueError(f"Tool input is empty/null string for {tool_name}")
        if isinstance(tool_input, dict):
            # Drop empty-string values
            cleaned = {k: v for k, v in tool_input.items() if not (isinstance(v, str) and v.strip() == "")}
            tool_input = cleaned
            tool_invoke_info = {**tool_invoke_info, "tool_input": tool_input}
            if len(tool_input) == 0:
                raise ValueError(f"Tool input is empty dict for {tool_name}")
        if isinstance(tool_input, list) and len(tool_input) == 0:
            raise ValueError(f"Tool input is empty list for {tool_name}")

        try:
            tool = self.tool_map.get(tool_name)
        except Exception as e:
            logging.error(f"Error accessing tool_map: {e}")

        if tool is None:
            logging.info(
                f"Tool {tool_name} not found in the tool pool(with {len(self.tool_map)} tools available)."
            )
        
        try:
            logging.info(f"Executing tool: {tool_name} with input: {tool_input}")
        except Exception as e:
            logging.error(f"Error logging tool execution: {e}")

        try:
            # Use the new call_tool method we added to OmniagentMCPToolClient
            result = await self.mcp_client.call_tool(tool_name, tool_input)
            result = self._normalize_tool_result(result)
        except Exception as e:
            logging.error(f"Error invoking tool {tool_name}, tool_input: {tool_input}, tool_invoke_info: {tool_invoke_info}, error: {e}")
            raise e

        try:
            tool_source = self.mcp_client.tool2source.get(tool_name, "Unknown")
        except Exception as e:
            logging.error(f"Error extracting tool_source: {e}")
            tool_source = "Unknown"

        try:
            tool_calling_result = {
                # IMPORTANT: str(list/dict) is not valid JSON (single quotes), and will break downstream parsers.
                # Persist a JSON string when possible so ToolResultParser can json.loads reliably.
                "content": (
                    json.dumps(result, ensure_ascii=False)
                    if isinstance(result, (dict, list))
                    else str(result)
                ),
                "tool_name": tool_name,
                "toolsuite": tool_source,
                "success": self.judge_output_is_meaningful(result)
            }
        except Exception as e:
            logging.error(f"Error building tool_calling_result: {e}")
            raise e

        try:
            tool_calling_result = self.extract_additional_info(tool_calling_result)
        except Exception as e:
            logging.error(f"Error extracting additional info: {e}")
            raise e

        return tool_calling_result

    def _preprocess_tool_calls(self, tool_invoke_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed_tool_invoke_list = []

        for tool_invoke_info in tool_invoke_list:
            processed_tool_invoke_info = {**tool_invoke_info}
            tool_name = tool_invoke_info.get("tool")
            if not tool_name:
                logger.error("tool name is None")
                continue
            tool_input = tool_invoke_info.get("tool_input")
            converter = tool_input_convert_map.get(tool_name)
            if converter:
                new_tool_input = converter(tool_input)
                processed_tool_invoke_info["tool_input"] = new_tool_input
                logger.info(f"{tool_name} Parameter preprocessing completed: {tool_input} -> {new_tool_input}")
            processed_tool_invoke_list.append(processed_tool_invoke_info)
        return processed_tool_invoke_list

    def extract_additional_info(self, tool_calling_result):
        try:
            result = tool_calling_result.get("content", {})
        
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except:
                    tool_calling_result["additional_info_type"] = "Others"
                    tool_calling_result["additional_info"] = ""
                    return tool_calling_result

            if isinstance(result, dict):
                if "SMILES" in result.keys() and result.get("SMILES"):
                    smiles = result["SMILES"]
                    info_type = "Compound"
                    info_content = smiles
                    info_urls = f"https://pubchem.ncbi.nlm.nih.gov/#query={urllib.parse.quote(smiles)}"
                elif "pdb_id" in result.keys() and result.get("pdb_id"):
                    pdb_id = result["pdb_id"]
                    info_type = "Protein"
                    info_content = pdb_id
                    info_urls = f"https://www.rcsb.org/structure/{urllib.parse.quote(pdb_id)}"
                elif tool_calling_result.get("tool_name") == "tavily_search":
                    content_urls = [each["url"] for each in result.get("results", [])]
                    info_type = "URL"
                    info_content = content_urls
                    info_urls = content_urls
                else:
                    info_type = "Others"
                    info_content = ""
                    info_urls = ""
            else:
                info_type = "Others"
                info_content = ""
                info_urls = ""

        except Exception as e:
            logging.error(f"Error extracting additional info: {e}")
            info_type = "Others"
            info_content = ""
            info_urls = ""

        tool_calling_result["additional_info_type"] = info_type
        tool_calling_result["additional_info"] = info_content
        tool_calling_result["additional_urls"] = info_urls

        return tool_calling_result
    
    def judge_output_is_meaningful(self, tool_response):
        if isinstance(tool_response, dict):
            if tool_response.get("ok") is True:
                return True
            result = tool_response.get("result")
            if isinstance(result, dict) and result.get("success") is True:
                return True
            if tool_response.get("success") is True:
                return True

        if not isinstance(tool_response, str):
            try:
                tool_response = str(tool_response)
            except Exception as e:
                log_msg = f"[{datetime.now().isoformat()}] Tool response is not a string: {tool_response}"
                write_log_process_safe(self.error_log_path, log_msg)
                return False
        
        prompt = f"""
# Your Task
You are a senior data scientist. I currently have an MCP service with many tools. Please help me determine whether the tool execution was successful based on the tool's output.

# Tool calling result
Top 500 tokens of the full content: {tool_response[:500]}

# Instructions
1. The returned result may be a JSON-formatted string (part of the full tool calling result), null, or an empty string.
2. If the returned result contains words like "error", "404", "400", or "No data found", it indicates that the request failed.

# Output Format
Return data in python dict format as follows:
```python
{{
    "success": [success status] # True if the tool ran successfully with substantial information returned, False if it failed
}}
```
"""
        response = self.llm_light.invoke(prompt)
        result = extract_and_convert_dict(response.content)
        
        if isinstance(result, dict) and "success" in result:
            return result["success"]
        else:
            return False

    def log_execution_failed_tool(self, tool_invoke_list, tool_calling_results):
        for tool_invoke_info, tool_calling_result in zip(tool_invoke_list, tool_calling_results):
            if tool_calling_result.get("success", False):
                continue
            
            error_info = tool_invoke_info.copy()
            error_info['tool_calling_result'] = tool_calling_result["content"]
            error_info['log_time'] = datetime.now().isoformat()
            write_json_log_process_safe(self.execution_failed_log_path, error_info)

    async def run(self, tool_invoke_list):
        """
        Execute the tools.
        """
        processed_list = self._preprocess_tool_calls(tool_invoke_list)

        async def _bounded_execute(info):
            async with self.semaphore:
                return await self.execute_tool_with_timeout(info)

        execute_tasks = [
            _bounded_execute(info) for info in processed_list
        ]
        tool_calling_results = await asyncio.gather(*execute_tasks)
        self.log_execution_failed_tool(processed_list, tool_calling_results)
        return tool_calling_results
