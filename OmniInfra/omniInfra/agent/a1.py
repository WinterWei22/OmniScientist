import ast
import glob
import hashlib
import inspect
import json
import os
import re
import subprocess
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from omniInfra.agent.execution_policy import (
    A1RunControl,
    ExecutionAdmissionController,
    normalize_observation,
    solution_invariant_violation,
)
from omniInfra.config import default_config
from omniInfra.know_how import KnowHowLoader
from omniInfra.llm import SourceType, get_llm
from omniInfra.model.retriever import ToolRetriever
from omniInfra.tool.support_tools import run_python_repl
from omniInfra.tool.tool_registry import ToolRegistry
from omniInfra.utils import (
    check_and_download_s3_files,
    clean_message_content,
    convert_markdown_to_pdf,
    create_parsing_error_html,
    find_matching_execution,
    format_execute_tags_in_content,
    format_lists_in_text,
    format_observation_as_terminal,
    function_to_api_schema,
    has_execution_results,
    inject_custom_functions_to_repl,
    parse_tool_calls_from_code,
    parse_tool_calls_with_modules,
    pretty_print,
    read_module2api,
    run_bash_script,
    run_r_code,
    run_with_timeout,
    should_skip_message,
    textify_api_dict,
)

if os.path.exists(".env"):
    load_dotenv(".env", override=False)
    print("Loaded environment variables from .env")


class AgentState(TypedDict):
    messages: list[BaseMessage]
    next_step: str | None


class A1:
    # Tools backed by a provider-specific API must not be exposed to an agent
    # running on an incompatible model.  Otherwise the tool retriever can
    # repeatedly select a function which is guaranteed to fail at execution.
    _CLAUDE_ONLY_TOOLS = {"advanced_web_search_claude"}
    # Temporarily keep Claude-only tools out of every A1 run.  This is separate
    # from provider compatibility so removing the temporary block later restores
    # the original Claude behavior without changing the compatibility rule.
    _TEMPORARILY_DISABLED_TOOLS = {"advanced_web_search_claude"}

    def __init__(
        self,
        path: str | None = None,
        llm: str | None = None,
        source: SourceType | None = None,
        use_tool_retriever: bool | None = None,
        timeout_seconds: int | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        commercial_mode: bool | None = None,
        expected_data_lake_files: list | None = None,
        max_output_tokens: int | None = None,
        context_window_tokens: int | None = None,
        token_safety_margin: int | None = None,
        network_policy: str | None = None,
        max_execute_rounds: int = 16,
        max_tool_calls: int = 24,
        max_policy_rejections: int = 2,
        max_tool_recoveries: int = 1,
        max_endpoint_discoveries: int = 2,
        max_fallback_requests: int = 4,
        max_consecutive_no_evidence: int = 3,
        max_response_format_failures: int = 4,
        max_generated_code_failures: int = 6,
    ):
        """Initialize the omniInfra agent.

        Args:
            path: Path to the data
            llm: LLM to use for the agent
            source (str): Source provider such as "OpenAI", "Anthropic", "MiniMax", "Qwen", "Ollama", or "Custom"
            use_tool_retriever: If True, use a tool retriever
            timeout_seconds: Timeout for code execution in seconds
            max_output_tokens: Maximum tokens reserved for each agent response
            context_window_tokens: Total model context window used for prompt budgeting
            token_safety_margin: Tokens kept unused for tokenizer/provider variance
            network_policy: controlled (default), offline, or legacy_open
            base_url: Base URL for custom model serving (e.g., "http://localhost:8000/v1")
            api_key: API key for the custom LLM
            commercial_mode: If True, excludes datasets that require commercial licenses or are non-commercial only

        """
        # Use default_config values for unspecified parameters
        if path is None:
            path = default_config.path
        if llm is None:
            llm = default_config.llm
        else:
            # Keep nested LLM-backed tools (for example, natural-language database
            # query builders) on the same model as the agent constructor override.
            default_config.llm = llm
        if source is None:
            source = default_config.source
        if use_tool_retriever is None:
            use_tool_retriever = default_config.use_tool_retriever
        if timeout_seconds is None:
            timeout_seconds = default_config.timeout_seconds
        if max_output_tokens is None:
            max_output_tokens = default_config.max_output_tokens
        if context_window_tokens is None:
            context_window_tokens = default_config.context_window_tokens
        if token_safety_margin is None:
            token_safety_margin = default_config.token_safety_margin
        if base_url is None:
            base_url = default_config.base_url
        if api_key is None:
            api_key = default_config.api_key if default_config.api_key else "EMPTY"
        if commercial_mode is None:
            commercial_mode = default_config.commercial_mode
        if network_policy is None:
            network_policy = default_config.network_policy

        # Import appropriate env_desc based on commercial_mode
        if commercial_mode:
            from omniInfra.env_desc_cm import data_lake_dict, library_content_dict

            print("🏢 Commercial mode: Using commercial-licensed datasets only")
        else:
            from omniInfra.env_desc import data_lake_dict, library_content_dict

            print("🎓 Academic mode: Using all datasets (including non-commercial)")

        # Store as instance attributes for later use
        self.data_lake_dict = data_lake_dict
        self.library_content_dict = library_content_dict
        self.commercial_mode = commercial_mode

        # Display configuration in a nice, readable format
        print("\n" + "=" * 50)
        print("🔧 OMNIINFRA CONFIGURATION")
        print("=" * 50)

        # Get the actual LLM values that will be used by the agent
        agent_llm = llm if llm is not None else default_config.llm
        agent_source = source if source is not None else default_config.source

        # Show default config (database LLM)
        print("📋 DEFAULT CONFIG (Including Database LLM):")
        config_dict = default_config.to_dict()
        for key, value in config_dict.items():
            if value is not None:
                # Special formatting for commercial_mode
                if key == "commercial_mode":
                    mode_text = "Commercial (licensed datasets only)" if value else "Academic (all datasets)"
                    print(f"  {key.replace('_', ' ').title()}: {mode_text}")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {value}")

        # Show agent-specific LLM if different from default
        if agent_llm != default_config.llm or agent_source != default_config.source:
            print("\n🤖 AGENT LLM (Constructor Override):")
            print(f"  LLM Model: {agent_llm}")
            if agent_source is not None:
                print(f"  Source: {agent_source}")
            if base_url is not None:
                print(f"  Base URL: {base_url}")
            if api_key is not None and api_key != "EMPTY":
                print(f"  API Key: {'*' * 8 + api_key[-4:] if len(api_key) > 8 else '***'}")

        print("=" * 50 + "\n")

        self.path = path

        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")

        # --- Begin custom folder/file checks ---
        benchmark_dir = os.path.join(path, "omniInfra_data", "benchmark")
        data_lake_dir = os.path.join(path, "omniInfra_data", "data_lake")

        # Create the omniInfra_data directory structure
        os.makedirs(benchmark_dir, exist_ok=True)
        os.makedirs(data_lake_dir, exist_ok=True)

        if expected_data_lake_files is None or expected_data_lake_files:
            check_all_data_lake_files = expected_data_lake_files is None
            files_to_check = list(self.data_lake_dict.keys()) if check_all_data_lake_files else expected_data_lake_files

            # Check and download the requested data lake files
            print("Checking and downloading missing data lake files...")
            check_and_download_s3_files(
                s3_bucket_url="https://omniInfra-release.s3.amazonaws.com",
                local_data_lake_path=data_lake_dir,
                expected_files=files_to_check,
                folder="data_lake",
            )

            if check_all_data_lake_files:
                # Check if benchmark directory structure is complete
                benchmark_ok = False
                if os.path.isdir(benchmark_dir):
                    patient_gene_detection_dir = os.path.join(benchmark_dir, "hle")
                    if os.path.isdir(patient_gene_detection_dir):
                        benchmark_ok = True

                if not benchmark_ok:
                    print("Checking and downloading benchmark files...")
                    check_and_download_s3_files(
                        s3_bucket_url="https://omniInfra-release.s3.amazonaws.com",
                        local_data_lake_path=benchmark_dir,
                        expected_files=[],  # Empty list - will download entire folder
                        folder="benchmark",
                    )
        else:
            print("Skipping datalake download (load_datalake=False)")
            print("Note: Some tools may require datalake files to function properly.")

        self.path = os.path.join(path, "omniInfra_data")
        self.model_name = str(llm)
        self.module2api = self._filter_available_module2api(read_module2api())
        self._observed_tool_outputs: set[str] = set()
        self.network_policy = str(network_policy).strip().lower()
        self.execution_admission = ExecutionAdmissionController(self.network_policy)
        self._run_control_defaults = {
            "max_execute_rounds": max_execute_rounds,
            "max_tool_calls": max_tool_calls,
            "max_policy_rejections": max_policy_rejections,
            "max_tool_recoveries": max_tool_recoveries,
            "max_endpoint_discoveries": max_endpoint_discoveries,
            "max_fallback_requests": max_fallback_requests,
            "max_consecutive_no_evidence": max_consecutive_no_evidence,
            "max_response_format_failures": max_response_format_failures,
            "max_generated_code_failures": max_generated_code_failures,
        }
        if any(not isinstance(value, int) or value <= 0 for value in self._run_control_defaults.values()):
            raise ValueError("All A1 execution and evidence budgets must be positive integers.")
        self._reset_run_control()

        self.max_output_tokens = max(1, int(max_output_tokens))
        self.context_window_tokens = None if context_window_tokens is None else max(1, int(context_window_tokens))
        self.token_safety_margin = max(0, int(token_safety_margin))
        if (
            self.context_window_tokens is not None
            and self.max_output_tokens + self.token_safety_margin >= self.context_window_tokens
        ):
            raise ValueError(
                "A1 token budget is invalid: max_output_tokens + token_safety_margin "
                "must be smaller than context_window_tokens."
            )

        self.llm = get_llm(
            llm,
            stop_sequences=["</execute>", "</solution>"],
            max_tokens=self.max_output_tokens,
            source=source,
            base_url=base_url,
            api_key=api_key,
        )
        self.use_tool_retriever = use_tool_retriever

        # The registry is always available for bounded local semantic retrieval.
        # Qwen3 embedding recall and Qwen3 reranking always run; the
        # use_tool_retriever flag controls only the optional final LLM selection
        # over that bounded candidate set.
        self.tool_registry = ToolRegistry(self.module2api)
        self.retriever = ToolRetriever()

        # Initialize know-how loader
        self.know_how_loader = KnowHowLoader()

        # Filter know-how documents based on commercial mode
        if commercial_mode:
            self._filter_know_how_for_commercial_mode()

        print(f"📚 Loaded {len(self.know_how_loader.documents)} know-how documents")

        # Add timeout parameter
        self.timeout_seconds = timeout_seconds  # 10 minutes default timeout
        self.configure()

    def add_tool(self, api):
        """Add a new tool to the agent's tool registry and make it available for retrieval.

        Args:
            api: A callable function to be added as a tool

        """
        try:
            # Get function information
            function_code = inspect.getsource(api)
            module_name = api.__module__ if hasattr(api, "__module__") else "custom_tools"
            function_name = api.__name__ if hasattr(api, "__name__") else str(api)

            # Generate API schema using the existing utility function
            schema = function_to_api_schema(function_code, self.llm)

            # Ensure the schema has all required fields for the tool registry
            if not isinstance(schema, dict):
                raise ValueError("Generated schema is not a dictionary")

            # Validate and enhance the schema

            # Set default values if missing
            if "name" not in schema:
                schema["name"] = function_name
            if "description" not in schema:
                schema["description"] = f"Custom tool: {function_name}"
            if "required_parameters" not in schema:
                # Try to extract from parameters if available
                if "parameters" in schema and isinstance(schema["parameters"], dict):
                    required_params = []
                    params = schema["parameters"]
                    if "properties" in params:
                        for param_name in params["properties"]:
                            if param_name in params.get("required", []):
                                required_params.append(param_name)
                    schema["required_parameters"] = required_params
                else:
                    schema["required_parameters"] = []

            # Add module information to the schema
            schema["module"] = module_name

            if not self._is_tool_available(schema["name"]):
                raise ValueError("This custom tool is unavailable in the current A1 deployment")

            # Add the tool to the tool registry if it exists
            if hasattr(self, "tool_registry") and self.tool_registry is not None:
                try:
                    self.tool_registry.register_tool(schema)
                    print(f"Successfully registered tool '{schema['name']}' in tool registry")
                except Exception as e:
                    print(f"Warning: Failed to register tool in registry: {e}")
                    # Continue with adding to module2api even if registry fails

            # Add the tool to module2api structure for system prompt generation
            if not hasattr(self, "module2api") or self.module2api is None:
                self.module2api = {}

            if module_name not in self.module2api:
                self.module2api[module_name] = []

            # Check if tool already exists in module2api to avoid duplicates
            existing_tool = None
            for existing in self.module2api[module_name]:
                if existing.get("name") == schema["name"]:
                    existing_tool = existing
                    break

            if existing_tool:
                # Update existing tool
                existing_tool.update(schema)
                print(f"Updated existing tool '{schema['name']}' in module '{module_name}'")
            else:
                # Add new tool
                self.module2api[module_name].append(schema)
                print(f"Added new tool '{schema['name']}' to module '{module_name}'")

            # Update the tool registry's document dataframe if it exists
            if hasattr(self, "tool_registry") and self.tool_registry is not None:
                try:
                    # Rebuild the document dataframe
                    docs = []
                    for tool_id in range(len(self.tool_registry.tools)):
                        docs.append(
                            [
                                int(tool_id),
                                self.tool_registry.get_tool_by_id(int(tool_id)),
                            ]
                        )
                    self.tool_registry.document_df = pd.DataFrame(docs, columns=["docid", "document_content"])
                except Exception as e:
                    print(f"Warning: Failed to update tool registry document dataframe: {e}")

            # Store the original function for potential future use
            if not hasattr(self, "_custom_functions"):
                self._custom_functions = {}
            self._custom_functions[schema["name"]] = api

            # Also store in _custom_tools for highlighting
            if not hasattr(self, "_custom_tools"):
                self._custom_tools = {}
            self._custom_tools[schema["name"]] = {
                "name": schema["name"],
                "description": schema["description"],
                "module": module_name,
            }

            # Make the function available in the global namespace for execution
            import builtins

            if not hasattr(builtins, "_omniInfra_custom_functions"):
                builtins._omniInfra_custom_functions = {}
            builtins._omniInfra_custom_functions[schema["name"]] = api

            print(
                f"Tool '{schema['name']}' successfully added and ready for use in both direct execution and retrieval"
            )
            self.configure()
            return schema

        except Exception as e:
            print(f"Error adding tool: {e}")
            import traceback

            traceback.print_exc()
            raise

    def add_mcp(self, config_path: str | Path = "./tutorials/examples/mcp_config.yaml") -> None:
        """
        Add MCP (Model Context Protocol) tools from configuration file.

        This method dynamically registers MCP server tools as callable functions within
        the omniInfra agent system. Each MCP server is loaded as an independent module
        with its tools exposed as synchronous wrapper functions.

        Supports both manual tool definitions and automatic tool discovery from MCP servers.

        Args:
            config_path: Path to the MCP configuration YAML file containing server
                        definitions and tool specifications.

        Raises:
            FileNotFoundError: If the config file doesn't exist
            yaml.YAMLError: If the config file is malformed
            RuntimeError: If MCP server initialization fails
        """
        import asyncio
        import os
        import sys
        import types
        from pathlib import Path

        import nest_asyncio
        import yaml
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        nest_asyncio.apply()

        def discover_mcp_tools_sync(server_params: StdioServerParameters) -> list[dict]:
            """Discover available tools from MCP server synchronously."""
            try:

                async def _discover_async():
                    async with stdio_client(server_params) as (reader, writer):
                        async with ClientSession(reader, writer) as session:
                            await session.initialize()

                            # Get available tools
                            tools_result = await session.list_tools()
                            tools = tools_result.tools if hasattr(tools_result, "tools") else tools_result

                            discovered_tools = []
                            for tool in tools:
                                if hasattr(tool, "name"):
                                    discovered_tools.append(
                                        {
                                            "name": tool.name,
                                            "description": tool.description,
                                            "inputSchema": tool.inputSchema,
                                        }
                                    )
                                else:
                                    print(f"Warning: Skipping tool with no name attribute: {tool}")

                            return discovered_tools

                return asyncio.run(_discover_async())
            except Exception as e:
                print(f"Failed to discover tools: {e}")
                return []

        def make_mcp_wrapper(cmd: str, args: list[str], tool_name: str, doc: str, env_vars: dict = None):
            """Create a synchronous wrapper for an async MCP tool call."""

            def sync_tool_wrapper(**kwargs):
                """Synchronous wrapper for MCP tool execution."""
                try:
                    server_params = StdioServerParameters(command=cmd, args=args, env=env_vars)

                    async def async_tool_call():
                        async with stdio_client(server_params) as (reader, writer):
                            async with ClientSession(reader, writer) as session:
                                await session.initialize()
                                result = await session.call_tool(tool_name, kwargs)
                                content = result.content[0]
                                if hasattr(content, "json"):
                                    return content.json()
                                return content.text

                    try:
                        loop = asyncio.get_running_loop()
                        return loop.create_task(async_tool_call())
                    except RuntimeError:
                        return asyncio.run(async_tool_call())

                except Exception as e:
                    raise RuntimeError(f"MCP tool execution failed for '{tool_name}': {e}") from e

            sync_tool_wrapper.__name__ = tool_name
            sync_tool_wrapper.__doc__ = doc
            return sync_tool_wrapper

        # Initialize registries if they don't exist
        self._custom_functions = getattr(self, "_custom_functions", {})
        self._custom_tools = getattr(self, "_custom_tools", {})

        # Load and validate configuration
        try:
            config_content = Path(config_path).read_text(encoding="utf-8")
            cfg: dict[str, Any] = yaml.safe_load(config_content) or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"MCP config file not found: {config_path}") from None
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in MCP config: {e}") from e

        mcp_servers: dict[str, Any] = cfg.get("mcp_servers", {})
        if not mcp_servers:
            print("Warning: No MCP servers found in configuration")
            return

        # Process each MCP server configuration
        for server_name, server_meta in mcp_servers.items():
            if not server_meta.get("enabled", True):
                continue

            # Validate command configuration
            cmd_list = server_meta.get("command", [])
            if not cmd_list or not isinstance(cmd_list, list):
                print(f"Warning: Invalid command configuration for server '{server_name}'")
                continue

            cmd, *args = cmd_list

            # Process environment variables
            env_vars = server_meta.get("env", {})
            if env_vars:
                processed_env = {}
                for key, value in env_vars.items():
                    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                        var_name = value[2:-1]
                        processed_env[key] = os.getenv(var_name, "")
                    else:
                        processed_env[key] = value
                env_vars = processed_env

            # Create module namespace for this MCP server
            mcp_module_name = f"mcp_servers.{server_name}"
            if mcp_module_name not in sys.modules:
                sys.modules[mcp_module_name] = types.ModuleType(mcp_module_name)
            server_module = sys.modules[mcp_module_name]

            tools_config = server_meta.get("tools", [])

            if not tools_config:
                try:
                    server_params = StdioServerParameters(command=cmd, args=args, env=env_vars)
                    tools_config = discover_mcp_tools_sync(server_params)

                    if tools_config:
                        print(f"Discovered {len(tools_config)} tools from {server_name} MCP server")
                    else:
                        print(f"Warning: No tools discovered from {server_name} MCP server")
                        continue

                except Exception as e:
                    print(f"Failed to discover tools for {server_name}: {e}")
                    continue

            # Register each tool
            for tool_meta in tools_config:
                if isinstance(tool_meta, dict) and "omniInfra_name" in tool_meta:
                    # Manual tool definition
                    tool_name = tool_meta.get("omniInfra_name")
                    description = tool_meta.get("description", f"MCP tool: {tool_name}")
                    parameters = tool_meta.get("parameters", {})
                    # For manual tools, check if each parameter has a "required" field
                    required_param_names = []
                    for param_name, param_spec in parameters.items():
                        if param_spec.get("required", False):
                            required_param_names.append(param_name)
                else:
                    # Auto-discovered tool
                    tool_name = tool_meta.get("name")
                    description = tool_meta.get("description", f"MCP tool: {tool_name}")
                    input_schema = tool_meta.get("inputSchema", {})
                    parameters = input_schema.get("properties", {})
                    # For auto-discovered tools, get required list from inputSchema top level
                    required_param_names = input_schema.get("required", [])

                if not tool_name:
                    print(f"Warning: Skipping tool with no name in {server_name}")
                    continue
                if not self._is_tool_available(tool_name):
                    print("Warning: Skipping an unavailable MCP tool")
                    continue

                # Create wrapper function
                wrapper_function = make_mcp_wrapper(cmd, args, tool_name, description, env_vars)

                # Add to module namespace
                setattr(server_module, tool_name, wrapper_function)

                # Build parameter lists
                required_params, optional_params = [], []
                for param_name, param_spec in parameters.items():
                    param_info = {
                        "name": param_name,
                        "type": str(param_spec.get("type", "string")),
                        "description": param_spec.get("description", ""),
                        "default": param_spec.get("default", None),
                    }

                    # Check if parameter is required based on the required_param_names list
                    if param_name in required_param_names:
                        required_params.append(param_info)
                    else:
                        optional_params.append(param_info)

                # Create tool schema
                tool_schema = {
                    "name": tool_name,
                    "description": description,
                    "parameters": parameters,
                    "required_parameters": required_params,
                    "optional_parameters": optional_params,
                    "module": mcp_module_name,
                    "fn": wrapper_function,
                }

                # Register in tool registry
                self.tool_registry.register_tool(tool_schema)

                # Add to module2api mapping
                if mcp_module_name not in self.module2api:
                    self.module2api[mcp_module_name] = []
                self.module2api[mcp_module_name].append(tool_schema)

                # Add to instance registries
                self._custom_functions[tool_name] = wrapper_function
                self._custom_tools[tool_name] = {
                    "name": tool_name,
                    "description": description,
                    "module": mcp_module_name,
                }

        # Update agent configuration
        self.configure()

    def get_custom_tool(self, name):
        """Get a custom tool by name.

        Args:
            name: The name of the custom tool

        Returns:
            The custom tool function if found, None otherwise

        """
        if hasattr(self, "_custom_functions") and name in self._custom_functions:
            return self._custom_functions[name]
        return None

    def list_custom_tools(self):
        """List all custom tools that have been added.

        Returns:
            A list of custom tool names

        """
        if hasattr(self, "_custom_functions"):
            return list(self._custom_functions.keys())
        return []

    def remove_custom_tool(self, name):
        """Remove a custom tool.

        Args:
            name: The name of the custom tool to remove

        Returns:
            True if the tool was removed, False if it wasn't found

        """
        removed = False

        # Remove from custom functions
        if hasattr(self, "_custom_functions") and name in self._custom_functions:
            del self._custom_functions[name]
            removed = True

        # Remove from custom tools (for highlighting)
        if hasattr(self, "_custom_tools") and name in self._custom_tools:
            del self._custom_tools[name]
            removed = True

        # Remove from global namespace
        import builtins

        if hasattr(builtins, "_omniInfra_custom_functions") and name in builtins._omniInfra_custom_functions:
            del builtins._omniInfra_custom_functions[name]

        # Remove from tool registry
        if hasattr(self, "tool_registry") and self.tool_registry is not None:
            if self.tool_registry.remove_tool_by_name(name):
                removed = True
                # Rebuild the document dataframe
                try:
                    docs = []
                    for tool_id in range(len(self.tool_registry.tools)):
                        docs.append(
                            [
                                int(tool_id),
                                self.tool_registry.get_tool_by_id(int(tool_id)),
                            ]
                        )
                    self.tool_registry.document_df = pd.DataFrame(docs, columns=["docid", "document_content"])
                except Exception as e:
                    print(f"Warning: Failed to update tool registry document dataframe: {e}")

        # Remove from module2api
        if hasattr(self, "module2api"):
            for tools in self.module2api.values():
                for i, tool in enumerate(tools):
                    if tool.get("name") == name:
                        del tools[i]
                        removed = True
                        break

        if removed:
            print(f"Custom tool '{name}' has been removed")
        else:
            print(f"Custom tool '{name}' was not found")

        return removed

    def add_data(self, data):
        """Add new data to the data lake.

        Args:
            data: Dictionary with file path as key and description as value
                  e.g., {'my_dataset.csv': 'A dataset containing gene expression data'}
                  or {'path/to/file.txt': 'Description of the file'}

        """
        try:
            if not isinstance(data, dict):
                raise ValueError("Data must be a dictionary with file path as key and description as value")

            # Initialize custom data storage if it doesn't exist
            if not hasattr(self, "_custom_data"):
                self._custom_data = {}

            # Add each data item
            for file_path, description in data.items():
                if not isinstance(file_path, str) or not isinstance(description, str):
                    print("Warning: Skipping invalid data entry - file_path and description must be strings")
                    continue

                # Extract filename from path for storage
                filename = os.path.basename(file_path) if "/" in file_path else file_path

                # Store the data with both the full path and description
                self._custom_data[filename] = {
                    "path": file_path,
                    "description": description,
                }

                # Also add to the data_lake_dict for consistency
                self.data_lake_dict[filename] = description

                print(f"Added data item '{filename}': {description}")
            self.configure()
            print(f"Successfully added {len(data)} data item(s) to the data lake")
            return True

        except Exception as e:
            print(f"Error adding data: {e}")
            import traceback

            traceback.print_exc()
            return False

    def get_custom_data(self, name):
        """Get a custom data item by name.

        Args:
            name: The name of the custom data item

        Returns:
            The custom data item info if found, None otherwise

        """
        if hasattr(self, "_custom_data") and name in self._custom_data:
            return self._custom_data[name]
        return None

    def list_custom_data(self):
        """List all custom data items that have been added.

        Returns:
            A list of custom data item names and descriptions

        """
        if hasattr(self, "_custom_data"):
            return [(name, info["description"]) for name, info in self._custom_data.items()]
        return []

    def remove_custom_data(self, name):
        """Remove a custom data item.

        Args:
            name: The name of the custom data item to remove

        Returns:
            True if the data item was removed, False if it wasn't found

        """
        removed = False

        # Remove from custom data
        if hasattr(self, "_custom_data") and name in self._custom_data:
            del self._custom_data[name]
            removed = True

        # Remove from data_lake_dict
        if hasattr(self, "data_lake_dict") and name in self.data_lake_dict:
            del self.data_lake_dict[name]
            removed = True

        if removed:
            print(f"Custom data item '{name}' has been removed")
        else:
            print(f"Custom data item '{name}' was not found")

        return removed

    def add_software(self, software):
        """Add new software to the software library.

        Args:
            software: Dictionary with software name as key and description as value
                     e.g., {'custom_tool': 'A custom analysis tool for processing data'}
                     or {'my_package': 'Description of the package functionality'}

        """
        try:
            if not isinstance(software, dict):
                raise ValueError("Software must be a dictionary with software name as key and description as value")

            # Initialize custom software storage if it doesn't exist
            if not hasattr(self, "_custom_software"):
                self._custom_software = {}

            # Add each software item
            for software_name, description in software.items():
                if not isinstance(software_name, str) or not isinstance(description, str):
                    print("Warning: Skipping invalid software entry - software_name and description must be strings")
                    continue

                # Store the software with description
                self._custom_software[software_name] = {
                    "name": software_name,
                    "description": description,
                }

                # Also add to the library_content_dict for consistency
                self.library_content_dict[software_name] = description

                print(f"Added software '{software_name}': {description}")

            print(f"Successfully added {len(software)} software item(s) to the library")
            self.configure()
            return True

        except Exception as e:
            print(f"Error adding software: {e}")
            import traceback

            traceback.print_exc()
            return False

    def get_custom_software(self, name):
        """Get a custom software item by name.

        Args:
            name: The name of the custom software item

        Returns:
            The custom software item info if found, None otherwise

        """
        if hasattr(self, "_custom_software") and name in self._custom_software:
            return self._custom_software[name]
        return None

    def list_custom_software(self):
        """List all custom software items that have been added.

        Returns:
            A list of custom software item names and descriptions

        """
        if hasattr(self, "_custom_software"):
            return [(name, info["description"]) for name, info in self._custom_software.items()]
        return []

    def remove_custom_software(self, name):
        """Remove a custom software item.

        Args:
            name: The name of the custom software item to remove

        Returns:
            True if the software item was removed, False if it wasn't found

        """
        removed = False

        # Remove from custom software
        if hasattr(self, "_custom_software") and name in self._custom_software:
            del self._custom_software[name]
            removed = True

        # Remove from library_content_dict
        if hasattr(self, "library_content_dict") and name in self.library_content_dict:
            del self.library_content_dict[name]
            removed = True

        if removed:
            print(f"Custom software item '{name}' has been removed")
        else:
            print(f"Custom software item '{name}' was not found")

        return removed

    def _filter_know_how_for_commercial_mode(self):
        """Filter out know-how documents that don't allow commercial use.

        This method removes documents from the know-how loader that have
        commercial use restrictions when the agent is in commercial mode.
        """
        docs_to_remove = []

        for doc_id, doc in self.know_how_loader.documents.items():
            metadata = doc.get("metadata", {})
            commercial_use = metadata.get("commercial_use", "")

            # Check if commercial use is NOT allowed
            if "❌" in commercial_use or "Not Allowed" in commercial_use or "Non-Commercial" in commercial_use:
                docs_to_remove.append(doc_id)

        # Remove documents that don't allow commercial use
        for doc_id in docs_to_remove:
            doc_name = self.know_how_loader.documents[doc_id]["name"]
            self.know_how_loader.remove_document(doc_id)
            print(f"  ⚠️  Excluded know-how '{doc_name}' (non-commercial license)")

    def _generate_system_prompt(
        self,
        tool_desc,
        data_lake_content,
        library_content_list,
        self_critic=False,
        is_retrieval=False,
        custom_tools=None,
        custom_data=None,
        custom_software=None,
        know_how_docs=None,
    ):
        """Generate the system prompt based on the provided resources.

        Args:
            tool_desc: Dictionary of tool descriptions
            data_lake_content: List of data lake items
            library_content_list: List of libraries
            self_critic: Whether to include self-critic instructions
            is_retrieval: Whether this is for retrieval (True) or initial configuration (False)
            custom_tools: List of custom tools to highlight
            custom_data: List of custom data items to highlight
            custom_software: List of custom software items to highlight
            know_how_docs: List of know-how documents with best practices and protocols

        Returns:
            The generated system prompt

        """
        if isinstance(tool_desc, dict):
            tool_desc = self._filter_available_module2api(tool_desc)

        def format_item_with_description(name, description):
            """Format an item with its description in a readable way."""
            # Handle None or empty descriptions
            if not description:
                description = f"Data lake item: {name}"

            # Check if the item is already formatted (contains a colon)
            if isinstance(name, str) and ": " in name:
                return name

            # Wrap long descriptions to make them more readable
            max_line_length = 80
            if len(description) > max_line_length:
                # Simple wrapping for long descriptions
                wrapped_desc = []
                words = description.split()
                current_line = ""

                for word in words:
                    if len(current_line) + len(word) + 1 <= max_line_length:
                        if current_line:
                            current_line += " " + word
                        else:
                            current_line = word
                    else:
                        wrapped_desc.append(current_line)
                        current_line = word

                if current_line:
                    wrapped_desc.append(current_line)

                # Join with newlines and proper indentation
                formatted_desc = f"{name}:\n  " + "\n  ".join(wrapped_desc)
                return formatted_desc
            else:
                return f"{name}: {description}"

        # Separate custom and default resources
        default_data_lake_content = []
        default_library_content_list = []

        # Filter out custom items from default lists
        custom_data_names = set()
        custom_software_names = set()

        if custom_data:
            custom_data_names = {item.get("name") if isinstance(item, dict) else item for item in custom_data}
        if custom_software:
            custom_software_names = {item.get("name") if isinstance(item, dict) else item for item in custom_software}

        # Separate default data lake items
        for item in data_lake_content:
            if isinstance(item, dict):
                name = item.get("name", "")
                if name not in custom_data_names:
                    default_data_lake_content.append(item)
            elif item not in custom_data_names:
                default_data_lake_content.append(item)

        # Separate default library items
        for lib in library_content_list:
            if isinstance(lib, dict):
                name = lib.get("name", "")
                if name not in custom_software_names:
                    default_library_content_list.append(lib)
            elif lib not in custom_software_names:
                default_library_content_list.append(lib)

        # Format the default data lake content
        if isinstance(default_data_lake_content, list) and all(
            isinstance(item, str) for item in default_data_lake_content
        ):
            # Simple list of strings - check if they already have descriptions
            data_lake_formatted = []
            for item in default_data_lake_content:
                # Check if the item already has a description (contains a colon)
                if ": " in item:
                    data_lake_formatted.append(item)
                else:
                    description = self.data_lake_dict.get(item, f"Data lake item: {item}")
                    data_lake_formatted.append(format_item_with_description(item, description))
        else:
            # List with descriptions
            data_lake_formatted = []
            for item in default_data_lake_content:
                if isinstance(item, dict):
                    name = item.get("name", "")
                    description = self.data_lake_dict.get(name, f"Data lake item: {name}")
                    data_lake_formatted.append(format_item_with_description(name, description))
                # Check if the item already has a description (contains a colon)
                elif isinstance(item, str) and ": " in item:
                    data_lake_formatted.append(item)
                else:
                    description = self.data_lake_dict.get(item, f"Data lake item: {item}")
                    data_lake_formatted.append(format_item_with_description(item, description))

        # Format the default library content
        if isinstance(default_library_content_list, list) and all(
            isinstance(item, str) for item in default_library_content_list
        ):
            if (
                len(default_library_content_list) > 0
                and isinstance(default_library_content_list[0], str)
                and "," not in default_library_content_list[0]
            ):
                # Simple list of strings
                libraries_formatted = []
                for lib in default_library_content_list:
                    description = self.library_content_dict.get(lib, f"Software library: {lib}")
                    libraries_formatted.append(format_item_with_description(lib, description))
            else:
                # Already formatted string
                libraries_formatted = default_library_content_list
        else:
            # List with descriptions
            libraries_formatted = []
            for lib in default_library_content_list:
                if isinstance(lib, dict):
                    name = lib.get("name", "")
                    description = self.library_content_dict.get(name, f"Software library: {name}")
                    libraries_formatted.append(format_item_with_description(name, description))
                else:
                    description = self.library_content_dict.get(lib, f"Software library: {lib}")
                    libraries_formatted.append(format_item_with_description(lib, description))

        # Format custom resources with highlighting
        custom_tools_formatted = []
        if custom_tools:
            for tool in custom_tools:
                if isinstance(tool, dict):
                    name = tool.get("name", "Unknown")
                    desc = tool.get("description", "")
                    module = tool.get("module", "custom_tools")
                    custom_tools_formatted.append(f"🔧 {name} (from {module}): {desc}")
                else:
                    custom_tools_formatted.append(f"🔧 {str(tool)}")

        custom_data_formatted = []
        if custom_data:
            for item in custom_data:
                if isinstance(item, dict):
                    name = item.get("name", "Unknown")
                    desc = item.get("description", "")
                    custom_data_formatted.append(f"📊 {format_item_with_description(name, desc)}")
                else:
                    desc = self.data_lake_dict.get(item, f"Custom data: {item}")
                    custom_data_formatted.append(f"📊 {format_item_with_description(item, desc)}")

        custom_software_formatted = []
        if custom_software:
            for item in custom_software:
                if isinstance(item, dict):
                    name = item.get("name", "Unknown")
                    desc = item.get("description", "")
                    custom_software_formatted.append(f"⚙️ {format_item_with_description(name, desc)}")
                else:
                    desc = self.library_content_dict.get(item, f"Custom software: {item}")
                    custom_software_formatted.append(f"⚙️ {format_item_with_description(item, desc)}")

        # Format know-how documents - include FULL content (metadata already stripped)
        know_how_formatted = []
        if know_how_docs:
            for doc in know_how_docs:
                if isinstance(doc, dict):
                    name = doc.get("name", "Unknown")
                    content = doc.get("content", "")
                    # Include full content in system prompt (metadata already removed)
                    know_how_formatted.append(f"📚 {name}:\n{content}")

        # Base prompt
        prompt_modifier = """
You are a helpful biomedical assistant assigned with the task of problem-solving.
To achieve this, you will be using an interactive coding environment equipped with a variety of tool functions, data, and softwares to assist you throughout the process.

Given a task, make a plan first. The plan should be a numbered list of steps that you will take to solve the task. Be specific and detailed.
Format your plan as a checklist with empty checkboxes like this:
1. [ ] First step
2. [ ] Second step
3. [ ] Third step

Follow the plan step by step. After completing each step, update the checklist by replacing the empty checkbox with a checkmark:
1. [✓] First step (completed)
2. [ ] Second step
3. [ ] Third step

If a step fails or needs modification, mark it with an X and explain why:
1. [✓] First step (completed)
2. [✗] Second step (failed because...)
3. [ ] Modified second step
4. [ ] Third step

Always show the updated plan after each step so the user can track progress.

At each turn, you should first provide your thinking and reasoning given the conversation history.
After that, you have two options:

1) Interact with a programming environment and receive the corresponding output within <observe></observe>. Your code should be enclosed using "<execute>" tag, for example: <execute> print("Hello World!") </execute>. IMPORTANT: You must end the code block with </execute> tag.
   - For Python code (default): <execute> print("Hello World!") </execute>
   - For R code: <execute> #!R\nlibrary(ggplot2)\nprint("Hello from R") </execute>
   - For Bash scripts and commands: <execute> #!BASH\necho "Hello from Bash"\nls -la </execute>
   - For CLI softwares, use Bash scripts.

2) When you think it is ready, directly provide a solution that adheres to the required format for the given task to the user. Your solution should be enclosed using "<solution>" tag, for example: The answer is <solution> A </solution>. IMPORTANT: You must end the solution block with </solution> tag.

You have many chances to interact with the environment to receive the observation. So you can decompose your code into multiple steps.
Don't overcomplicate the code. Keep it simple and easy to understand.
When writing the code, please print out the steps and results in a clear and concise manner, like a research log.
Save each Function Dictionary result. For a new output shape, call once and MUST `print()` (bare expressions show nothing)
the saved result; end step, inspect that observation, then reuse it next step without `.get()` or `[]` beforehand.
Do not guess fields or require dictionary output.

EVIDENCE AND MULTI-STEP QUERY RULES:
- Never present a nearby source or version as an exact match. If exact evidence is unavailable, a useful bounded proxy may be shown only with its mismatch and limitations stated explicitly.
- When filtering a prior candidate set, keep that set unless the user requests expansion. For STRING, pass all prior candidates to query_stringdb(identifiers=[...]), not only the top gene.
- Prefer a tool's high-level structured parameters over hand-written GraphQL, REST, or RCSB request objects.
- Never infer absence or failed validation from missing fields, empty results, or local fallbacks (`[]`, `{{}}`, `None`, `N/A`, `No summary`); HTTP success is not scientific success.
- Use evidence only if its observed type addresses the predicate; otherwise find a suitable source or report insufficiency.
- Use only schema parameters and observed fields; do not copy raw output into source code.

For R code, use the #!R marker at the beginning of your code block to indicate it's R code.
For Bash scripts and commands, use the #!BASH marker at the beginning of your code block. This allows for both simple commands and multi-line scripts with variables, loops, conditionals, loops, and other Bash features.

In each response, you must include EITHER <execute> or <solution> tag. Not both at the same time. Do not respond with messages without any tags. No empty messages.
"""

        network_policy = getattr(self, "network_policy", None)
        if network_policy == "controlled":
            prompt_modifier += """

NETWORK EXECUTION POLICY:
- Do not use raw Python, R, or shell networking. Use a listed OmniInfra tool.
- If no specialized tool is available, use query_public_endpoint for bounded read-only access.
- After printing a tool observation, reuse the saved result for analysis; do not call the same tool again with the same arguments.
- Preserve task goals, constraints, and required outputs. Treat supplied URLs, code, and tool parameters as execution proposals that still require admission.
- A 404 proves only that the specified resource was not found. Empty or failed queries are not scientific negative evidence.
"""
        elif network_policy == "offline":
            prompt_modifier += """

NETWORK EXECUTION POLICY:
- This run is offline. Do not make external network requests, including through listed network tools.
- Use local data and analysis only, and report when external evidence cannot be retrieved.
"""
        elif network_policy == "legacy_open":
            prompt_modifier += """

NETWORK EXECUTION POLICY:
- This run uses the explicit legacy_open compatibility mode. Raw networking is audited.
- Prefer listed OmniInfra tools and never invent endpoint explanations after a failed request.
"""

        if self._TEMPORARILY_DISABLED_TOOLS:
            prompt_modifier += """

TOOL AVAILABILITY:
- The Function Dictionary is the complete set of tools available to this A1 run.
- Do not guess, reconstruct, or dynamically import tools that are not listed there.
- If no listed tool can retrieve the required evidence, report that limitation explicitly.
"""

        # Add self-critic instructions if needed
        if self_critic:
            prompt_modifier += """
You may or may not receive feedbacks from human. If so, address the feedbacks by following the same procedure of multiple rounds of thinking, execution, and then coming up with a new solution.
"""

        # Add protocol generation instructions
        prompt_modifier += """
PROTOCOL GENERATION:
If the user requests an experimental protocol, use the compatible literature and protocol tools listed above to generate an accurate protocol. Include details such as reagents (with catalog numbers if available), equipment specifications, replicate requirements, error handling, and troubleshooting - but ONLY include information found in these resources. Do not make up specifications, catalog numbers, or equipment details. Prioritize accuracy over completeness.
"""

        # Add custom resources section first (highlighted)
        has_custom_resources = any(
            [custom_tools_formatted, custom_data_formatted, custom_software_formatted, know_how_formatted]
        )

        if has_custom_resources:
            prompt_modifier += """

PRIORITY CUSTOM RESOURCES
===============================
IMPORTANT: The following custom resources have been specifically added for your use.
    PRIORITIZE using these resources as they are directly relevant to your task.
    Always consider these FIRST and in the meantime using default resources.

"""

            if know_how_formatted:
                prompt_modifier += """
📚 KNOW-HOW DOCUMENTS (BEST PRACTICES & PROTOCOLS - ALREADY LOADED):
{know_how_docs}

IMPORTANT: These documents are ALREADY AVAILABLE in your context. You do NOT need to
retrieve them or "review" them as a separate step. You can DIRECTLY reference and use
the information from these documents to answer questions, provide protocols, suggest
parameters, and offer troubleshooting guidance.

These documents contain expert knowledge, protocols, and troubleshooting guidance.
Reference them directly for experimental design, methodology, and problem-solving.

"""

            if custom_tools_formatted:
                prompt_modifier += """
🔧 CUSTOM TOOLS (USE THESE FIRST):
{custom_tools}

"""

            if custom_data_formatted:
                prompt_modifier += """
📊 CUSTOM DATA (PRIORITIZE THESE DATASETS):
{custom_data}

"""

            if custom_software_formatted:
                prompt_modifier += """
⚙️ CUSTOM SOFTWARE (USE THESE LIBRARIES):
{custom_software}

"""

            prompt_modifier += """===============================
"""

        # Add concrete environment resources only after retrieval. The initial
        # prompt remains resource-independent so A1 startup never embeds the
        # complete registry, data lake, software catalog, or know-how corpus.
        has_listed_resources = bool(tool_desc or data_lake_formatted or libraries_formatted)
        if has_listed_resources:
            prompt_modifier += """

Environment Resources:

- Function Dictionary:
{function_intro}
---
{tool_desc}
---

{import_instruction}
  - Import functions only from the exact module shown in the Function Dictionary; do not guess a module path. In particular, query_pubmed is from omniInfra.tool.literature.

- Biological data lake
You can access a biological data lake at the following path: {data_lake_path}.
  - Always build local dataset paths from this exact path; do not check dataset filenames relative to the current working directory.
{data_lake_intro}
Each item is listed with its description to help you understand its contents.
----
{data_lake_content}
----

- Software Library:
{library_intro}
Each library is listed with its description to help you understand its functionality.
----
{library_content_formatted}
----

- Note on using R packages and Bash scripts:
  - R packages: Use subprocess.run(['Rscript', '-e', 'your R code here']) in Python, or use the #!R marker in your execute block.
  - Bash scripts and commands: Use the #!BASH marker in your execute block for both simple commands and complex shell scripts with variables, loops, conditionals, etc.
        """
        else:
            prompt_modifier += """

Environment Access:
- Relevant tools, datasets, software libraries, and know-how are selected per task and will be listed when available.
- Biological data lake root: {data_lake_path}
- Import and call only resources explicitly listed for the current task; do not guess tool names or module paths.
- For R use the #!R marker. For Bash use the #!BASH marker.
        """

        # Set appropriate text based on whether this is initial configuration or after retrieval
        if is_retrieval:
            function_intro = "Based on your query, I've identified the following most relevant functions that you can use in your code:"
            data_lake_intro = "Based on your query, I've identified the following most relevant datasets:"
            library_intro = (
                "Based on your query, I've identified the following most relevant libraries that you can use:"
            )
            import_instruction = "IMPORTANT: When using any function, you MUST first import it from its module. For example:\nfrom [module_name] import [function_name]"
        else:
            function_intro = "In your code, you will need to import the function location using the following dictionary of functions:"
            data_lake_intro = "You can write code to understand the data, process and utilize it for the task. Here is the list of datasets:"
            library_intro = "The environment supports a list of libraries that can be directly used. Do not forget the import statement:"
            import_instruction = ""

        # Format the content consistently for both initial and retrieval cases
        library_content_formatted = "\n".join(libraries_formatted)
        data_lake_content_formatted = "\n".join(data_lake_formatted)

        # Format the prompt with the appropriate values
        format_dict = {
            "function_intro": function_intro,
            "tool_desc": textify_api_dict(tool_desc) if isinstance(tool_desc, dict) else tool_desc,
            "import_instruction": import_instruction,
            "data_lake_path": self.path + "/data_lake",
            "data_lake_intro": data_lake_intro,
            "data_lake_content": data_lake_content_formatted,
            "library_intro": library_intro,
            "library_content_formatted": library_content_formatted,
        }

        # Add custom resources to format dict if they exist
        if know_how_formatted:
            format_dict["know_how_docs"] = "\n\n".join(know_how_formatted)
        if custom_tools_formatted:
            format_dict["custom_tools"] = "\n".join(custom_tools_formatted)
        if custom_data_formatted:
            format_dict["custom_data"] = "\n".join(custom_data_formatted)
        if custom_software_formatted:
            format_dict["custom_software"] = "\n".join(custom_software_formatted)

        formatted_prompt = prompt_modifier.format(**format_dict)

        return formatted_prompt

    def _is_tool_available(self, tool_name: str) -> bool:
        """Return whether a tool can run with this agent's configured model."""
        if tool_name == "run_python_repl":
            return False
        if tool_name in self._TEMPORARILY_DISABLED_TOOLS:
            return False
        if tool_name in self._CLAUDE_ONLY_TOOLS:
            return "claude" in self.model_name.lower()
        return True

    def _filter_available_module2api(self, module2api: dict) -> dict:
        """Return an isolated schema mapping containing only available tools."""
        filtered = {}
        for module_name, tool_schemas in module2api.items():
            available_schemas = [
                dict(tool_schema)
                for tool_schema in tool_schemas
                if isinstance(tool_schema, dict)
                and tool_schema.get("name")
                and self._is_tool_available(tool_schema["name"])
            ]
            if available_schemas:
                filtered[module_name] = available_schemas
        return filtered

    def _disabled_tools_in_code(self, code: str) -> list[str]:
        """Return temporarily disabled tool names referenced by generated code."""
        return sorted(
            tool_name
            for tool_name in self._TEMPORARILY_DISABLED_TOOLS
            if re.search(rf"\b{re.escape(tool_name)}\b", code)
        )

    def _reset_run_control(self) -> None:
        """Reset bounded execution and evidence state for one user task."""
        self._run_control = A1RunControl(**self._run_control_defaults)
        self._tool_call_aliases: dict[str, str] = {}

    def _selected_network_tool_names(self) -> set[str]:
        """Return explicitly selected tools whose descriptors declare public networking."""
        selected = (getattr(self, "_active_selected_resources", None) or {}).get("tools") or []
        if self.network_policy == "offline":
            # Until every descriptor declares its runtime dependencies, offline
            # mode admits only tools explicitly reviewed as offline-capable.
            return {
                str(tool.get("name"))
                for tool in selected
                if isinstance(tool, dict) and "offline_capable" not in (tool.get("capabilities") or [])
            }
        return {
            str(tool.get("name"))
            for tool in selected
            if isinstance(tool, dict) and "public_read_only_network" in (tool.get("capabilities") or [])
        }

    def _recover_tools_for_policy_block(self, code: str) -> list[str]:
        """Add a bounded descriptor match after raw networking is rejected."""
        run_control = self._run_control
        active = getattr(self, "_active_selected_resources", None)
        if not active or run_control.tool_recoveries >= run_control.max_tool_recoveries:
            return []

        active_names = {tool.get("name") if isinstance(tool, dict) else str(tool) for tool in active.get("tools", [])}
        hints = f"{getattr(self, 'user_task', '')}\n{code}".lower()
        exact_matches: list[dict] = []
        fallback_matches: list[dict] = []
        for module_name, schemas in self.module2api.items():
            for schema in schemas:
                if schema.get("deprecated") or not self._is_tool_available(str(schema.get("name", ""))):
                    continue
                candidate = dict(schema)
                candidate["module"] = module_name
                source_match = any(str(source).lower() in hints for source in candidate.get("source_ids") or [])
                alias_match = any(
                    re.search(rf"\b{re.escape(str(alias).lower())}\b", hints)
                    for alias in candidate.get("aliases") or []
                )
                name_match = str(candidate.get("name", "")).lower() in hints
                if source_match or alias_match or name_match:
                    exact_matches.append(candidate)
                elif candidate.get("name") == "query_public_endpoint":
                    fallback_matches.append(candidate)

        ordered = exact_matches + fallback_matches
        alternatives: list[str] = []
        added: list[str] = []
        for candidate in ordered:
            name = str(candidate.get("name"))
            if name in alternatives:
                continue
            alternatives.append(name)
            if name not in active_names and len(added) < 3:
                active.setdefault("tools", []).append(candidate)
                active_names.add(name)
                added.append(name)
            if len(alternatives) >= 3:
                break

        if added:
            run_control.tool_recoveries += 1
            run_control.events.append(
                {
                    "event": "tool_recovery",
                    "tools": added,
                    "reason": "raw_network_denied",
                }
            )
        return alternatives

    def _termination_solution(self) -> str:
        reason = self._run_control.termination_reason or "evidence_insufficient"
        return (
            "<solution>Execution stopped because "
            f"{reason}. The available admitted evidence is insufficient for a stronger conclusion.</solution>"
        )

    def _omniInfra_tool_actions(self, code: str) -> list[tuple[str, str | None, bool]]:
        """Return stable native-tool call fingerprints for budgets and repetition checks."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        registered_names = {
            str(tool.get("name"))
            for tools in self.module2api.values()
            for tool in tools
            if isinstance(tool, dict) and tool.get("name")
        }
        aliases = {name: name for name in registered_names}
        aliases.update(getattr(self, "_tool_call_aliases", {}))
        aliases.update(self._omniInfra_tool_aliases(code, registered_names=registered_names))

        actions: list[tuple[str, str | None, bool]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            tool_name = aliases.get(node.func.id)
            if not tool_name:
                continue
            arguments_are_literal = all(self._literal_ast_value(argument) for argument in node.args) and all(
                keyword.arg is not None and self._literal_ast_value(keyword.value) for keyword in node.keywords
            )
            action_hash = None
            if arguments_are_literal:
                signature = ast.dump(
                    ast.Call(
                        func=ast.Name(id=tool_name, ctx=ast.Load()),
                        args=node.args,
                        keywords=sorted(node.keywords, key=lambda keyword: keyword.arg or ""),
                    ),
                    annotate_fields=True,
                    include_attributes=False,
                )
                action_hash = hashlib.sha256(signature.encode()).hexdigest()
            has_documentation = tool_name == "query_public_endpoint" and any(
                keyword.arg == "documentation_url"
                and not (isinstance(keyword.value, ast.Constant) and keyword.value.value is None)
                for keyword in node.keywords
            )
            actions.append((tool_name, action_hash, has_documentation))
        return actions

    def _omniInfra_tool_aliases(
        self,
        code: str,
        *,
        registered_names: set[str] | None = None,
    ) -> dict[str, str]:
        """Return exact OmniInfra tool aliases imported by one generated block."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {}
        if registered_names is None:
            registered_names = {
                str(tool.get("name"))
                for tools in self.module2api.values()
                for tool in tools
                if isinstance(tool, dict) and tool.get("name")
            }
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("omniInfra.tool."):
                for imported in node.names:
                    if imported.name in registered_names:
                        aliases[imported.asname or imported.name] = imported.name
        return aliases

    @staticmethod
    def _literal_ast_value(node: ast.AST) -> bool:
        """Return whether an expression has a stable literal value before execution."""
        try:
            ast.literal_eval(node)
        except (ValueError, TypeError, SyntaxError):
            return False
        return True

    def _validate_omniInfra_tool_imports(self, code: str) -> str | None:
        """Reject unavailable, guessed, or dynamically imported OmniInfra tools."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None

        module2api = getattr(self, "module2api", {})
        tool_to_module = {
            tool["name"]: module_name
            for module_name, tools in module2api.items()
            for tool in tools
            if isinstance(tool, dict) and tool.get("name")
        }
        active_resources = getattr(self, "_active_selected_resources", None)
        selected_tools = (active_resources or {}).get("tools") or []
        selected_names = {tool.get("name") if isinstance(tool, dict) else str(tool) for tool in selected_tools}
        enforce_selection = active_resources is not None
        call_aliases = {name: name for name in tool_to_module}
        call_aliases.update(getattr(self, "_tool_call_aliases", {}))
        call_aliases.update(self._omniInfra_tool_aliases(code, registered_names=set(tool_to_module)))
        importlib_aliases = {"importlib"}
        builtins_aliases = {"builtins"}
        dynamic_import_names = {"__import__"}

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "omniInfra.tool" or alias.name.startswith("omniInfra.tool."):
                        return "Tool import blocked: use only exact imports from the current Function Dictionary."
                    if alias.name == "importlib":
                        importlib_aliases.add(alias.asname or "importlib")
                    if alias.name == "builtins":
                        builtins_aliases.add(alias.asname or "builtins")
                continue

            if not isinstance(node, ast.ImportFrom):
                continue
            module_name = node.module or ""
            if module_name == "importlib":
                for alias in node.names:
                    if alias.name == "import_module":
                        dynamic_import_names.add(alias.asname or alias.name)
                continue
            if module_name == "builtins":
                for alias in node.names:
                    if alias.name == "__import__":
                        dynamic_import_names.add(alias.asname or alias.name)
                continue
            if module_name == "omniInfra" and any(alias.name == "tool" for alias in node.names):
                return "Tool import blocked: use only exact imports from the current Function Dictionary."
            if module_name != "omniInfra.tool" and not module_name.startswith("omniInfra.tool."):
                continue
            if module_name == "omniInfra.tool":
                return "Tool import blocked: use the exact module shown in the current Function Dictionary."

            for alias in node.names:
                tool_name = alias.name
                if tool_name == "*":
                    return "Tool import blocked: wildcard OmniInfra tool imports are not allowed."
                registered_module = tool_to_module.get(tool_name)
                if registered_module is None or not self._is_tool_available(tool_name):
                    return "Tool import blocked: the requested function is not available to this A1 run."
                if registered_module != module_name:
                    return "Tool import blocked: use the exact module shown in the current Function Dictionary."
                if enforce_selection and tool_name not in selected_names:
                    return "Tool import blocked: use only tools selected in the current Function Dictionary."

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                if node.func.id in dynamic_import_names:
                    return "Dynamic imports are blocked in A1-generated Python code."
                called_tool = call_aliases.get(node.func.id)
                if called_tool is not None:
                    if not self._is_tool_available(called_tool):
                        return "Tool call blocked: the requested function is not available to this A1 run."
                    if enforce_selection and called_tool not in selected_names:
                        return "Tool call blocked: use only tools selected in the current Function Dictionary."
            if not isinstance(node.func, ast.Attribute):
                continue
            if (
                node.func.attr == "import_module"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in importlib_aliases
            ) or (
                node.func.attr == "__import__"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in builtins_aliases
            ):
                return "Dynamic imports are blocked in A1-generated Python code."

        return None

    def _validate_omniInfra_tool_arguments(self, code: str) -> str | None:
        """Reject unknown keyword arguments for registered OmniInfra tools."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None
        schemas = {}
        for tools in getattr(self, "module2api", {}).values():
            for tool in tools:
                if not isinstance(tool, dict) or not tool.get("name"):
                    continue
                names = {
                    str(item.get("name"))
                    for key in ("required_parameters", "optional_parameters")
                    for item in tool.get(key, []) or []
                    if isinstance(item, dict) and item.get("name")
                }
                schemas[str(tool["name"])] = names
        aliases = {name: name for name in schemas}
        aliases.update(getattr(self, "_tool_call_aliases", {}))
        aliases.update(self._omniInfra_tool_aliases(code, registered_names=set(schemas)))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            tool_name = aliases.get(node.func.id)
            if tool_name is None or tool_name not in schemas:
                continue
            for keyword in node.keywords:
                if keyword.arg is not None and keyword.arg not in schemas[tool_name]:
                    return f"Tool argument blocked: {tool_name} does not declare keyword '{keyword.arg}'."
        return None

    @staticmethod
    def _compact_execution_result(result: str, limit: int = 4000) -> str:
        """Keep tool/runtime failures bounded before adding them to A1 context."""
        if len(result) <= limit:
            return result
        if result.lstrip().startswith(("Error:", "Traceback", "{")):
            tail = min(800, limit // 4)
            marker = "\n...[execution output elided]...\n"
            return result[: limit - tail - len(marker)] + marker + result[-tail:]
        marker = "\n...[execution output elided]..."
        return result[: limit - len(marker)] + marker

    @staticmethod
    def _call_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    @classmethod
    def _assignment_names(cls, target: ast.AST) -> set[str]:
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, (ast.Tuple, ast.List)):
            return {name for item in target.elts for name in cls._assignment_names(item)}
        if isinstance(target, ast.Subscript):
            return cls._assignment_names(target.value)
        return set()

    def _registered_tool_names(self) -> set[str]:
        names = {
            str(tool.get("name"))
            for tools in getattr(self, "module2api", {}).values()
            for tool in tools
            if isinstance(tool, dict) and tool.get("name")
        }
        names.update(getattr(self, "_custom_functions", {}))
        return names

    def _first_use_output_inspection(self, code: str) -> tuple[str | None, set[str]]:
        """Require one observable tool result before field-specific access.

        The guard is deliberately output-shape agnostic: it accepts dictionary,
        list, string, MCP, and custom-tool values. It only rejects a first-use
        block that does not print the saved value or reads fields before the
        model has received an observation.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None, set()

        registered = self._registered_tool_names()
        observed = getattr(self, "_observed_tool_outputs", set())
        call_aliases = {name: name for name in registered}
        call_aliases.update(getattr(self, "_tool_call_aliases", {}))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            for imported in node.names:
                if imported.name in registered:
                    call_aliases[imported.asname or imported.name] = imported.name

        first_use_calls: dict[int, str] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            tool_name = call_aliases.get(self._call_name(node.func))
            if tool_name is not None and tool_name not in observed:
                first_use_calls[id(node)] = tool_name

        if not first_use_calls:
            return None, set()

        first_use_assignments: dict[str, set[str]] = {}
        assigned_call_ids: set[int] = set()

        for node in ast.walk(tree):
            value = None
            targets: list[ast.AST] = []
            if isinstance(node, ast.Assign):
                value = node.value
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign):
                value = node.value
                targets = [node.target]
            elif isinstance(node, ast.NamedExpr):
                value = node.value
                targets = [node.target]
            if not isinstance(value, ast.Call) or id(value) not in first_use_calls:
                continue
            tool_name = first_use_calls[id(value)]
            assigned = {name for target in targets for name in self._assignment_names(target)}
            if assigned:
                assigned_call_ids.add(id(value))
                first_use_assignments.setdefault(tool_name, set()).update(assigned)

        unassigned_tools = sorted(
            {tool for call_id, tool in first_use_calls.items() if call_id not in assigned_call_ids}
        )
        if unassigned_tools:
            return (
                "Error: First-use tool output inspection required before execution. "
                f"Save each call to {', '.join(unassigned_tools)} in a variable, then use only print(type(result)) "
                "and print(result). End the execute block before using .get() or [] to read fields.",
                set(),
            )

        result_names = {name for names in first_use_assignments.values() for name in names}
        tainted_by = {name: {name} for name in result_names}

        def roots_in(node: ast.AST | None) -> set[str]:
            if node is None:
                return set()
            return {
                root
                for nested in ast.walk(node)
                if isinstance(nested, ast.Name)
                for root in tainted_by.get(nested.id, set())
            }

        def add_taint(target: ast.AST, roots: set[str]) -> bool:
            changed = False
            for name in self._assignment_names(target):
                previous = tainted_by.setdefault(name, set())
                if not roots <= previous:
                    previous.update(roots)
                    changed = True
            return changed

        changed = True
        while changed:
            changed = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    roots = roots_in(node.value)
                    changed |= any(add_taint(target, roots) for target in node.targets)
                elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
                    changed |= add_taint(node.target, roots_in(node.value))
                elif isinstance(node, (ast.For, ast.AsyncFor)):
                    if isinstance(node.target, (ast.Tuple, ast.List)) and isinstance(node.iter, (ast.Tuple, ast.List)):
                        rows = [item for item in node.iter.elts if isinstance(item, (ast.Tuple, ast.List))]
                        if rows and all(len(row.elts) == len(node.target.elts) for row in rows):
                            for index, target in enumerate(node.target.elts):
                                roots = {root for row in rows for root in roots_in(row.elts[index])}
                                changed |= add_taint(target, roots)
                            continue
                    changed |= add_taint(node.target, roots_in(node.iter))

        printed_names: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or self._call_name(node.func) != "print":
                continue
            for argument in node.args:
                printed_names.update(roots_in(argument))

        unprinted = sorted(result_names - printed_names)
        if unprinted:
            tools = ", ".join(sorted(first_use_assignments))
            variables = ", ".join(unprinted)
            example_variable = unprinted[0]
            return (
                "Error: First-use tool output inspection required before execution. "
                f"Save {variables} from {tools}, then use only print(type({example_variable})) and "
                f"print({example_variable}); bare expressions show nothing. "
                "End the execute block after printing, then reuse the saved variable to read fields in the next turn; "
                "do not call the tool again with the same arguments.",
                set(),
            )

        unsafe_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
                unsafe_names.update(roots_in(node.func.value))
            elif (
                isinstance(node, ast.Subscript)
                and isinstance(node.ctx, ast.Load)
                and not isinstance(node.slice, ast.Slice)
            ):
                unsafe_names.update(roots_in(node.value))

        if unsafe_names:
            sorted_unsafe_names = sorted(unsafe_names)
            variables = ", ".join(sorted_unsafe_names)
            example_variable = sorted_unsafe_names[0]
            return (
                "Error: First-use tool output fields were read before observation. "
                f"For {variables}, use only print(type({example_variable})) and "
                f"print({example_variable}) in this block. Do not use .get() or []; end the execute block "
                "and inspect fields from the saved variable in the next turn.",
                set(),
            )

        return None, set(first_use_assignments)

    _CONSERVATIVE_CHARS_PER_TOKEN = 3
    _MESSAGE_OVERHEAD_CHARS = 512
    _RESOURCE_TEMPLATE_OVERHEAD_CHARS = 2048
    _MAX_RETRIEVED_KNOW_HOW_CHARS = 3000
    _MIN_LATER_ROUND_RESOURCE_CHARS = 2500
    _MAX_COMPACTED_OBSERVATION_CHARS = 1600
    _MAX_COMPACTED_RECENT_MESSAGE_CHARS = 2400

    @classmethod
    def _estimate_tokens(cls, character_count: int) -> int:
        return (max(0, character_count) + cls._CONSERVATIVE_CHARS_PER_TOKEN - 1) // cls._CONSERVATIVE_CHARS_PER_TOKEN

    @staticmethod
    def _message_character_count(messages: list[BaseMessage]) -> int:
        return sum(len(str(message.content)) for message in messages)

    def _resource_char_budget(
        self,
        prompt: str,
        messages: list[BaseMessage] | None = None,
        *,
        extra_system_chars: int = 0,
    ) -> int | None:
        """Return the current prompt space available for retrieved resources.

        ``messages`` is supplied immediately before each provider call. This is
        intentionally different from budgeting once from the original user
        prompt: ReAct replies and observations consume the same context window
        and therefore reduce the resource allowance on later rounds.
        """
        if self.context_window_tokens is None:
            return None
        input_tokens = self.context_window_tokens - self.max_output_tokens - self.token_safety_margin
        input_chars = input_tokens * self._CONSERVATIVE_CHARS_PER_TOKEN
        message_chars = len(prompt) if messages is None else self._message_character_count(messages)
        fixed_chars = (
            len(self.base_system_prompt)
            + message_chars
            + self._MESSAGE_OVERHEAD_CHARS
            + self._RESOURCE_TEMPLATE_OVERHEAD_CHARS
            + max(0, int(extra_system_chars))
        )
        return max(0, input_chars - fixed_chars)

    def _tool_module_name(self, tool: dict) -> str:
        module_name = tool.get("module")
        if module_name:
            return str(module_name)
        for module, apis in self.module2api.items():
            if any(api.get("name") == tool.get("name") for api in apis):
                return module
        return "omniInfra.tool.scRNA_tools"

    def _resource_prompt_cost(self, category: str, item) -> int:
        if category == "tools" and isinstance(item, dict):
            module_name = self._tool_module_name(item)
            return len(textify_api_dict({module_name: [item]}))
        if category == "data_lake":
            name = item.get("name", "") if isinstance(item, dict) else str(item)
            description = self.data_lake_dict.get(name, f"Data lake item: {name}")
            return len(name) + len(description) + 32
        if category == "libraries":
            name = item.get("name", "") if isinstance(item, dict) else str(item)
            description = self.library_content_dict.get(name, f"Software library: {name}")
            return len(name) + len(description) + 32
        if category == "know_how" and isinstance(item, dict):
            return len(str(item.get("name", ""))) + len(str(item.get("content", ""))) + 64
        return len(str(item)) + 32

    def _limit_selected_resources(
        self,
        selected_resources: dict,
        prompt: str,
        messages: list[BaseMessage] | None = None,
        *,
        extra_system_chars: int = 0,
    ) -> dict:
        """Pack ranked resources into the model-specific input budget.

        Tools are packed first because their schemas are required to execute the
        task. Lower-priority data, library, and know-how context consumes only
        the space left after the ranked tool candidates. The method is called on
        every generation, so later ReAct rounds automatically shed optional
        resources as the message history grows.
        """
        categories = ("tools", "data_lake", "libraries", "know_how")
        if self.context_window_tokens is None:
            return {category: list(selected_resources.get(category, [])) for category in categories}
        packed = {category: [] for category in categories}
        remaining = self._resource_char_budget(
            prompt,
            messages,
            extra_system_chars=extra_system_chars,
        )
        skipped = 0

        for category in categories:
            for item in selected_resources.get(category, []):
                cost = self._resource_prompt_cost(category, item)
                if cost <= remaining:
                    packed[category].append(item)
                    remaining -= cost
                else:
                    skipped += 1
        if skipped:
            print(f"Context budget omitted {skipped} retrieved resource(s); {remaining} resource characters remain.")
        return packed

    @staticmethod
    def _selected_resource_names(selected_resources: dict) -> dict[str, list[str]]:
        names: dict[str, list[str]] = {}
        for category, items in selected_resources.items():
            names[category] = [
                str(item.get("name", item.get("id", ""))) if isinstance(item, dict) else str(item) for item in items
            ]
        return names

    @staticmethod
    def _copy_message_with_content(message: BaseMessage, content: str) -> BaseMessage:
        if hasattr(message, "model_copy"):
            return message.model_copy(update={"content": content})
        return message.copy(update={"content": content})

    @staticmethod
    def _response_finish_reason(response: Any) -> str | None:
        """Return a provider finish reason without depending on one SDK shape."""
        for container_name in ("response_metadata", "generation_info", "additional_kwargs"):
            container = getattr(response, container_name, None)
            if not isinstance(container, dict):
                continue
            for key in ("finish_reason", "stop_reason"):
                value = container.get(key)
                if value is not None:
                    return str(value).strip().lower()
        return None

    @staticmethod
    def _finish_reason_is_truncated(finish_reason: str | None) -> bool:
        """Recognize provider reasons that mean the generated response was cut off."""
        if finish_reason is None:
            return False
        normalized = finish_reason.replace("-", "_").replace(" ", "_")
        return normalized in {
            "length",
            "max_length",
            "max_tokens",
            "max_output_tokens",
            "token_limit",
        }

    @staticmethod
    def _finish_reason_allows_stop_tag_completion(finish_reason: str | None) -> bool:
        """Allow closing a tag only when a provider reports a normal stop."""
        return finish_reason in {"stop", "end_turn", "complete", "completed"}

    @staticmethod
    def _python_syntax_error(code: str) -> str | None:
        """Validate generated Python before reserving execution budget."""
        try:
            compile(code, "<a1-generated>", "exec")
        except SyntaxError as exc:
            location = f"line {exc.lineno}" if exc.lineno is not None else "unknown line"
            return f"{exc.msg} ({location})"
        return None

    @staticmethod
    def _bash_syntax_error(code: str) -> str | None:
        """Validate generated Bash syntax before reserving execution budget."""
        try:
            completed = subprocess.run(
                ["bash", "-n"],
                input=code,
                text=True,
                capture_output=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return f"Bash syntax checker unavailable: {type(exc).__name__}: {exc}"
        if completed.returncode:
            detail = (completed.stderr or completed.stdout or "syntax check failed").strip().splitlines()
            return detail[-1][:300] if detail else "Bash syntax check failed"
        return None

    @classmethod
    def _compact_failed_provider_turns(cls, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Compact known failed attempts even when no context window is configured."""
        compacted = list(messages)
        recent_start = max(1, len(compacted) - 4)
        for index, message in enumerate(compacted):
            additional_kwargs = getattr(message, "additional_kwargs", {})
            if isinstance(additional_kwargs, dict) and (
                additional_kwargs.get("a1_response_error")
                or additional_kwargs.get("a1_execution_error")
            ):
                compacted[index] = cls._truncate_provider_message(message, 800)
                continue

            content = str(message.content)
            if index >= recent_start or "[a1 normalized observation: failure_kind=" not in content.lower():
                continue
            compacted[index] = cls._truncate_provider_message(message, cls._MAX_COMPACTED_OBSERVATION_CHARS)
            previous_index = index - 1
            if previous_index > 0 and "<execute>" in str(compacted[previous_index].content).lower():
                compacted[previous_index] = cls._truncate_provider_message(
                    compacted[previous_index],
                    cls._MAX_COMPACTED_RECENT_MESSAGE_CHARS,
                )
        return compacted

    @classmethod
    def _truncate_provider_message(cls, message: BaseMessage, limit: int) -> BaseMessage:
        content = str(message.content)
        if len(content) <= limit:
            return message
        marker = "\n...[provider context compacted; full message remains in the A1 execution log]...\n"
        available = max(0, limit - len(marker))
        head_chars = (available * 3) // 4
        compacted = content[:head_chars] + marker + content[-(available - head_chars) :]
        return cls._copy_message_with_content(message, compacted)

    def _provider_history_char_budget(self) -> int | None:
        if self.context_window_tokens is None:
            return None
        available_input_chars = (
            self.context_window_tokens - self.max_output_tokens - self.token_safety_margin
        ) * self._CONSERVATIVE_CHARS_PER_TOKEN
        resource_reserve = (
            self._MIN_LATER_ROUND_RESOURCE_CHARS
            if getattr(
                self,
                "_active_selected_resources",
                None,
            )
            else 0
        )
        return max(
            len(getattr(self, "user_task", "")) + 512,
            available_input_chars
            - len(self.base_system_prompt)
            - self._MESSAGE_OVERHEAD_CHARS
            - self._RESOURCE_TEMPLATE_OVERHEAD_CHARS
            - resource_reserve,
        )

    def _compact_messages_for_provider(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Build an auditable bounded view while retaining the full graph state.

        The original task, recent turns, and scientific observations are
        prioritized. Oversized observations carry an explicit compaction marker;
        nothing is deleted from LangGraph state or the execution log.
        """
        messages = self._compact_failed_provider_turns(messages)
        budget = self._provider_history_char_budget()
        if budget is None:
            return messages
        if self._message_character_count(messages) <= budget:
            return messages

        selected_by_index: dict[int, BaseMessage] = {}
        if messages:
            selected_by_index[0] = messages[0]
        for index, message in enumerate(messages[1:], start=1):
            if "<observation>" in str(message.content).lower():
                selected_by_index[index] = self._truncate_provider_message(
                    message,
                    self._MAX_COMPACTED_OBSERVATION_CHARS,
                )
        for index in range(max(1, len(messages) - 4), len(messages)):
            selected_by_index[index] = self._truncate_provider_message(
                messages[index],
                self._MAX_COMPACTED_RECENT_MESSAGE_CHARS,
            )

        notice = HumanMessage(
            content=(
                "[Provider context notice: older reasoning turns were omitted and oversized observations were "
                "explicitly compacted for the configured context window. Full messages remain in the execution log.]"
            )
        )
        compacted = [selected_by_index[index] for index in sorted(selected_by_index)]
        compacted.insert(1 if compacted else 0, notice)

        while self._message_character_count(compacted) > budget and len(compacted) > 2:
            candidates = [
                (len(str(message.content)), index)
                for index, message in enumerate(compacted)
                if index not in {0, 1, len(compacted) - 1}
            ]
            if not candidates:
                break
            length, index = max(candidates)
            excess = self._message_character_count(compacted) - budget
            target = max(400, length - excess)
            if length <= 400:
                compacted.pop(index)
            else:
                compacted[index] = self._truncate_provider_message(compacted[index], target)

        if self._message_character_count(compacted) > budget and compacted:
            final_index = len(compacted) - 1
            allowed = max(
                400,
                budget - self._message_character_count(compacted[:-1]),
            )
            compacted[final_index] = self._truncate_provider_message(compacted[final_index], allowed)
        return compacted

    def _repack_system_prompt_for_messages(
        self,
        messages: list[BaseMessage],
        *,
        extra_system_chars: int = 0,
    ) -> None:
        """Rebuild the dynamic resource block for the current ReAct history."""
        selected_resources = getattr(self, "_active_selected_resources", None)
        if not selected_resources:
            self.system_prompt = self.base_system_prompt
            return

        packed = self._limit_selected_resources(
            selected_resources,
            getattr(self, "user_task", ""),
            messages,
            extra_system_chars=extra_system_chars,
        )
        self.update_system_prompt_with_selected_resources(packed)

        if not hasattr(self, "context_budget_log"):
            self.context_budget_log = []
        input_chars = (
            len(self.system_prompt)
            + self._message_character_count(messages)
            + self._MESSAGE_OVERHEAD_CHARS
            + max(0, int(extra_system_chars))
        )
        self.context_budget_log.append(
            {
                "round": len(self.context_budget_log) + 1,
                "message_chars": self._message_character_count(messages),
                "system_prompt_chars": len(self.system_prompt),
                "estimated_input_tokens": self._estimate_tokens(input_chars),
                "available_input_tokens": (
                    None
                    if self.context_window_tokens is None
                    else self.context_window_tokens - self.max_output_tokens - self.token_safety_margin
                ),
                "selected_resources": self._selected_resource_names(packed),
            }
        )
        latest_budget = self.context_budget_log[-1]
        available = latest_budget["available_input_tokens"]
        budget_text = "unbounded" if available is None else f"{available} tokens"
        print(
            "A1 context round "
            f"{latest_budget['round']}: system={latest_budget['system_prompt_chars']} chars, "
            f"messages={latest_budget['message_chars']} chars, "
            f"estimated_input={latest_budget['estimated_input_tokens']}, available={budget_text}."
        )

    def _validate_context_budget(self, system_prompt: str, messages: list[BaseMessage]) -> None:
        if self.context_window_tokens is None:
            return
        message_chars = sum(len(str(message.content)) for message in messages)
        input_chars = len(system_prompt) + message_chars + self._MESSAGE_OVERHEAD_CHARS
        estimated_input_tokens = self._estimate_tokens(input_chars)
        available_input_tokens = self.context_window_tokens - self.max_output_tokens - self.token_safety_margin
        if estimated_input_tokens > available_input_tokens:
            raise ValueError(
                "A1 prompt exceeds the configured context budget before provider invocation: "
                f"estimated_input_tokens={estimated_input_tokens}, "
                f"available_input_tokens={available_input_tokens}, "
                f"max_output_tokens={self.max_output_tokens}, "
                f"context_window_tokens={self.context_window_tokens}."
            )

    def configure(self, self_critic=False, test_time_scale_round=0):
        """Configure the agent with the initial system prompt and workflow.

        Args:
            self_critic: Whether to enable self-critic mode
            test_time_scale_round: Number of rounds for test time scaling

        """
        # Store self_critic for later use
        self.self_critic = self_critic

        # Build a resource-independent base prompt. Concrete resources are
        # selected and budgeted in go()/go_stream() immediately before use.
        self.system_prompt = self._generate_system_prompt(
            tool_desc={},
            data_lake_content=[],
            library_content_list=[],
            self_critic=self_critic,
            is_retrieval=False,
        )
        self.base_system_prompt = self.system_prompt

        # Define the nodes
        def generate(state: AgentState) -> AgentState:
            # Add OpenAI-specific formatting reminders if using OpenAI models
            system_prompt_suffix = ""
            if hasattr(self.llm, "model_name") and (
                "gpt" in str(self.llm.model_name).lower() or "openai" in str(type(self.llm)).lower()
            ):
                system_prompt_suffix = "\n\nIMPORTANT FOR GPT MODELS: You MUST use XML tags <execute> or <solution> in EVERY response. Do not use markdown code blocks (```) - use <execute> tags instead."

            provider_messages = self._compact_messages_for_provider(state["messages"])
            self._repack_system_prompt_for_messages(
                provider_messages,
                extra_system_chars=len(system_prompt_suffix),
            )
            system_prompt = self.system_prompt + system_prompt_suffix

            self._validate_context_budget(system_prompt, provider_messages)
            messages = [SystemMessage(content=system_prompt)] + provider_messages
            response = self.llm.invoke(messages)
            finish_reason = self._response_finish_reason(response)

            # Normalize Responses API content blocks (list of dicts) into a plain string
            content = response.content
            if isinstance(content, list):
                # Concatenate textual parts; ignore tool_use or other non-text blocks
                text_parts: list[str] = []
                for block in content:
                    try:
                        if isinstance(block, dict):
                            btype = block.get("type")
                            if btype in ("text", "output_text", "redacted_text"):
                                part = block.get("text") or block.get("content") or ""
                                if isinstance(part, str):
                                    text_parts.append(part)
                    except Exception:
                        # Be conservative; skip malformed blocks
                        continue
                msg = "".join(text_parts)
            else:
                # Fallback to string conversion for legacy content
                msg = str(content)

            response_error: str | None = None
            if self._finish_reason_is_truncated(finish_reason):
                response_error = "response_truncated"
            elif self._finish_reason_allows_stop_tag_completion(finish_reason):
                # Provider stop sequences are not included in returned content.
                # Reinsert only the exact configured stop tag after a normal stop;
                # never repair content that ended because of a token limit.
                if "<execute>" in msg and "</execute>" not in msg:
                    msg += "</execute>"
                if "<solution>" in msg and "</solution>" not in msg:
                    msg += "</solution>"
                if "<think>" in msg and "</think>" not in msg:
                    msg += "</think>"
            elif any(
                opening in msg and closing not in msg
                for opening, closing in (
                    ("<execute>", "</execute>"),
                    ("<solution>", "</solution>"),
                    ("<think>", "</think>"),
                )
            ):
                response_error = "response_incomplete"

            # More flexible pattern matching for different LLM styles
            think_match = re.search(r"<think>(.*?)</think>", msg, re.DOTALL | re.IGNORECASE)
            execute_match = re.search(r"<execute>(.*?)</execute>", msg, re.DOTALL | re.IGNORECASE)
            answer_match = re.search(r"<solution>(.*?)</solution>", msg, re.DOTALL | re.IGNORECASE)

            if response_error:
                execute_match = None
                answer_match = None
                think_match = None

            if self._run_control.termination_reason and execute_match:
                msg = self._termination_solution()
                execute_match = None
                answer_match = re.search(r"<solution>(.*?)</solution>", msg, re.DOTALL | re.IGNORECASE)

            # Preserve provider termination metadata so truncated responses are
            # auditable and can be compacted before the next provider request.
            state["messages"].append(
                AIMessage(
                    content=msg.strip(),
                    additional_kwargs={
                        "a1_finish_reason": finish_reason,
                        **({"a1_response_error": response_error} if response_error else {}),
                    },
                )
            )

            if response_error:
                print(f"response error: {response_error} (finish_reason={finish_reason or 'unknown'})")
                remaining = self._run_control.record_response_format_failure(response_error)
                if self._run_control.termination_reason:
                    state["messages"].append(AIMessage(content=self._termination_solution()))
                    state["next_step"] = "end"
                else:
                    state["messages"].append(
                        HumanMessage(
                            content=(
                                "The previous response was incomplete and was not executed. "
                                "Regenerate a shorter, complete response with exactly one closed "
                                "<execute>...</execute> or <solution>...</solution> block. "
                                f"Remaining format retries: {remaining}."
                            )
                        )
                    )
                    state["next_step"] = "generate"
            elif answer_match:
                violation = solution_invariant_violation(
                    answer_match.group(1),
                    self._run_control.observation_kinds,
                )
                if violation and self._run_control.solution_rewrites < 1:
                    self._run_control.solution_rewrites += 1
                    state["messages"].append(
                        HumanMessage(
                            content=(
                                "Rewrite the solution once using only observed evidence. "
                                f"Policy correction: {violation}"
                            )
                        )
                    )
                    state["next_step"] = "generate"
                elif violation:
                    self._run_control.termination_reason = "evidence_insufficient"
                    state["messages"].append(AIMessage(content=self._termination_solution()))
                    state["next_step"] = "end"
                else:
                    state["next_step"] = "end"
            elif execute_match:
                state["next_step"] = "execute"
            elif think_match:
                state["next_step"] = "generate"
            else:
                print("parsing error...")
                remaining = self._run_control.record_response_format_failure("response_format_error")
                state["messages"][-1].additional_kwargs["a1_response_error"] = "response_format_error"

                if self._run_control.termination_reason:
                    print("Detected repeated parsing errors, ending conversation")
                    state["next_step"] = "end"
                    state["messages"].append(AIMessage(content=self._termination_solution()))
                else:
                    state["messages"].append(
                        HumanMessage(
                            content=(
                                "The previous response did not follow the required protocol and was not executed. "
                                "Regenerate a shorter response with exactly one closed <execute>...</execute> "
                                "or <solution>...</solution> block. "
                                f"Remaining format retries: {remaining}."
                            )
                        )
                    )
                    state["next_step"] = "generate"
            return state

        def execute(state: AgentState) -> AgentState:
            last_message = state["messages"][-1].content
            execute_match = re.search(r"<execute>(.*?)</execute>", last_message, re.DOTALL | re.IGNORECASE)
            if execute_match:
                code = execute_match.group(1)
                observed_after_execution: set[str] = set()

                # Set timeout duration (10 minutes = 600 seconds)
                timeout = self.timeout_seconds

                stripped = code.strip()
                if stripped.startswith(("#!R", "# R code", "# R script")):
                    execution_language = "r"
                elif stripped.startswith(("#!BASH", "# Bash script", "#!CLI")):
                    execution_language = "bash"
                else:
                    execution_language = "python"

                if execution_language == "python":
                    syntax_error = self._python_syntax_error(code)
                elif execution_language == "bash":
                    syntax_error = self._bash_syntax_error(code)
                else:
                    syntax_error = None
                tool_actions = self._omniInfra_tool_actions(code) if execution_language == "python" else []
                admission = self.execution_admission.inspect(
                    code,
                    language=execution_language,
                    offline_network_tools=self._selected_network_tool_names(),
                )

                def admit_execution() -> str | None:
                    """Reserve run budget only after all non-executing guards pass."""
                    budget_reason = self._run_control.admit_action(code, tool_actions=tool_actions)
                    if budget_reason is None:
                        return None
                    return json.dumps(
                        {
                            "success": False,
                            "failure_kind": budget_reason,
                            "error": "A1 execution budget terminated this run",
                        },
                        ensure_ascii=False,
                    )

                disabled_tools = self._disabled_tools_in_code(code)
                if syntax_error:
                    if state["messages"] and isinstance(state["messages"][-1], AIMessage):
                        state["messages"][-1].additional_kwargs["a1_execution_error"] = "syntax_error"
                    remaining_code_failures = self._run_control.record_generated_code_failure()
                    result = json.dumps(
                        {
                            "success": False,
                            "failure_kind": "invalid_generated_code",
                            "error": f"Generated {execution_language.capitalize()} was not executed: {syntax_error}",
                            "remaining_code_failures": remaining_code_failures,
                        },
                        ensure_ascii=False,
                    )
                    self._run_control.events.append(
                        {
                            "event": "generated_code_rejected",
                            "reason": "syntax_error",
                            "error": syntax_error,
                        }
                    )
                elif not admission.allowed:
                    alternatives = self._recover_tools_for_policy_block(code)
                    remaining = self._run_control.record_policy_rejection()
                    result = admission.observation(
                        alternatives=alternatives,
                        remaining_policy_budget=remaining,
                    )
                    self._run_control.events.append(
                        {
                            "event": "action_rejected",
                            "reason": admission.reason_code,
                            "alternatives": alternatives,
                        }
                    )
                elif disabled_tools:
                    result = "Error: A1 execution blocked an unavailable tool. Use a listed tool instead."
                # Check if the code is R code
                elif (
                    code.strip().startswith("#!R")
                    or code.strip().startswith("# R code")
                    or code.strip().startswith("# R script")
                ):
                    result = admit_execution()
                    if result is None:
                        # Remove the R marker and run as R code
                        r_code = re.sub(r"^#!R|^# R code|^# R script", "", code, count=1).strip()
                        result = run_with_timeout(run_r_code, [r_code], timeout=timeout)
                # Check if the code is a Bash script or CLI command
                elif (
                    code.strip().startswith("#!BASH")
                    or code.strip().startswith("# Bash script")
                    or code.strip().startswith("#!CLI")
                ):
                    result = admit_execution()
                    if result is None:
                        # Handle both Bash scripts and CLI commands with the same function
                        if code.strip().startswith("#!CLI"):
                            # For CLI commands, extract the command and run it as a simple bash script
                            cli_command = re.sub(r"^#!CLI", "", code, count=1).strip()
                            # Remove any newlines to ensure it's a single command
                            cli_command = cli_command.replace("\n", " ")
                            result = run_with_timeout(run_bash_script, [cli_command], timeout=timeout)
                        else:
                            # For Bash scripts, remove the marker and run as a bash script
                            bash_script = re.sub(r"^#!BASH|^# Bash script", "", code, count=1).strip()
                            result = run_with_timeout(run_bash_script, [bash_script], timeout=timeout)
                # Otherwise, run as Python code
                else:
                    import_error = self._validate_omniInfra_tool_imports(code)
                    if import_error:
                        result = f"Error: {import_error}"
                    else:
                        argument_error = self._validate_omniInfra_tool_arguments(code)
                        if argument_error:
                            result = f"Error: {argument_error}"
                        else:
                            inspection_error, observed_after_execution = self._first_use_output_inspection(code)
                            if inspection_error:
                                result = inspection_error
                            else:
                                result = admit_execution()
                                if result is None:
                                    # Clear any previous plots before execution
                                    self._clear_execution_plots()

                                    # Inject custom functions into the Python execution environment
                                    self._inject_custom_functions_to_repl()
                                    self._tool_call_aliases.update(self._omniInfra_tool_aliases(code))
                                    result = run_with_timeout(run_python_repl, [code], timeout=timeout)

                                    if (
                                        observed_after_execution
                                        and result.strip()
                                        and not result.lstrip().startswith("Error:")
                                    ):
                                        self._observed_tool_outputs.update(observed_after_execution)

                    # Plots are now captured directly in the execution entry above

                if admission.audit_warning:
                    self._run_control.events.append(
                        {
                            "event": "legacy_open_network",
                            "detections": list(admission.detections),
                        }
                    )

                failure_kind, evidence_guidance = normalize_observation(result)
                self._run_control.record_observation(failure_kind)
                if failure_kind:
                    result += (
                        f"\n\n[A1 normalized observation: failure_kind={failure_kind}; guidance={evidence_guidance}]"
                    )
                if self._run_control.termination_reason:
                    result += f"\n[A1 termination_reason={self._run_control.termination_reason}]"

                if len(result) > 10000:
                    result = (
                        "The output is too long to be added to context. Here are the first 10K characters...\n"
                        + result[:10000]
                    )
                result = self._compact_execution_result(result)

                # Store the execution result with the triggering message
                if not hasattr(self, "_execution_results"):
                    self._execution_results = []

                # Get any plots that were generated during this execution
                execution_plots = []
                try:
                    from omniInfra.tool.support_tools import get_captured_plots

                    current_plots = get_captured_plots()
                    execution_plots = current_plots.copy()
                except Exception as e:
                    print(f"Warning: Could not capture plots from execution: {e}")
                    execution_plots = []

                # Store the execution result with metadata
                execution_entry = {
                    "triggering_message": last_message,  # The AI message that contained <execute>
                    "images": execution_plots,  # Base64 encoded images from this execution
                    "timestamp": datetime.now().isoformat(),
                    "network_policy": self.network_policy,
                    "termination_reason": self._run_control.termination_reason,
                }
                self._execution_results.append(execution_entry)

                observation = f"\n<observation>{result}</observation>"
                state["messages"].append(AIMessage(content=observation.strip()))

            return state

        def routing_function(
            state: AgentState,
        ) -> Literal["execute", "generate", "end"]:
            next_step = state.get("next_step")
            if next_step == "execute":
                return "execute"
            elif next_step == "generate":
                return "generate"
            elif next_step == "end":
                return "end"
            else:
                raise ValueError(f"Unexpected next_step: {next_step}")

        def routing_function_self_critic(
            state: AgentState,
        ) -> Literal["generate", "end"]:
            next_step = state.get("next_step")
            if next_step == "generate":
                return "generate"
            elif next_step == "end":
                return "end"
            else:
                raise ValueError(f"Unexpected next_step: {next_step}")

        def execute_self_critic(state: AgentState) -> AgentState:
            if self.critic_count < test_time_scale_round:
                # Generate feedback based on message history
                messages = state["messages"]
                feedback_prompt = f"""
                Here is a reminder of what is the user requested: {self.user_task}
                Examine the previous executions, reaosning, and solutions.
                Critic harshly on what could be improved?
                Be specific and constructive.
                Think hard what are missing to solve the task.
                No question asked, just feedbacks.
                """
                feedback_messages = messages + [HumanMessage(content=feedback_prompt)]
                self._validate_context_budget("", feedback_messages)
                feedback = self.llm.invoke(feedback_messages)

                # Add feedback as a new message
                state["messages"].append(
                    HumanMessage(
                        content=f"Wait... this is not enough to solve the task. Here are some feedbacks for improvement:\n{feedback.content}"
                    )
                )
                self.critic_count += 1
                state["next_step"] = "generate"
            else:
                state["next_step"] = "end"

            return state

        # Create the workflow
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("generate", generate)
        workflow.add_node("execute", execute)

        if self_critic:
            workflow.add_node("self_critic", execute_self_critic)
            # Add conditional edges
            workflow.add_conditional_edges(
                "generate",
                routing_function,
                path_map={
                    "execute": "execute",
                    "generate": "generate",
                    "end": "self_critic",
                },
            )
            workflow.add_conditional_edges(
                "self_critic",
                routing_function_self_critic,
                path_map={"generate": "generate", "end": END},
            )
        else:
            # Add conditional edges
            workflow.add_conditional_edges(
                "generate",
                routing_function,
                path_map={"execute": "execute", "generate": "generate", "end": END},
            )
        workflow.add_edge("execute", "generate")
        workflow.add_edge(START, "generate")

        # Compile the workflow
        self.app = workflow.compile()
        self.checkpointer = MemorySaver()
        self.app.checkpointer = self.checkpointer
        # display(Image(self.app.get_graph().draw_mermaid_png()))

    def _prepare_resources_for_retrieval(self, prompt):
        """Prepare resources for retrieval and return selected resource names.

        Args:
            prompt: The user's query

        Returns:
            dict: Dictionary containing selected resource names for tools, data_lake, and libraries
        """
        # Gather all available resources
        # 1. Tools from the registry
        all_tools = (
            [tool for tool in self.tool_registry.tools if self._is_tool_available(tool["name"])]
            if hasattr(self, "tool_registry")
            else []
        )

        # 2. Data lake items with descriptions
        data_lake_path = self.path + "/data_lake"
        data_lake_content = glob.glob(data_lake_path + "/*")
        data_lake_items = [x.split("/")[-1] for x in data_lake_content]

        # Create data lake descriptions for retrieval
        data_lake_descriptions = []
        for item in data_lake_items:
            description = self.data_lake_dict.get(item, f"Data lake item: {item}")
            data_lake_descriptions.append({"name": item, "description": description})

        # Add custom data items to retrieval if they exist
        if hasattr(self, "_custom_data") and self._custom_data:
            for name, info in self._custom_data.items():
                data_lake_descriptions.append({"name": name, "description": info["description"]})

        # 3. Libraries with descriptions - use library_content_dict directly
        library_descriptions = []
        for lib_name, lib_desc in self.library_content_dict.items():
            library_descriptions.append({"name": lib_name, "description": lib_desc})

        # Add custom software items to retrieval if they exist
        if hasattr(self, "_custom_software") and self._custom_software:
            for name, info in self._custom_software.items():
                # Check if it's not already in the library descriptions to avoid duplicates
                if not any(lib["name"] == name for lib in library_descriptions):
                    library_descriptions.append({"name": name, "description": info["description"]})

        # 4. Know-how documents
        know_how_summaries = self.know_how_loader.get_document_summaries()

        # Use retrieval to get relevant resources
        resources = {
            "tools": all_tools,
            "data_lake": data_lake_descriptions,
            "libraries": library_descriptions,
            "know_how": know_how_summaries,
        }

        catalog_char_budget = self._resource_char_budget(prompt)
        if catalog_char_budget is not None and catalog_char_budget <= 0:
            selected_resources = {"tools": [], "data_lake": [], "libraries": [], "know_how": []}
            retrieval_mode = "base prompt only because no resource budget remains"
        elif self.use_tool_retriever:
            # Optional LLM selection is applied only after local Qwen3 semantic
            # recall and reranking, and uses a small response allowance.
            selected_resources = self.retriever.prompt_based_retrieval(
                prompt,
                resources,
                llm=self.llm,
                catalog_char_budget=catalog_char_budget,
                max_output_tokens=min(512, self.max_output_tokens),
            )
            retrieval_mode = "Qwen3 embedding + reranker + bounded LLM selection"
        else:
            selected_resources = self.retriever.local_retrieval(
                prompt,
                resources,
                catalog_char_budget=catalog_char_budget,
            )
            retrieval_mode = "Qwen3 embedding + reranker"
        self.retrieval_diagnostics = self.retriever.last_retrieval_diagnostics
        print("\n" + "=" * 60)
        print("🔍 RESOURCE RETRIEVAL")
        print("=" * 60)
        print(f"Using {retrieval_mode}")
        tool_diagnostics = self.retrieval_diagnostics.get("categories", {}).get("tools", [])
        if tool_diagnostics:
            semantic_top = sorted(tool_diagnostics, key=lambda item: item["embedding_rank"])[:16]
            reranked_top = sorted(tool_diagnostics, key=lambda item: item["final_rank"])[:16]
            print("Embedding tool candidates: " + ", ".join(item["name"] for item in semantic_top))
            print("Embedding/reranker fused tools: " + ", ".join(item["name"] for item in reranked_top))

        # Extract the names from the selected resources for the system prompt
        selected_resources_names = {
            "tools": selected_resources["tools"],
            "data_lake": [],
            "libraries": [lib["name"] if isinstance(lib, dict) else lib for lib in selected_resources["libraries"]],
            "know_how": [],
        }

        # Process data lake items to extract just the names
        for item in selected_resources["data_lake"]:
            if isinstance(item, dict):
                selected_resources_names["data_lake"].append(item["name"])
            elif isinstance(item, str) and ": " in item:
                # If the item already has a description, extract just the name
                name = item.split(": ")[0]
                selected_resources_names["data_lake"].append(name)
            else:
                selected_resources_names["data_lake"].append(item)

        # Process know-how documents - get the full content for selected documents
        if "know_how" in selected_resources and selected_resources["know_how"]:
            print("\n📚 Know-How Documents Retrieved:")
            for item in selected_resources["know_how"]:
                if isinstance(item, dict):
                    doc_id = item["id"]
                    doc = self.know_how_loader.get_document_by_id(doc_id)
                    if doc:
                        print(f"  ✓ {doc['name']}")
                        content = doc["content_without_metadata"]
                        if self.context_window_tokens is not None and len(content) > self._MAX_RETRIEVED_KNOW_HOW_CHARS:
                            content = (
                                content[: self._MAX_RETRIEVED_KNOW_HOW_CHARS]
                                + "\n\n[Know-how content truncated to fit the context budget.]"
                            )
                        # Create a copy with content_without_metadata for agent context
                        doc_for_agent = {
                            "id": doc["id"],
                            "name": doc["name"],
                            "description": doc["description"],
                            "content": content,
                            "metadata": doc["metadata"],
                        }
                        selected_resources_names["know_how"].append(doc_for_agent)
        else:
            print("\n📚 Know-How: None retrieved for this query")

        # Print summary of what was retrieved
        print("\n" + "-" * 60)
        print("📊 RETRIEVAL SUMMARY:")
        print(f"  🔧 Tools: {len(selected_resources_names['tools'])} selected")
        print(f"  📊 Data Lake: {len(selected_resources_names['data_lake'])} selected")
        print(f"  ⚙️  Libraries: {len(selected_resources_names['libraries'])} selected")
        print(f"  📚 Know-How: {len(selected_resources_names['know_how'])} selected")
        print("=" * 60 + "\n")

        return selected_resources_names

    def go(self, prompt):
        """Execute the agent with the given prompt.

        Args:
            prompt: The user's query

        """
        self.critic_count = 0
        self.user_task = prompt
        self.context_budget_log = []
        self._observed_tool_outputs = set()
        self._reset_run_control()

        selected_resources_names = self._prepare_resources_for_retrieval(prompt)
        self._active_selected_resources = selected_resources_names
        self.system_prompt = self.base_system_prompt

        inputs = {"messages": [HumanMessage(content=prompt)], "next_step": None}
        config = {"recursion_limit": 500, "configurable": {"thread_id": 42}}
        self.log = []

        # Store the final conversation state for markdown generation
        final_state = None

        for s in self.app.stream(inputs, stream_mode="values", config=config):
            message = s["messages"][-1]
            out = pretty_print(message)
            self.log.append(out)
            final_state = s  # Store the latest state

        # Store the conversation state for markdown generation
        self._conversation_state = final_state

        return self.log, message.content

    def go_stream(self, prompt) -> Generator[dict, None, None]:
        """Execute the agent with the given prompt and return a generator that yields each step.

        This function returns a generator that yields each step of the agent's execution,
        allowing for real-time monitoring of the agent's progress.

        Args:
            prompt: The user's query

        Yields:
            dict: Each step of the agent's execution containing the current message and state
        """
        self.critic_count = 0
        self.user_task = prompt
        self.context_budget_log = []
        self._observed_tool_outputs = set()
        self._reset_run_control()

        selected_resources_names = self._prepare_resources_for_retrieval(prompt)
        self._active_selected_resources = selected_resources_names
        self.system_prompt = self.base_system_prompt

        inputs = {"messages": [HumanMessage(content=prompt)], "next_step": None}
        config = {"recursion_limit": 500, "configurable": {"thread_id": 42}}
        self.log = []

        # Store the final conversation state for markdown generation
        final_state = None

        for s in self.app.stream(inputs, stream_mode="values", config=config):
            message = s["messages"][-1]
            out = pretty_print(message)
            self.log.append(out)
            final_state = s  # Store the latest state

            # Yield the current step
            yield {"output": out}

        # Store the conversation state for markdown generation
        self._conversation_state = final_state

    def update_system_prompt_with_selected_resources(self, selected_resources):
        """Update the system prompt with the selected resources."""
        # Extract tool descriptions for the selected tools
        tool_desc = {}
        for tool in selected_resources["tools"]:
            tool_name = tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", str(tool))
            if not self._is_tool_available(tool_name):
                continue
            # Get the module name from the tool
            if isinstance(tool, dict):
                module_name = tool.get("module", None)

                # If module is not specified, try to find it in the module2api
                if not module_name and hasattr(self, "module2api"):
                    for mod, apis in self.module2api.items():
                        for api in apis:
                            if api.get("name") == tool.get("name"):
                                module_name = mod
                                # Update the tool with the module information
                                tool["module"] = module_name
                                break
                        if module_name:
                            break

                # If still not found, use a default
                if not module_name:
                    module_name = "omniInfra.tool.scRNA_tools"  # Default to scRNA_tools as a fallback
                    tool["module"] = module_name
            else:
                module_name = getattr(tool, "module_name", None)

                # If module is not specified, try to find it in the module2api
                if not module_name and hasattr(self, "module2api"):
                    tool_name = getattr(tool, "name", str(tool))
                    for mod, apis in self.module2api.items():
                        for api in apis:
                            if api.get("name") == tool_name:
                                module_name = mod
                                # Set the module_name attribute
                                tool.module_name = module_name
                                break
                        if module_name:
                            break

                # If still not found, use a default
                if not module_name:
                    module_name = "omniInfra.tool.scRNA_tools"  # Default to scRNA_tools as a fallback
                    tool.module_name = module_name

            if module_name not in tool_desc:
                tool_desc[module_name] = []

            # Add the tool to the appropriate module
            if isinstance(tool, dict):
                # Ensure the module is included in the tool description
                if "module" not in tool:
                    tool["module"] = module_name
                tool_desc[module_name].append(tool)
            else:
                # Convert tool object to dictionary
                tool_dict = {
                    "name": getattr(tool, "name", str(tool)),
                    "description": getattr(tool, "description", ""),
                    "parameters": getattr(tool, "parameters", {}),
                    "module": module_name,  # Explicitly include the module
                }
                tool_desc[module_name].append(tool_dict)

        # Prepare data lake items with descriptions
        data_lake_with_desc = []
        for item in selected_resources["data_lake"]:
            description = self.data_lake_dict.get(item, f"Data lake item: {item}")
            data_lake_with_desc.append({"name": item, "description": description})

        # Highlight only custom resources that survived retrieval and context
        # packing. Adding every custom resource here would bypass the same
        # budget that protects built-in resources.
        # Selected custom tools already appear in the Function Dictionary with
        # their full schema. Repeating them in the priority block would charge
        # the context twice for the same resource.
        custom_tools = []

        selected_data_names = set(selected_resources["data_lake"])
        custom_data = [
            {"name": name, "description": info["description"]}
            for name, info in getattr(self, "_custom_data", {}).items()
            if name in selected_data_names
        ]

        selected_library_names = set(selected_resources["libraries"])
        custom_software = [
            {"name": name, "description": info["description"]}
            for name, info in getattr(self, "_custom_software", {}).items()
            if name in selected_library_names
        ]

        # Extract know-how documents if present
        know_how_docs = selected_resources.get("know_how", [])

        self.system_prompt = self._generate_system_prompt(
            tool_desc=tool_desc,
            data_lake_content=data_lake_with_desc,
            library_content_list=selected_resources["libraries"],
            self_critic=getattr(self, "self_critic", False),
            is_retrieval=True,
            custom_tools=custom_tools if custom_tools else None,
            custom_data=custom_data if custom_data else None,
            custom_software=custom_software if custom_software else None,
            know_how_docs=know_how_docs if know_how_docs else None,
        )

        # Print the raw system prompt for debugging
        # print("\n" + "="*20 + " RAW SYSTEM PROMPT FROM AGENT " + "="*20)
        # print(self.system_prompt)
        # print("="*70 + "\n")

    def result_formatting(self, output_class, task_intention):
        self.format_check_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are evaluateGPT, tasked with extract and parse the task output based on the history of an agent. "
                        "Review the entire history of messages provided. "
                        "Here is the task output requirement: \n"
                        f"'{task_intention.replace('{', '{{').replace('}', '}}')}'.\n"
                    ),
                ),
                ("placeholder", "{messages}"),
            ]
        )

        checker_llm = self.format_check_prompt | self.llm.with_structured_output(output_class)
        result = checker_llm.invoke({"messages": [("user", str(self.log))]}).dict()
        return result

    def _parse_tool_calls_from_code(self, code: str) -> list[str]:
        """Parse code to detect imported tools by looking for import statements.

        Args:
            code: The Python code to parse

        Returns:
            List of detected tool names
        """
        module2api = getattr(self, "module2api", {})
        custom_functions = getattr(self, "_custom_functions", {})
        return parse_tool_calls_from_code(code, module2api, custom_functions)

    def _parse_tool_calls_with_modules(self, code: str) -> list[tuple[str, str]]:
        """Parse code to detect imported tools and their modules.

        Args:
            code: The Python code to parse

        Returns:
            List of tuples (tool_name, module_name)
        """
        module2api = getattr(self, "module2api", {})
        custom_functions = getattr(self, "_custom_functions", {})
        return parse_tool_calls_with_modules(code, module2api, custom_functions)

    def _inject_custom_functions_to_repl(self):
        """Inject custom functions into the Python REPL execution environment.
        This makes custom tools available during code execution.
        """
        custom_functions = getattr(self, "_custom_functions", {})
        inject_custom_functions_to_repl(custom_functions)

    def create_mcp_server(self, tool_modules=None):
        """
        Create an MCP server object that exposes internal OmniInfra tools.
        This gives you control over when and how to run the server.

        Args:
            tool_modules: List of module names to expose (default: all in self.module2api)

        Returns:
            FastMCP server object that you can run manually
        """
        import importlib

        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("OmniInfraTools")
        modules = tool_modules or list(self.module2api.keys())

        registered_tools = 0

        for module_name in modules:
            try:
                # Import the actual module
                module = importlib.import_module(module_name)
                # Get tools for this module
                module_tools = self.module2api.get(module_name, [])

                for tool_schema in module_tools:
                    tool_name = tool_schema.get("name")
                    if not tool_name or not self._is_tool_available(tool_name):
                        continue

                    try:
                        # Get the actual function
                        fn = getattr(module, tool_name, None)
                        if fn is None:
                            fn = getattr(self, "_custom_functions", {}).get(tool_name)

                        if fn is None:
                            print(f"Warning: Could not find function '{tool_name}' in module '{module_name}'")
                            continue

                        # Extract parameters from your specific schema format
                        required_params = tool_schema.get("required_parameters", [])
                        optional_params = tool_schema.get("optional_parameters", [])

                        # Generate the wrapper function
                        wrapper_func = self._generate_mcp_wrapper_from_omniInfra_schema(
                            fn, tool_name, required_params, optional_params
                        )

                        # Register with MCP
                        mcp.tool()(wrapper_func)
                        registered_tools += 1

                    except Exception as e:
                        print(f"Warning: Failed to register tool '{tool_name}': {e}")
                        continue

            except ImportError as e:
                print(f"Warning: Could not import module '{module_name}': {e}")
                continue

        print(f"Created MCP server with {registered_tools} tools")
        return mcp

    def save_conversation_history(self, filepath: str, include_images: bool = True, save_pdf: bool = True) -> None:
        """Save the complete conversation history as PDF only.

        This function generates and saves the complete conversation history from the agent's
        log and conversation state. It creates a temporary markdown file with formatted content
        including steps, code execution, observations, and optionally images, then converts it
        to PDF format. The markdown file is automatically cleaned up after PDF conversion.

        Args:
            filepath: Path where to save the PDF file (without extension). If the path doesn't
                    end with .pdf, it will be automatically appended.
            include_images: Whether to include captured plots and images in the output.
                          Defaults to True.
            save_pdf: Whether to save as PDF. Defaults to True. If False, no file is saved.

        Note:
            The function includes a 60-second timeout for PDF generation to prevent
            hanging. A temporary markdown file is created and automatically deleted.
        """
        import os
        import tempfile

        if not save_pdf:
            print("PDF saving is disabled. No file will be saved.")
            return

        # Ensure directory exists
        directory = os.path.dirname(filepath)
        if directory:  # Only create directory if it's not empty
            os.makedirs(directory, exist_ok=True)

        # Create PDF file path - use the user's filename and add .pdf extension
        if filepath.endswith(".pdf"):
            pdf_path = filepath
        else:
            # Remove any existing .md extension if present, then add .pdf
            base_name = filepath
            if base_name.endswith(".md"):
                base_name = base_name[:-3]  # Remove .md extension
            pdf_path = f"{base_name}.pdf"

        # Create markdown content
        markdown_content = self._generate_markdown_content(include_images)

        # Create a temporary markdown file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as temp_file:
            temp_file.write(markdown_content)
            temp_markdown_path = temp_file.name

        try:
            # Add timeout for PDF generation to prevent hanging
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("PDF generation timed out")

            # Set timeout to 60 seconds
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)

            try:
                self._convert_markdown_to_pdf(temp_markdown_path, pdf_path)
                print(f"Conversation history saved as PDF: {pdf_path}")
                print(f"Total steps recorded: {len(self.log)}")
            finally:
                signal.alarm(0)  # Cancel the alarm

        except TimeoutError:
            print("Warning: PDF generation timed out after 60 seconds")
        except Exception as e:
            print(f"Warning: Could not convert to PDF: {e}")
        finally:
            # Clean up the temporary markdown file
            try:
                os.unlink(temp_markdown_path)
            except OSError:
                pass  # File might already be deleted

    def _generate_markdown_content(self, include_images: bool = True) -> str:
        """Generate markdown content from conversation history using both log and conversation state.

        This function processes the agent's conversation history from either the conversation
        state (if available) or the internal log to create a formatted markdown document.
        It handles step numbering, message processing, and content formatting.

        Args:
            include_images: Whether to include captured plots and images in the output.
                          Defaults to True.

        Returns:
            Formatted markdown string containing the complete conversation history
            with proper step numbering and content structure.
        """

        # Initialize content and tracking variables
        content = """# OmniInfra Agent Conversation History

"""
        added_plots = set()
        step_number = 0
        first_human_shown = False

        # Get data source (conversation state or log)
        messages = self._get_messages_for_processing()

        # Process all messages using unified logic
        for message_data in messages:
            content, step_number, first_human_shown = self._process_message(
                message_data, content, step_number, first_human_shown, added_plots, include_images
            )

        return content

    def _get_messages_for_processing(self):
        """Get messages from conversation state or fallback to log.

        This function determines the best source for conversation messages, prioritizing
        the conversation state if available, otherwise falling back to the internal log.
        It normalizes the messages into a unified format for processing.

        Returns:
            List of normalized message dictionaries with 'content', 'type', and 'original' keys
        """
        conversation_state = getattr(self, "_conversation_state", None)

        if conversation_state and hasattr(conversation_state, "get") and "messages" in conversation_state:
            print(f"DEBUG: Using conversation state with {len(conversation_state['messages'])} messages")
            return self._normalize_conversation_state_messages(conversation_state["messages"])
        else:
            print(f"DEBUG: Using self.log with {len(self.log)} entries")
            return self._normalize_log_messages(self.log)

    def _normalize_conversation_state_messages(self, messages):
        """Convert conversation state messages to unified format.

        This function takes LangChain message objects from the conversation state and
        converts them into a standardized dictionary format that the markdown generation
        system can work with. It extracts content and determines message types.

        Args:
            messages: List of LangChain message objects (HumanMessage, AIMessage, etc.)

        Returns:
            List of normalized message dictionaries with 'content', 'type', and 'original' keys
        """
        normalized = []
        for message in messages:
            if hasattr(message, "content"):
                content = str(message.content)
            else:
                content = str(message)

            # Determine message type
            if isinstance(message, HumanMessage):
                msg_type = "human"
            elif isinstance(message, AIMessage):
                msg_type = "ai"
            else:
                msg_type = "other"

            normalized.append({"content": content, "type": msg_type, "original": message})

        return normalized

    def _normalize_log_messages(self, log_entries):
        """Convert log entries to unified format.

        This function takes internal log entries and converts them into the same
        standardized format as conversation state messages. It parses the log format
        to determine message types and extract content.

        Args:
            log_entries: List of log entry strings from the agent's internal log

        Returns:
            List of normalized message dictionaries with 'content', 'type', and 'original' keys
        """
        normalized = []
        for log_entry in log_entries:
            content = str(log_entry)

            # Determine message type from log format
            if "Human Message" in content:
                msg_type = "human"
            elif "Ai Message" in content:
                msg_type = "ai"
            else:
                msg_type = "other"

            normalized.append({"content": content, "type": msg_type, "original": log_entry})

        return normalized

    def _process_message(self, message_data, content, step_number, first_human_shown, added_plots, include_images):
        """Process a single message and return updated state.

        This function is the main dispatcher for processing individual messages in the
        conversation history. It determines the message type and delegates to the
        appropriate processing function.

        Args:
            message_data: Dictionary containing 'content', 'type', and 'original' keys
            content: Current markdown content string
            step_number: Current step number counter
            first_human_shown: Boolean flag indicating if first human message was shown
            added_plots: Set of already added plot data to avoid duplicates
            include_images: Whether to include images in the output

        Returns:
            Tuple of (updated_content, updated_step_number, updated_first_human_shown)
        """
        clean_output = clean_message_content(message_data["content"])
        msg_type = message_data["type"]

        if msg_type == "human":
            return self._process_human_message(clean_output, content, step_number, first_human_shown)
        elif msg_type == "ai":
            return self._process_ai_message(clean_output, content, step_number, added_plots, include_images)
        else:
            return self._process_other_message(
                clean_output, content, step_number, first_human_shown, added_plots, include_images
            )

    def _process_human_message(self, clean_output, content, step_number, first_human_shown):
        """Process human messages.

        This function handles human messages in the conversation history. It identifies
        parsing error messages and displays them appropriately, or formats the first
        human prompt as a special section.

        Args:
            clean_output: Cleaned message content with ANSI codes removed
            content: Current markdown content string
            step_number: Current step number counter (unchanged for human messages)
            first_human_shown: Boolean flag indicating if first human message was shown

        Returns:
            Tuple of (updated_content, step_number, updated_first_human_shown)

        Note:
            Human messages don't increment the step counter as they are not considered
            steps in the agent's process.
        """
        if "each response must include thinking process" in clean_output.lower():
            parsing_error_content = create_parsing_error_html()
            content += f"{parsing_error_content}\n\n"
        elif not first_human_shown:
            content += "#### Human Prompt\n\n"
            content += f"*{clean_output}*\n\n"
            first_human_shown = True

        return content, step_number, first_human_shown  # step_number unchanged

    def _process_ai_message(self, clean_output, content, step_number, added_plots, include_images):
        """Process AI messages.

        This function handles AI messages in the conversation history. It can process
        both regular AI responses and messages containing observation tags. It handles
        step numbering, execution results, and content formatting.

        Args:
            clean_output: Cleaned message content with ANSI codes removed
            content: Current markdown content string
            step_number: Current step number counter
            added_plots: Set of already added plot data to avoid duplicates
            include_images: Whether to include images in the output

        Returns:
            Tuple of (updated_content, updated_step_number, True)

        Note:
            This function can split messages containing observation tags and process
            each part separately, with observations formatted as terminal blocks.
        """
        # Check if this message contains observation tags and process accordingly
        import re

        observation_pattern = r"<observation>(.*?)</observation>"
        observation_matches = re.findall(observation_pattern, clean_output, re.DOTALL | re.IGNORECASE)

        if observation_matches:
            # Extract content before, between, and after observation tags
            parts = re.split(observation_pattern, clean_output, flags=re.DOTALL | re.IGNORECASE)

            # Process each part
            for i, part in enumerate(parts):
                if i % 2 == 0:  # Even indices are non-observation content
                    if part.strip():
                        # This is regular content - process it normally
                        if not should_skip_message(part):
                            if part.strip():
                                step_number += 1
                                content += f"#### Step {step_number}\n\n"

                                # Handle execution results if present
                                execution_results = getattr(self, "_execution_results", None)
                                if has_execution_results(part, execution_results):
                                    content, added_plots = self._process_execution_with_results(
                                        part, content, added_plots, include_images, execution_results
                                    )
                                else:
                                    content = self._process_regular_ai_message(part, content)
                else:  # Odd indices are observation content
                    if part.strip():
                        # This is observation content - format as terminal
                        formatted_observation = format_observation_as_terminal(f"<observation>{part}</observation>")
                        if formatted_observation is not None:
                            content += f"{formatted_observation}\n\n"

            return content, step_number, True

        # Skip empty or error messages
        if should_skip_message(clean_output):
            return content, step_number, True

        if clean_output.strip():
            step_number += 1
            content += f"#### Step {step_number}\n\n"

            # Handle execution results if present
            execution_results = getattr(self, "_execution_results", None)
            if has_execution_results(clean_output, execution_results):
                content, added_plots = self._process_execution_with_results(
                    clean_output, content, added_plots, include_images, execution_results
                )
            else:
                content = self._process_regular_ai_message(clean_output, content)

        return content, step_number, True

    def _process_other_message(
        self, clean_output, content, step_number, first_human_shown, added_plots, include_images
    ):
        """Process other message types.

        This function handles message types that are neither human nor AI messages.
        It checks for observation tags and processes them accordingly, or adds the
        content as regular text.

        Args:
            clean_output: Cleaned message content with ANSI codes removed
            content: Current markdown content string
            step_number: Current step number counter
            first_human_shown: Boolean flag indicating if first human message was shown
            added_plots: Set of already added plot data to avoid duplicates
            include_images: Whether to include images in the output

        Returns:
            Tuple of (updated_content, step_number, first_human_shown)
        """
        # Check if this is actually an observation (has <observation> tags)
        import re

        if not re.search(r"<observation>", clean_output, re.IGNORECASE):
            content += f"{clean_output}\n\n"
        return content, step_number, first_human_shown

    def _process_execution_with_results(self, clean_output, content, added_plots, include_images, execution_results):
        """Process AI message with execution results.

        This function handles AI messages that have associated execution results.
        It finds the matching execution result and adds any captured plots or images
        to the content.

        Args:
            clean_output: Cleaned message content with ANSI codes removed
            content: Current markdown content string
            added_plots: Set of already added plot data to avoid duplicates
            include_images: Whether to include images in the output
            execution_results: List of execution result dictionaries

        Returns:
            Tuple of (updated_content, updated_added_plots)
        """
        matching_execution = find_matching_execution(clean_output, execution_results)

        if matching_execution:
            content = self._format_and_add_content(clean_output, content)
            content, added_plots = self._add_execution_plots(matching_execution, content, added_plots, include_images)
        else:
            content = self._format_and_add_content(clean_output, content)

        return content, added_plots

    def _format_and_add_content(self, clean_output, content):
        """Format and add content to markdown.

        This function applies formatting to AI message content before adding it to the
        markdown. It processes lists, execute tags, and tool calls to create properly
        formatted markdown content.

        Args:
            clean_output: Cleaned message content with ANSI codes removed
            content: Current markdown content string

        Returns:
            Updated markdown content string with formatted content added
        """
        # Process lists first, then execute tags
        formatted_content = format_lists_in_text(clean_output)

        # Create a wrapper function for the tool parsing
        def parse_tool_calls_wrapper(code):
            return self._parse_tool_calls_with_modules(code)

        formatted_content = format_execute_tags_in_content(formatted_content, parse_tool_calls_wrapper)
        return content + f"{formatted_content}\n\n"

    def _add_execution_plots(self, matching_execution, content, added_plots, include_images):
        """Add plots from execution results.

        This function adds captured plots and images from execution results to the
        markdown content. It prevents duplicate plots from being added multiple times.

        Args:
            matching_execution: Execution result dictionary containing image data
            content: Current markdown content string
            added_plots: Set of already added plot data to avoid duplicates
            include_images: Whether to include images in the output

        Returns:
            Tuple of (updated_content, updated_added_plots)
        """
        if include_images and matching_execution.get("images"):
            for plot_data in matching_execution["images"]:
                if plot_data not in added_plots:
                    content += f"![Plot]({plot_data})\n\n"
                    added_plots.add(plot_data)
        return content, added_plots

    def _process_regular_ai_message(self, clean_output, content):
        """Process regular AI message without execution results.

        This function handles AI messages that don't have associated execution results.
        It applies standard formatting and adds the content to the markdown.

        Args:
            clean_output: Cleaned message content with ANSI codes removed
            content: Current markdown content string

        Returns:
            Updated markdown content string with formatted content added
        """
        return self._format_and_add_content(clean_output, content)

    def _convert_markdown_to_pdf(self, markdown_path: str, pdf_path: str) -> None:
        """Convert markdown file to PDF using weasyprint or markdown2pdf.

        This function is a wrapper around the utility function for converting markdown
        to PDF. It provides a clean interface for the agent to convert conversation
        history to PDF format.

        Args:
            markdown_path: Path to the input markdown file
            pdf_path: Path where the output PDF file should be saved

        Note:
            This function delegates to the convert_markdown_to_pdf utility function
            which handles multiple PDF conversion libraries and fallbacks.
        """
        convert_markdown_to_pdf(markdown_path, pdf_path)

    def _clear_execution_plots(self):
        """Clear execution plots before new execution.

        This function clears any previously captured plots from the execution environment
        before starting a new execution. This prevents old plots from appearing in
        new execution results.

        Note:
            This function calls the clear_captured_plots utility function and handles
            any exceptions gracefully to prevent execution failures.
        """
        try:
            from omniInfra.tool.support_tools import clear_captured_plots

            clear_captured_plots()
        except Exception as e:
            print(f"Warning: Could not clear execution plots: {e}")

    def _generate_mcp_wrapper_from_omniInfra_schema(self, original_func, func_name, required_params, optional_params):
        """Generate wrapper function based on OmniInfra schema format."""
        import inspect

        # Combine all parameters
        all_params = required_params + optional_params

        if not all_params:
            # No parameters
            def wrapper() -> dict:
                try:
                    result = original_func()
                    if isinstance(result, dict):
                        return result
                    return {"result": result}
                except Exception as e:
                    return {"error": str(e)}

            wrapper.__name__ = func_name
            wrapper.__doc__ = original_func.__doc__
            return wrapper

        else:
            # Has parameters
            def wrapper(**kwargs) -> dict:
                try:
                    # Build arguments dict
                    filtered_kwargs = {}

                    # Add required parameters
                    for param_info in required_params:
                        param_name = param_info["name"]
                        if param_name in kwargs and kwargs[param_name] is not None:
                            filtered_kwargs[param_name] = kwargs[param_name]

                    # Add optional parameters only if provided and not None
                    for param_info in optional_params:
                        param_name = param_info["name"]
                        if param_name in kwargs and kwargs[param_name] is not None:
                            filtered_kwargs[param_name] = kwargs[param_name]

                    result = original_func(**filtered_kwargs)
                    if isinstance(result, dict):
                        return result
                    return {"result": result}
                except Exception as e:
                    return {"error": str(e)}

            # Set function metadata
            wrapper.__name__ = func_name
            wrapper.__doc__ = original_func.__doc__

            # Create proper signature
            new_params = []

            # Map your types to Python types
            type_map = {"str": str, "int": int, "float": float, "bool": bool, "List[str]": list[str], "dict": dict}

            # Add required parameters
            for param_info in required_params:
                param_name = param_info["name"]
                param_type_str = param_info["type"]
                param_type = type_map.get(param_type_str, str)

                new_params.append(inspect.Parameter(param_name, inspect.Parameter.KEYWORD_ONLY, annotation=param_type))

            # Add optional parameters
            for param_info in optional_params:
                param_name = param_info["name"]
                param_type_str = param_info["type"]
                param_type = type_map.get(param_type_str, str)

                default = param_info.get("default")
                optional_type = param_type | None if default is None else param_type

                new_params.append(
                    inspect.Parameter(
                        param_name, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=optional_type
                    )
                )

            # Set the signature
            wrapper.__signature__ = inspect.Signature(new_params, return_annotation=dict)

            return wrapper

    def launch_gradio_demo(self, thread_id=42, share=False, server_name="0.0.0.0", require_verification=False):
        """Launch a full-featured Gradio UI for the A1 agent (adapted from codeact_copilot).

        Args:
            thread_id: Thread ID for the conversation
            share: Whether to create a public shareable link
            server_name: Server name/IP to bind to (default: "0.0.0.0")
            require_verification: If True, requires access code verification

        Example:
            >>> agent = A1()
            >>> agent.launch_gradio_demo()
        """
        try:
            import gradio as gr
            from gradio import ChatMessage
        except ImportError:
            raise ImportError("Gradio is not installed. Please install it with: pip install gradio") from None

        import os
        from time import time

        # Define supported file extensions
        SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".pdf")

        self.main_history_copy = []

        # Available access codes (if verification is required)
        available_access_codes = ["OmniInfra2025"]

        # Function for verification page
        def verify_access_code(code):
            if code in available_access_codes:
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
            else:
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(value="Incorrect access code. Please check your access code.", visible=True),
                )

        def generate_response(prompt_input, inner_history=None, main_history=None):
            if main_history is None:
                main_history = []
            if inner_history is None:
                inner_history = []
            text_input = prompt_input.get("text", "")
            files = prompt_input.get("files", [])

            self.main_history_copy += [{"role": "user", "content": text_input}]
            main_history.append(ChatMessage(role="user", content=text_input if text_input else "[Uploaded file]"))

            # Add "Executor is working on it" message
            main_history.append(ChatMessage(role="assistant", content="Executor is working on it 👉"))
            yield inner_history, main_history

            # Process uploaded files if any
            for file_info in files:
                file_path = file_info
                text_input += f"\n\n User uploaded this file: {file_path}\n Please use it if needed."

            agent_messages = []
            for msg in self.main_history_copy:
                if msg["role"] == "user":
                    agent_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    if msg["content"] not in ["Executor is working on it 👉"]:
                        agent_messages.append(AIMessage(content=msg["content"]))

            agent_messages.append(HumanMessage(content=text_input))

            # Prepare inputs for the agent
            inputs = {"messages": agent_messages, "next_step": None}
            config = {"recursion_limit": 500, "configurable": {"thread_id": thread_id}}

            # Stream the agent's responses
            t = time()
            solution_found = False

            # Qwen3 semantic retrieval always runs; use_tool_retriever only adds
            # the optional bounded LLM selection stage.
            if self.use_tool_retriever:
                print("Using tool retriever...")
                inner_history.append(
                    ChatMessage(
                        role="assistant",
                        content="Retrieving relevant tools, data lake items, and libraries...",
                    )
                )
                yield inner_history, main_history

            try:
                selected_resources_names = self._prepare_resources_for_retrieval(text_input)
                self.user_task = text_input
                self.context_budget_log = []
                self._reset_run_control()
                self._active_selected_resources = selected_resources_names
                self.system_prompt = self.base_system_prompt
            except Exception as e:
                self._active_selected_resources = None
                self.system_prompt = self.base_system_prompt
                print(f"Warning: Bounded resource retrieval failed: {e}")
                print("Continuing with the resource-independent base prompt...")
                if self.use_tool_retriever:
                    inner_history.append(
                        ChatMessage(
                            role="assistant",
                            content="Resource retrieval unavailable; proceeding without optional resources.",
                        )
                    )
                    yield inner_history, main_history

            # Keep track of code execution messages
            code_execution_messages = []

            # Stream the agent's responses
            for s in self.app.stream(inputs, stream_mode="values", config=config):
                t_step = time() - t
                message = s["messages"][-1]

                # Skip the first message which is the input task
                if message.content == text_input:
                    t = time()
                    continue

                # Process the message
                if isinstance(message.content, str):
                    # Extract thinking/reasoning part (text before any tags)
                    tag_positions = []
                    for tag in ["<execute>", "<solution>", "<observation>"]:
                        pos = message.content.find(tag)
                        if pos != -1:
                            tag_positions.append(pos)

                    # If there are tags, extract the text before the first tag
                    if tag_positions:
                        first_tag_pos = min(tag_positions)
                        thinking = message.content[:first_tag_pos].strip()
                        if thinking:
                            inner_history.append(
                                ChatMessage(
                                    role="assistant",
                                    content=f"{thinking}",
                                    metadata={"title": "🤔 Reasoning", "log": "Agent's thinking process"},
                                )
                            )
                            yield inner_history, main_history

                    # Check for solution tag
                    solution_match = re.search(r"<solution>(.*?)</solution>", message.content, re.DOTALL)
                    if solution_match and not solution_found:
                        solution = solution_match.group(1).strip()
                        main_history.append(
                            ChatMessage(
                                role="assistant",
                                content=solution,
                                metadata={"title": "✅ Answer", "log": "Final answer provided by the agent"},
                            )
                        )
                        self.main_history_copy += [{"role": "assistant", "content": solution}]
                        solution_found = True
                        yield inner_history, main_history

                    # Check for execute tag
                    execute_match = re.search(r"<execute>(.*?)</execute>", message.content, re.DOTALL)
                    if execute_match:
                        code = execute_match.group(1).strip()
                        language = "python"
                        if code.strip().startswith("#!R"):
                            language = "r"
                            code = re.sub(r"^#!R", "", code, count=1).strip()
                        elif code.strip().startswith("#!BASH") or code.strip().startswith("#!CLI"):
                            language = "bash"
                            code = re.sub(r"^#!BASH|^#!CLI", "", code, count=1).strip()

                        code_msg = ChatMessage(
                            role="assistant",
                            content=f"##### Code: \n```{language}\n{code}\n```",
                            metadata={
                                "title": "🛠️ Executing code...",
                                "log": f"Executing {language.capitalize()} code block...",
                                "status": "pending",
                                "start_time": t,
                            },
                        )
                        inner_history.append(code_msg)
                        code_execution_messages.append(code_msg)
                        yield inner_history, main_history

                    # Check for observation
                    observation_match = re.search(r"<observation>(.*?)</observation>", message.content, re.DOTALL)
                    if observation_match:
                        observation = observation_match.group(1).strip()

                        # Update the status of the most recent code execution message
                        if code_execution_messages:
                            code_msg = code_execution_messages[-1]
                            code_msg.metadata.update(
                                {
                                    "status": "done",
                                    "duration": t_step,
                                    "log": f"Code execution completed in {t_step:.2f}s",
                                }
                            )

                        # Create a new message for the observation
                        inner_history.append(
                            ChatMessage(
                                role="assistant",
                                content=f"##### Observation: \n```\n{observation}\n```",
                                metadata={
                                    "status": "done",
                                    "duration": t_step,
                                    "log": "Observation from code execution",
                                    "collapsed": True,
                                    "collapsible": True,
                                },
                            )
                        )
                        yield inner_history, main_history

                        # Check for file paths in the observation
                        if isinstance(observation, str) and any(ext in observation for ext in SUPPORTED_EXTENSIONS):
                            matches = re.findall(r"(\S+?(?:\.png|\.jpg|\.jpeg|\.gif|\.bmp|\.webp|\.pdf))", observation)

                            valid_matches = []
                            for match in matches:
                                if not (
                                    match.startswith("Warning:") or match.startswith("Error:") or match.startswith("'")
                                ):
                                    if not match.startswith("."):
                                        valid_matches.append(match)

                            if valid_matches:
                                inner_history.append(
                                    ChatMessage(
                                        role="assistant",
                                        content="",
                                        metadata={"title": "📁 Files", "log": "Files generated by the agent"},
                                    )
                                )

                                for file_path in valid_matches:
                                    file_path = file_path.strip("\"'").strip()

                                    abs_path = None
                                    if os.path.isabs(file_path) and os.path.exists(file_path):
                                        abs_path = file_path
                                    elif os.path.exists(os.path.join(os.getcwd(), file_path)):
                                        abs_path = os.path.join(os.getcwd(), file_path)
                                    elif (
                                        hasattr(self, "path")
                                        and self.path
                                        and os.path.exists(os.path.join(self.path, file_path))
                                    ):
                                        abs_path = os.path.join(self.path, file_path)

                                    if abs_path:
                                        if file_path.lower().endswith(".pdf"):
                                            inner_history.append(
                                                ChatMessage(
                                                    role="assistant",
                                                    content=f"Found PDF at: {abs_path}",
                                                    metadata={"title": "📄 PDF File"},
                                                )
                                            )
                                        else:
                                            inner_history.append(
                                                ChatMessage(
                                                    role="assistant",
                                                    content=gr.Image(abs_path),
                                                    metadata={"title": "🖼️ Image Preview"},
                                                )
                                            )

                                yield inner_history, main_history

                t = time()

            # If no solution was found, add the final message
            if not solution_found:
                final_message = s["messages"][-1].content if s["messages"] else ""
                solution_match = re.search(r"<solution>(.*?)</solution>", final_message, re.DOTALL)
                if solution_match:
                    solution = solution_match.group(1).strip()
                    main_history.append(
                        ChatMessage(role="assistant", content=solution, metadata={"title": "✅ Solution"})
                    )
                    self.main_history_copy += [{"role": "assistant", "content": solution}]
                else:
                    cleaned_content = re.sub(r"<execute>.*?</execute>", "", final_message, flags=re.DOTALL)
                    cleaned_content = re.sub(r"<observation>.*?</observation>", "", cleaned_content, flags=re.DOTALL)
                    cleaned_content = re.sub(r"\n\s*\n", "\n\n", cleaned_content)

                    if cleaned_content.strip():
                        main_history.append(
                            ChatMessage(
                                role="assistant", content=cleaned_content.strip(), metadata={"title": "📝 Summary"}
                            )
                        )
                        self.main_history_copy += [{"role": "assistant", "content": cleaned_content.strip()}]
                    else:
                        main_history.append(
                            ChatMessage(
                                role="assistant",
                                content="Task completed. Please check the execution log for details.",
                                metadata={"title": "📝 Summary"},
                            )
                        )
                        self.main_history_copy += [{"role": "assistant", "content": "Task completed."}]

            # Add completion message
            inner_history.append(
                ChatMessage(
                    role="assistant",
                    content="👈 Returning the result to the main interface...",
                    metadata={"title": "🔄 Complete"},
                )
            )
            yield inner_history, main_history

        def like(data: gr.LikeData):
            print("User liked the response")
            print(f"Index: {data.index}, Liked: {data.liked}")

        # Create the Gradio interface
        with gr.Blocks() as demo:
            # Verification page (if enabled)
            verification_container = gr.Group(visible=require_verification)
            main_interface_container = gr.Group(visible=not require_verification)

            with verification_container:
                gr.Markdown("# OmniInfra A1 Agent - Access Verification")
                gr.Markdown("Please enter your access code to continue.")
                access_code_input = gr.Textbox(label="Access Code", type="password")
                access_error_msg = gr.Markdown(visible=False)
                verify_btn = gr.Button("Verify Access")
                verify_btn.click(
                    fn=verify_access_code,
                    inputs=[access_code_input],
                    outputs=[verification_container, main_interface_container, access_error_msg],
                )

            # Main interface
            with main_interface_container:
                with gr.Row():
                    with gr.Column(scale=1):
                        main_chatbot = gr.Chatbot(
                            label="OmniInfra A1 Agent",
                            type="messages",
                            height=800,
                            show_copy_button=True,
                            show_share_button=True,
                        )
                    with gr.Column(scale=1):
                        innerloop_chatbot = gr.Chatbot(
                            label="OmniInfra Executor",
                            type="messages",
                            height=800,
                            show_copy_button=True,
                            show_share_button=True,
                        )

                with gr.Row():
                    prompt_input = gr.MultimodalTextbox(
                        interactive=True,
                        file_count="multiple",
                        placeholder="Ask something or upload a file...",
                        show_label=False,
                    )

                # Bind submission
                prompt_input.submit(
                    generate_response,
                    [prompt_input, innerloop_chatbot, main_chatbot],
                    [innerloop_chatbot, main_chatbot],
                ).then(lambda: gr.MultimodalTextbox(value=None), None, [prompt_input])
                main_chatbot.like(like)

        # Launch
        print(f"Launching Gradio demo on {server_name}:7860")
        demo.launch(share=share, server_name=server_name)
