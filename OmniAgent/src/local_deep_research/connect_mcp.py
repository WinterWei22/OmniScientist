import re
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from .config import mcp_url, settings

MCP_SERVICE_URL = mcp_url
tool_packages = [
    "chembl_mcp",
    "kegg_mcp",
    "string_mcp",
    "search_mcp",
    "pubchem_mcp",
    "ncbi_mcp",
    "uniprot_mcp",
    "tcga_mcp",
    "ensembl_mcp",
    "ucsc_mcp",
    "fda_drug_mcp",
    "opentargets_mcp",
    "depmap_mcp",
    "monarch_mcp",
    "clinicaltrials_mcp",
    "pdb_mcp",
    "dbsearch_mcp",
    #   "zhihuiya_mcp",
]


mcp_servers = {
    package: {
        "transport": "streamable_http",
        "url": f"{MCP_SERVICE_URL}/{package}/mcp/",
    }
    for package in tool_packages
}


class OmniagentMCPToolClient:
    def __init__(self, mcp_servers: dict[str, Any], specified_tools: list = None):
        self.mcp_servers = mcp_servers
        self.mcp_tools = None
        self.mcp_tool_map = {}
        self.available_tools = specified_tools
        self.biomni_gateway_tools = {}
        self.biomni_tool_name_map = {}

    @staticmethod
    def _safe_biomni_tool_name(qualified_name: str) -> str:
        name = qualified_name
        if name.startswith("biomni.tool."):
            name = name[len("biomni.tool.") :]
        name = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        return f"{settings.biomni_mcp.tool_prefix}_{name}"

    @staticmethod
    def _extract_text_payload(result: Any) -> Any:
        content = getattr(result, "content", None)
        if content and len(content) == 1 and hasattr(content[0], "text"):
            return content[0].text
        if isinstance(result, list) and len(result) == 1:
            item = result[0]
            if hasattr(item, "text"):
                return item.text
            if isinstance(item, dict) and item.get("type") == "text":
                return item.get("text")
        return result

    @classmethod
    def _parse_mcp_json_result(cls, result: Any) -> Any:
        import json

        payload = cls._extract_text_payload(result)
        if isinstance(payload, str):
            return json.loads(payload)
        return payload

    async def _initialize_biomni_tools(self):
        if not settings.biomni_mcp.enabled:
            return []

        biomni_servers = {
            "biomni": {
                "transport": settings.biomni_mcp.transport,
                "url": settings.biomni_mcp.url,
            }
        }
        biomni_client = MultiServerMCPClient(biomni_servers)
        biomni_gateway_tools = await biomni_client.get_tools()
        self.biomni_gateway_tools = {tool.name: tool for tool in biomni_gateway_tools}

        if not settings.biomni_mcp.expose_internal_tools:
            return biomni_gateway_tools

        list_tool = self.biomni_gateway_tools.get("biomni_list_tools")
        if list_tool is None:
            raise RuntimeError(
                "Biomni MCP does not expose biomni_list_tools; restart Biomni MCP with the updated gateway."
            )

        catalog = self._parse_mcp_json_result(
            await list_tool.ainvoke({"include_schema": True})
        )

        tools = []
        for record in catalog.get("tools", []):
            qualified_name = record.get("qualified_name")
            if not qualified_name:
                continue
            tool_name = self._safe_biomni_tool_name(qualified_name)
            self.biomni_tool_name_map[tool_name] = qualified_name
            tools.append(
                BiomniInternalTool(
                    name=tool_name,
                    qualified_name=qualified_name,
                    description=record.get("description", ""),
                    args_schema=record.get("input_schema", {}),
                )
            )
        return tools

    async def initialize(self):
        """Initialize async components"""
        client = MultiServerMCPClient(self.mcp_servers)

        self.tool2source = {}
        for pkg_name in self.mcp_servers.keys():
            async with client.session(pkg_name) as session:
                tools = await load_mcp_tools(session)
                self.tool2source.update(
                    {tool.name: pkg_name.replace("_mcp", "") for tool in tools}
                )

        self.mcp_tools = await client.get_tools()
        biomni_tools = await self._initialize_biomni_tools()
        if biomni_tools:
            self.mcp_tools.extend(biomni_tools)

        if self.available_tools:
            self.mcp_tools = [
                tool for tool in self.mcp_tools if tool.name in self.available_tools
            ]
        self.mcp_tool_map = {tool.name: tool for tool in self.mcp_tools}
        self.tool2source.update({tool.name: "biomni" for tool in biomni_tools})
        print(f"MCP server connected! Found {len(self.mcp_tools)} tools")

    async def call_tool(self, tool_name: str, args: dict[str, Any]):
        if tool_name in self.biomni_tool_name_map:
            invoke_tool = self.biomni_gateway_tools.get("biomni_invoke_tool")
            if invoke_tool is None:
                raise ValueError("Biomni gateway tool biomni_invoke_tool not found")
            return await invoke_tool.ainvoke(
                {
                    "tool_name": self.biomni_tool_name_map[tool_name],
                    "arguments": args,
                }
            )

        tool = self.mcp_tool_map.get(tool_name)
        if not tool:
            raise ValueError(f"Tool {tool_name} not found")
        return await tool.ainvoke(args)


class BiomniInternalTool:
    def __init__(
        self,
        name: str,
        qualified_name: str,
        description: str,
        args_schema: dict[str, Any],
    ):
        self.name = name
        self.qualified_name = qualified_name
        self.description = f"{description} [Biomni internal tool: {qualified_name}]"
        self.args_schema = args_schema
