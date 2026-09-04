import asyncio
import logging
import re
from collections import defaultdict

import networkx as nx
import pandas as pd
from langchain_openai import ChatOpenAI

from .search_system_support import extract_and_convert_list
from .tool_embedding_retriever import ToolEmbeddingRetriever
from .utils import (
    biological_entities,
    exact_match_entity_type,
    extract_and_convert_dict,
    generate_tools_descriptions,
)
from .config import settings


class KGNetwork:
    def __init__(self, tool_info_path: str, available_tools: list = None):
        """
        Args:
            tool_info_path: path to the tool info excel file
        """
        self.available_tools = available_tools
        self.G, self.edge_tools, self.tool2package = self.build_mcp_network(
            tool_info_path
        )

    def build_mcp_network(self, excel_file_path):
        """
        Build the MCP tool network from the Excel file, and associate the related tool information with each node

        Args:
            excel_file_path: Excel file path

        Returns:
            networkx.Graph: The built network, with each node containing related tool information
            dict: The edge tool information dictionary
        """

        # Create a graph
        G = nx.DiGraph()

        # Initialize nodes and add empty tool sets
        for entity in biological_entities:
            G.add_node(entity, related_tools=set())

        # Read all sheets of the Excel file
        excel_data = pd.read_excel(excel_file_path, sheet_name=None)

        # Dictionary to store edge and related tools
        edge_tools = defaultdict(set)
        # Dictionary to store nodes and related tools
        node_tools = defaultdict(set)

        tool2package = {}

        # Iterate through each sheet
        for sheet_name, df in excel_data.items():
            # print(f"Processing tool package: {sheet_name}")

            # Iterate through each row of data
            for idx, row in df.iterrows():
                tool_name = row["tool_name"]
                input_entity = row["input_entity"]
                output_entity = row["output_entity"]
                tool2package[tool_name] = sheet_name

                if self.available_tools and tool_name not in self.available_tools:
                    continue

                # Process input entities
                input_entities = []
                if pd.notna(input_entity) and input_entity.strip():
                    input_entities = [
                        entity.strip() for entity in str(input_entity).split(";")
                    ]

                # Process output entities
                output_entities = []
                if pd.notna(output_entity) and output_entity.strip():
                    output_entities = [
                        entity.strip() for entity in str(output_entity).split(";")
                    ]

                # Associate tools with input and output entities
                for input_ent in input_entities:
                    if input_ent in biological_entities:
                        node_tools[input_ent].add(tool_name)
                    elif input_ent:
                        logging.info(
                            f"Error input_entity: #{input_ent}#, tool_name: {tool_name}"
                        )

                # Create edges
                for input_ent in input_entities:
                    for output_ent in output_entities:
                        # Ensure entities are in the predefined list
                        if not input_ent or not output_ent:
                            continue
                        elif (
                            input_ent in biological_entities
                            and output_ent in biological_entities
                        ):
                            # Create edge keys (sorted alphabetically to ensure consistency)
                            edge_key = (input_ent, output_ent)
                            edge_tools[edge_key].add(tool_name)
                        else:
                            if input_ent not in biological_entities:
                                logging.info(
                                    f"Error input_entity: #{input_ent}#, tool_name: {tool_name}"
                                )
                            if output_ent not in biological_entities:
                                logging.info(
                                    f"Error output_entity: #{output_ent}#, tool_name: {tool_name}"
                                )

        # Add edges to the graph
        for edge_key, tools in edge_tools.items():
            if len(edge_key) == 2:  # Ensure it's a valid edge
                node1, node2 = edge_key
                G.add_edge(node1, node2, related_tools=list(tools))

        # Add tool information to nodes
        for node, tools in node_tools.items():
            G.nodes[node]["related_tools"] = list(tools)

        return G, edge_tools, tool2package

    def get_node_tools(self, node_name):
        """
        Query the list of tools associated with a specified node

        Args:
            G: networkx.Graph, MCP network graph
            node_name: str, the name of the node to query

        Returns:
            list: the list of tools associated with the node, empty list if the node does not exist
        """
        if node_name in self.G.nodes:
            return self.G.nodes[node_name].get("related_tools", [])
        else:
            logging.info(f"Node '{node_name}' does not exist in the network graph")
            return []

    def get_edge_tools(self, node1, node2):
        """
        Query the list of tools associated with a specified edge

        Args:
            G: networkx.Graph, MCP network graph
            node1: str, the name of the first node of the edge
            node2: str, the name of the second node of the edge

        Returns:
            list: the list of tools associated with the edge, empty list if the edge does not exist
        """
        if self.G.has_edge(node1, node2):
            return self.G[node1][node2].get("related_tools", [])
        else:
            logging.info(f"Edge ({node1}, {node2}) does not exist in the network graph")
            return []


class ExpertToolSelector:
    def __init__(
        self,
        llm,
        mcp_client,
        tool_embedding_retriever: ToolEmbeddingRetriever,
        kg_network: KGNetwork,
        available_tools: list = None,
    ):
        self.llm = llm
        self.mcp_client = mcp_client
        # Map tool name to mcp tool object
        self.tool_map = mcp_client.mcp_tool_map
        self.kg_network = kg_network
        self.tool_embedding_retriever = tool_embedding_retriever
        self.available_tools = available_tools
        if not available_tools:
            self.available_tools = list(self.tool_map.keys())
        else:
            self.available_tools = [
                tool for tool in self.available_tools if tool in self.tool_map.keys()
            ]

    def extract_entity(self, query: str):
        entity_extraction_propmt = f"""
## Task Description
You are a professional biomedical text analysis expert. Please analyze the given question to extract biological entities and identify the logical relationships between these entities.

## Main question
{query}

## Biological Entity Categories
Please identify entities in the main question from the following predefined categories: {biological_entities}

## Relationship Types
Please identity the relationship type between the identified entities according to the main question.

## Instructions
1. Entity Extraction: Identify all biological entities in the question and classify them according to the 23 categories above.
2. Relationship Analysis: Determine what type of relationship the question is asking about between the identified entities.

## Note
1. Do not select overly generic terms as entities, such as "Mutation", "Drug", "Gene", "Disease", etc.
2. Don't split entities too finely. For example, for the following question: Which domain of FAK is responsible for its localization to focal adhesions?
In this case, domain of FAK and localization to focal adhesions should both be treated as entities. Do not only split them into domain, FAK, localization, and focal adhesions.

## Output Format
Please provide your analysis in the following JSON format:
```json
{{
  "question": "[original question]",
  "entities": [
     ["[entity_1 text]", "[category of entity 1]"],
     ["[entity_2 text]", "[category of entity 2]"],
     ["[entity_3 text]", "[category of entity 3]"],
  ],
  "relationships": [
      ["[entity_1 text]", "[entity_2 text]"],
      ["[entity_1 text]", "[entity_3 text]"],
  ]
}}
```
"""
        result = self.llm.invoke(entity_extraction_propmt)
        result = extract_and_convert_dict(result.content)
        if result is not None:
            result["entities"] = [
                [entity[0], exact_match_entity_type(entity[1])]
                for entity in result["entities"]
            ]

        return result

    def entity_filter(self, query: str, entity_and_relationship: dict):
        entities = entity_and_relationship["entities"]

        prompt = f"""
## Task Description
You are a professional disease biologist. Please filter out entities without clear specificity from the input information and return the filtered information.

## Input
Original query: {query}
Extracted entities: {entities}

## Instructions
1. Remove entities without clear specificity: The retained entities should have meaningful information and be proper nouns rather than generic terms. For example, for drug-related entities, we should be able to find a specific drug or a class of drugs through the entity name, rather than a simple generic term (such as "effective drug", "drug molecule"). Terms like "appropriate targets" and "related disease" should not be considered as informative entities.
2. Only remove entities, do not add new entities or modify the retained entities.

## Output Format
Please provide your output in the following list format:
[
    [[entity_1 text]", "[category of entity 1]"],
    [[entity_2 text]", "[category of entity 2]"],
    ...
]
"""
        result = self.llm.invoke(prompt)
        filtered_entities = extract_and_convert_list(result.content)
        # Defensive: LLM sometimes returns invalid JSON and parser returns None.
        # In that case, fall back to original extracted entities to avoid crashing.
        if filtered_entities is None:
            filtered_entities = entities
        entity_and_relationship["entities"] = filtered_entities

        entity_set = set([entity[0] for entity in filtered_entities])
        entity_and_relationship["relationships"] = [
            relationship
            for relationship in entity_and_relationship["relationships"]
            if relationship[0] in entity_set and relationship[1] in entity_set
        ]

        return entity_and_relationship

    async def run(self, query: str):
        """
        Retrieve the most relevant tools for a given query using the KG network

        Args:
            query: The input query string

        Returns:
            List of tool names sorted by relevance
        """
        logging.info("Extracting entities and relationships from the query...")
        entity_and_relationship = self.extract_entity(query)
        entity_and_relationship = self.entity_filter(query, entity_and_relationship)
        logging.info("Coarse retrieval of tools...")
        entity2tools = self.coarse_retrieval(entity_and_relationship, explain_item=True)
        logging.info("Precise retrieval of tools...")

        # Create tasks for parallel execution
        tasks = []
        entity_list = list(entity2tools.keys())
        for item in entity_list:
            tool_candidates = entity2tools[item]
            tasks.append(self.precise_retrieval(item, tool_candidates))

        # Execute all tasks in parallel and gather results
        results = await asyncio.gather(*tasks)
        results = [result for result in results if result]

        return results

    def cache_entity_explanation(self, entity_and_relationship: dict):
        entity_list = [item for item, _ in entity_and_relationship["entities"]]
        relationship_list = [
            f"Relationship between **{e1}** and **{e2}**"
            for e1, e2 in entity_and_relationship["relationships"]
        ]
        item_list = entity_list + relationship_list
        item_explanation_dict = self.tool_embedding_retriever.batch_explain_item(
            item_list
        )
        return item_explanation_dict

    def coarse_retrieval(
        self, entity_and_relationship: dict, explain_item: bool = False
    ):
        entity2tools = {}
        entity2type = {}

        if explain_item:
            item_explanation_dict = self.cache_entity_explanation(
                entity_and_relationship
            )

        for entity in entity_and_relationship["entities"]:
            item, entity_type = entity
            entity2type[item] = entity_type
            # All related tools extracted from the mini knowledge graph
            tool_pool = self.kg_network.get_node_tools(entity_type)
            if self.available_tools:
                tool_pool = [tool for tool in tool_pool if tool in self.available_tools]

            # Retrieve the most relevant 10 tools from the tool embedding retriever
            if len(tool_pool) > 10:
                if explain_item and item in item_explanation_dict.keys():
                    query = item_explanation_dict[item]
                    # Do not explain the item again
                    explain_item_ = False
                else:
                    query = item
                    explain_item_ = True
                tools_name_list, _ = (
                    self.tool_embedding_retriever.retrieve_tools_from_candidates(
                        query, tool_pool, top_k=10, explain_item=explain_item_
                    )
                )
            else:
                tools_name_list = tool_pool
            entity2tools[item] = tools_name_list

        for relationship in entity_and_relationship["relationships"]:
            e1, e2 = relationship
            r1, r2 = entity2type[e1], entity2type[e2]
            tool_pool = self.kg_network.get_edge_tools(r1, r2)

            if self.available_tools:
                tool_pool = [tool for tool in tool_pool if tool in self.available_tools]

            if len(tool_pool) == 0:
                continue
            elif len(tool_pool) > 5:
                key = f"Relationship between **{e1}** and **{e2}**"
                if explain_item and key in item_explanation_dict.keys():
                    query = item_explanation_dict[key]
                    # Do not explain the item again
                    explain_item_ = False
                else:
                    query = key
                    explain_item_ = True
                tools_name_list, _ = (
                    self.tool_embedding_retriever.retrieve_tools_from_candidates(
                        query, tool_pool, top_k=10, explain_item=explain_item_
                    )
                )
            else:
                tools_name_list = tool_pool
            entity2tools[(e1, e2)] = tools_name_list
        return entity2tools

    async def precise_retrieval(self, item, tool_candidates: list):
        tool_list = [
            self.tool_map[tool_name]
            for tool_name in tool_candidates
            if tool_name in self.available_tools
        ]
        tool_desc = generate_tools_descriptions(tool_list)

        tool_selection_propmt = f"""
## Task Description
You are a professional biomedical text analysis expert. Please select the most appropriate tool to call and generate input parameters for the tool.

## Input
{item}

## Available Tools
{tool_desc}

## Instructions
1. Tool Selection: Choose the most appropriate tool to call and generate input parameters for the tool.
2. Tool Input Generation: Generate json format input to invoke the tool 
3. If you think none of the candidate tools are suitable, please return an empty dictionary.

## Notes
1. Do not select overly generic terms as tool input, such as "Mutation" should not be used as the parameter value of the tool.

## Output Format
Please provide your analysis in the following JSON format:
```json
{{
    "item": "[item1]", 
    "tool": "[tool1]",
    "tool_input": {{
      "parameter_1": "[parameter_1 content]",
      "parameter_2": "[parameter_2 content]",
      ....
    }}
}}
```
"""
        result = await self.llm.ainvoke(tool_selection_propmt)
        tool_invoke_list = extract_and_convert_dict(result.content)
        return tool_invoke_list


class GeneralToolSelector:
    # Candidate general tools
    GENERAL_TOOLS_NAME = [
        "tavily_search",
        'paper_search',
        "pubmed_search",
        "search_target",
        "search_assay",
        "search_activity",
        "tcga_immune_correlation_analysis",
        "get_general_info_by_compound_name",
        "get_general_info_by_protein_or_gene_name",
        "get_general_info_by_disease_name",
        # "zhihuiya_biologist_llm",
        # "search_papers",
        'get_target_gene_ontology_by_name',
        'get_target_classes_by_name',
        'get_associated_diseases_phenotypes_by_target_name'
    ]

    def __init__(self, llm_light, llm_reasoning, mcp_client):
        self.llm_light = llm_light
        self.llm_reasoning = llm_reasoning
        # Map tool name to mcp tool object
        self.mcp_client = mcp_client
        self.tool_map = mcp_client.mcp_tool_map
        logging.info(f"tool_map: {len(self.tool_map)}")
        self.general_tools = [
            self.tool_map[tool_name]
            for tool_name in self.GENERAL_TOOLS_NAME
            if tool_name in self.tool_map.keys()
        ]
        self.general_tools_desc = generate_tools_descriptions(self.general_tools)

    async def run(self, query):
        """
        Select the most relevant tools for the query.
        LLM will extract the keywords from the query and prepare the input for the selected tools.

        Args:
            query: The query to select the tools for. The input query in this function should be easy logic to understand.

        Returns:
            A list of tool calling messages.
        """

        tool_selection_propmt = f"""
## Task Description
You are a professional biomedical text analysis expert. Please analyze the given question and break it down into keywords to call specific tools for searching

## Main question
{query}

## Available Tools
{self.general_tools_desc}

## Instructions
1. Entity Extraction: Identify all keywords in the question and list the relationship deserved to search
2. Tool Selection: Choose appropriate tools to search each item you listed.
3. Tool Input Generation: Generate json format input to invoke the tool 

## Notes
1. Do not use overly broad terms such as "drug," "target," "cancer," or "disease".
2. When generating keywords, the content should be as consistent as possible with the original content of the question.
3. When the tool has only one input parameter, the content of the input parameter should be the same as the keyword.
4. In the Entity Extraction step, try to extract short and meaningful items that are worth searching.
5. Do not generate placeholder entities, and do not use the output of other tools as the input of a tool.

## Output Format
Please provide your analysis in the following JSON format:
```json
[
{{
    "item": "[item1]", 
    "tool": "[tool1]",
    "tool_input": {{
      "parameter_1": "[parameter_1 content]",
      "parameter_2": "[parameter_2 content]",
      ....
    }}
}},
{{
    "item": "[item2]", 
    "tool": "[tool2]",
    "tool_input": {{
      "parameter_1": "[parameter_1 content]",
      "parameter_2": "[parameter_2 content]",
      ....
    }}
}},
...
]
```

## Example
**Question**: "Identify novel kinase targets upregulated in Breast Cancer and assess their druggability."

**Output**:
```json
[
  {{
    "item": "Harvest a candidate name list of extracellularly reachable signaling control points in Glioblastoma (surfaceome / ligand–receptor / receptor atlas)",
    "tool": "tavily_search",
    "tool_input": {{
      "query": "glioblastoma surfaceome extracellularly accessible signaling receptors ligand-receptor atlas candidate targets list"
    }}
    }},
  {{
    "item": "Add tumor-intrinsic functional anchors for candidates (perturbation/fitness, cell-fate gating, stress adaptation, resistance)",
    "tool": "tavily_search",
    "tool_input": {{
      "query": "glioblastoma tumor-intrinsic dependency knockout knockdown CRISPR fitness cell surface signaling receptor apoptosis cell cycle stress resistance"
    }}
    }}
  {{
    "item": "Deep search: pull key mechanistic paper(s) for a shortlisted candidate (how it gates signaling and cell-fate programs in GBM)",
    "tool": "paper_search",
    "tool_input":   {{
      "query": "EGFR glioblastoma mechanism signaling receptor cell cycle apoptosis stress response"
    }}
    }}
  {{
    "item": "Confirm target class / localization metadata for the shortlisted candidate",
    "tool": "get_general_info_by_protein_or_gene_name",
    "tool_input":   {{
      "name": "EGFR"
    }}
  }},
  {{
    "item": "First-in-class check: flag clear approved/late-stage direct modulators; otherwise mark FIC_uncertain if needed",
    "tool": "pubmed_search",
    "tool_input":   {{
      "query": "EGFR clinical trial Phase II Phase III approved direct modulator antibody antagonist inhibitor"
    }}
    }}
``` 
"""
        if len(self.general_tools) == 0:
            logging.info("No general tools are selected by the user.")
            return []

        logging.info("Choose general tools...")
        response = await self.llm_light.ainvoke(tool_selection_propmt)
        tool_invoke_list = extract_and_convert_list(response.content)

        if tool_invoke_list is None:
            tool_invoke_list = []

        return tool_invoke_list


class ToolSelector:
    DYNAMIC_TOOL_EXCLUDE = {
        "call_biomni",
        "biomni_search_tools",
        "biomni_list_tools",
        "biomni_invoke_tool",
    }
    BIOMNI_KG_HINTS = {
        "kg",
        "knowledge graph",
        "link prediction",
        "enclosing subgraph",
        "metapath",
        "random walk",
        "pagerank",
    }

    def __init__(
        self,
        llm_light: ChatOpenAI,
        llm_reasoning: ChatOpenAI,
        mcp_tool_client,
        tool_info_data: str,
        embedding_api_key: str,
        embedding_cache: str = None,
        available_tools: list = None,
    ):
        """
        Args:
            llm_light: DeepSeek V3, fast.
            llm_reasoning: DeepSeek R1, for reasoning and part of the reasoning.
            mcp_servers: The MCP servers.
            tool_info_data: The tool info data, including the related entity types of each tool.
            embedding_api_key: The API key for the embedding model.
            embedding_cache: The embedding cache of all tool's description.
        """
        self.llm_light = llm_light
        self.llm_reasoning = llm_reasoning
        self.mcp_tool_client = mcp_tool_client
        self.kg_network = KGNetwork(tool_info_data, available_tools)
        self.tool_embedding_retriever = ToolEmbeddingRetriever(
            self.llm_light,
            self.mcp_tool_client,
            embedding_api_key,
            embedding_cache,
            available_tools,
        )
        self.expert_tool_selector = ExpertToolSelector(
            self.llm_light,
            self.mcp_tool_client,
            self.tool_embedding_retriever,
            self.kg_network,
        )
        self.general_tool_selector = GeneralToolSelector(
            self.llm_light, self.llm_reasoning, self.mcp_tool_client
        )

    async def dynamic_tool_retrieval(self, query: str, top_k: int = 10):
        """
        Retrieve directly from all MCP tools by semantic similarity.

        This supplements the curated OmniAgent tool graph so newly exposed Biomni
        tools can be selected without editing cache_data/tool_info.xlsx first.
        Final tool choice still remains inside OmniAgent's selector.
        """
        lower_query = query.lower()
        hinted_candidates = []
        if any(hint in lower_query for hint in self.BIOMNI_KG_HINTS):
            hinted_candidates = [
                name
                for name in self.mcp_tool_client.mcp_tool_map.keys()
                if name.startswith("biomni_knowledge_graph_")
            ]

        try:
            candidates, _ = self.tool_embedding_retriever.retrieve_tools(
                query, top_k=top_k, explain_item=False
            )
        except Exception as e:
            logging.warning(f"Dynamic tool retrieval failed: {e}")
            candidates = []

        candidates = hinted_candidates + [
            tool_name
            for tool_name in candidates
            if tool_name not in self.DYNAMIC_TOOL_EXCLUDE
        ]
        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            return []

        selected = await self.expert_tool_selector.precise_retrieval(query, candidates)
        if isinstance(selected, dict) and selected.get("tool"):
            return [selected]

        fallback = self._kg_tool_fallback(query, candidates)
        if fallback:
            return [fallback]
        return []

    def _kg_tool_fallback(self, query: str, candidates: list[str]):
        lower_query = query.lower()
        tool_name = "biomni_knowledge_graph_extract_enclosing_subgraph"
        if tool_name not in candidates:
            return None
        if not any(hint in lower_query for hint in self.BIOMNI_KG_HINTS):
            return None

        entities = re.findall(
            r"\b(?:Gene|Disease|Drug|Pathway|Protein|Compound):[A-Za-z0-9_.-]+",
            query,
        )
        entities = [entity.rstrip(".,;:!?)]}") for entity in entities]
        if len(entities) < 2:
            return None

        kg_path = getattr(settings.biomni_mcp, "default_kg_path", "")
        if not kg_path:
            return None

        return {
            "item": query,
            "tool": tool_name,
            "tool_input": {
                "kg_path": kg_path,
                "head_entity": entities[0],
                "tail_entity": entities[1],
                "max_hops": 2,
                "max_nodes_per_hop": 200,
                "remove_direct_link": True,
                "bidirectional": True,
            },
        }

    async def run(self, query):
        """
        Run the tool selector for a single (sub-)query.

        Args:
            query: The (sub-)query to select the tools for. The input query in this function should be easy logic to understand.

        Returns:
            A list of tool calling messages.
        """
        # Choose the general tools to call
        general_tool_invoke_list = await self.general_tool_selector.run(query)

        # Choose the specific tools to call
        expert_tool_invoke_list = await self.expert_tool_selector.run(query)
        dynamic_tool_invoke_list = await self.dynamic_tool_retrieval(query)
        tool_invoke_list = (
            general_tool_invoke_list + expert_tool_invoke_list + dynamic_tool_invoke_list
        )
        # Filter out obviously invalid tool calls (especially for search tools that require a non-empty query).
        # This prevents cases like paper_search being called with tool_input="[]" or other empty artifacts.
        cleaned: list[dict] = []
        for each in tool_invoke_list:
            if not isinstance(each, dict):
                continue
            tool = each.get("tool")
            tool_input = each.get("tool_input")
            if not tool:
                continue

            # Search tools must have a real query string inside a dict.
            if tool in {"paper_search", "tavily_search", "pubmed_search"}:
                if not isinstance(tool_input, dict):
                    continue
                q = tool_input.get("query")
                if not (isinstance(q, str) and q.strip()):
                    continue
                cleaned.append(each)
                continue

            # Guardrail: ontology lookup is frequently called with plain-English phrases (e.g., "co-expression network")
            # which tends to 400 at the Ensembl endpoint. Keep only short, token-like terms.
            if tool == "get_ontology_name":
                if not isinstance(tool_input, dict):
                    continue
                name = tool_input.get("name") or tool_input.get("query") or tool_input.get("term")
                if not (isinstance(name, str) and name.strip()):
                    continue
                name = name.strip()
                if len(name) > 40 or any(ch.isspace() for ch in name):
                    logging.warning(
                        f"Discarding get_ontology_name tool call due to unlikely ontology term (too long or contains whitespace): {tool_input}"
                    )
                    continue
                cleaned.append(each)
                continue

            # Guardrail: gene-specific expression tool is prone to failing when input is not a clean HGNC-like symbol.
            if tool == "get_gene_specific_expression_in_cancer_type":
                if not isinstance(tool_input, dict):
                    continue
                gene = tool_input.get("gene") or tool_input.get("name") or tool_input.get("query")
                if not (isinstance(gene, str) and gene.strip()):
                    continue
                gene = gene.strip()
                if not (2 <= len(gene) <= 20) or not all((c.isalnum() or c in {"-", "_"}) for c in gene):
                    logging.warning(
                        f"Discarding get_gene_specific_expression_in_cancer_type tool call due to invalid gene input: {tool_input}"
                    )
                    continue
                cleaned.append(each)
                continue

            # Default: keep any non-empty tool_input (dict or scalar) for non-search tools.
            if tool_input:
                cleaned.append(each)

        tool_invoke_list = cleaned

        return tool_invoke_list

    def tool_input_filter(self, tool_invoke_list: list):
        prompt = f"""
## Task Description
You are a professional disease biologist. Please check if each tool usage is reasonable.

## Input
Tool invoke list: {tool_invoke_list}

## Instructions
1. The input is a list where each element is a dict containing three keys: item (entity to search), tool (tool to call), and tool_input (tool input parameters).
2. Check if each tool's input is reasonable based on the tool name and input parameters. Remove tool calls that are unreasonable. Unreasonable cases include:
    (a) Input type mismatch with tool type: For example, if the tool is get_general_info_by_compound_name but the input parameter is a target name;
    (b) Expected input and actual input have different information levels: For example, if the tool is get_clinical_pharmacology_by_drug_name but the input parameter is "drug molecule" (should input specific drug name instead of a general term);

## Notes
1. Only remove unreasonable elements from the tool invoke list, do not add new elements or modify the retained elements.

## Output Format
Please provide your analysis in the following JSON format:
"""
        result = self.llm_light.invoke(prompt)
        filtered_tool_invoke_list = extract_and_convert_list(result.content)
        return filtered_tool_invoke_list

    async def run_batch(self, query_list):
        """
        Run the tool selector for a list of sub-queries in parallel.

        Args:
            query_list: A list of sub-queries.

        Returns:
            A list of tool calling messages.
        """
        tasks = [self.run(query) for query in query_list]
        results = await asyncio.gather(*tasks)
        return results
