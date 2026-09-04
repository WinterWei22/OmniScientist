description = [
    {
        "name": "load_biomedical_kg",
        "description": "Load a biomedical knowledge graph from a triple file or PrimeKG-style CSV and return graph statistics.",
        "required_parameters": [
            {
                "name": "kg_path",
                "type": "str",
                "description": "Path to the KG file.",
                "default": None,
            }
        ],
        "optional_parameters": [
            {"name": "format", "type": "str", "description": "File format. Use 'csv' or 'tsv'.", "default": "csv"},
            {"name": "delimiter", "type": "str", "description": "Column delimiter for triple files.", "default": "\t"},
            {"name": "has_header", "type": "bool", "description": "Whether the first line is a header row.", "default": False},
            {"name": "schema", "type": "str", "description": "Input schema: 'auto', 'triples', or 'primekg'.", "default": "auto"},
            {"name": "use_cache", "type": "bool", "description": "Cache the parsed graph in memory.", "default": True},
        ],
    },
    {
        "name": "random_walk_with_restart",
        "description": "Run Random Walk with Restart on a biomedical knowledge graph to rank nodes by proximity to seed entities.",
        "required_parameters": [
            {"name": "kg_path", "type": "str", "description": "Path to the KG triple file.", "default": None},
            {"name": "seed_nodes", "type": "List[str]", "description": "Seed entity IDs to start the walk from.", "default": None},
        ],
        "optional_parameters": [
            {"name": "restart_prob", "type": "float", "description": "Restart probability in (0, 1).", "default": 0.7},
            {"name": "max_iter", "type": "int", "description": "Maximum number of iterations.", "default": 100},
            {"name": "epsilon", "type": "float", "description": "Convergence threshold on L1 delta.", "default": 1e-6},
            {"name": "top_k", "type": "int", "description": "Number of top-ranked nodes to return.", "default": 200},
            {"name": "return_subgraph", "type": "bool", "description": "Also return the induced top-k subgraph.", "default": True},
        ],
    },
    {
        "name": "extract_khop_subgraph",
        "description": "Extract the k-hop neighbourhood subgraph around seed entities via BFS expansion.",
        "required_parameters": [
            {"name": "kg_path", "type": "str", "description": "Path to the KG triple file.", "default": None},
            {"name": "seed_nodes", "type": "List[str]", "description": "Seed entity IDs to center the extraction on.", "default": None},
        ],
        "optional_parameters": [
            {"name": "k", "type": "int", "description": "Number of hops to expand.", "default": 2},
            {"name": "relation_filter", "type": "List[str]", "description": "Optional relation types allowed during BFS expansion and in returned edges.", "default": None},
            {"name": "bidirectional", "type": "bool", "description": "Follow both outgoing and incoming edges when true; otherwise follow outgoing edges only.", "default": True},
        ],
    },
    {
        "name": "extract_metapaths",
        "description": "Find metapath patterns and their path instances connecting two entities in a biomedical knowledge graph. "
        "Here, a metapath means the relation sequence along a simple path, for example binds → associated_with.",
        "required_parameters": [
            {"name": "kg_path", "type": "str", "description": "Path to the KG triple file.", "default": None},
            {"name": "head_entity", "type": "str", "description": "Source entity ID.", "default": None},
            {"name": "tail_entity", "type": "str", "description": "Target entity ID.", "default": None},
        ],
        "optional_parameters": [
            {"name": "max_length", "type": "int", "description": "Maximum number of hops in a path.", "default": 4},
            {"name": "max_paths", "type": "int", "description": "Maximum number of simple paths to enumerate.", "default": 100},
            {"name": "bidirectional", "type": "bool", "description": "Traverse edges in both directions.", "default": True},
        ],
    },
    {
        "name": "inspect_metapath_length_limits",
        "description": "Inspect practical max_length limits for extract_metapaths on one or more KG files, including structural upper bounds and diameter-style guidance.",
        "required_parameters": [],
        "optional_parameters": [
            {"name": "kg_paths", "type": "List[str]", "description": "KG files to inspect. Defaults to primekg.csv, kg.csv, and data/kg.csv when omitted.", "default": None},
            {"name": "sample_pairs", "type": "int", "description": "Number of random node pairs to sample when estimating shortest-path scale.", "default": 64},
            {"name": "random_seed", "type": "int", "description": "Random seed used for pair sampling.", "default": 0},
        ],
    },
    {
        "name": "traverse_metapath",
        "description": "Traverse the KG from a head entity using an exact relation sequence or one pattern item returned by extract_metapaths; when a pattern is supplied, enforce its typed node sequence.",
        "required_parameters": [
            {"name": "kg_path", "type": "str", "description": "Path to the KG triple file.", "default": None},
            {"name": "head_entity", "type": "str", "description": "Starting entity ID.", "default": None},
        ],
        "optional_parameters": [
            {"name": "metapath", "type": "List[str]", "description": "Ordered list of relation types to follow. Required unless pattern is supplied.", "default": None},
            {"name": "max_results", "type": "int", "description": "Maximum number of tail entities to return.", "default": 50},
            {"name": "bidirectional", "type": "bool", "description": "Allow INV_ prefixes for reverse traversal.", "default": True},
            {"name": "pattern", "type": "dict", "description": "One item from extract_metapaths.result.metapath_patterns. Its relation_pattern drives traversal and node_type_pattern constrains each layer.", "default": None},
        ],
    },
    {
        "name": "extract_enclosing_subgraph",
        "description": "Extract the enclosing subgraph around a head-tail entity pair via bidirectional BFS.",
        "required_parameters": [
            {"name": "kg_path", "type": "str", "description": "Path to the KG triple file.", "default": None},
            {"name": "head_entity", "type": "str", "description": "Source entity ID.", "default": None},
            {"name": "tail_entity", "type": "str", "description": "Target entity ID.", "default": None},
        ],
        "optional_parameters": [
            {"name": "max_hops", "type": "int", "description": "Maximum hop radius from each endpoint.", "default": 3},
            {"name": "max_nodes_per_hop", "type": "int", "description": "Maximum number of new nodes per BFS layer.", "default": 200},
            {"name": "remove_direct_link", "type": "bool", "description": "Exclude direct head-tail edges from the returned subgraph.", "default": True},
            {"name": "bidirectional", "type": "bool", "description": "Follow both outgoing and incoming edges.", "default": True},
        ],
    },
    {
        "name": "compute_pagerank",
        "description": "Compute global PageRank centrality on the biomedical knowledge graph.",
        "required_parameters": [
            {"name": "kg_path", "type": "str", "description": "Path to the KG triple file.", "default": None},
        ],
        "optional_parameters": [
            {"name": "damping", "type": "float", "description": "PageRank damping factor in (0, 1).", "default": 0.85},
            {"name": "top_k", "type": "int", "description": "Number of top-ranked nodes to return.", "default": 100},
            {"name": "relation_filter", "type": "List[str]", "description": "Optional list of allowed relation types.", "default": None},
            {"name": "bidirectional", "type": "bool", "description": "Treat edges as undirected when ranking.", "default": True},
            {"name": "max_iter", "type": "int", "description": "Maximum number of power-iteration steps.", "default": 100},
            {"name": "tolerance", "type": "float", "description": "Convergence tolerance for PageRank.", "default": 1e-6},
        ],
    },
]
