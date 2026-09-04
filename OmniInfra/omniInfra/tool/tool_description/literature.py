description = [
    {
        "description": "Fetches supplementary information for a paper given its DOI "
        "and saves it to a specified directory.",
        "name": "fetch_supplementary_info_from_doi",
        "optional_parameters": [
            {
                "default": "supplementary_info",
                "description": "Directory to save supplementary files",
                "name": "output_dir",
                "type": "str",
            },
            {
                "default": 30,
                "description": "HTTP timeout in seconds",
                "name": "timeout_seconds",
                "type": "int",
            },
            {
                "default": 52428800,
                "description": "Maximum bytes allowed for each supplementary download",
                "name": "max_download_bytes",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "The paper DOI",
                "name": "doi",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query arXiv for papers based on the provided search query.",
        "name": "query_arxiv",
        "optional_parameters": [
            {
                "default": 10,
                "description": "The maximum number of papers to retrieve.",
                "name": "max_papers",
                "type": "int",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "The search query string.",
                "name": "query",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query bioRxiv preprint metadata through the official bioRxiv API by DOI or by locally matching keywords within a bounded date interval.",
        "name": "query_biorxiv",
        "optional_parameters": [
            {
                "default": 10,
                "description": "Maximum number of matching preprints to return (1-100).",
                "name": "max_papers",
                "type": "int",
            },
            {
                "default": None,
                "description": "Inclusive interval start in YYYY-MM-DD format; provide together with end_date. Defaults to the most recent 30 days when both dates are omitted.",
                "name": "start_date",
                "type": "str",
            },
            {
                "default": None,
                "description": "Inclusive interval end in YYYY-MM-DD format; provide together with start_date.",
                "name": "end_date",
                "type": "str",
            },
            {
                "default": None,
                "description": "Optional official bioRxiv subject category, for example cell biology or neuroscience.",
                "name": "category",
                "type": "str",
            },
            {
                "default": 300,
                "description": "Maximum number of API metadata records to scan for a keyword query (30-3000).",
                "name": "max_records",
                "type": "int",
            },
            {
                "default": 30,
                "description": "Per-request HTTP timeout in seconds (1-120).",
                "name": "timeout_seconds",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Non-empty keyword query or a bioRxiv DOI beginning with 10.1101/.",
                "name": "query",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query PubMed for papers based on the provided search query.",
        "name": "query_pubmed",
        "canonical_name": "omniInfra.tool.literature.query_pubmed",
        "aliases": ["pubmed_search", "literature_search", "biomedical_literature_retrieval"],
        "capabilities": ["literature_retrieval", "biomedical_evidence", "public_read_only_network"],
        "source_ids": ["pubmed"],
        "evidence_types": ["peer_reviewed_literature"],
        "side_effect": "read_only",
        "optional_parameters": [
            {
                "default": 10,
                "description": "The maximum number of papers to retrieve.",
                "name": "max_papers",
                "type": "int",
            },
            {
                "default": 3,
                "description": "Maximum number of retry attempts with modified queries.",
                "name": "max_retries",
                "type": "int",
            },
            {
                "default": None,
                "description": "NCBI contact email; defaults to OMNIINFRA_PUBMED_EMAIL when set",
                "name": "email",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "The search query string.",
                "name": "query",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query OpenAlex for the first scholarly work matching a search query using a custom, local-default, or anonymous free quota.",
        "name": "query_scholar",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "The search query string.",
                "name": "query",
                "type": "str",
            }
        ],
    },
    {
        "description": "Deprecated compatibility alias that searches the web with Bing and returns structured results.",
        "name": "search_google",
        "optional_parameters": [
            {
                "default": 3,
                "description": "Number of results to return",
                "name": "num_results",
                "type": "int",
            },
            {
                "default": "en",
                "description": "Language code for search results",
                "name": "language",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": 'The search query (e.g., "protocol text or search question")',
                "name": "query",
                "type": "str",
            }
        ],
    },
    {
        "description": "Search the web with Bing or Baidu and return structured title, URL, and snippet results.",
        "name": "search_web",
        "aliases": ["web_search", "official_documentation_search"],
        "capabilities": ["web_search", "official_documentation_discovery", "public_read_only_network"],
        "source_ids": ["bing", "baidu"],
        "evidence_types": ["web_search_result"],
        "side_effect": "read_only",
        "optional_parameters": [
            {"default": 3, "description": "Number of results to return (1-20)", "name": "num_results", "type": "int"},
            {"default": "en", "description": "Language code", "name": "language", "type": "str"},
            {"default": "bing", "description": "Search provider: bing or baidu", "name": "provider", "type": "str"},
            {"default": 15, "description": "HTTP timeout in seconds", "name": "timeout_seconds", "type": "int"},
        ],
        "required_parameters": [
            {"default": None, "description": "Non-empty web search query", "name": "query", "type": "str"}
        ],
    },
    {
        "description": "Extract the text content of a webpage using requests and BeautifulSoup.",
        "name": "extract_url_content",
        "aliases": ["read_webpage", "extract_official_documentation"],
        "capabilities": ["web_content_retrieval", "official_documentation_discovery", "public_read_only_network"],
        "source_ids": ["public_web"],
        "evidence_types": ["web_document"],
        "side_effect": "read_only",
        "optional_parameters": [
            {"default": 30, "description": "HTTP timeout in seconds", "name": "timeout_seconds", "type": "int"},
            {
                "default": 5242880,
                "description": "Maximum response body size in bytes",
                "name": "max_content_bytes",
                "type": "int",
            },
            {
                "default": 100000,
                "description": "Maximum extracted characters to return",
                "name": "max_characters",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Webpage URL to extract content from",
                "name": "url",
                "type": "str",
            }
        ],
    },
    {
        "description": "Perform one bounded read-only GET request to a registered public API endpoint or an endpoint proven by official documentation.",
        "name": "query_public_endpoint",
        "aliases": ["public_api_query", "documented_endpoint_query"],
        "capabilities": ["public_api_query", "documented_endpoint_query", "public_read_only_network"],
        "source_ids": ["public_web"],
        "evidence_types": ["public_api_record"],
        "side_effect": "read_only",
        "optional_parameters": [
            {"default": None, "description": "Optional GET query parameters", "name": "params", "type": "dict"},
            {
                "default": None,
                "description": "Official documentation URL required for an unregistered endpoint",
                "name": "documentation_url",
                "type": "str",
            },
            {
                "default": 30,
                "description": "HTTP timeout in seconds (1-120)",
                "name": "timeout_seconds",
                "type": "int",
            },
            {
                "default": 1048576,
                "description": "Maximum response bytes (up to 5 MiB)",
                "name": "max_content_bytes",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Public HTTP(S) endpoint URL",
                "name": "url",
                "type": "str",
            }
        ],
    },
    {
        "description": "Extract text content from a PDF file.",
        "name": "extract_pdf_content",
        "optional_parameters": [
            {"default": 30, "description": "HTTP timeout in seconds", "name": "timeout_seconds", "type": "int"},
            {
                "default": 52428800,
                "description": "Maximum PDF download size in bytes",
                "name": "max_download_bytes",
                "type": "int",
            },
            {"default": 100, "description": "Maximum PDF pages to process", "name": "max_pages", "type": "int"},
            {
                "default": 200000,
                "description": "Maximum extracted characters to return",
                "name": "max_characters",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "URL of the PDF file",
                "name": "url",
                "type": "str",
            }
        ],
    },
    {
        "description": "Initiate an advanced web search by launching a specialized agent to collect relevant information and citations through multiple rounds of web searches for a given query.",
        "name": "advanced_web_search_claude",
        "optional_parameters": [
            {
                "default": 1,
                "description": "Maximum number of searches",
                "name": "max_searches",
                "type": "int",
            },
            {
                "default": 3,
                "description": "Maximum number of retry attempts with modified queries.",
                "name": "max_retries",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "The search query string.",
                "name": "query",
                "type": "str",
            }
        ],
    },
]
