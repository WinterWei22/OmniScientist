description = [
    {
        "description": "Read target-linked BindingDB TSV records with malformed-row tolerance and parse statistics.",
        "name": "query_bindingdb",
        "required_parameters": [
            {"name": "gene_symbol", "type": "str", "description": "Target gene/protein name", "default": None},
            {"name": "data_lake_path", "type": "str", "description": "Data lake directory containing BindingDB_All_202409.tsv", "default": None},
        ],
        "optional_parameters": [
            {"name": "max_results", "type": "int", "description": "Maximum matching records", "default": 20},
        ],
    },
    {
        "description": "Query public ZINC compound records, automatically falling back from browser-blocked ZINC15 to the asynchronous Cartblanche22/ZINC22 API.",
        "name": "query_zinc15_compounds",
        "optional_parameters": [
            {
                "name": "search_type",
                "type": "str",
                "description": "Query interpretation: smiles, zinc_id, or similarity.",
                "default": "smiles",
            },
            {
                "name": "max_results",
                "type": "int",
                "description": "Maximum number of compound records to return, from 1 to 1000.",
                "default": 20,
            },
            {"name": "timeout_seconds", "type": "int", "description": "ZINC15 HTTP timeout in seconds.", "default": 30},
            {
                "name": "task_id",
                "type": "str",
                "description": "Existing Cartblanche22 task ID to continue polling after a previous pending response.",
                "default": "",
            },
            {
                "name": "poll_interval_seconds",
                "type": "float",
                "description": "Delay between Cartblanche22 result polls.",
                "default": 1.0,
            },
        ],
        "required_parameters": [
            {
                "name": "query",
                "type": "str",
                "description": "Non-empty SMILES or ZINC ID; when task_id is supplied this value is retained only as query context.",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the UniProt REST API using either natural language or a direct endpoint.",
        "name": "query_uniprot",
        "optional_parameters": [
            {
                "default": None,
                "description": 'Natural language query about proteins (e.g., "Find information about human insulin")',
                "name": "prompt",
                "type": "str",
            },
            {
                "default": None,
                "description": "Full or partial UniProt API endpoint URL to query directly (e.g., 'https://rest.uniprot.org/uniprotkb/P01308')",
                "name": "endpoint",
                "type": "str",
            },
            {"default": 5, "description": "Maximum number of results to return", "name": "max_results", "type": "int"},
        ],
        "required_parameters": [],
    },
    {
        "description": "Query the AlphaFold Database API for protein structure predictions or metadata; optionally download structures.",
        "name": "query_alphafold",
        "optional_parameters": [
            {
                "name": "endpoint",
                "type": "str",
                "description": "Endpoint: 'prediction', 'summary', or 'annotations'",
                "default": "prediction",
            },
            {"name": "residue_range", "type": "str", "description": "Residue range as 'start-end'", "default": None},
            {"name": "download", "type": "bool", "description": "Whether to download structure file", "default": False},
            {"name": "output_dir", "type": "str", "description": "Directory to save downloaded files", "default": None},
            {"name": "file_format", "type": "str", "description": "Download format 'pdb' or 'cif'", "default": "pdb"},
            {
                "name": "model_version",
                "type": "str",
                "description": "Optional version override (e.g., v6); by default use the API-provided URL",
                "default": None,
            },
            {"name": "model_number", "type": "int", "description": "Model number (1-5)", "default": 1},
        ],
        "required_parameters": [
            {
                "name": "uniprot_id",
                "type": "str",
                "description": "UniProt accession ID (e.g., 'P12345')",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the InterPro REST API using natural language or a direct endpoint.",
        "name": "query_interpro",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "Endpoint path or full URL", "default": None},
            {"name": "max_results", "type": "int", "description": "Max results per page", "default": 3},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about protein domains/families",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the RCSB PDB database using natural language or a direct structured query.",
        "name": "query_pdb",
        "optional_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about protein structures",
                "default": None,
            },
            {"name": "query", "type": "dict", "description": "Direct RCSB Search API query JSON", "default": None},
            {"name": "max_results", "type": "int", "description": "Maximum results to return", "default": 3},
            {
                "name": "gene_symbol",
                "type": "str",
                "description": "Gene symbol or protein name for a deterministic full-text query",
                "default": None,
            },
            {
                "name": "organism",
                "type": "str",
                "description": "Scientific organism name, such as Homo sapiens",
                "default": None,
            },
            {
                "name": "experimental_method",
                "type": "str",
                "description": "Exact experimental method, such as SOLUTION NMR",
                "default": None,
            },
            {
                "name": "released_before",
                "type": "str",
                "description": "Inclusive initial-release cutoff in YYYY-MM-DD format",
                "default": None,
            },
        ],
        "required_parameters": [],
    },
    {
        "description": "Retrieve RCSB PDB entry, entity, instance, assembly, or Chemical Component Dictionary records and optionally download entry structure files.",
        "name": "query_pdb_identifiers",
        "optional_parameters": [
            {
                "name": "return_type",
                "type": "str",
                "description": "Record type: entry, assembly, polymer_entity, nonpolymer_entity, polymer_instance, or mol_definition. Use mol_definition for Chemical Component Dictionary IDs such as ATP or 7NL.",
                "default": "entry",
            },
            {
                "name": "download",
                "type": "bool",
                "description": "Download coordinate files for four-character PDB entry identifiers, using PDB when compatible and mmCIF otherwise; Chemical Component records are metadata-only.",
                "default": False,
            },
            {
                "name": "attributes",
                "type": "List[str]",
                "description": "Specific attributes to retrieve",
                "default": None,
            },
            {
                "name": "output_dir",
                "type": "str",
                "description": "Directory for downloaded coordinate files. Defaults to OMNIINFRA_TASK_OUTPUT_DIR when set; PDB-incompatible entries are saved as mmCIF.",
                "default": None,
            },
        ],
        "required_parameters": [
            {
                "name": "identifiers",
                "type": "List[str]",
                "description": "Identifiers matching return_type, for example 7NLL for an entry, 7NLL_1 for a polymer entity, or 7NL for a Chemical Component record.",
                "default": None,
            }
        ],
    },
    {
        "description": "Retrieve compact PDB structure and ligand metadata for an exact UniProt accession, with deterministic method and resolution filtering and automatic entity enumeration.",
        "name": "query_pdb_target_ligand_structures",
        "required_parameters": [
            {
                "name": "uniprot_accession",
                "type": "str",
                "description": "Exact UniProt accession to verify against PDB polymer entities, for example P22607.",
                "default": None,
            }
        ],
        "optional_parameters": [
            {
                "name": "max_resolution",
                "type": "float",
                "description": "Maximum X-ray/cryo-EM resolution in angstroms; entries without resolution are excluded when set.",
                "default": None,
            },
            {
                "name": "experimental_method",
                "type": "str",
                "description": "Exact PDB experimental method, such as X-RAY DIFFRACTION.",
                "default": "X-RAY DIFFRACTION",
            },
            {
                "name": "ligand_names",
                "type": "List[str]",
                "description": "Optional ligand names or Chemical Component IDs to match within the filtered structures.",
                "default": None,
            },
            {
                "name": "ligand_inchikeys",
                "type": "Dict",
                "description": "Optional mapping from requested ligand names to exact InChIKeys for identity matching.",
                "default": None,
            },
            {
                "name": "require_ligand",
                "type": "bool",
                "description": "Only return entries containing at least one nonpolymer entity.",
                "default": True,
            },
            {
                "name": "max_results",
                "type": "int",
                "description": "Maximum exact-UniProt entries to inspect, from 1 through 200.",
                "default": 50,
            },
            {
                "name": "timeout_seconds",
                "type": "int",
                "description": "Timeout for each RCSB HTTP request, from 1 through 120 seconds.",
                "default": 30,
            },
        ],
    },
    {
        "description": "Take a natural language prompt and convert it to a structured KEGG API query.",
        "name": "query_kegg",
        "optional_parameters": [
            {"name": "prompt", "type": "str", "description": "Natural language query about KEGG data", "default": None},
            {"name": "endpoint", "type": "str", "description": "Direct KEGG endpoint to query", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [],
    },
    {
        "description": "Query the STRING protein interaction database using natural language or direct endpoint.",
        "name": "query_stringdb",
        "optional_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about protein interactions",
                "default": None,
            },
            {"name": "endpoint", "type": "str", "description": "Full URL to query directly", "default": None},
            {
                "name": "download_image",
                "type": "bool",
                "description": "Download image results if endpoint is image",
                "default": False,
            },
            {"name": "output_dir", "type": "str", "description": "Directory to save downloaded files", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
            {
                "name": "identifiers",
                "type": "List[str]",
                "description": "Complete candidate gene/protein set to compare. Pass the full earlier ranked set, not only its top gene; results never add outside nodes.",
                "default": None,
            },
            {"name": "species", "type": "int", "description": "NCBI taxonomy identifier", "default": 9606},
            {
                "name": "required_score",
                "type": "int",
                "description": "Minimum STRING interaction score from 0 to 1000",
                "default": 400,
            },
        ],
        "required_parameters": [],
    },
    {
        "description": "Query the IUCN Red List API using natural language or a direct endpoint.",
        "name": "query_iucn",
        "optional_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about species conservation status",
                "default": None,
            },
            {"name": "endpoint", "type": "str", "description": "Endpoint name or full URL", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [{"name": "token", "type": "str", "description": "IUCN API token", "default": ""}],
    },
    {
        "description": "Query the Paleobiology Database (PBDB) API using natural language or a direct endpoint.",
        "name": "query_paleobiology",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "API endpoint name or full URL", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about fossil records",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the JASPAR REST API for transcription factor binding profiles.",
        "name": "query_jaspar",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "API endpoint path or full URL", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about TF binding profiles",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the World Register of Marine Species (WoRMS) REST API using natural language or a direct endpoint.",
        "name": "query_worms",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "Full URL or endpoint specification", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about marine species",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the cBioPortal REST API using natural language or a direct endpoint.",
        "name": "query_cbioportal",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "API endpoint path or full URL", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about cancer genomics",
                "default": None,
            }
        ],
    },
    {
        "description": "Convert a natural language prompt into a structured ClinVar search query and run it.",
        "name": "query_clinvar",
        "optional_parameters": [
            {"name": "search_term", "type": "str", "description": "Direct ClinVar search term", "default": None},
            {"name": "max_results", "type": "int", "description": "Maximum number of results", "default": 3},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about genetic variants",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the NCBI GEO database (GDS/GEOPROFILES) using natural language or direct search term.",
        "name": "query_geo",
        "optional_parameters": [
            {"name": "search_term", "type": "str", "description": "Direct GEO search term", "default": None},
            {"name": "max_results", "type": "int", "description": "Maximum number of results", "default": 3},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about expression data",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the NCBI dbSNP database using natural language or direct search term.",
        "name": "query_dbsnp",
        "optional_parameters": [
            {"name": "search_term", "type": "str", "description": "Direct dbSNP search term", "default": None},
            {"name": "max_results", "type": "int", "description": "Maximum number of results", "default": 3},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about SNPs/variants",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the UCSC Genome Browser API using natural language or a direct endpoint.",
        "name": "query_ucsc",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "Full URL or endpoint spec", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about genomic data",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the Ensembl REST API using natural language or a direct endpoint.",
        "name": "query_ensembl",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "Direct Ensembl endpoint or full URL", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about genomic data",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the OpenTargets Platform API using natural language or a direct GraphQL query.",
        "name": "query_opentarget",
        "optional_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about targets/diseases",
                "default": None,
            },
            {"name": "query", "type": "str", "description": "Direct GraphQL query string", "default": None},
            {"name": "variables", "type": "dict", "description": "Variables for GraphQL", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": False},
            {
                "name": "disease_name",
                "type": "str",
                "description": "Disease name to resolve and rank associated targets by Open Targets overall association score",
                "default": None,
            },
            {
                "name": "disease_id",
                "type": "str",
                "description": "Open Targets disease identifier; bypasses disease-name resolution",
                "default": None,
            },
            {
                "name": "max_results",
                "type": "int",
                "description": "Maximum associated targets to return",
                "default": 10,
            },
        ],
        "required_parameters": [],
    },
    {
        "description": "Query the Monarch Initiative API using natural language or a direct endpoint.",
        "name": "query_monarch",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "Direct endpoint or full URL", "default": None},
            {"name": "max_results", "type": "int", "description": "Max results (adds limit param)", "default": 2},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": False},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about genes/diseases/phenotypes",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the OpenFDA API using natural language or direct parameters.",
        "name": "query_openfda",
        "optional_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about OpenFDA data",
                "default": None,
            },
            {"name": "endpoint", "type": "str", "description": "Direct endpoint or full URL", "default": None},
            {"name": "max_results", "type": "int", "description": "Max results (limit)", "default": 100},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
            {"name": "search_params", "type": "dict", "description": "Search parameters mapping", "default": None},
            {"name": "sort_params", "type": "dict", "description": "Sort parameters mapping", "default": None},
            {"name": "count_params", "type": "str", "description": "Field to count", "default": None},
            {"name": "skip_results", "type": "int", "description": "Skip for pagination", "default": 0},
        ],
        "required_parameters": [],
    },
    {
        "description": "Query the GWAS Catalog API using natural language or a direct endpoint.",
        "name": "query_gwas_catalog",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "Endpoint name (e.g., 'studies')", "default": None},
            {"name": "max_results", "type": "int", "description": "Max results per page (size)", "default": 3},
        ],
        "required_parameters": [
            {"name": "prompt", "type": "str", "description": "Natural language query about GWAS data", "default": None}
        ],
    },
    {
        "description": "Query gnomAD for variants in a gene using natural language or direct gene symbol.",
        "name": "query_gnomad",
        "optional_parameters": [
            {"name": "gene_symbol", "type": "str", "description": "Gene symbol (e.g., 'BRCA1')", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about genetic variants",
                "default": None,
            }
        ],
    },
    {
        "description": "Identify a DNA or protein sequence using NCBI BLAST.",
        "name": "blast_sequence",
        "optional_parameters": [],
        "required_parameters": [
            {"name": "sequence", "type": "str", "description": "Query sequence", "default": None},
            {"name": "database", "type": "str", "description": "BLAST database (e.g., core_nt or nr)", "default": None},
            {"name": "program", "type": "str", "description": "BLAST program (blastn or blastp)", "default": None},
        ],
    },
    {
        "description": "Query the Reactome database using natural language or a direct endpoint; optionally download pathway diagrams.",
        "name": "query_reactome",
        "optional_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about biological pathways",
                "default": None,
            },
            {"name": "endpoint", "type": "str", "description": "Direct endpoint or full URL", "default": None},
            {
                "name": "download",
                "type": "bool",
                "description": "Download pathway diagram if available",
                "default": False,
            },
            {"name": "output_dir", "type": "str", "description": "Directory to save downloads", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [],
    },
    {
        "description": "Query the RegulomeDB database using natural language or direct endpoint.",
        "name": "query_regulomedb",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "Direct RegulomeDB endpoint URL", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": False},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about regulatory elements",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the PRIDE proteomics database using natural language or a direct endpoint.",
        "name": "query_pride",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "Full endpoint to query", "default": None},
            {"name": "max_results", "type": "int", "description": "Maximum number of results", "default": 3},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about proteomics data",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the Guide to PHARMACOLOGY (GtoPdb) database using natural language or a direct endpoint.",
        "name": "query_gtopdb",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "Full API endpoint to query", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about drug targets/ligands/interactions",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the ReMap database for regulatory elements and transcription factor binding.",
        "name": "query_remap",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "Full API endpoint to query", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about TF binding sites",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the Mouse Phenome Database (MPD) using natural language or a direct endpoint.",
        "name": "query_mpd",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "Full API endpoint to query", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about mouse phenotypes/strains",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the Electron Microscopy Data Bank (EMDB) using natural language or a direct endpoint.",
        "name": "query_emdb",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "Full API endpoint to query", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about EM structures",
                "default": None,
            }
        ],
    },
    {
        "description": "Query Synapse REST API for biomedical datasets/files using natural language or structured search parameters. Supports optional authentication via SYNAPSE_AUTH_TOKEN.",
        "name": "query_synapse",
        "optional_parameters": [
            {
                "name": "query_term",
                "type": "str|list[str]",
                "description": "Search term(s) (AND logic across list)",
                "default": None,
            },
            {
                "name": "return_fields",
                "type": "list[str]",
                "description": "Fields to return",
                "default": ["name", "node_type", "description"],
            },
            {"name": "max_results", "type": "int", "description": "Max results (20 typical, up to 50)", "default": 20},
            {
                "name": "query_type",
                "type": "str",
                "description": "'dataset', 'file', or 'folder'",
                "default": "dataset",
            },
            {
                "name": "verbose",
                "type": "bool",
                "description": "Return full API response or formatted",
                "default": True,
            },
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about biomedical data",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the PubChem PUG-REST API using natural language or a direct endpoint.",
        "name": "query_pubchem",
        "optional_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about chemical compounds",
                "default": None,
            },
            {
                "name": "endpoint",
                "type": "str",
                "description": "Direct PubChem API endpoint or full URL",
                "default": None,
            },
            {"name": "max_results", "type": "int", "description": "Max results (rate-limited to 5 rps)", "default": 5},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [],
    },
    {
        "description": "Query the ChEMBL REST API via natural language, direct endpoint, or identifiers (chembl_id, smiles, molecule_name).",
        "name": "query_chembl",
        "canonical_name": "omniInfra.tool.database.query_chembl",
        "aliases": ["chembl_query", "chembl_search"],
        "capabilities": [
            "chemical_database_query",
            "drug_mechanism_retrieval",
            "bioactivity_retrieval",
            "public_read_only_network",
        ],
        "source_ids": ["chembl"],
        "evidence_types": ["curated_database_record"],
        "side_effect": "read_only",
        "optional_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about bioactivity data",
                "default": None,
            },
            {
                "name": "endpoint",
                "type": "str",
                "description": "Direct ChEMBL API endpoint or full URL",
                "default": None,
            },
            {"name": "chembl_id", "type": "str", "description": "ChEMBL ID (e.g., 'CHEMBL25')", "default": None},
            {"name": "smiles", "type": "str", "description": "SMILES for similarity/substructure", "default": None},
            {"name": "molecule_name", "type": "str", "description": "Molecule name to search", "default": None},
            {"name": "max_results", "type": "int", "description": "Max results", "default": 20},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [],
    },
    {
        "description": "Query the UniChem 2.0 REST API using natural language or a direct endpoint.",
        "name": "query_unichem",
        "optional_parameters": [
            {"name": "prompt", "type": "str", "description": "Natural language query about chemical cross-references", "default": None},
            {"name": "endpoint", "type": "str", "description": "Direct UniChem endpoint or full URL", "default": None},
            {"name": "method", "type": "str", "description": "HTTP method; defaults to POST for compounds/connectivity and GET otherwise", "default": None},
            {"name": "data", "type": "dict", "description": "JSON body for UniChem POST requests", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [],
    },
    {
        "description": "Query the ClinicalTrials.gov API v2 using natural language or a direct endpoint.",
        "name": "query_clinicaltrials",
        "optional_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about clinical trials",
                "default": None,
            },
            {
                "name": "endpoint",
                "type": "str",
                "description": "Direct ClinicalTrials.gov endpoint or full URL",
                "default": None,
            },
            {"name": "max_results", "type": "int", "description": "Page size for results (pageSize)", "default": 10},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [],
    },
    {
        "description": "Query the DailyMed RESTful API using natural language or a direct endpoint.",
        "name": "query_dailymed",
        "optional_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about drug labeling",
                "default": None,
            },
            {"name": "endpoint", "type": "str", "description": "Direct DailyMed endpoint or full URL", "default": None},
            {"name": "format", "type": "str", "description": "'json' or 'xml'", "default": "json"},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [],
    },
    {
        "description": "Query the QuickGO API using natural language or a direct endpoint.",
        "name": "query_quickgo",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "Direct QuickGO endpoint or full URL", "default": None},
            {"name": "max_results", "type": "int", "description": "Max results (limit, up to 100)", "default": 25},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about GO terms/annotations",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the ENCODE Portal API to locate functional genomics data (experiments, files, biosamples, datasets).",
        "name": "query_encode",
        "optional_parameters": [
            {
                "name": "endpoint",
                "type": "str",
                "description": "Direct ENCODE Portal endpoint or full URL",
                "default": None,
            },
            {
                "name": "max_results",
                "type": "int|str",
                "description": "Limit for search endpoints (number or 'all')",
                "default": 25,
            },
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about functional genomics data",
                "default": None,
            }
        ],
    },
    {
        "description": "Given genomic coordinates, retrieve intersecting ENCODE SCREEN cCREs.",
        "name": "region_to_ccre_screen",
        "optional_parameters": [
            {"name": "assembly", "type": "str", "description": "Genome assembly (e.g., 'GRCh38')", "default": "GRCh38"}
        ],
        "required_parameters": [
            {"name": "coord_chrom", "type": "str", "description": "Chromosome (e.g., 'chr12')", "default": None},
            {"name": "coord_start", "type": "int", "description": "Start coordinate", "default": None},
            {"name": "coord_end", "type": "int", "description": "End coordinate", "default": None},
        ],
    },
    {
        "description": "Given a cCRE accession, return k nearest genes sorted by distance.",
        "name": "get_genes_near_ccre",
        "optional_parameters": [
            {"name": "k", "type": "int", "description": "Number of nearby genes to return", "default": 10}
        ],
        "required_parameters": [
            {
                "name": "accession",
                "type": "str",
                "description": "ENCODE cCRE accession ID (e.g., 'EH38E1516980')",
                "default": None,
            },
            {"name": "assembly", "type": "str", "description": "Genome assembly (e.g., 'GRCh38')", "default": None},
            {
                "name": "chromosome",
                "type": "str",
                "description": "Chromosome of the cCRE (e.g., 'chr12')",
                "default": None,
            },
        ],
    },
    {
        "name": "query_pharos_target",
        "description": "Retrieve Pharos evidence for one known, non-empty human HGNC gene symbol. This is not a disease-first target discovery tool; never pass gene_symbol=None.",
        "required_parameters": [
            {
                "name": "gene_symbol",
                "type": "str",
                "description": "One required, non-empty human HGNC gene symbol, such as EGFR; None is invalid.",
                "default": None,
            }
        ],
        "optional_parameters": [
            {
                "name": "disease_name",
                "type": "str",
                "description": "Optional disease-name substring used to filter Pharos disease associations.",
                "default": None,
            },
            {
                "name": "max_results",
                "type": "int",
                "description": "Maximum disease associations and ligand or drug records to return.",
                "default": 10,
            },
        ],
    },
    {
        "name": "query_opentargets_genetic_evidence",
        "description": "Retrieve Open Targets Platform GWAS studies, credible sets, Locus2Gene predictions, and colocalisation evidence for one human HGNC gene symbol.",
        "required_parameters": [
            {
                "name": "gene_symbol",
                "type": "str",
                "description": "One human HGNC gene symbol, such as EGFR.",
                "default": None,
            }
        ],
        "optional_parameters": [
            {
                "name": "disease_name",
                "type": "str",
                "description": "Optional disease or trait substring used to filter studies.",
                "default": None,
            },
            {
                "name": "max_results",
                "type": "int",
                "description": "Maximum records returned in each genetic-evidence section.",
                "default": 10,
            },
        ],
    },
    {
        "name": "query_gtex_expression",
        "description": "Read the complete normal-tissue expression profile for one human HGNC gene symbol from the registered GTEx data-lake Parquet snapshot.",
        "required_parameters": [
            {"name": "gene_symbol", "type": "str", "description": "One human HGNC gene symbol.", "default": None},
            {
                "name": "data_lake_path",
                "type": "str",
                "description": "Directory containing gtex_tissue_gene_tpm.parquet.",
                "default": None,
            },
        ],
        "optional_parameters": [
            {
                "name": "top_n",
                "type": "int",
                "description": "Number of highest-expression tissues to repeat in the top-tissues summary.",
                "default": 10,
            }
        ],
    },
    {
        "name": "query_depmap_gene_dependency",
        "description": "Read CRISPR gene effect and dependency probability for one human HGNC gene symbol from registered DepMap data-lake snapshots.",
        "required_parameters": [
            {"name": "gene_symbol", "type": "str", "description": "One human HGNC gene symbol.", "default": None},
            {
                "name": "data_lake_path",
                "type": "str",
                "description": "Directory containing the registered DepMap CSV snapshots.",
                "default": None,
            },
        ],
        "optional_parameters": [
            {
                "name": "lineage",
                "type": "str",
                "description": "Optional exact OncoTree lineage value from DepMap model metadata.",
                "default": None,
            },
            {
                "name": "include_expression",
                "type": "bool",
                "description": "Whether to include matching expression values from the optional DepMap expression snapshot.",
                "default": False,
            },
            {
                "name": "top_n",
                "type": "int",
                "description": "Maximum number of model records and expression records to return.",
                "default": 20,
            },
        ],
    },
]
