description = [
    {
        "name": "screen_antibiotic_candidates_with_antibioticsai",
        "description": "Score a CSV of candidate SMILES with the published 20-model Antibiotics-AI antibacterial, HepG2, HSkMC, and IMR-90 Chemprop ensembles, then apply the source library's strict antibacterial threshold and all three strict cytotoxicity thresholds. This is a prediction-stage screen, not experimental validation or reproduction of the published 283-compound selection.",
        "required_parameters": [
            {"name": "input_csv", "type": "str", "description": "CSV path containing a case-sensitive SMILES column; input order and duplicate SMILES are preserved by stable row ID.", "default": None},
            {"name": "output_dir", "type": "str", "description": "Writable parent directory where a unique run directory, four raw predictions, combined CSV, and summary JSON are created.", "default": None},
        ],
        "optional_parameters": [
            {"name": "asset_dir", "type": "str", "description": "Antibiotics-AI deployment root containing .venv/bin/chemprop_predict and four 20-checkpoint ensembles; defaults to ANTIBIOTICSAI_ROOT.", "default": ""},
            {"name": "library", "type": "str", "description": "Source-library threshold policy: broad uses antibacterial score >0.2; mcule uses >0.4. Both require HepG2, HSkMC, and IMR-90 scores each <0.2.", "default": "broad"},
            {"name": "timeout_seconds", "type": "int", "description": "Maximum total runtime for all four CPU ensembles in seconds.", "default": 3600},
        ],
    },
    {
        "description": "Screen candidate SMILES against a cropped binding pocket using real, offline DrugCLIP inference and return normalized embedding-similarity rankings, not calibrated affinity or binding probability.",
        "name": "screen_compounds_with_drugclip",
        "optional_parameters": [
            {"name": "output_dir", "type": "str", "description": "Parent directory for a unique run directory containing generated LMDB inputs, logs, and ranked scores; defaults to tools_pkg/DrugCLIP/outputs/runs.", "default": ""},
            {"name": "max_results", "type": "int", "description": "Maximum ranked compounds returned, from 1 to the official retrieval limit of 10000.", "default": 10000},
            {"name": "timeout_seconds", "type": "int", "description": "Maximum retrieval runtime in seconds.", "default": 3600},
        ],
        "required_parameters": [
            {"name": "smiles_list", "type": "List[str]", "description": "Candidate compound SMILES. The isolated runner validates each SMILES and generates its RDKit 3-D conformer and molecule LMDB.", "default": None},
            {"name": "pocket_path", "type": "str", "description": "Path to an already cropped binding-pocket PDB/ENT file containing protein ATOM coordinates, or an advanced-user official DrugCLIP pocket LMDB. A full receptor is not automatically pocket-cropped.", "default": None},
        ],
    },
    {
        "description": "Rank candidate compounds using a real DeepPurpose pretrained BindingDB drug-target interaction model.",
        "name": "rank_hits_with_deeppurpose",
        "optional_parameters": [
            {"name": "model_type", "type": "str", "description": "DeepPurpose model architecture: CNN-CNN, MPNN-CNN, Morgan-AAC, or Daylight-AAC.", "default": "MPNN-CNN"},
            {"name": "pretrained_models_dir", "type": "str", "description": "Directory containing uploaded official DeepPurpose model folders or ZIP archives; defaults to DEEPPURPOSE_PRETRAINED_ROOT or Biomni/DeepPurpose_models/pretrained_models. Missing local models fall back to official download.", "default": ""},
            {"name": "top_k", "type": "int", "description": "Maximum ranked hits returned.", "default": 20},
        ],
        "required_parameters": [
            {"name": "smiles_list", "type": "List[str]", "description": "Candidate compound SMILES strings.", "default": None},
            {"name": "protein_sequence", "type": "str", "description": "Target amino-acid sequence.", "default": None},
        ],
    },
    {
        "description": "Rank candidate compounds with an official pretrained GraphDTA Davis or KIBA benchmark model.",
        "name": "rank_hits_with_graphdta",
        "optional_parameters": [
            {"name": "checkpoint_path", "type": "str", "description": "Path to a real compatible checkpoint; defaults to GRAPH_DTA_CHECKPOINT or the pinned official upstream pretrained/model_<model>_<dataset>.model.", "default": ""},
            {"name": "model_name", "type": "str", "description": "Official architecture: GINConvNet, GATNet, GAT_GCN, or GCNNet.", "default": "GINConvNet"},
            {"name": "dataset_name", "type": "str", "description": "Checkpoint training dataset, normally davis or kiba.", "default": "davis"},
            {"name": "top_k", "type": "int", "description": "Maximum ranked hits returned.", "default": 20},
            {"name": "timeout_seconds", "type": "int", "description": "Maximum inference runtime in seconds.", "default": 1800},
            {"name": "repo_path", "type": "str", "description": "Official GraphDTA checkout; defaults to GRAPH_DTA_REPO or tools_pkg/GraphDTA/upstream.", "default": ""},
            {"name": "python_executable", "type": "str", "description": "GraphDTA environment Python; defaults to GRAPH_DTA_PYTHON or tools_pkg/GraphDTA/.conda/bin/python.", "default": ""},
        ],
        "required_parameters": [
            {"name": "smiles_list", "type": "List[str]", "description": "Candidate compound SMILES strings.", "default": None},
            {"name": "protein_sequence", "type": "str", "description": "Target amino-acid sequence, truncated by GraphDTA to 1000 residues.", "default": None},
        ],
    },
    {
        "description": "Submit a Pharmit pharmacophore query through its real startquery/getdata FastCGI protocol.",
        "name": "query_pharmit_pharmacophores",
        "optional_parameters": [
            {"name": "subset", "type": "str", "description": "Pharmit library subdir to search; use an available subset returned by the service.", "default": ""},
            {"name": "max_results", "type": "int", "description": "Maximum result rows returned, from 1 to 1000.", "default": 20},
            {"name": "pharmit_url", "type": "str", "description": "FastCGI endpoint; defaults to PHARMIT_URL or the public Pharmit service.", "default": ""},
            {"name": "timeout_seconds", "type": "int", "description": "Overall query and HTTP timeout in seconds.", "default": 60},
            {"name": "poll_interval_seconds", "type": "float", "description": "Delay between getdata polling requests.", "default": 1.0},
        ],
        "required_parameters": [
            {"name": "pharmacophore", "type": "dict", "description": "Pharmit saved-session query object containing a non-empty points list.", "default": None}
        ],
    },
    {
        "description": "Use RDKit ChemicalFeatures to design a pharmacophore feature set from a SMILES, optionally with a 3-D conformer.",
        "name": "design_rdkit_pharmacophore",
        "optional_parameters": [
            {"name": "include_3d", "type": "bool", "description": "Generate a UFF-optimized 3-D conformer and return atom coordinates.", "default": False}
        ],
        "required_parameters": [
            {"name": "smiles", "type": "str", "description": "Input molecule as a valid SMILES string.", "default": None}
        ],
    },
    {
        "description": "Run DiffDock molecular docking using a protein PDB file and "
        "a SMILES string for the ligand, executing the process in a "
        "Docker container.",
        "name": "run_diffdock_with_smiles",
        "optional_parameters": [
            {
                "default": None,
                "description": "GPU device ID；留空时由 MCP 服务自动选择空闲 GPU。",
                "name": "gpu_device",
                "type": "int",
            },
            {
                "default": True,
                "description": "Whether to use GPU acceleration for docking",
                "name": "use_gpu",
                "type": "bool",
            },
            {
                "default": None,
                "description": "Host directory containing score_model and confidence_model; if omitted, use DIFFDOCK_MODEL_DIR or auto-discover data/diffdock_models/v1.1. Its adjacent torch_cache persists downloaded ESM weights unless DIFFDOCK_CACHE_DIR overrides it",
                "name": "model_dir",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the protein PDB file for docking",
                "name": "pdb_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "SMILES string representation of the ligand molecule",
                "name": "smiles_string",
                "type": "str",
            },
            {
                "default": None,
                "description": "Local directory path where docking results will be saved",
                "name": "local_output_dir",
                "type": "str",
            },
        ],
    },
    {
        "description": "Performs AutoDock Vina docking and persists ranked PDBQT poses with docking scores.",
        "name": "docking_autodock_vina",
        "optional_parameters": [
            {
                "default": 1,
                "description": "Number of CPU cores to use for docking",
                "name": "ncpu",
                "type": "int",
            },
            {
                "default": None,
                "description": "Directory for persisted pose files; defaults to BIOMNI_TASK_OUTPUT_DIR or docking_results",
                "name": "output_dir",
                "type": "str",
            },
            {
                "default": 5,
                "description": "Number of ranked poses to persist per ligand",
                "name": "n_poses",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "List of SMILES strings representing small molecules to dock",
                "name": "smiles_list",
                "type": "List[str]",
            },
            {
                "default": None,
                "description": "Path to the receptor protein structure PDB file",
                "name": "receptor_pdb_file",
                "type": "str",
            },
            {
                "default": None,
                "description": "3D coordinates [x, y, z] of the docking box center",
                "name": "box_center",
                "type": "List[float]",
            },
            {
                "default": None,
                "description": "Dimensions [x, y, z] of the docking box",
                "name": "box_size",
                "type": "List[float]",
            },
        ],
    },
    {
        "description": "Runs the official Scripps ADFR Suite AutoSite commands on a receptor PDB, using ADFR_PREPARE_RECEPTOR and AUTOSITE_BIN when configured, and returns the generated pocket-ranking summary.",
        "name": "run_autosite",
        "optional_parameters": [
            {
                "default": 1.0,
                "description": "Positive grid spacing parameter for the AutoSite calculation",
                "name": "spacing",
                "type": "float",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the input PDB file",
                "name": "pdb_file",
                "type": "str",
            },
            {
                "default": None,
                "description": "Directory where AutoSite results will be saved",
                "name": "output_dir",
                "type": "str",
            },
        ],
    },
    {
        "description": "Retrieves top drug-repurposing candidates from trusted, precomputed TxGNN deployment artifacts and returns sigmoid-transformed ranking scores that are not calibrated probabilities.",
        "name": "retrieve_topk_repurposing_drugs_from_disease_txgnn",
        "optional_parameters": [
            {
                "default": 5,
                "description": "The number of top drug predictions to return",
                "name": "k",
                "type": "int",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "The name of the disease for which to retrieve drug predictions",
                "name": "disease_name",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to a trusted data lake containing txgnn_name_mapping.pkl and txgnn_prediction.pkl",
                "name": "data_lake_path",
                "type": "str",
            },
        ],
    },
    {
        "description": "Predicts 16 ADMET endpoints for compounds using validated local DeepPurpose checkpoints, with network download disabled by default.",
        "name": "predict_admet_properties",
        "optional_parameters": [
            {
                "default": "MPNN",
                "description": "Type of model to use for ADMET prediction (options: 'MPNN', 'CNN', 'Morgan')",
                "name": "ADMET_model_type",
                "type": "str",
            },
            {
                "default": None,
                "description": "Directory containing one extracted DeepPurpose checkpoint directory per ADMET endpoint; defaults to DEEPPURPOSE_PRETRAINED_ROOT or DeepPurpose_models/pretrained_models",
                "name": "models_root",
                "type": "str",
            },
            {
                "default": False,
                "description": "Whether missing checkpoints may use DeepPurpose's upstream download fallback; disabled by default for reproducible offline execution",
                "name": "allow_download",
                "type": "bool",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "List of SMILES strings representing chemical compounds to analyze",
                "name": "smiles_list",
                "type": "List[str]",
            }
        ],
    },
    {
        "description": "Predicts binding affinity between small molecules and a "
        "protein sequence using pre-trained deep learning models.",
        "name": "predict_binding_affinity_protein_1d_sequence",
        "optional_parameters": [
            {
                "default": "MPNN-CNN",
                "description": "Deep learning model architecture to "
                "use for binding affinity prediction "
                "(options: CNN-CNN, MPNN-CNN, "
                "Morgan-CNN, Morgan-AAC, "
                "Daylight-AAC)",
                "name": "affinity_model_type",
                "type": "str",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "List of SMILES strings representing chemical compounds",
                "name": "smiles_list",
                "type": "List[str]",
            },
            {
                "default": None,
                "description": "Protein sequence in amino acid format",
                "name": "amino_acid_sequence",
                "type": "str",
            },
        ],
    },
    {
        "description": "Analyzes the stability of pharmaceutical formulations under accelerated storage conditions.",
        "name": "analyze_accelerated_stability_of_pharmaceutical_formulations",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "List of formulation dictionaries "
                "containing name, active ingredient, "
                "concentration, and excipients",
                "name": "formulations",
                "type": "List[dict]",
            },
            {
                "default": None,
                "description": "List of storage condition "
                "dictionaries containing "
                "temperature, humidity (optional), "
                "and description",
                "name": "storage_conditions",
                "type": "List[dict]",
            },
            {
                "default": None,
                "description": "List of time points in days to evaluate stability",
                "name": "time_points",
                "type": "List[int]",
            },
        ],
    },
    {
        "description": "Generates a detailed protocol for performing a 3D "
        "chondrogenic aggregate culture assay to evaluate compounds' "
        "effects on chondrogenesis.",
        "name": "run_3d_chondrogenic_aggregate_assay",
        "optional_parameters": [
            {
                "default": 21,
                "description": "Total duration of the culture period in days",
                "name": "culture_duration_days",
                "type": "int",
            },
            {
                "default": 7,
                "description": "Interval in days between measurements",
                "name": "measurement_intervals",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Dictionary with cell information "
                "including 'source', "
                "'passage_number', and "
                "'cell_density'",
                "name": "chondrocyte_cells",
                "type": "dict",
            },
            {
                "default": None,
                "description": "List of compounds to test, each with 'name', 'concentration', and 'vehicle' keys",
                "name": "test_compounds",
                "type": "list of dict",
            },
        ],
    },
    {
        "description": "Grade and monitor adverse events in animal studies using the VCOG-CTCAE standard.",
        "name": "grade_adverse_events_using_vcog_ctcae",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to a CSV file containing "
                "clinical evaluation data with "
                "columns: subject_id, time_point, "
                "symptom, severity, measurement "
                "(optional)",
                "name": "clinical_data_file",
                "type": "str",
            }
        ],
    },
    {
        "description": "Analyze biodistribution and pharmacokinetic profile of radiolabeled antibodies.",
        "name": "analyze_radiolabeled_antibody_biodistribution",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "Time points (hours) at which measurements were taken",
                "name": "time_points",
                "type": "List[float] or numpy.ndarray",
            },
            {
                "default": None,
                "description": "Dictionary where keys are tissue "
                "names and values are lists/arrays "
                "of %IA/g measurements corresponding "
                "to time_points. Must include "
                "'tumor' as one of the keys",
                "name": "tissue_data",
                "type": "dict",
            },
        ],
    },
    {
        "description": "Estimate radiation absorbed doses to tumor and normal organs "
        "for alpha-particle radiotherapeutics using the Medical "
        "Internal Radiation Dose (MIRD) schema.",
        "name": "estimate_alpha_particle_radiotherapy_dosimetry",
        "optional_parameters": [
            {
                "default": "dosimetry_results.csv",
                "description": "Filename to save the dosimetry results",
                "name": "output_file",
                "type": "str",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Dictionary containing organ/tissue "
                "names as keys and a list of "
                "time-activity measurements as "
                "values. Each measurement should be "
                "a tuple of (time_hours, "
                "percent_injected_activity). Must "
                "include entries for all relevant "
                "organs including 'tumor'.",
                "name": "biodistribution_data",
                "type": "dict",
            },
            {
                "default": None,
                "description": "Dictionary containing radiation "
                "parameters for the alpha-emitting "
                "radionuclide including "
                "'radionuclide', 'half_life_hours', "
                "'energy_per_decay_MeV', "
                "'radiation_weighting_factor', and "
                "'S_factors'.",
                "name": "radiation_parameters",
                "type": "dict",
            },
        ],
    },
    {
        "description": "Perform a Methylome-wide Association Study (MWAS) to "
        "identify CpG sites significantly associated with CYP2C19 "
        "metabolizer status.",
        "name": "perform_mwas_cyp2c19_metabolizer_status",
        "optional_parameters": [
            {
                "default": None,
                "description": "Path to CSV or TSV file containing "
                "covariates to adjust for in the "
                "regression model (e.g., age, sex, "
                "smoking status).",
                "name": "covariates_path",
                "type": "str",
            },
            {
                "default": 0.05,
                "description": "P-value threshold for significance after multiple testing correction.",
                "name": "pvalue_threshold",
                "type": "float",
            },
            {
                "default": "significant_cpg_sites.csv",
                "description": "Filename to save significant CpG sites.",
                "name": "output_file",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to CSV or TSV file containing "
                "DNA methylation beta values. Rows "
                "should be samples, columns should "
                "be CpG sites.",
                "name": "methylation_data_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to CSV or TSV file containing "
                "CYP2C19 metabolizer status for each "
                "sample. Should have a sample ID "
                "column and a status column.",
                "name": "metabolizer_status_path",
                "type": "str",
            },
        ],
    },
    {
        "description": "Calculate key physicochemical properties of a drug candidate molecule.",
        "name": "calculate_physicochemical_properties",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "The molecular structure in SMILES format",
                "name": "smiles_string",
                "type": "str",
            }
        ],
    },
    {
        "description": "Analyze tumor growth inhibition in xenograft models across different treatment groups.",
        "name": "analyze_xenograft_tumor_growth_inhibition",
        "optional_parameters": [
            {
                "default": "./results",
                "description": "Directory to save output files",
                "name": "output_dir",
                "type": "str",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to CSV or TSV file containing tumor volume measurements",
                "name": "data_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Name of the column containing time points",
                "name": "time_column",
                "type": "str",
            },
            {
                "default": None,
                "description": "Name of the column containing tumor volume measurements",
                "name": "volume_column",
                "type": "str",
            },
            {
                "default": None,
                "description": "Name of the column containing treatment group labels",
                "name": "group_column",
                "type": "str",
            },
            {
                "default": None,
                "description": "Name of the column containing subject/mouse identifiers",
                "name": "subject_column",
                "type": "str",
            },
        ],
    },
    {
        "description": "Analyze western blot or DNA electrophoresis images and return pixel distribution statistics including intensity statistics, percentiles, and brightness distribution. Use this to determine appropriate threshold values for find_roi_from_image.",
        "name": "analyze_pixel_distribution",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the input grayscale image. Automatically appends .png if no suffix is provided.",
                "name": "image_path",
                "type": "str",
            }
        ],
    },
    {
        "description": "Find the ROIs (regions of interest) of protein bands from a Western blot or DNA electrophoresis image using threshold-based blob detection. Returns annotated image path and list of ROI coordinates. Use analyze_pixel_distribution first to determine appropriate threshold values. The returned ROI list can be converted to target_bands format for analyze_western_blot.",
        "name": "find_roi_from_image",
        "optional_parameters": [
            {
                "default": True,
                "description": "If True, draw green contours (hulls) and blue keypoint boxes for debugging.",
                "name": "debug",
                "type": "bool",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the input image.",
                "name": "image_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Pixel intensities lower than this value are used to make the binary image. Use analyze_pixel_distribution to determine appropriate values.",
                "name": "lower_threshold",
                "type": "int",
            },
            {
                "default": None,
                "description": "Pixel intensities greater than or equal to this value are used to make the binary image. Use analyze_pixel_distribution to determine appropriate values.",
                "name": "upper_threshold",
                "type": "int",
            },
            {
                "default": None,
                "description": "The actual number of bands expected in the image.",
                "name": "number_of_bands",
                "type": "int",
            },
        ],
    },
    {
        "description": "Performs densitometric analysis of Western blot images to "
        "quantify relative protein expression.",
        "name": "analyze_western_blot",
        "optional_parameters": [
            {
                "default": "./results",
                "description": "Directory to save output files",
                "name": "output_dir",
                "type": "str",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the Western blot image file",
                "name": "blot_image_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "List of dictionaries containing "
                "information about target protein "
                "bands, each with 'name' and 'roi' "
                "(region of interest as [x, y, "
                "width, height])",
                "name": "target_bands",
                "type": "list of dict",
            },
            {
                "default": None,
                "description": "Dictionary with 'name' and 'roi' "
                "for the loading control protein "
                "(e.g., β-actin, GAPDH)",
                "name": "loading_control_band",
                "type": "dict",
            },
            {
                "default": None,
                "description": "Dictionary containing information "
                "about antibodies used with "
                "'primary' and 'secondary' keys",
                "name": "antibody_info",
                "type": "dict",
            },
        ],
    },
    {
        "description": "Query drug-drug interactions from DDInter database to identify potential interactions, mechanisms, and severity levels between specified drugs.",
        "name": "query_drug_interactions",
        "required_parameters": [
            {
                "default": None,
                "description": "List of drug names to query for interactions",
                "name": "drug_names",
                "type": "List[str]",
            }
        ],
        "optional_parameters": [
            {
                "default": None,
                "description": "Filter results by specific interaction types",
                "name": "interaction_types",
                "type": "List[str]",
            },
            {
                "default": None,
                "description": "Filter results by severity levels (Major, Moderate, Minor)",
                "name": "severity_levels",
                "type": "List[str]",
            },
            {
                "default": None,
                "description": "Path to data lake directory containing DDInter data",
                "name": "data_lake_path",
                "type": "str",
            },
        ],
    },
    {
        "description": "Analyze safety of a drug combination for potential interactions using DDInter database with comprehensive risk assessment and clinical recommendations.",
        "name": "check_drug_combination_safety",
        "required_parameters": [
            {
                "default": None,
                "description": "List of drugs to analyze for combination safety",
                "name": "drug_list",
                "type": "List[str]",
            }
        ],
        "optional_parameters": [
            {
                "default": True,
                "description": "Include interaction mechanism descriptions in results",
                "name": "include_mechanisms",
                "type": "bool",
            },
            {
                "default": True,
                "description": "Include management recommendations in results",
                "name": "include_management",
                "type": "bool",
            },
            {
                "default": None,
                "description": "Path to data lake directory containing DDInter data",
                "name": "data_lake_path",
                "type": "str",
            },
        ],
    },
    {
        "description": "Analyze interaction mechanisms between two specific drugs providing detailed mechanistic insights and clinical significance assessment.",
        "name": "analyze_interaction_mechanisms",
        "required_parameters": [
            {
                "default": None,
                "description": "Pair of drug names to analyze (drug1, drug2)",
                "name": "drug_pair",
                "type": "Tuple[str, str]",
            }
        ],
        "optional_parameters": [
            {
                "default": True,
                "description": "Include detailed mechanistic information in analysis",
                "name": "detailed_analysis",
                "type": "bool",
            },
            {
                "default": None,
                "description": "Path to data lake directory containing DDInter data",
                "name": "data_lake_path",
                "type": "str",
            },
        ],
    },
    {
        "description": "Find alternative drugs that don't interact with contraindicated drugs using DDInter database for safer therapeutic substitutions.",
        "name": "find_alternative_drugs_ddinter",
        "required_parameters": [
            {
                "default": None,
                "description": "Drug to find alternatives for",
                "name": "target_drug",
                "type": "str",
            },
            {
                "default": None,
                "description": "List of drugs to avoid interactions with",
                "name": "contraindicated_drugs",
                "type": "List[str]",
            },
        ],
        "optional_parameters": [
            {
                "default": None,
                "description": "Limit search to specific therapeutic class",
                "name": "therapeutic_class",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to data lake directory containing DDInter data",
                "name": "data_lake_path",
                "type": "str",
            },
        ],
    },
    {
        "description": "Query FDA adverse event reports for specific drugs from the OpenFDA database to identify potential safety signals, reaction patterns, and regulatory intelligence.",
        "name": "query_fda_adverse_events",
        "required_parameters": [
            {
                "default": None,
                "description": "Name of the drug to query for adverse events",
                "name": "drug_name",
                "type": "str",
            },
        ],
        "optional_parameters": [
            {
                "default": None,
                "description": "Optional date range as (start_date, end_date) in YYYY-MM-DD format",
                "name": "date_range",
                "type": "Tuple[str, str]",
            },
            {
                "default": None,
                "description": "Optional filter by severity levels ['serious', 'non_serious']",
                "name": "severity_filter",
                "type": "List[str]",
            },
            {
                "default": None,
                "description": "Optional filter by outcomes ['life_threatening', 'hospitalization', 'death']",
                "name": "outcome_filter",
                "type": "List[str]",
            },
            {
                "default": 100,
                "description": "Maximum number of results to return",
                "name": "limit",
                "type": "int",
            },
        ],
    },
    {
        "description": "Retrieve FDA drug label information including indications, contraindications, warnings, and dosage information from the OpenFDA database.",
        "name": "get_fda_drug_label_info",
        "required_parameters": [
            {
                "default": None,
                "description": "Name of the drug to query for label information",
                "name": "drug_name",
                "type": "str",
            },
        ],
        "optional_parameters": [
            {
                "default": None,
                "description": "Optional list of specific sections to retrieve ['indications_and_usage', 'contraindications', 'warnings', 'dosage_and_administration']",
                "name": "sections",
                "type": "List[str]",
            },
        ],
    },
    {
        "description": "Check for FDA drug recalls and enforcement actions from the OpenFDA database to identify safety concerns and regulatory actions.",
        "name": "check_fda_drug_recalls",
        "required_parameters": [
            {
                "default": None,
                "description": "Name of the drug to check for recalls",
                "name": "drug_name",
                "type": "str",
            },
        ],
        "optional_parameters": [
            {
                "default": None,
                "description": "Optional filter by recall class ['Class I', 'Class II', 'Class III']",
                "name": "classification",
                "type": "List[str]",
            },
            {
                "default": None,
                "description": "Optional date range for recalls as (start_date, end_date)",
                "name": "date_range",
                "type": "Tuple[str, str]",
            },
        ],
    },
    {
        "description": "Analyze safety signals across multiple drugs using OpenFDA adverse event data to identify patterns and comparative risk profiles.",
        "name": "analyze_fda_safety_signals",
        "required_parameters": [
            {
                "default": None,
                "description": "List of drug names to analyze for safety signals",
                "name": "drug_list",
                "type": "List[str]",
            },
        ],
        "optional_parameters": [
            {
                "default": None,
                "description": "Optional comparison time period as (start_date, end_date)",
                "name": "comparison_period",
                "type": "Tuple[str, str]",
            },
            {
                "default": 2.0,
                "description": "Threshold for signal detection",
                "name": "signal_threshold",
                "type": "float",
            },
        ],
    },
    {
        "description": "Predict macro-pKa from a SMILES string or micro-pKa from paired acid/base microstate SMILES using the external TripKa repository and the unipka conda environment.",
        "name": "predict_pka_with_tripka",
        "required_parameters": [],
        "optional_parameters": [
            {
                "default": "",
                "description": "SMILES string for macro-pKa prediction. Leave empty when using micro_a and micro_b.",
                "name": "smiles",
                "type": "str",
            },
            {
                "default": "",
                "description": "Acid microstate SMILES for micro-pKa prediction. Must be provided together with micro_b.",
                "name": "micro_a",
                "type": "str",
            },
            {
                "default": "",
                "description": "Base microstate SMILES for micro-pKa prediction. Must be provided together with micro_a.",
                "name": "micro_b",
                "type": "str",
            },
            {
                "default": "biomni_tripka",
                "description": "Dataset/task name prefix used by TripKa for intermediate and output files.",
                "name": "dataset",
                "type": "str",
            },
            {
                "default": 4,
                "description": "Number of TripKa enumeration iterations for macro-pKa prediction.",
                "name": "iter_num",
                "type": "int",
            },
            {
                "default": 4,
                "description": "Number of conformer samples used by TripKa. Must be at least 2.",
                "name": "sample_num",
                "type": "int",
            },
            {
                "default": 0,
                "description": "CUDA device index passed to TripKa inference.",
                "name": "cuda_idx",
                "type": "int",
            },
            {
                "default": "A",
                "description": "Macro-pKa enumeration mode, either 'A' for acid-first or 'B' for base-first.",
                "name": "mode",
                "type": "str",
            },
            {
                "default": 1800,
                "description": "Maximum runtime in seconds before terminating the TripKa subprocess.",
                "name": "timeout_seconds",
                "type": "int",
            },
            {
                "default": "",
                "description": "TripKa checkout; defaults to TRIPKA_REPO or tools_pkg/TripKa/upstream.",
                "name": "tripka_repo",
                "type": "str",
            },
            {
                "default": "",
                "description": "TripKa environment Python; defaults to TRIPKA_PYTHON or tools_pkg/TripKa/.conda/bin/python.",
                "name": "python_executable",
                "type": "str",
            },
            {
                "default": "",
                "description": "Parent directory for isolated per-call TripKa workspaces; defaults to TRIPKA_RUNTIME_ROOT or output/tripka_runtime.",
                "name": "runtime_root",
                "type": "str",
            },
        ],
    },
    {
        "name": "generate_conformers_with_geodiff",
        "description": "Generate multiple 3-D conformers for a supplied molecular graph using an official GeoDiff checkpoint in an isolated environment.",
        "required_parameters": [
            {"name": "smiles", "type": "str", "description": "Valid SMILES defining the molecular graph whose conformers will be generated.", "default": None},
            {"name": "output_dir", "type": "str", "description": "Host directory for generated SDF files and run metadata.", "default": None},
        ],
        "optional_parameters": [
            {"name": "num_conformers", "type": "int", "description": "Number of conformers to generate, from 1 to 1000.", "default": 10},
            {"name": "checkpoint_path", "type": "str", "description": "Official GeoDiff checkpoint; defaults to GEODIFF_CHECKPOINT or the deployed drugs_default checkpoint.", "default": ""},
            {"name": "repo_path", "type": "str", "description": "GeoDiff pretrain-branch checkout; defaults to GEODIFF_REPO or tools_pkg/GeoDiff/upstream.", "default": ""},
            {"name": "python_executable", "type": "str", "description": "GeoDiff environment Python; defaults to GEODIFF_PYTHON or tools_pkg/GeoDiff/.conda/bin/python.", "default": ""},
            {"name": "gpu_device", "type": "int", "description": "Host CUDA device index；留空时由 MCP 服务自动选择空闲 GPU。", "default": None},
            {"name": "use_gpu", "type": "bool", "description": "Use CUDA when true; use CPU when false.", "default": True},
            {"name": "num_steps", "type": "int", "description": "GeoDiff sampling steps, from 1 to 10000; lower values are faster but may reduce quality.", "default": 5000},
            {"name": "timeout_seconds", "type": "int", "description": "Maximum subprocess runtime in seconds.", "default": 7200},
        ],
    },
    {
        "name": "generate_ligands_with_targetdiff",
        "description": "Generate reconstructable 3-D ligands conditioned on a prepared protein-pocket PDB using TargetDiff.",
        "required_parameters": [
            {"name": "pocket_pdb_path", "type": "str", "description": "Prepared PDB containing only the target pocket atoms, normally cropped around a reference ligand.", "default": None},
            {"name": "output_dir", "type": "str", "description": "Host directory for TargetDiff SDF files and run metadata.", "default": None},
        ],
        "optional_parameters": [
            {"name": "num_samples", "type": "int", "description": "Number of ligand samples to request, from 1 to 1000.", "default": 20},
            {"name": "config_path", "type": "str", "description": "Official TargetDiff sampling YAML; defaults to TARGETDIFF_CONFIG or the deployed upstream sampling config.", "default": ""},
            {"name": "checkpoint_path", "type": "str", "description": "Official TargetDiff checkpoint; defaults to TARGETDIFF_CHECKPOINT or the deployed pretrained_diffusion checkpoint.", "default": ""},
            {"name": "repo_path", "type": "str", "description": "TargetDiff checkout; defaults to TARGETDIFF_REPO or tools_pkg/TargetDiff/upstream.", "default": ""},
            {"name": "python_executable", "type": "str", "description": "TargetDiff environment Python; defaults to TARGETDIFF_PYTHON or tools_pkg/TargetDiff/.conda/bin/python.", "default": ""},
            {"name": "gpu_device", "type": "int", "description": "Host CUDA device index；留空时由 MCP 服务自动选择，-1 表示 CPU。", "default": None},
            {"name": "batch_size", "type": "int", "description": "Sampling batch size, from 1 to 1000.", "default": 20},
            {"name": "timeout_seconds", "type": "int", "description": "Maximum subprocess runtime in seconds.", "default": 7200},
        ],
    },
    {
        "name": "generate_ligands_with_pocket2mol",
        "description": "Generate 3-D molecules with Pocket2Mol inside a cubic binding-pocket region of a protein structure.",
        "required_parameters": [
            {"name": "protein_pdb_path", "type": "str", "description": "Protein PDB used to extract pocket atoms.", "default": None},
            {"name": "pocket_center", "type": "list[float]", "description": "Pocket center coordinates [x, y, z] in angstroms.", "default": None},
            {"name": "output_dir", "type": "str", "description": "Host directory for Pocket2Mol SDF files and run metadata.", "default": None},
        ],
        "optional_parameters": [
            {"name": "num_samples", "type": "int", "description": "Number of molecules to request, from 1 to 1000.", "default": 20},
            {"name": "bbox_size", "type": "float", "description": "Side length of the cubic pocket box in angstroms.", "default": 23.0},
            {"name": "config_path", "type": "str", "description": "Official Pocket2Mol sampling YAML; defaults to POCKET2MOL_CONFIG or the deployed upstream sampling config.", "default": ""},
            {"name": "checkpoint_path", "type": "str", "description": "Official Pocket2Mol checkpoint; defaults to POCKET2MOL_CHECKPOINT or the deployed pretrained_Pocket2Mol checkpoint.", "default": ""},
            {"name": "repo_path", "type": "str", "description": "Pocket2Mol checkout; defaults to POCKET2MOL_REPO or tools_pkg/Pocket2Mol/upstream.", "default": ""},
            {"name": "python_executable", "type": "str", "description": "Pocket2Mol environment Python; defaults to POCKET2MOL_PYTHON or tools_pkg/Pocket2Mol/.conda/bin/python.", "default": ""},
            {"name": "gpu_device", "type": "int", "description": "Host CUDA device index；留空时由 MCP 服务自动选择，-1 表示 CPU。", "default": None},
            {"name": "timeout_seconds", "type": "int", "description": "Maximum subprocess runtime in seconds.", "default": 7200},
        ],
    },
    {
        "name": "optimize_ligands_with_autogrow4",
        "description": "Grow and optimize seed SMILES against a prepared receptor and docking box using the official AutoGrow4 workflow.",
        "required_parameters": [
            {"name": "receptor_pdb_path", "type": "str", "description": "Prepared receptor PDB accepted by AutoGrow4.", "default": None},
            {"name": "source_smiles", "type": "List[str]", "description": "Non-empty seed SMILES list used for mutation and crossover.", "default": None},
            {"name": "box_center", "type": "list[float]", "description": "Docking-box center coordinates [x, y, z] in angstroms.", "default": None},
            {"name": "box_size", "type": "list[float]", "description": "Positive docking-box dimensions [x, y, z] in angstroms.", "default": None},
            {"name": "output_dir", "type": "str", "description": "Host directory for the AutoGrow4 run and generated ligands.", "default": None},
        ],
        "optional_parameters": [
            {"name": "num_generations", "type": "int", "description": "Evolutionary generations, from 1 to 100.", "default": 3},
            {"name": "population_size", "type": "int", "description": "Mutants and crossovers requested per generation, from 2 to 10000.", "default": 10},
            {"name": "repo_path", "type": "str", "description": "AutoGrow4 checkout; defaults to AUTOGROW4_REPO or tools_pkg/AutoGrow4/upstream.", "default": ""},
            {"name": "python_executable", "type": "str", "description": "AutoGrow4 environment Python; defaults to AUTOGROW4_PYTHON or tools_pkg/AutoGrow4/.conda/bin/python.", "default": ""},
            {"name": "timeout_seconds", "type": "int", "description": "Maximum subprocess runtime in seconds.", "default": 14400},
        ],
    },
    {
        "name": "link_fragments_with_syntalinker",
        "description": "Generate complete molecules by linking two single-attachment fragments under a shortest-linker-bond-distance constraint using SyntaLinker.",
        "required_parameters": [
            {"name": "fragment_a_smiles", "type": "str", "description": "First valid fragment SMILES containing exactly one dummy attachment atom (*).", "default": None},
            {"name": "fragment_b_smiles", "type": "str", "description": "Second valid fragment SMILES containing exactly one dummy attachment atom (*).", "default": None},
            {"name": "linker_length", "type": "int", "description": "Requested shortest linker bond distance from 2 through 20, matching the published training domain.", "default": None},
            {"name": "output_dir", "type": "str", "description": "Host directory for predictions and run metadata.", "default": None},
        ],
        "optional_parameters": [
            {"name": "num_samples", "type": "int", "description": "Number of ranked predictions to request, from 1 to 100 and no greater than beam_size.", "default": 10},
            {"name": "beam_size", "type": "int", "description": "Beam-search width, from 1 to 100.", "default": 10},
            {"name": "max_length", "type": "int", "description": "Maximum decoded token length, from 1 to 1000.", "default": 200},
            {"name": "checkpoint_path", "type": "str", "description": "Reviewed SyntaLinker model checkpoint; defaults to SYNTALINKER_CHECKPOINT. Upstream does not publish pretrained weights.", "default": ""},
            {"name": "repo_path", "type": "str", "description": "SyntaLinker checkout; defaults to SYNTALINKER_REPO or tools_pkg/SyntaLinker/upstream.", "default": ""},
            {"name": "python_executable", "type": "str", "description": "SyntaLinker environment Python; defaults to SYNTALINKER_PYTHON or tools_pkg/SyntaLinker/.conda/bin/python.", "default": ""},
            {"name": "gpu_device", "type": "int", "description": "Host CUDA device index, or -1 for the default CPU deployment.", "default": -1},
            {"name": "timeout_seconds", "type": "int", "description": "Maximum subprocess runtime in seconds.", "default": 3600},
        ],
    },
    {
        "name": "detect_protein_pockets_with_fpocket",
        "description": "Detect pockets in one local PDB structure with the Docker-only fpocket backend and return native pocket descriptors.",
        "required_parameters": [
            {
                "name": "pdb_file_path",
                "type": "str",
                "description": "Path to a local PDB file; this tool does not search or download structures.",
                "default": None,
            },
            {
                "name": "output_dir",
                "type": "str",
                "description": "Dedicated local directory for native fpocket output files.",
                "default": None,
            },
        ],
        "optional_parameters": [
            {
                "name": "top_n",
                "type": "int",
                "description": "Maximum pockets to return, sorted by native fpocket score.",
                "default": 10,
            }
        ],
    },
    {
        "name": "score_protein_pockets_with_dogsite",
        "description": "Upload one local PDB structure to ProteinsPlus and return native DoGSiteScorer pocket and druggability descriptors.",
        "required_parameters": [
            {
                "name": "pdb_file_path",
                "type": "str",
                "description": "Path to a local PDB file that will be uploaded to ProteinsPlus; this tool does not search or download structures.",
                "default": None,
            },
            {
                "name": "output_dir",
                "type": "str",
                "description": "Dedicated local directory for downloaded DoGSiteScorer result files.",
                "default": None,
            },
        ],
        "optional_parameters": [
            {
                "name": "chain_id",
                "type": "str",
                "description": "Optional PDB chain identifier; None analyzes all chains.",
                "default": None,
            },
            {
                "name": "include_subpockets",
                "type": "bool",
                "description": "Whether ProteinsPlus should return subpocket analysis as well as pockets.",
                "default": False,
            },
            {
                "name": "top_n",
                "type": "int",
                "description": "Maximum pockets to return, sorted by native DoGSiteScorer drug score.",
                "default": 10,
            },
        ],
    },
    {
        "name": "generate_molecules_with_reinvent",
        "description": "Run a REINVENT 4 sampling or property-optimization configuration in the deployed CUDA 12.1 environment, with optional CPU fallback.",
        "required_parameters": [
            {"name": "config_path", "type": "str", "description": "Existing REINVENT TOML, JSON, or YAML configuration defining the prior, run mode, and target-property scoring components.", "default": None},
            {"name": "output_dir", "type": "str", "description": "Directory for the Biomni REINVENT log and result references.", "default": None},
        ],
        "optional_parameters": [
            {"name": "gpu_device", "type": "int", "description": "CUDA device index；留空时由 MCP 服务自动选择。", "default": None},
            {"name": "use_gpu", "type": "bool", "description": "Use CUDA when true; use CPU fallback when false.", "default": True},
            {"name": "timeout_seconds", "type": "int", "description": "Maximum backend runtime in seconds.", "default": 1800},
        ],
    },
    {
        "name": "generate_smiles_with_molgpt",
        "description": "Generate RDKit-validated SMILES unconditionally or from a MOSES-vocabulary SMILES prefix using the deployed official MolGPT checkpoint; natural-language property prompts are not supported.",
        "required_parameters": [
            {"name": "smiles_prefix", "type": "str", "description": "Use 'unconditional' (or an empty string) for unconditional sampling, otherwise provide a prefix fully composed of MOSES-vocabulary SMILES tokens; do not provide natural language.", "default": None},
            {"name": "output_dir", "type": "str", "description": "Directory for the request and generated SMILES.", "default": None},
        ],
        "optional_parameters": [
            {"name": "num_molecules", "type": "int", "description": "Number of SMILES requested.", "default": 10},
            {"name": "gpu_device", "type": "int", "description": "CUDA device index；留空时由 MCP 服务自动选择。", "default": None},
            {"name": "use_gpu", "type": "bool", "description": "Use CUDA when true; use CPU fallback when false.", "default": True},
            {"name": "timeout_seconds", "type": "int", "description": "Maximum backend runtime in seconds.", "default": 1800},
        ],
    },
    {
        "name": "generate_molecules_with_graphaf",
        "description": "Generate RDKit-valid SMILES with the deployed anonymous ICLR2020 GraphAF checkpoint, or an explicitly configured command or Docker image.",
        "required_parameters": [
            {"name": "prompt", "type": "str", "description": "Generation task label; the deployed checkpoint performs unconditional sampling and does not condition on natural-language properties.", "default": None},
            {"name": "output_dir", "type": "str", "description": "Writable directory for graphaf_request.json and generated_smiles.txt; the returned result_path identifies the final SMILES file.", "default": None},
        ],
        "optional_parameters": [
            {"name": "num_molecules", "type": "int", "description": "Number of molecules requested.", "default": 10},
            {"name": "gpu_device", "type": "int", "description": "CUDA device index；留空时由 MCP 服务自动选择。", "default": None},
            {"name": "use_gpu", "type": "bool", "description": "Use CUDA when available; fall back to CPU when false or when CUDA is unavailable.", "default": True},
            {"name": "timeout_seconds", "type": "int", "description": "Maximum backend runtime in seconds.", "default": 1800},
        ],
    },
    {
        "name": "edit_molecule_with_rdkit",
        "description": "Apply a SMARTS replacement to edit a molecule using RDKit.",
        "required_parameters": [
            {"name": "smiles", "type": "str", "description": "Input molecule SMILES.", "default": None},
            {"name": "smarts_pattern", "type": "str", "description": "SMARTS substructure to replace.", "default": None},
            {"name": "replacement_smiles", "type": "str", "description": "SMILES fragment used as replacement.", "default": None},
        ],
        "optional_parameters": [],
    },
    {
        "name": "search_rdkit_scaffold_network",
        "description": "Enumerate the RDKit Scaffold Network for an input molecule.",
        "required_parameters": [{"name": "smiles", "type": "str", "description": "Input molecule SMILES.", "default": None}],
        "optional_parameters": [{"name": "output_dir", "type": "str", "description": "Optional directory for scaffold_network.json.", "default": ""}],
    },
    {
        "name": "hop_scaffolds_with_rdkit",
        "description": "Perform scaffold hopping by matching a query molecule's Murcko scaffold against a candidate compound library and ranking replacements by scaffold fingerprint similarity.",
        "required_parameters": [
            {"name": "query_smiles", "type": "str", "description": "Query molecule SMILES whose scaffold is to be replaced.", "default": None},
            {"name": "candidate_compounds", "type": "List[dict]", "description": "Candidate compounds, each an object with an ID and SMILES string.", "default": None},
        ],
        "optional_parameters": [
            {"name": "top_k", "type": "int", "description": "Maximum number of ranked scaffold-hop candidates to return.", "default": 10},
            {"name": "min_score", "type": "float", "description": "Minimum Morgan-2 Tanimoto similarity between query and candidate scaffolds.", "default": 0.0},
            {"name": "include_same_scaffold", "type": "bool", "description": "Include candidates whose scaffold is identical to the query scaffold.", "default": False},
            {"name": "output_dir", "type": "str", "description": "Optional directory for scaffold_hopping.json.", "default": ""},
        ],
    },
    {
        "name": "map_reaction_atoms_with_rxnmapper",
        "description": "Assign atom-map numbers to one complete reaction SMILES with the official RXNMapper attention model and return its model confidence and structural validation; confidence is not proof of chemical correctness.",
        "required_parameters": [
            {
                "name": "reaction_smiles",
                "type": "str",
                "description": "Complete reaction SMILES in reactants>reagents>products form; reactant and product sections must be non-empty, while the reagent section may be empty.",
                "default": None,
            }
        ],
        "optional_parameters": [
            {
                "name": "python_executable",
                "type": "str",
                "description": "RXNMapper environment Python; defaults to RXNMAPPER_PYTHON or tools_pkg/RXNMapper/.conda/bin/python.",
                "default": "",
            },
            {
                "name": "timeout_seconds",
                "type": "int",
                "description": "Maximum isolated RXNMapper inference runtime in seconds.",
                "default": 120,
            },
        ],
    },
    {
        "name": "predict_reaction_centers_with_retroxpert",
        "description": "Use the corrected canonical-product RetroXpert stage-1 model to rank existing product bonds as retrosynthetic disconnection centers; this does not generate reactants or a multistep route.",
        "required_parameters": [
            {
                "name": "product_smiles",
                "type": "str",
                "description": "Valid product SMILES whose existing bonds will be ranked as possible retrosynthetic disconnections; atom-map numbers are preserved in the result when supplied.",
                "default": None,
            }
        ],
        "optional_parameters": [
            {
                "name": "reaction_class",
                "type": "int",
                "description": "USPTO-50K reaction class from 1 through 10 for typed inference, or -1 to use the untyped model when the class is unknown.",
                "default": -1,
            },
            {
                "name": "top_k",
                "type": "int",
                "description": "Maximum ranked product bonds returned, from 1 through 100.",
                "default": 10,
            },
            {
                "name": "checkpoint_path",
                "type": "str",
                "description": "Reviewed typed or untyped RetroXpert checkpoint; defaults to RETROXPERT_CHECKPOINT or the matching official checkpoint in the pinned checkout.",
                "default": "",
            },
            {
                "name": "repo_path",
                "type": "str",
                "description": "Pinned RetroXpert canonical_product checkout; defaults to RETROXPERT_REPO or tools_pkg/RetroXpert/upstream.",
                "default": "",
            },
            {
                "name": "python_executable",
                "type": "str",
                "description": "RetroXpert environment Python; defaults to RETROXPERT_PYTHON or tools_pkg/RetroXpert/.conda/bin/python.",
                "default": "",
            },
            {
                "name": "timeout_seconds",
                "type": "int",
                "description": "Maximum isolated RetroXpert inference runtime in seconds.",
                "default": 300,
            },
        ],
    },
    {
        "name": "predict_reaction_products_with_molecular_transformer",
        "description": "Predict ranked product SMILES from reactants and optional reagents using a reviewed checkpoint for the original Molecular Transformer; decoded scores are not exposed as calibrated probabilities.",
        "required_parameters": [
            {
                "name": "reactants_smiles",
                "type": "str",
                "description": "One or more valid reactant SMILES separated by dots, without > reaction separators.",
                "default": None,
            },
            {
                "name": "output_dir",
                "type": "str",
                "description": "Dedicated directory for tokenized model input and native prediction output.",
                "default": None,
            },
        ],
        "optional_parameters": [
            {
                "name": "reagents_smiles",
                "type": "str",
                "description": "Optional dot-separated reagent and catalyst SMILES, supplied separately from reactants.",
                "default": "",
            },
            {
                "name": "top_k",
                "type": "int",
                "description": "Maximum unique RDKit-valid predictions returned, from 1 through 50.",
                "default": 5,
            },
            {
                "name": "beam_size",
                "type": "int",
                "description": "Legacy OpenNMT beam width from 1 through 100 and no smaller than top_k.",
                "default": 10,
            },
            {
                "name": "max_length",
                "type": "int",
                "description": "Maximum decoded token length from 1 through 1000.",
                "default": 200,
            },
            {
                "name": "checkpoint_path",
                "type": "str",
                "description": "Reviewed official averaged checkpoint matching the chosen training/input mode; defaults to MOLECULARTRANSFORMER_CHECKPOINT or the imported MIT mixed averaged-20 model.",
                "default": "",
            },
            {
                "name": "repo_path",
                "type": "str",
                "description": "Pinned original Molecular Transformer checkout; defaults to MOLECULARTRANSFORMER_REPO or tools_pkg/MolecularTransformer/upstream.",
                "default": "",
            },
            {
                "name": "python_executable",
                "type": "str",
                "description": "Reviewed legacy-compatible environment Python; defaults to MOLECULARTRANSFORMER_PYTHON or tools_pkg/MolecularTransformer/.conda/bin/python.",
                "default": "",
            },
            {
                "name": "gpu_device",
                "type": "int",
                "description": "CUDA device index seen by the isolated runtime, or -1 for CPU.",
                "default": -1,
            },
            {
                "name": "timeout_seconds",
                "type": "int",
                "description": "Maximum isolated inference runtime in seconds.",
                "default": 600,
            },
        ],
    },
    {
        "name": "plan_retrosynthesis_with_askcos",
        "description": "Request ranked multistep buyable-path proposals from a separately deployed ASKCOS v2 tree-search service; routes are computational suggestions requiring expert review.",
        "required_parameters": [
            {
                "name": "target_smiles",
                "type": "str",
                "description": "Valid target-molecule SMILES for multistep retrosynthetic planning.",
                "default": None,
            }
        ],
        "optional_parameters": [
            {
                "name": "max_depth",
                "type": "int",
                "description": "Maximum synthesis-tree depth from 1 through 20.",
                "default": 5,
            },
            {
                "name": "max_routes",
                "type": "int",
                "description": "Maximum ranked routes requested and returned, from 1 through 100.",
                "default": 5,
            },
            {
                "name": "expansion_time_seconds",
                "type": "int",
                "description": "ASKCOS tree-expansion budget in seconds, from 1 through 3600.",
                "default": 120,
            },
            {
                "name": "backend",
                "type": "str",
                "description": "ASKCOS v2 tree-search backend: mcts or retro_star.",
                "default": "mcts",
            },
            {
                "name": "api_url",
                "type": "str",
                "description": "ASKCOS v2 API-gateway base URL; defaults to ASKCOS_API_URL. Credentials must not be embedded in this URL.",
                "default": "",
            },
            {
                "name": "timeout_seconds",
                "type": "int",
                "description": "HTTP timeout in seconds; must be at least expansion_time_seconds.",
                "default": 300,
            },
        ],
    },
    {
        "name": "score_candidates_qed",
        "description": "Calculate official RDKit QED drug-likeness scores (0-1).",
        "required_parameters": [{"name": "smiles", "type": "List[str]", "description": "Candidate SMILES.", "default": None}],
        "optional_parameters": [],
    },
    {
        "name": "score_candidates_sa",
        "description": "Estimate synthetic accessibility with an explicitly heuristic RDKit complexity score (1 easy, 10 hard).",
        "required_parameters": [{"name": "smiles", "type": "List[str]", "description": "Candidate SMILES.", "default": None}],
        "optional_parameters": [],
    },
    {
        "name": "analyze_matched_molecular_pairs",
        "description": "Perform pairwise MMP-style analysis using RDKit maximum common substructures.",
        "required_parameters": [{"name": "smiles", "type": "List[str]", "description": "Candidate SMILES.", "default": None}],
        "optional_parameters": [{"name": "max_pairs", "type": "int", "description": "Maximum pairs to return.", "default": 100}],
    },
    {
        "name": "benchmark_moleculenet_qsar",
        "description": "Run a reproducible MoleculeNet-style QSAR descriptor baseline with scikit-learn.",
        "required_parameters": [{"name": "smiles", "type": "List[str]", "description": "Molecule SMILES.", "default": None}, {"name": "labels", "type": "List[float]", "description": "Property or class labels.", "default": None}],
        "optional_parameters": [{"name": "task_type", "type": "str", "description": "regression or classification.", "default": "regression"}, {"name": "test_fraction", "type": "float", "description": "Held-out fraction.", "default": 0.2}, {"name": "random_seed", "type": "int", "description": "Split/model seed.", "default": 42}],
    },
    {
        "name": "optimize_molecules_multiobjective",
        "description": "Rank candidates by deterministic weighted QED and heuristic synthetic accessibility objectives.",
        "required_parameters": [{"name": "smiles", "type": "List[str]", "description": "Candidate SMILES.", "default": None}],
        "optional_parameters": [{"name": "qed_weight", "type": "float", "description": "QED objective weight.", "default": 0.5}, {"name": "sa_weight", "type": "float", "description": "SA objective weight.", "default": 0.5}, {"name": "top_k", "type": "int", "description": "Number of candidates.", "default": 10}],
    },
    {
        "name": "optimize_molecules_with_optuna",
        "description": "Optional Optuna optimizer entry point; reports a structured dependency error when Optuna is unavailable.",
        "required_parameters": [{"name": "smiles", "type": "List[str]", "description": "Candidate SMILES.", "default": None}],
        "optional_parameters": [{"name": "n_trials", "type": "int", "description": "Optuna trial count.", "default": 20}],
    },
    {
        "name": "select_next_molecule_botorch",
        "description": "Select a next candidate using a BoTorch Gaussian-process acquisition over RDKit descriptors.",
        "required_parameters": [{"name": "smiles", "type": "List[str]", "description": "Candidate SMILES.", "default": None}, {"name": "objective_values", "type": "List[float]", "description": "Observed objective values aligned to SMILES.", "default": None}],
        "optional_parameters": [{"name": "maximize", "type": "bool", "description": "Whether larger objective is better.", "default": True}],
    },
]
