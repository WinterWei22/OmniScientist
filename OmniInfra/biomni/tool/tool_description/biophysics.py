description = [
    {
        "description": "Predicts intrinsically disordered regions (IDRs) in a protein sequence using IUPred2A.",
        "name": "predict_protein_disorder_regions",
        "optional_parameters": [
            {
                "default": 0.5,
                "description": "The disorder score threshold above which a residue is considered disordered",
                "name": "threshold",
                "type": "float",
            },
            {
                "default": "disorder_prediction_results.csv",
                "description": "Filename to save the per-residue disorder scores",
                "name": "output_file",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "The amino acid sequence of the protein to analyze",
                "name": "protein_sequence",
                "type": "str",
            }
        ],
    },
    {
        "description": "Quantifies cell morphology and cytoskeletal organization from fluorescence microscopy images.",
        "name": "analyze_cell_morphology_and_cytoskeleton",
        "optional_parameters": [
            {
                "default": "./results",
                "description": "Directory to save output files",
                "name": "output_dir",
                "type": "str",
            },
            {
                "default": "otsu",
                "description": "Method for cell segmentation ('otsu', 'adaptive', or 'manual')",
                "name": "threshold_method",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the fluorescence microscopy image file",
                "name": "image_path",
                "type": "str",
            }
        ],
    },
    {
        "description": "Quantify tissue deformation and flow dynamics from microscopy image sequence.",
        "name": "analyze_tissue_deformation_flow",
        "optional_parameters": [
            {
                "default": "results",
                "description": "Directory to save results",
                "name": "output_dir",
                "type": "str",
            },
            {
                "default": 1.0,
                "description": "Physical scale of pixels (e.g., μm/pixel) for proper scaling of metrics",
                "name": "pixel_scale",
                "type": "float",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Sequence of microscopy images "
                "(either a list of file paths or a "
                "3D numpy array [time, height, "
                "width])",
                "name": "image_sequence",
                "type": "list or numpy.ndarray",
            }
        ],
    },
    {
        "name": "prepare_proteinmd_inference",
        "description": "Validate an Atlas protein or MISATO protein-ligand ProteinMD request without starting GPU inference, and return a short-lived submission token.",
        "required_parameters": [
            {
                "name": "dataset",
                "type": "str",
                "description": "ProteinMD dataset mode: exactly 'atlas' or 'misato'.",
                "default": None,
            },
            {
                "name": "system_id",
                "type": "str",
                "description": "Managed ProteinMD system identifier containing only letters, digits, underscores, and hyphens.",
                "default": None,
            },
        ],
        "optional_parameters": [
            {
                "name": "execution_profile",
                "type": "str",
                "description": "Administrator-defined profile: 'smoke', 'standard', or 'production'.",
                "default": "smoke",
            },
            {
                "name": "replica",
                "type": "int",
                "description": "Atlas trajectory replica number: 1, 2, or 3.",
                "default": 1,
            },
            {
                "name": "pdb_only",
                "type": "bool",
                "description": "For Atlas, ignore XTC and initialize from PDB only.",
                "default": False,
            },
            {"name": "seed", "type": "int", "description": "Non-negative deterministic random seed.", "default": 0},
        ],
    },
    {
        "name": "submit_proteinmd_inference",
        "description": "Submit a short-lived validated ProteinMD specification as an independent GPU task; never accepts commands, paths, GPU IDs, or arbitrary environment variables.",
        "required_parameters": [
            {
                "name": "validation_token",
                "type": "str",
                "description": "Short-lived token returned by prepare_proteinmd_inference.",
                "default": None,
            },
        ],
        "optional_parameters": [
            {
                "name": "gpu_policy",
                "type": "str",
                "description": "GPU scheduling policy; ordinary callers must use 'auto'.",
                "default": "auto",
            },
        ],
    },
    {
        "name": "get_proteinmd_inference",
        "description": "Query an independently queued ProteinMD task and return status plus validated result references without loading trajectory binaries into context.",
        "required_parameters": [
            {
                "name": "task_id",
                "type": "str",
                "description": "ProteinMD task UUID returned by submit_proteinmd_inference.",
                "default": None,
            },
        ],
        "optional_parameters": [],
    },
]
