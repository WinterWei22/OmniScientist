"""Internal queue-only entry point for ProteinMD GPU execution."""

from typing import Any

from omniInfra.tool._proteinmd_runtime import execute_request


def _execute_proteinmd_inference(validation_token: str, gpu_device: int) -> dict[str, Any]:
    """Execute a validated ProteinMD request under the worker's exact GPU lease."""
    return execute_request(validation_token, gpu_device)
