# OmniAgent

OmniAgent is a research-oriented agent runtime for scientific discovery
workflows. It provides a closed-loop execution layer that coordinates planning,
tool selection, model calls, structured results, and workflow state.

## Scope

- Closed-loop scientific planning and execution
- Structured tool and capability interfaces
- Result validation, persistence, and traceability
- Model-provider and MCP integration points
- HTTP API support for agent runs

The implementation is organized under `src/local_deep_research/`, with the HTTP
wrapper in `api/`.

