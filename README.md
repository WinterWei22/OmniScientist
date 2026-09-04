# OmniScientist

OmniScientist develops reusable foundations for scientific AI systems. We focus
on reliable agent runtimes, structured tool interfaces, and practical software
for research-oriented workflows.

## Projects

- [`OmniInfra`](./OmniInfra): infrastructure for building biomedical AI agents.
- [`OmniAgent`](./OmniAgent): a closed-loop scientific research agent runtime.

## Environment

- Python 3.11 or newer (OmniAgent supports Python 3.11–3.13)
- A virtual environment is recommended for each project
- Network access and the credentials required by the selected model/provider

Create an environment, for example:

```bash
python3.11 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

## Dependencies

Install each project independently from its own directory:

```bash
cd OmniInfra
python -m pip install -e .
```

```bash
cd OmniAgent
python -m pip install -e .
```

`OmniInfra` provides the core package dependencies (`pydantic`, `langchain`,
and `python-dotenv`). `OmniAgent` additionally declares its research, model,
workflow, MCP, and API dependencies in [`OmniAgent/pyproject.toml`](./OmniAgent/pyproject.toml).

## Running OmniInfra

Verify the package installation:

```bash
cd OmniInfra
python -c "import omniInfra; print(omniInfra.__name__)"
```

The Python package is imported as `omniInfra`.

## Running OmniAgent

Show the closed-loop runner options:

```bash
cd OmniAgent
PYTHONPATH=src python -m local_deep_research.closed_loop --help
```

To run the HTTP API, install an ASGI server and start it from the project root:

```bash
cd OmniAgent
python -m pip install "uvicorn[standard]"
PYTHONPATH=src:. uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Runtime settings are supplied through environment variables. Keep API keys and
tokens outside the repository; do not commit local secret files or generated
run artifacts.
