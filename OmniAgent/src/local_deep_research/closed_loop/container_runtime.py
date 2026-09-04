from __future__ import annotations

import socket
import subprocess
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


class ContainerRuntimeError(RuntimeError):
    """Raised when the local container runtime fails to build or run A1."""


@dataclass(frozen=True, slots=True)
class CommandSpec:
    argv: tuple[str, ...]
    cwd: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"argv": list(self.argv), "cwd": self.cwd}

    def to_powershell(self) -> str:
        quoted = [_quote_powershell_arg(item) for item in self.argv]
        return "& " + " ".join(quoted)


@dataclass(frozen=True, slots=True)
class A1ImageBuildSpec:
    context_dir: Path
    dockerfile: Path
    image_tag: str = "omniagent-biomni-a1-runtime:phase1"
    runtime: str = "docker"
    pull: bool = False
    build_args: dict[str, str] = field(default_factory=dict)

    def command(self) -> CommandSpec:
        argv = [self.runtime, "build", "--tag", self.image_tag]
        if self.pull:
            argv.append("--pull")
        for key, value in sorted(self.build_args.items()):
            argv.extend(["--build-arg", f"{key}={value}"])
        argv.extend(["--file", str(self.dockerfile), str(self.context_dir)])
        return CommandSpec(tuple(argv))


@dataclass(frozen=True, slots=True)
class A1ContainerSpec:
    workspace: Path
    image_tag: str = "omniagent-biomni-a1-runtime:phase1"
    container_name: str = "omniagent-biomni-a1"
    host_port: int = 18100
    container_port: int = 18000
    runtime: str = "docker"
    command: tuple[str, ...] = ()
    public_data: Path | None = None
    cpus: str = "4"
    memory: str = "16g"
    readonly_rootfs: bool = True
    extra_env: dict[str, str] = field(default_factory=dict)

    @property
    def endpoint_url(self) -> str:
        return f"http://127.0.0.1:{self.host_port}/sse"

    def run_command(self, *, detach: bool = True) -> CommandSpec:
        workspace = self.workspace.resolve(strict=True)
        argv = [
            self.runtime,
            "run",
            "--rm",
            "--name",
            self.container_name,
            "--publish",
            f"127.0.0.1:{self.host_port}:{self.container_port}",
            "--mount",
            f"type=bind,source={workspace},target=/workspace/task",
            "--workdir",
            "/workspace/task",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PYTHONUNBUFFERED=1",
            "--env",
            "BIOMNI_ALLOWED_WORKSPACE=/workspace/task",
            "--env",
            "BIOMNI_OUTPUT_DIR=/workspace/task",
            "--cpus",
            self.cpus,
            "--memory",
            self.memory,
            "--pids-limit",
            "512",
            "--security-opt",
            "no-new-privileges",
            "--cap-drop",
            "ALL",
            "--tmpfs",
            "/tmp:rw,nosuid,size=2g",
            "--tmpfs",
            "/var/tmp:rw,nosuid,size=1g",
        ]
        if detach:
            argv.append("--detach")
        if self.readonly_rootfs:
            argv.append("--read-only")
        if self.public_data is not None:
            public_data = self.public_data.resolve(strict=True)
            argv.extend(
                [
                    "--mount",
                    f"type=bind,source={public_data},target=/biomni-data,readonly",
                    "--env",
                    "BIOMNI_DATA_ROOT=/biomni-data",
                ]
            )
        for key, value in sorted(self.extra_env.items()):
            argv.extend(["--env", f"{key}={value}"])
        argv.append(self.image_tag)
        argv.extend(self.command)
        return CommandSpec(tuple(argv))

    def stop_command(self) -> CommandSpec:
        return CommandSpec((self.runtime, "stop", self.container_name))


class A1ContainerRuntime:
    """Thin wrapper around Docker/Podman command execution."""

    def run_checked(self, command: CommandSpec) -> None:
        completed = subprocess.run(
            command.argv,
            cwd=command.cwd,
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise ContainerRuntimeError(
                "Command failed with exit code "
                f"{completed.returncode}: {' '.join(command.argv)}\n"
                f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )

    def start_detached(self, spec: A1ContainerSpec) -> None:
        self.run_checked(spec.run_command(detach=True))

    def stop(self, spec: A1ContainerSpec) -> None:
        completed = subprocess.run(
            spec.stop_command().argv,
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0 and "No such container" not in completed.stderr:
            raise ContainerRuntimeError(
                f"Failed to stop A1 container {spec.container_name}: {completed.stderr}"
            )

    @staticmethod
    def wait_for_port(host: str, port: int, *, timeout_seconds: float = 60.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        last_error: OSError | None = None
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((host, port), timeout=1.0):
                    return
            except OSError as exc:
                last_error = exc
                time.sleep(0.5)
        raise ContainerRuntimeError(
            f"Timed out waiting for {host}:{port}: {last_error}"
        )


def _quote_powershell_arg(value: str) -> str:
    if value == "":
        return "''"
    if not any(char.isspace() or char in "'\"&|<>^" for char in value):
        return value
    return "'" + value.replace("'", "''") + "'"

