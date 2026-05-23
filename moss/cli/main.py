"""MOSS CLI: `moss evo` subcommands."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="moss", help="MOSS: Self-Evolution through Source-Level Rewriting")
evo_app = typer.Typer(name="evo", help="Evolution control commands")
app.add_typer(evo_app)

console = Console()

# --- Helpers ---


def _gateway_url() -> str:
    from moss.config import load_config

    cfg = load_config()
    return f"http://{cfg.gateway.host}:{cfg.gateway.port}"


def _http_get(path: str) -> dict:
    import httpx

    resp = httpx.get(f"{_gateway_url()}{path}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def _http_post(path: str, json: dict | None = None) -> dict:
    import httpx

    resp = httpx.post(f"{_gateway_url()}{path}", json=json or {}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _daemon_rpc(method: str, params: dict | None = None) -> dict:
    """Send an RPC call to the host daemon via Unix socket."""
    import json
    import socket

    from moss.config import load_config

    cfg = load_config()
    sock_path = cfg.daemon.socket_path

    request = json.dumps({"method": method, "params": params or {}})

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(sock_path)
        sock.sendall(request.encode() + b"\n")
        data = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
        return json.loads(data.decode())
    finally:
        sock.close()


# --- Gateway-proxied commands ---


@evo_app.command()
def status() -> None:
    """Show current evolution state."""
    data = _http_get("/evo/status")
    if not data or data.get("status") == "idle":
        console.print("[dim]No evolution running.[/dim]")
        return

    table = Table(title="Evolution Status")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    for key, val in data.items():
        table.add_row(key, str(val))
    console.print(table)


@evo_app.command()
def batches() -> None:
    """List all batches."""
    data = _http_get("/evo/batches")
    batch_list = data.get("batches", [])
    if not batch_list:
        console.print("[dim]No batches found.[/dim]")
        return

    table = Table(title="Batches")
    table.add_column("ID", style="cyan")
    table.add_column("Status")
    table.add_column("Chunks", justify="right")
    table.add_column("Created")
    for b in batch_list:
        table.add_row(b["id"], b["status"], str(b.get("chunk_count", 0)), b.get("created_at", ""))
    console.print(table)


@evo_app.command()
def batch(batch_id: str = typer.Argument(..., help="Batch ID to inspect")) -> None:
    """Show details of a specific batch."""
    data = _http_get(f"/evo/batch/{batch_id}")
    console.print_json(data=data)


@evo_app.command()
def start(
    batch_id: str = typer.Argument(..., help="Batch ID to evolve against"),
    depth: str = typer.Option("standard", help="Depth dial: light, standard, deep"),
) -> None:
    """Start an evolution run."""
    data = _http_post("/evo/start", json={"batch_id": batch_id, "depth": depth})
    console.print(f"[green]Evolution started:[/green] {data.get('evo_id', 'unknown')}")


@evo_app.command()
def stop() -> None:
    """Stop the current evolution run."""
    data = _http_post("/evo/stop")
    console.print(f"[yellow]Evolution stopped:[/yellow] {data.get('message', 'done')}")


@evo_app.command()
def restart(
    evo_id: str = typer.Argument(..., help="Evolution ID to restart"),
) -> None:
    """Restart a failed evolution run."""
    data = _http_post("/evo/restart", json={"evo_id": evo_id})
    console.print(f"[green]Evolution restarted:[/green] {data.get('evo_id', 'unknown')}")


@evo_app.command(name="apply")
def apply_cmd(
    image_tag: str = typer.Argument(..., help="Docker image tag to swap to"),
) -> None:
    """Write a swap-request to apply a candidate image."""
    data = _http_post("/evo/apply", json={"image_tag": image_tag})
    console.print(f"[green]Swap request written:[/green] {data.get('request_path', '')}")


# --- Daemon-direct commands ---


@evo_app.command()
def flag(
    session_id: str = typer.Argument(..., help="Session ID to scan"),
) -> None:
    """Scan a single session from its cursor to EOF."""
    result = _daemon_rpc("scan.flag", {"session_id": session_id})
    if result.get("error"):
        console.print(f"[red]Error:[/red] {result['error']}")
    else:
        console.print(f"[green]Flagged:[/green] {result.get('chunks_added', 0)} chunks added")


@evo_app.command()
def catch_up() -> None:
    """Scan all sessions from their cursors."""
    result = _daemon_rpc("scan.catch_up")
    if result.get("error"):
        console.print(f"[red]Error:[/red] {result['error']}")
    else:
        console.print(
            f"[green]Catch-up complete:[/green] "
            f"{result.get('sessions_scanned', 0)} sessions, "
            f"{result.get('chunks_added', 0)} chunks"
        )


if __name__ == "__main__":
    app()
