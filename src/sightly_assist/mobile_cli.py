"""Command-line entry point for the Expo Go mobile bridge."""

from __future__ import annotations

from typing import Annotated

import typer

from sightly_assist.mobile_api import discover_lan_ipv4_addresses, run_mobile_server

app = typer.Typer(no_args_is_help=True, help="Sightly Assist mobile preview tools.")


@app.command()
def serve(
    host: Annotated[
        str,
        typer.Option("--host", help="Interface to bind; use 0.0.0.0 for phone access."),
    ] = "0.0.0.0",
    port: Annotated[
        int,
        typer.Option("--port", help="HTTP and WebSocket port."),
    ] = 8765,
    demo: Annotated[
        bool,
        typer.Option(
            "--demo/--no-demo",
            help="Publish a deterministic hazard sequence without camera hardware.",
        ),
    ] = True,
) -> None:
    """Serve live telemetry to the Expo Go application."""

    typer.echo("Sightly Assist mobile bridge")
    typer.echo(f"Binding: http://{host}:{port}")
    addresses = discover_lan_ipv4_addresses()
    if addresses:
        typer.echo("Enter one of these URLs in the phone app:")
        for address in addresses:
            typer.echo(f"  http://{address}:{port}")
    else:
        typer.echo("No LAN IPv4 address was detected; check ipconfig/ifconfig manually.")
    typer.echo(f"Demo publisher: {'enabled' if demo else 'disabled'}")
    run_mobile_server(host, port, demo=demo)


if __name__ == "__main__":
    app()
