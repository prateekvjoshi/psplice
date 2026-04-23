"""
Interactive chat REPL.

Provides an interactive terminal chat session backed by the psplice daemon.
Tokens are streamed as they arrive so the terminal feels responsive.

Slash commands
--------------
/exit  — quit the session
/reset — clear conversation history
/status — print active interventions and model info
"""

from __future__ import annotations

import sys
from typing import Optional

from rich.console import Console
from rich.prompt import Prompt
from rich.rule import Rule

from psplice.client.daemon_client import DaemonClient, DaemonUnavailableError
from psplice.ui.display import error, print_chat_header, print_status

console = Console()


def run_chat(
    client: DaemonClient,
    system_prompt: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
) -> None:
    """
    Start an interactive chat REPL.

    The conversation history is accumulated across turns so the model has
    multi-turn context.  Use /reset to clear history.
    """
    try:
        status = client.status()
    except Exception as exc:
        error(f"Could not fetch daemon status: {exc}")
        sys.exit(1)

    print_chat_header(status)
    console.print()

    history: list[dict[str, str]] = []

    while True:
        # Get user input
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Exiting chat.[/dim]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            handled = _handle_slash(user_input, client, history)
            if handled == "exit":
                break
            continue

        # Stream response from daemon
        console.print("[bold green]Assistant[/bold green]", end="  ")
        full_response = ""
        try:
            for chunk in client.generate_streaming(
                prompt=user_input,
                system_prompt=system_prompt,
                conversation_history=history,
                max_new_tokens=max_new_tokens,
            ):
                console.print(chunk, end="", markup=False)
                full_response += chunk
        except DaemonUnavailableError as exc:
            console.print()
            error(str(exc))
            break
        except RuntimeError as exc:
            console.print()
            error(str(exc))
            continue
        except KeyboardInterrupt:
            console.print("\n[dim](interrupted)[/dim]")
            continue

        console.print()  # newline after streamed output

        # Accumulate conversation history for multi-turn context
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": full_response})

        console.print()


def _handle_slash(command: str, client: DaemonClient, history: list) -> Optional[str]:
    """
    Handle a slash command.

    Returns "exit" if the session should end, None otherwise.
    """
    cmd = command.lower().strip()

    if cmd in ("/exit", "/quit", "/q"):
        console.print("[dim]Goodbye.[/dim]")
        return "exit"

    if cmd == "/reset":
        history.clear()
        console.print("[dim]Conversation history cleared.[/dim]")
        return None

    if cmd == "/status":
        try:
            status = client.status()
            print_status(status)
        except Exception as exc:
            error(str(exc))
        return None

    console.print(f"[dim]Unknown command: {command!r}[/dim]")
    console.print("[dim]Available: /exit  /reset  /status[/dim]")
    return None
