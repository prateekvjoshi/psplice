"""
Rich-based display utilities.

All terminal output lives here so that CLI command handlers stay focused on
logic rather than formatting.
"""

from __future__ import annotations

from typing import Any, Optional

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

console = Console()
err_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Model / daemon summary
# ---------------------------------------------------------------------------

def print_model_summary(status: dict[str, Any]) -> None:
    """Print a formatted model summary table after `psplice load`."""
    t = Table(
        show_header=False,
        box=box.SIMPLE,
        padding=(0, 1),
        show_edge=False,
    )
    t.add_column("key", style="bold cyan", min_width=22)
    t.add_column("value", style="white")

    param_b = status.get("param_count", 0) / 1e9
    param_str = f"{param_b:.1f}B" if param_b >= 1 else f"{status.get('param_count', 0) / 1e6:.0f}M"

    rows = [
        ("Model", status.get("model_id", "?")),
        ("Architecture", f"{status.get('arch_family', '?')}  ({status.get('arch_class', '?')})"),
        ("Parameters", param_str),
        ("Layers", str(status.get("num_layers", "?"))),
        ("Attention heads", str(status.get("num_attention_heads", "?"))),
        ("Hidden size", str(status.get("hidden_size", "?"))),
        ("Max context", str(status.get("max_position_embeddings", "?"))),
        ("Device", status.get("device", "?")),
        ("dtype", status.get("dtype", "?")),
        ("Attention impl", status.get("attention_impl", "?")),
    ]

    for k, v in rows:
        t.add_row(k, v)

    console.print(
        Panel(t, title="[bold green]psplice — model loaded[/bold green]", border_style="green")
    )

    if status.get("eager_attn"):
        console.print(
            "  [dim]Head masking enabled (eager attention)[/dim]"
        )
    else:
        console.print(
            "  [dim yellow]Tip: load with --eager-attn to enable head masking[/dim yellow]"
        )


def print_status(status: dict[str, Any]) -> None:
    """Print the full daemon status panel."""
    # Model info
    model_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1), show_edge=False)
    model_table.add_column("k", style="bold cyan", min_width=22)
    model_table.add_column("v")

    param_b = status.get("param_count", 0) / 1e9
    param_str = f"{param_b:.1f}B" if param_b >= 1 else f"{status.get('param_count', 0) / 1e6:.0f}M"

    model_table.add_row("Model", status.get("model_id", "?"))
    model_table.add_row("Architecture", f"{status.get('arch_family', '?')}  ({status.get('arch_class', '?')})")
    model_table.add_row("Parameters", param_str)
    model_table.add_row("Device", status.get("device", "?"))
    model_table.add_row("dtype", status.get("dtype", "?"))
    model_table.add_row("Attention impl", status.get("attention_impl", "?"))
    model_table.add_row("Active preset", status.get("active_preset") or "[dim]none[/dim]")
    model_table.add_row("Active LoRA", status.get("active_lora") or "[dim]none[/dim]")
    console.print(Panel(model_table, title="[bold]Model[/bold]", border_style="blue"))

    # Decode settings
    ds = status.get("decode_settings", {})
    ds_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1), show_edge=False)
    ds_table.add_column("k", style="bold cyan", min_width=22)
    ds_table.add_column("v")
    ds_table.add_row("max_new_tokens", str(ds.get("max_new_tokens", 512)))
    ds_table.add_row("temperature", str(ds.get("temperature")) if ds.get("temperature") else "[dim]default[/dim]")
    ds_table.add_row("top_p", str(ds.get("top_p")) if ds.get("top_p") else "[dim]default[/dim]")
    ds_table.add_row("top_k", str(ds.get("top_k")) if ds.get("top_k") else "[dim]default[/dim]")
    ds_table.add_row("repetition_penalty", str(ds.get("repetition_penalty")) if ds.get("repetition_penalty") else "[dim]default[/dim]")
    console.print(Panel(ds_table, title="[bold]Decode settings[/bold]", border_style="blue"))

    # Interventions
    ivs = status.get("interventions", [])
    if ivs:
        iv_table = Table(
            "Name", "Type", "Details",
            box=box.SIMPLE,
            padding=(0, 1),
            show_edge=False,
        )
        iv_table.columns[0].style = "bold yellow"
        iv_table.columns[1].style = "cyan"
        for iv in ivs:
            details = _format_iv_details(iv)
            iv_table.add_row(iv.get("name", "?"), iv.get("type", "?"), details)
        console.print(Panel(iv_table, title="[bold yellow]Active interventions[/bold yellow]", border_style="yellow"))
    else:
        console.print(
            Panel(
                "[dim]No active interventions[/dim]",
                title="[bold]Interventions[/bold]",
                border_style="dim",
            )
        )


def _format_iv_details(iv: dict) -> str:
    t = iv.get("type", "")
    if t == "steering":
        return f"layers={iv.get('layers', [])}  scale={iv.get('scale', 1.0)}"
    if t == "head_mask":
        lh = iv.get("layer_heads", {})
        pairs = ", ".join(f"L{k}:{v}" for k, v in lh.items())
        return f"masked: {pairs}"
    if t == "layer_skip":
        return f"skip from layer {iv.get('skip_from', '?')}"
    if t == "lora":
        return iv.get("adapter_path", "?")
    return str(iv)


# ---------------------------------------------------------------------------
# Chat header
# ---------------------------------------------------------------------------

def print_chat_header(status: dict[str, Any]) -> None:
    """Print a compact header at the start of a chat session."""
    model_id = status.get("model_id", "?")
    ivs = status.get("interventions", [])
    iv_str = ", ".join(
        f"{iv.get('name')}({iv.get('type')})" for iv in ivs
    ) if ivs else "none"
    lora = status.get("active_lora")
    lora_str = f"  LoRA: {lora}" if lora else ""

    console.print(
        Panel(
            f"[bold cyan]{model_id}[/bold cyan]  "
            f"[dim]interventions:[/dim] [yellow]{iv_str}[/yellow]{lora_str}\n"
            f"[dim]Commands: /exit  /reset  /status[/dim]",
            title="[bold]psplice chat[/bold]",
            border_style="cyan",
        )
    )


# ---------------------------------------------------------------------------
# Compare output
# ---------------------------------------------------------------------------

def print_compare(
    base_text: str,
    mod_text: str,
    base_stats: Optional[dict] = None,
    mod_stats: Optional[dict] = None,
) -> None:
    """
    Print base vs modified responses with a compact stats footer.

    The stats line is always shown — token counts, latency, and a word-level
    similarity score give immediate signal on whether the intervention is
    doing anything and in which direction.
    """
    base_tok = base_stats.get("tokens_generated", 0) if base_stats else 0
    mod_tok = mod_stats.get("tokens_generated", 0) if mod_stats else 0
    base_time = base_stats.get("time_seconds", 0.0) if base_stats else 0.0
    mod_time = mod_stats.get("time_seconds", 0.0) if mod_stats else 0.0

    console.print(
        Rule(
            f"[bold]Base[/bold]  [dim]no behaviors active · greedy · {base_tok} tok  {base_time:.1f}s[/dim]",
            style="dim",
        )
    )
    console.print(base_text.strip())

    console.print()
    console.print(
        Rule(
            f"[bold yellow]Modified[/bold yellow]  "
            f"[dim]with active behaviors · greedy · {mod_tok} tok  {mod_time:.1f}s[/dim]",
            style="yellow",
        )
    )
    console.print(mod_text.strip())
    console.print()

    # Summary stats line — always shown so the user gets feedback on effect size
    _print_compare_stats(base_text, mod_text, base_tok, mod_tok)


def _print_compare_stats(
    base_text: str,
    mod_text: str,
    base_tok: int,
    mod_tok: int,
) -> None:
    """Print a one-line diff summary under the compare output."""
    import difflib

    base_words = base_text.split()
    mod_words = mod_text.split()

    similarity = difflib.SequenceMatcher(None, base_words, mod_words).ratio()
    tok_delta = mod_tok - base_tok
    tok_delta_str = f"+{tok_delta}" if tok_delta >= 0 else str(tok_delta)
    tok_pct = int(abs(tok_delta) / base_tok * 100) if base_tok else 0

    sim_color = "green" if similarity > 0.7 else ("yellow" if similarity > 0.3 else "red")
    delta_color = "dim"  # neutral — shorter isn't always better

    console.print(
        f"  [dim]similarity:[/dim] [{sim_color}]{similarity:.0%}[/{sim_color}]  "
        f"[dim]token delta:[/dim] [{delta_color}]{tok_delta_str} ({tok_pct}%)[/{delta_color}]"
    )

    if similarity > 0.97 and base_tok > 0:
        console.print(
            "  [dim yellow]Outputs are nearly identical — the active behavior may not be "
            "strong enough. Try: psplice behavior add <name> --strength strong[/dim yellow]"
        )

    console.print()


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------

def error(msg: str) -> None:
    err_console.print(f"[bold red]Error:[/bold red] {msg}")


def warn(msg: str) -> None:
    console.print(f"[bold yellow]Warning:[/bold yellow] {msg}")


def success(msg: str) -> None:
    console.print(f"[bold green]✓[/bold green] {msg}")


def info(msg: str) -> None:
    console.print(f"[dim]{msg}[/dim]")
