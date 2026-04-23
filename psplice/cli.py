"""
psplice CLI.

All commands are thin clients that talk to the local daemon.  The only
command that actually starts a process is `psplice load`, which spawns the
daemon and waits for it to become healthy.

Everyday commands
-----------------
  psplice doctor      Check hardware and validate your setup
  psplice load        Load a model and start the daemon
  psplice behavior    Apply named behaviors (concise, direct, formal, …)
  psplice chat        Interactive terminal chat
  psplice compare     Side-by-side base vs modified generation
  psplice preset      Save and restore configurations
  psplice status      Show what's active
  psplice stop        Stop the daemon

Advanced commands (for researchers)
------------------------------------
  psplice steer       Raw activation steering
  psplice heads       Attention head masking
  psplice layers      Layer skip / early exit
  psplice lora        LoRA adapter hot injection
  psplice decode      Generation sampling settings
  psplice vectors     Steering vector extraction utilities
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import box

from psplice.client.daemon_client import DaemonClient, DaemonUnavailableError
from psplice.ui.display import (
    console,
    error,
    info,
    print_compare,
    print_model_summary,
    print_status,
    success,
    warn,
)

app = typer.Typer(
    name="psplice",
    help=(
        "Change how your LLM behaves at runtime. No retraining, no code changes.\n\n"
        "  [bold]Get started:[/bold]\n"
        "    psplice doctor                      — check your setup\n"
        "    psplice load Qwen/Qwen2.5-7B-Instruct\n"
        "    psplice behavior add concise\n"
        "    psplice compare \"Explain TCP\"\n\n"
        "  [dim]Advanced controls (layers, heads, raw vectors) are under their own subcommands.[/dim]"
    ),
    add_completion=True,
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Sub-apps
behavior_app = typer.Typer(help="Apply named behaviors to the loaded model.", no_args_is_help=True)
steer_app = typer.Typer(help="[Advanced] Raw activation steering.", no_args_is_help=True)
heads_app = typer.Typer(help="[Advanced] Attention head masking.", no_args_is_help=True)
layers_app = typer.Typer(help="[Advanced] Layer skip / early exit.", no_args_is_help=True)
lora_app = typer.Typer(help="LoRA adapter hot injection.", no_args_is_help=True)
decode_app = typer.Typer(help="Generation sampling settings.", no_args_is_help=True)
preset_app = typer.Typer(help="Save and restore configurations.", no_args_is_help=True)
vectors_app = typer.Typer(help="[Advanced] Steering vector extraction utilities.", no_args_is_help=True)

app.add_typer(behavior_app, name="behavior")
app.add_typer(steer_app, name="steer")
app.add_typer(heads_app, name="heads")
app.add_typer(layers_app, name="layers")
app.add_typer(lora_app, name="lora")
app.add_typer(decode_app, name="decode")
app.add_typer(preset_app, name="preset")
app.add_typer(vectors_app, name="vectors")


# ===========================================================================
# psplice load
# ===========================================================================

@app.command()
def load(
    model_id: str = typer.Argument(
        ...,
        help="HuggingFace Hub model ID or local directory path.",
        metavar="MODEL_ID",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Device to load onto: auto | cuda | cuda:0 | mps | cpu",
    ),
    dtype: str = typer.Option(
        "auto",
        "--dtype",
        help="Weight dtype: auto | bfloat16 | float16 | float32",
    ),
    eager_attn: bool = typer.Option(
        False,
        "--eager-attn",
        help="Force eager attention (required for head masking).",
    ),
    trust_remote_code: bool = typer.Option(
        False,
        "--trust-remote-code",
        help="Allow custom model code from HuggingFace Hub.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Stop the existing daemon and start a fresh one.",
    ),
) -> None:
    """
    Load a model onto GPU and start the local daemon.

    The daemon keeps the model resident in VRAM so subsequent commands
    (chat, steer, compare, …) run instantly without reloading.

    Examples:

      psplice load Qwen/Qwen2.5-7B-Instruct

      psplice load meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16

      psplice load ./my-model --device cuda --eager-attn
    """
    # Validate --device and --dtype before doing anything expensive
    _validate_load_args(device, dtype, model_id)

    from psplice.daemon.manager import (
        DaemonAlreadyRunningError,
        DaemonStartupError,
        start,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Loading [bold]{model_id}[/bold] — this may take a minute…", total=None
        )

        try:
            start(
                model_id=model_id,
                device=device,
                dtype=dtype,
                eager_attn=eager_attn,
                trust_remote_code=trust_remote_code,
                force=force,
            )
        except DaemonAlreadyRunningError as exc:
            progress.stop()
            error(str(exc))
            raise typer.Exit(1)
        except DaemonStartupError as exc:
            progress.stop()
            error(str(exc))
            raise typer.Exit(1)

    # Fetch full status and display
    client = DaemonClient.require()
    try:
        status = client.status()
        print_model_summary(status)
    except Exception as exc:
        warn(f"Model loaded but could not fetch status: {exc}")


# ===========================================================================
# psplice doctor
# ===========================================================================

@app.command()
def doctor() -> None:
    """
    Check your environment and see what psplice can do on this machine.

    Detects hardware, validates PyTorch, reports available VRAM, and shows
    which features are available.  Run this before loading a model for the
    first time.
    """
    import os
    import platform

    try:
        import torch
        torch_ok = True
        torch_version = torch.__version__
    except ImportError:
        torch_ok = False
        torch_version = "not installed"

    # --- System table ---
    sys_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1), show_edge=False)
    sys_table.add_column("k", style="bold cyan", min_width=20)
    sys_table.add_column("v")

    sys_table.add_row("Python", f"{sys.version.split()[0]}")
    sys_table.add_row("Platform", platform.system() + " " + platform.machine())
    sys_table.add_row(
        "PyTorch",
        f"{torch_version} [green]✓[/green]" if torch_ok else f"[red]{torch_version}[/red]",
    )

    # --- Hardware detection ---
    cuda_ok = torch_ok and torch.cuda.is_available()
    mps_ok = torch_ok and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    cpu_only = not cuda_ok and not mps_ok

    gpu_name = ""
    vram_gb = 0.0
    if cuda_ok:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        sys_table.add_row("CUDA", torch.version.cuda or "available")  # type: ignore[attr-defined]
        sys_table.add_row("GPU", gpu_name)
        sys_table.add_row("VRAM", f"{vram_gb:.0f} GB")
    elif mps_ok:
        sys_table.add_row("Apple Silicon", "[green]MPS available[/green]")
    else:
        sys_table.add_row("GPU", "[yellow]none detected — CPU only[/yellow]")

    # HF token
    hf_token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    sys_table.add_row(
        "HF token",
        "[green]set[/green]" if hf_token else "[dim]not set (needed for gated models)[/dim]",
    )

    console.print(Panel(sys_table, title="[bold]System[/bold]", border_style="blue"))

    # --- Feature compatibility ---
    feat_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1), show_edge=False)
    feat_table.add_column("feat", min_width=32)
    feat_table.add_column("status")

    def _feat(label: str, ok: bool, note: str = "") -> None:
        mark = "[green]✓[/green]" if ok else "[yellow]~[/yellow]"
        val = note if note else ("[green]available[/green]" if ok else "[yellow]limited[/yellow]")
        feat_table.add_row(f"{mark} {label}", val)

    _feat("Chat and generation", True)
    _feat("Behavior add/remove (steering)", torch_ok)
    _feat("Layer skip", torch_ok)
    _feat("LoRA hot injection", torch_ok)
    _feat(
        "Head masking",
        cuda_ok or mps_ok,
        "[dim]load with --eager-attn[/dim]" if (cuda_ok or mps_ok) else "[dim]requires GPU[/dim]",
    )
    _feat(
        "GPU acceleration",
        cuda_ok or mps_ok,
        "[green]CUDA[/green]" if cuda_ok else ("[green]MPS[/green]" if mps_ok else "[yellow]CPU only[/yellow]"),
    )

    console.print(Panel(feat_table, title="[bold]Features[/bold]", border_style="blue"))

    # --- Model recommendations ---
    rec_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1), show_edge=False)
    rec_table.add_column("k", style="bold cyan", min_width=20)
    rec_table.add_column("v")

    if cuda_ok and vram_gb >= 40:
        rec_table.add_row("Good fit", "Qwen/Qwen2.5-7B-Instruct  (fast, recommended)")
        rec_table.add_row("", "meta-llama/Llama-3.1-8B-Instruct")
        rec_table.add_row("Larger", "Qwen/Qwen2.5-32B-Instruct")
    elif cuda_ok and vram_gb >= 16:
        rec_table.add_row("Good fit", "Qwen/Qwen2.5-7B-Instruct  (recommended)")
        rec_table.add_row("", "meta-llama/Llama-3.1-8B-Instruct")
        rec_table.add_row("Stretch", "Qwen/Qwen2.5-14B-Instruct  (may need bfloat16)")
    elif cuda_ok and vram_gb >= 8:
        rec_table.add_row("Good fit", "Qwen/Qwen2.5-7B-Instruct  --dtype bfloat16")
        rec_table.add_row("Smaller", "Qwen/Qwen2.5-3B-Instruct")
    elif mps_ok:
        rec_table.add_row("Good fit", "Qwen/Qwen2.5-3B-Instruct")
        rec_table.add_row("", "Qwen/Qwen2.5-7B-Instruct  --dtype bfloat16  (slower)")
    else:
        rec_table.add_row("CPU only", "Qwen/Qwen2.5-3B-Instruct  (expect slow generation)")

    console.print(Panel(rec_table, title="[bold]Recommended models[/bold]", border_style="blue"))

    # --- Daemon status ---
    from psplice.state.session import get_active_session
    session = get_active_session()
    if session:
        console.print(f"\n  [green]Daemon running[/green] — model: [bold]{session.model_id}[/bold]  pid={session.pid}")
    else:
        console.print("\n  [dim]No daemon running.[/dim]  Start one with:")
        console.print("  [cyan]  psplice load Qwen/Qwen2.5-7B-Instruct[/cyan]\n")


# ===========================================================================
# psplice stop
# ===========================================================================

@app.command()
def stop() -> None:
    """Stop the running daemon and release model memory."""
    from psplice.daemon.manager import stop as _stop

    result = _stop()
    if result:
        success("Daemon stopped. Model memory released.")
    else:
        error("No running daemon found.")
        raise typer.Exit(1)


# ===========================================================================
# psplice status
# ===========================================================================

@app.command()
def status() -> None:
    """Show daemon status, loaded model, and active interventions."""
    client = DaemonClient.require()
    try:
        s = client.status()
        print_status(s)
    except DaemonUnavailableError as exc:
        error(str(exc))
        raise typer.Exit(1)
    except RuntimeError as exc:
        error(str(exc))
        raise typer.Exit(1)


# ===========================================================================
# psplice chat
# ===========================================================================

@app.command()
def chat(
    system: Optional[str] = typer.Option(
        None,
        "--system",
        "-s",
        help="System prompt for the conversation.",
    ),
    max_new_tokens: Optional[int] = typer.Option(
        None,
        "--max-new-tokens",
        help="Override max tokens per response.",
    ),
) -> None:
    """
    Interactive terminal chat with the loaded model.

    Tokens are streamed as they are generated.  Active interventions are
    applied automatically.

    Slash commands inside chat:
      /exit   — quit
      /reset  — clear conversation history
      /status — show active interventions
    """
    client = DaemonClient.require()
    from psplice.ui.chat import run_chat
    run_chat(client, system_prompt=system, max_new_tokens=max_new_tokens)


# ===========================================================================
# psplice compare
# ===========================================================================

@app.command()
def compare(
    prompt: str = typer.Argument(..., help="Prompt to run through base and modified model."),
    system: Optional[str] = typer.Option(None, "--system", help="System prompt."),
    max_new_tokens: Optional[int] = typer.Option(None, "--max-new-tokens"),
) -> None:
    """
    Run the same prompt through the base model and the modified model
    (with active interventions) side by side.

    Always shows token counts, latency, and a word-similarity score so you
    can tell at a glance whether the intervention is doing anything.

    Example:

      psplice compare "Explain TCP congestion control simply"

      psplice compare "Is this approach safe?"
    """
    client = DaemonClient.require()
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=console,
        ) as progress:
            progress.add_task("[cyan]Generating comparison…", total=None)
            result = client.compare(
                prompt=prompt,
                system_prompt=system,
                max_new_tokens=max_new_tokens,
            )
        print_compare(
            base_text=result["base"]["text"],
            mod_text=result["modified"]["text"],
            base_stats=result["base"],
            mod_stats=result["modified"],
        )
    except DaemonUnavailableError as exc:
        error(str(exc))
        raise typer.Exit(1)
    except RuntimeError as exc:
        error(str(exc))
        raise typer.Exit(1)


# ===========================================================================
# psplice behavior
# ===========================================================================

_STRENGTH_CHOICES = ("mild", "moderate", "strong")


@behavior_app.command("list")
def behavior_list() -> None:
    """
    Show all available built-in behaviors.

    Behaviors are named presets that shift the model in a direction — concise,
    formal, skeptical, etc.  Apply one with `psplice behavior add <name>`.
    """
    from psplice.behaviors.catalog import CATALOG

    categories: dict[str, list] = {}
    for b in CATALOG.values():
        categories.setdefault(b.category, []).append(b)

    for cat, behaviors in sorted(categories.items()):
        t = Table(
            show_header=True,
            box=box.SIMPLE,
            padding=(0, 1),
            show_edge=False,
            title=f"[bold]{cat.upper()}[/bold]",
            title_justify="left",
        )
        t.add_column("Name", style="bold cyan", min_width=20)
        t.add_column("What it does")
        for b in behaviors:
            t.add_row(b.name, b.description)
        console.print(t)

    console.print(
        "  Apply one:  [cyan]psplice behavior add concise[/cyan]\n"
        "  Learn more: [cyan]psplice behavior describe concise[/cyan]"
    )


@behavior_app.command("describe")
def behavior_describe(
    name: str = typer.Argument(..., help="Behavior name, e.g. concise"),
) -> None:
    """Show what a behavior does and how to use it."""
    from psplice.behaviors.catalog import get_behavior, scale_for_strength

    b = get_behavior(name)
    if b is None:
        error(f"Unknown behavior: {name!r}.  Run `psplice behavior list` to see available behaviors.")
        raise typer.Exit(1)

    t = Table(show_header=False, box=box.SIMPLE, padding=(0, 1), show_edge=False)
    t.add_column("k", style="bold cyan", min_width=20)
    t.add_column("v")
    t.add_row("Name", b.name)
    t.add_row("Category", b.category)
    t.add_row("Description", b.description)
    t.add_row("Effect", b.what_it_does)
    if b.when_to_use:
        t.add_row("When to use", b.when_to_use)
    t.add_row("", "")
    t.add_row("Strength: mild",     f"scale={scale_for_strength(b, 'mild')}  — subtle, low risk of drift")
    t.add_row("Strength: moderate", f"scale={scale_for_strength(b, 'moderate')}  — balanced (default)")
    t.add_row("Strength: strong",   f"scale={scale_for_strength(b, 'strong')}  — noticeable, may affect coherence")

    console.print(t)
    console.print(f"\n  [cyan]psplice behavior add {name}[/cyan]")
    console.print(f"  [cyan]psplice behavior add {name} --strength strong[/cyan]")


@behavior_app.command("add")
def behavior_add(
    name: str = typer.Argument(..., help="Behavior name, e.g. concise"),
    strength: str = typer.Option(
        "moderate",
        "--strength",
        "-s",
        help="Effect intensity: mild | moderate | strong",
    ),
) -> None:
    """
    Apply a named behavior to the loaded model.

    psplice handles layer selection, vector extraction, and hook registration
    automatically.  Run `psplice behavior list` to see what's available.

    Strength controls how strongly the behavior is applied:
      mild      — subtle shift, very safe
      moderate  — balanced (default)
      strong    — noticeable effect, may occasionally affect coherence

    Examples:

      psplice behavior add concise
      psplice behavior add formal --strength mild
      psplice behavior add skeptical --strength strong

    After applying, run `psplice compare "your prompt"` to see the difference.
    """
    from psplice.behaviors.catalog import get_behavior, scale_for_strength
    from platformdirs import user_data_dir

    if strength not in _STRENGTH_CHOICES:
        error(f"--strength must be one of: {', '.join(_STRENGTH_CHOICES)}")
        raise typer.Exit(1)

    b = get_behavior(name)
    if b is None:
        error(
            f"Unknown behavior: {name!r}.\n"
            f"  Run `psplice behavior list` to see available behaviors."
        )
        raise typer.Exit(1)

    client = DaemonClient.require()

    try:
        model_status = client.status()
    except RuntimeError as exc:
        error(str(exc))
        raise typer.Exit(1)

    num_layers: int = model_status.get("num_layers", 32)
    model_id: str = model_status.get("model_id", "unknown")

    # Auto-select middle third — where behavioral features are most consistently encoded
    start = num_layers // 3
    end = (2 * num_layers) // 3
    layer_indices = list(range(start, end))

    scale = scale_for_strength(b, strength)

    # Cache vectors per model so extraction only happens once
    cache_root = Path(user_data_dir("psplice")) / "vectors"
    safe_model_id = model_id.replace("/", "_").replace(":", "_")
    vector_path = cache_root / safe_model_id / f"{name}.pt"

    cached = vector_path.exists()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        if cached:
            progress.add_task(f"[cyan]Applying '{name}' behavior ({strength})…", total=None)
        else:
            progress.add_task(
                f"[cyan]Learning '{name}' behavior from examples (first use, takes ~10–30s)…",
                total=None,
            )

        if not cached:
            try:
                client.extract_vector(
                    positive_prompts=b.positive_prompts,
                    negative_prompts=b.negative_prompts,
                    layer_indices=layer_indices,
                    output_path=str(vector_path),
                    token_aggregation="mean",
                )
            except (RuntimeError, DaemonUnavailableError) as exc:
                error(str(exc))
                raise typer.Exit(1)

        intervention_name = f"behavior:{name}"
        try:
            client.steer_add(
                name=intervention_name,
                vector_path=str(vector_path),
                layer_indices=layer_indices,
                scale=scale,
            )
        except RuntimeError as exc:
            if "already exists" in str(exc).lower() or "duplicate" in str(exc).lower():
                error(
                    f"Behavior '{name}' is already active.\n"
                    f"  Remove it first: psplice behavior remove {name}"
                )
            else:
                error(str(exc))
            raise typer.Exit(1)

    success(f"Behavior '{name}' applied ({strength})")
    console.print(
        f"  [dim]Test it:[/dim]  [cyan]psplice compare \"Your prompt here\"[/cyan]\n"
        f"  [dim]Undo it:[/dim]  [cyan]psplice behavior remove {name}[/cyan]"
    )


@behavior_app.command("remove")
def behavior_remove(
    name: str = typer.Argument(..., help="Behavior name to remove, e.g. concise"),
) -> None:
    """Remove an active behavior."""
    client = DaemonClient.require()
    intervention_name = f"behavior:{name}"
    try:
        client.steer_remove(intervention_name)
        success(f"Behavior '{name}' removed.")
    except RuntimeError as exc:
        if "not found" in str(exc).lower() or "404" in str(exc):
            error(
                f"Behavior '{name}' is not currently active.\n"
                f"  Run `psplice status` to see what's active."
            )
        else:
            error(str(exc))
        raise typer.Exit(1)


@behavior_app.command("active")
def behavior_active() -> None:
    """Show which behaviors are currently active."""
    client = DaemonClient.require()
    try:
        ivs = client.steer_list()
    except RuntimeError as exc:
        error(str(exc))
        raise typer.Exit(1)

    behavior_ivs = [iv for iv in ivs if iv.get("name", "").startswith("behavior:")]

    if not behavior_ivs:
        info("No behaviors active.  Add one with `psplice behavior add <name>`.")
        return

    t = Table("Behavior", "Strength (scale)", "Layers", box=box.SIMPLE, show_edge=False, padding=(0, 1))
    t.columns[0].style = "bold cyan"
    for iv in behavior_ivs:
        bname = iv.get("name", "?").removeprefix("behavior:")
        t.add_row(bname, str(iv.get("scale", "?")), str(iv.get("layers", [])))
    console.print(t)


# ===========================================================================
# psplice steer
# ===========================================================================

@steer_app.command("add")
def steer_add(
    name: str = typer.Argument(..., help="Unique name for this steering vector."),
    vector: Path = typer.Option(
        ...,
        "--vector",
        "-v",
        help="Path to a .pt file containing the steering vector.",
        exists=True,
        readable=True,
    ),
    layers: str = typer.Option(
        ...,
        "--layers",
        "-l",
        help="Comma-separated layer indices to steer, e.g. 10,11,12",
    ),
    scale: float = typer.Option(1.0, "--scale", "-s", help="Steering vector scale factor."),
) -> None:
    """
    Add an activation steering vector.

    The .pt file must contain either:
      • A 1D tensor of shape [hidden_size] (broadcast to all specified layers)
      • A dict mapping int layer indices to 1D [hidden_size] tensors

    Example:

      psplice steer add honesty --vector ./vectors/honesty.pt --layers 12,13,14 --scale 0.6
    """
    try:
        layer_indices = [int(x.strip()) for x in layers.split(",")]
    except ValueError:
        error("--layers must be comma-separated integers, e.g. 10,11,12")
        raise typer.Exit(1)

    client = DaemonClient.require()
    try:
        client.steer_add(
            name=name,
            vector_path=str(vector.resolve()),
            layer_indices=layer_indices,
            scale=scale,
        )
        success(f"Steering vector '{name}' applied to layers {layer_indices} (scale={scale})")
    except RuntimeError as exc:
        error(str(exc))
        raise typer.Exit(1)


@steer_app.command("remove")
def steer_remove(
    name: str = typer.Argument(..., help="Name of the steering vector to remove."),
) -> None:
    """Remove an active steering vector by name."""
    client = DaemonClient.require()
    try:
        client.steer_remove(name)
        success(f"Steering vector '{name}' removed.")
    except RuntimeError as exc:
        error(str(exc))
        raise typer.Exit(1)


@steer_app.command("list")
def steer_list() -> None:
    """List all active steering vectors."""
    client = DaemonClient.require()
    try:
        ivs = client.steer_list()
    except RuntimeError as exc:
        error(str(exc))
        raise typer.Exit(1)

    if not ivs:
        info("No active steering vectors.")
        return

    t = Table("Name", "Layers", "Scale", "Vector path", box=box.SIMPLE)
    for iv in ivs:
        t.add_row(
            iv.get("name", "?"),
            str(iv.get("layers", [])),
            str(iv.get("scale", "?")),
            iv.get("vector_path", "?"),
        )
    console.print(t)


# ===========================================================================
# psplice heads
# ===========================================================================

@heads_app.command("mask")
def heads_mask(
    layers: Optional[List[str]] = typer.Option(
        None,
        "--layers",
        "-l",
        help="Layer:head spec, e.g. --layers 3:0,1 --layers 7:4",
    ),
) -> None:
    """
    Mask attention heads at selected layers.

    REQUIRES --eager-attn when loading the model.

    Specify each layer and its heads with --layers LAYER:HEAD[,HEAD...].
    Multiple --layers flags are accepted.

    Example:

      psplice heads mask --layers 3:0,1 --layers 7:4
    """
    if not layers:
        error("Specify at least one --layers argument, e.g. --layers 3:0,1")
        raise typer.Exit(1)

    layer_heads: dict[int, list[int]] = {}
    for spec in layers:
        try:
            layer_part, head_part = spec.split(":", 1)
            layer_idx = int(layer_part.strip())
            heads = [int(h.strip()) for h in head_part.split(",")]
            layer_heads[layer_idx] = heads
        except (ValueError, TypeError):
            error(f"Invalid layer spec: {spec!r}. Expected format: LAYER:HEAD[,HEAD...]")
            raise typer.Exit(1)

    client = DaemonClient.require()
    try:
        client.heads_mask(layer_heads)
        summary = ", ".join(f"L{k}→{v}" for k, v in sorted(layer_heads.items()))
        success(f"Head mask applied: {summary}")
    except RuntimeError as exc:
        error(str(exc))
        raise typer.Exit(1)


@heads_app.command("clear")
def heads_clear() -> None:
    """Clear all active head masks."""
    client = DaemonClient.require()
    try:
        client.heads_clear()
        success("Head masks cleared.")
    except RuntimeError as exc:
        error(str(exc))
        raise typer.Exit(1)


@heads_app.command("list")
def heads_list() -> None:
    """List active head masks."""
    client = DaemonClient.require()
    try:
        ivs = client.heads_list()
    except RuntimeError as exc:
        error(str(exc))
        raise typer.Exit(1)

    if not ivs:
        info("No active head masks.")
        return

    for iv in ivs:
        lh = iv.get("layer_heads", {})
        console.print(f"  [bold]{iv.get('name')}[/bold]")
        for layer, h_list in lh.items():
            console.print(f"    Layer {layer}: heads {h_list}")


# ===========================================================================
# psplice layers
# ===========================================================================

@layers_app.command("skip")
def layers_skip(
    from_layer: int = typer.Option(
        ...,
        "--from-layer",
        help="Start bypassing layers from this index (inclusive).",
    ),
) -> None:
    """
    Skip (bypass) decoder layers from a given index onward.

    Residual-stream updates from skipped layers are discarded — the hidden
    state entering the skipped layer is passed through unchanged.

    Note: compute cost is not reduced in v1 (the forward still runs). This
    is an approximation suitable for behavioral exploration.

    Example:

      psplice layers skip --from-layer 24
    """
    client = DaemonClient.require()
    try:
        client.layers_skip(from_layer)
        success(f"Layer skip active: bypassing layers {from_layer}+ onward.")
    except RuntimeError as exc:
        error(str(exc))
        raise typer.Exit(1)


@layers_app.command("clear")
def layers_clear() -> None:
    """Clear the active layer skip intervention."""
    client = DaemonClient.require()
    try:
        client.layers_clear()
        success("Layer skip cleared.")
    except RuntimeError as exc:
        error(str(exc))
        raise typer.Exit(1)


@layers_app.command("info")
def layers_info() -> None:
    """Show current layer skip configuration."""
    client = DaemonClient.require()
    try:
        ivs = client.layers_info()
    except RuntimeError as exc:
        error(str(exc))
        raise typer.Exit(1)

    if not ivs:
        info("No layer skip active.")
        return
    for iv in ivs:
        console.print(f"  Layer skip from: [bold yellow]{iv.get('skip_from', '?')}[/bold yellow]")


# ===========================================================================
# psplice lora
# ===========================================================================

@lora_app.command("load")
def lora_load(
    adapter_path: Path = typer.Argument(
        ...,
        help="Path to a PEFT adapter directory.",
        exists=True,
    ),
) -> None:
    """
    Hot-inject a PEFT LoRA adapter onto the resident model.

    The adapter directory must contain adapter_config.json and
    adapter_model.safetensors (or adapter_model.bin).

    Only one adapter can be active at a time in v1.

    Example:

      psplice lora load ./adapters/math-lora
    """
    client = DaemonClient.require()
    try:
        client.lora_load(str(adapter_path.resolve()))
        success(f"LoRA adapter loaded: {adapter_path}")
    except RuntimeError as exc:
        error(str(exc))
        raise typer.Exit(1)


@lora_app.command("unload")
def lora_unload() -> None:
    """Unload the active LoRA adapter and restore base model behavior."""
    client = DaemonClient.require()
    try:
        client.lora_unload()
        success("LoRA adapter unloaded.")
    except RuntimeError as exc:
        error(str(exc))
        raise typer.Exit(1)


@lora_app.command("list")
def lora_list() -> None:
    """Show the active LoRA adapter, if any."""
    client = DaemonClient.require()
    try:
        info_data = client.lora_info()
    except RuntimeError as exc:
        error(str(exc))
        raise typer.Exit(1)

    if info_data.get("active"):
        console.print(f"  Active adapter: [bold cyan]{info_data.get('adapter_path')}[/bold cyan]")
    else:
        info("No LoRA adapter loaded.")


# ===========================================================================
# psplice decode
# ===========================================================================

@decode_app.command("set")
def decode_set(
    temperature: Optional[float] = typer.Option(None, "--temperature", "-t"),
    top_p: Optional[float] = typer.Option(None, "--top-p"),
    top_k: Optional[int] = typer.Option(None, "--top-k"),
    repetition_penalty: Optional[float] = typer.Option(None, "--repetition-penalty"),
    max_new_tokens: Optional[int] = typer.Option(None, "--max-new-tokens"),
) -> None:
    """
    Set generation / decode parameters.

    Only specified options are changed; others remain at their current values.

    Example:

      psplice decode set --temperature 0.7 --top-p 0.9 --top-k 40
    """
    kwargs: dict = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if top_k is not None:
        kwargs["top_k"] = top_k
    if repetition_penalty is not None:
        kwargs["repetition_penalty"] = repetition_penalty
    if max_new_tokens is not None:
        kwargs["max_new_tokens"] = max_new_tokens

    if not kwargs:
        warn("No options specified. Use --temperature, --top-p, --top-k, etc.")
        return

    client = DaemonClient.require()
    try:
        result = client.decode_set(**kwargs)
        settings = result.get("settings", {})
        t = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        t.add_column("k", style="cyan")
        t.add_column("v")
        for k, v in settings.items():
            t.add_row(k, str(v) if v is not None else "[dim]default[/dim]")
        console.print(t)
    except RuntimeError as exc:
        error(str(exc))
        raise typer.Exit(1)


@decode_app.command("show")
def decode_show() -> None:
    """Show current generation settings."""
    client = DaemonClient.require()
    try:
        settings = client.decode_show()
    except RuntimeError as exc:
        error(str(exc))
        raise typer.Exit(1)

    t = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    t.add_column("k", style="bold cyan", min_width=20)
    t.add_column("v")
    for k, v in settings.items():
        t.add_row(k, str(v) if v is not None else "[dim]default[/dim]")
    console.print(t)


@decode_app.command("reset")
def decode_reset() -> None:
    """Reset all generation settings to defaults."""
    client = DaemonClient.require()
    try:
        client.decode_reset()
        success("Decode settings reset to defaults.")
    except RuntimeError as exc:
        error(str(exc))
        raise typer.Exit(1)


# ===========================================================================
# psplice preset
# ===========================================================================

@preset_app.command("save")
def preset_save(
    name: str = typer.Argument(..., help="Name to save the current intervention stack as."),
) -> None:
    """
    Save the current interventions and decode settings as a named preset.

    Example:

      psplice preset save concise_math
    """
    client = DaemonClient.require()
    try:
        client.preset_save(name)
        success(f"Preset '{name}' saved.")
    except RuntimeError as exc:
        error(str(exc))
        raise typer.Exit(1)


@preset_app.command("load")
def preset_load(
    name: str = typer.Argument(..., help="Name of the preset to load."),
) -> None:
    """
    Load a named preset and activate its interventions.

    Example:

      psplice preset load concise_math
    """
    client = DaemonClient.require()
    try:
        result = client.preset_load(name)
        errs = result.get("errors", [])
        if errs:
            for e in errs:
                warn(e)
        success(f"Preset '{name}' loaded.")
    except RuntimeError as exc:
        error(str(exc))
        raise typer.Exit(1)


@preset_app.command("list")
def preset_list() -> None:
    """List all saved presets."""
    client = DaemonClient.require()
    try:
        names = client.preset_list()
    except RuntimeError as exc:
        error(str(exc))
        raise typer.Exit(1)

    if not names:
        info("No presets saved. Use `psplice preset save <name>` to create one.")
        return

    for n in names:
        console.print(f"  [cyan]{n}[/cyan]")


@preset_app.command("clear")
def preset_clear() -> None:
    """
    Deactivate all current interventions.

    This does NOT delete saved presets from disk — it only removes the
    interventions that are currently active in the daemon.  To delete a saved
    preset file, remove it from ~/.local/share/psplice/presets/ manually.
    """
    client = DaemonClient.require()
    try:
        client.preset_clear()
        success("All active interventions cleared.")
    except RuntimeError as exc:
        error(str(exc))
        raise typer.Exit(1)


# ===========================================================================
# psplice vectors
# ===========================================================================

@vectors_app.command("extract")
def vectors_extract(
    positive: List[str] = typer.Option(
        ...,
        "--positive",
        "-p",
        help="A prompt representing the concept/behaviour to steer toward. "
             "Repeat the flag for multiple prompts.",
    ),
    negative: List[str] = typer.Option(
        ...,
        "--negative",
        "-n",
        help="A prompt representing the opposite concept/behaviour. "
             "Repeat the flag for multiple prompts.",
    ),
    layers: str = typer.Option(
        ...,
        "--layers",
        "-l",
        help="Comma-separated layer indices to extract from, e.g. 14,15,16",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output path for the .pt file.",
    ),
    token_aggregation: str = typer.Option(
        "mean",
        "--aggregation",
        help="How to pool the sequence dimension: mean (default) or last.",
    ),
) -> None:
    """
    Extract a contrastive steering vector from the loaded model.

    Runs your positive and negative prompts through the model, collects
    hidden states at the specified layers, and saves the mean activation
    difference as a .pt file ready for `psplice steer add`.

    You need at least one prompt on each side.  More prompts (5–20 per side)
    produce more stable vectors.

    Examples:

      # Conciseness vector
      psplice vectors extract \\
          --positive "Be brief." --positive "Answer in one sentence." \\
          --negative "Explain in detail." --negative "Elaborate fully." \\
          --layers 14,15,16 \\
          --output ./vectors/concise.pt

      # Then immediately steer with it
      psplice steer add concise \\
          --vector ./vectors/concise.pt \\
          --layers 14,15,16 \\
          --scale 0.5

    Aggregation modes:

      mean  Average hidden states across all token positions (default).
            Works well for most use cases.

      last  Use only the last token position.  Better for instruction-tuned
            models where the final token carries the most context.
    """
    if token_aggregation not in ("mean", "last"):
        error(f"--aggregation must be 'mean' or 'last', got {token_aggregation!r}")
        raise typer.Exit(1)

    try:
        layer_indices = [int(x.strip()) for x in layers.split(",")]
    except ValueError:
        error("--layers must be comma-separated integers, e.g. 14,15,16")
        raise typer.Exit(1)

    # Resolve output path to absolute before sending to daemon
    output_abs = str(output.resolve())

    client = DaemonClient.require()

    # Show a clear summary of what's about to run
    console.print()
    console.print(f"  [bold]Positive prompts[/bold] ({len(positive)}):")
    for p in positive:
        console.print(f"    [green]+ {p}[/green]")
    console.print(f"  [bold]Negative prompts[/bold] ({len(negative)}):")
    for n in negative:
        console.print(f"    [red]- {n}[/red]")
    console.print(f"  [bold]Layers:[/bold] {layer_indices}")
    console.print(f"  [bold]Aggregation:[/bold] {token_aggregation}")
    console.print()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=console,
        ) as progress:
            progress.add_task(
                f"[cyan]Running {len(positive) + len(negative)} forward passes…",
                total=None,
            )
            result = client.extract_vector(
                positive_prompts=list(positive),
                negative_prompts=list(negative),
                layer_indices=layer_indices,
                output_path=output_abs,
                token_aggregation=token_aggregation,
            )
    except RuntimeError as exc:
        error(str(exc))
        raise typer.Exit(1)
    except DaemonUnavailableError as exc:
        error(str(exc))
        raise typer.Exit(1)

    success(f"Vector saved: {result['output_path']}")
    console.print(
        f"  [dim]layers={result['layer_indices']}  "
        f"hidden_size={result['hidden_size']}  "
        f"format={result['format']}[/dim]"
    )
    console.print()
    console.print("  To use it:")
    console.print(
        f"  [cyan]psplice steer add <name> "
        f"--vector {output_abs} "
        f"--layers {','.join(str(i) for i in result['layer_indices'])} "
        f"--scale 0.5[/cyan]"
    )


# ===========================================================================
# Entrypoint helpers
# ===========================================================================

_VALID_DEVICES = {"auto", "cpu", "cuda", "mps"}
_VALID_DTYPES = {"auto", "bfloat16", "float16", "float32"}


def _validate_load_args(device: str, dtype: str, model_id: str) -> None:
    """
    Fail fast with a clear error before spawning the daemon subprocess.

    This catches obvious mistakes (wrong --device name, unsupported --dtype)
    before the user waits 60+ seconds for a model load that will fail anyway.
    """
    # Allow cuda:N and cuda:device_name patterns
    device_base = device.split(":")[0] if ":" in device else device
    if device_base not in _VALID_DEVICES:
        error(
            f"Unknown --device value: {device!r}\n"
            f"  Valid values: auto, cpu, cuda, cuda:0, cuda:1, mps"
        )
        raise typer.Exit(1)

    if dtype not in _VALID_DTYPES:
        error(
            f"Unknown --dtype value: {dtype!r}\n"
            f"  Valid values: {', '.join(sorted(_VALID_DTYPES))}"
        )
        raise typer.Exit(1)

    # Local path check: if model_id looks like a path, verify it exists
    from pathlib import Path as _Path
    if model_id.startswith(".") or model_id.startswith("/"):
        p = _Path(model_id)
        if not p.exists():
            error(f"Local model path does not exist: {model_id}")
            raise typer.Exit(1)
        if not p.is_dir():
            error(f"Local model path is not a directory: {model_id}")
            raise typer.Exit(1)


# ===========================================================================
# Entrypoint
# ===========================================================================

if __name__ == "__main__":
    app()
