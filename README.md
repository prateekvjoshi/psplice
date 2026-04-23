# psplice

**Modify LLM behavior at runtime. No retraining. No code changes.**

psplice loads a model onto your GPU and lets you modify its behavior interactively — apply named behaviors, compare outputs, save configurations, and iterate in seconds instead of hours.

The everyday workflow (load, apply a behavior, compare, save) requires no ML knowledge. Some features — custom steering vectors, head masking, layer skip — have a learning curve and are documented separately.

---

## Requirements

- Python 3.11+
- A GPU with CUDA (Linux/Windows) or Apple Silicon MPS (Mac). CPU is supported for small models but is slow.
- 8–28 GB VRAM depending on the model (see [Recommended models](#recommended-models))

## Install

```bash
# With uv (recommended)
uv pip install git+https://github.com/prateekvjoshi/psplice.git

# Or with pip
pip install git+https://github.com/prateekvjoshi/psplice.git
```

Run `psplice doctor` after installing to validate your setup and get model recommendations for your hardware.

---

## What you can do in 5 minutes

```bash
# 1. Check your setup
psplice doctor

# 2. Load a model (stays in GPU memory)
psplice load Qwen/Qwen2.5-7B-Instruct

# 3. Apply a behavior
psplice behavior add concise

# 4. See what changed
psplice compare "Explain quantum entanglement."

# 5. Chat with the modified model
psplice chat

# 6. Save this configuration
psplice preset save concise-mode

# 7. Done for now
psplice stop
```

That's it. The first use of each behavior takes 10–30 seconds to learn from examples; subsequent uses are instant.

---

## Built-in behaviors

Behaviors are named presets that shift the model in a direction. psplice ships with a curated library:

| Name | What it does |
|------|-------------|
| `concise` | Shorter, more direct responses. Less padding. |
| `direct` | Fewer hedges and qualifications. Definitive answers. |
| `formal` | Professional, structured tone. |
| `casual` | Conversational, friendly tone. |
| `skeptical` | Questions assumptions. Surfaces failure modes. |
| `structured` | Organized output with headers and lists. |
| `creative` | More varied, original responses. |
| `technical` | Deeper technical detail. Assumes domain knowledge. |
| `cautious` | Surfaces risks and edge cases. More careful. |

```bash
# See all behaviors with descriptions
psplice behavior list

# Learn about a specific behavior
psplice behavior describe skeptical

# Apply with strength control (mild / moderate / strong)
psplice behavior add concise --strength mild
psplice behavior add concise --strength strong

# See what's currently active
psplice behavior active

# Remove a behavior
psplice behavior remove concise
```

Behaviors are extracted from your own model on the first use, then cached for subsequent runs. Extraction takes 5–30 seconds depending on model size.

---

## Compare: see exactly what changed

```bash
psplice compare "Explain TCP congestion control."
psplice compare "Should I always trust benchmarks?"
psplice compare "What are the risks of moving fast here?"
```

Compare runs the same prompt twice — once with no modifications (base) and once with your active behaviors — and shows both outputs side by side with:
- Token counts and latency for each
- A word-similarity score (how much the response actually changed)
- Token delta (did it get longer or shorter?)

This gives you immediate, quantitative feedback on whether a behavior is working.

---

## Presets: save and restore configurations

```bash
# Save everything that's active right now
psplice preset save support-bot

# Restore it later (in a new session)
psplice load Qwen/Qwen2.5-7B-Instruct
psplice preset load support-bot

# See what's saved
psplice preset list

# Clear active interventions without deleting presets
psplice preset clear
```

---

## Common tasks

### Make responses shorter
```bash
psplice behavior add concise
psplice compare "Explain REST APIs."
# Not short enough? Try stronger:
psplice behavior remove concise
psplice behavior add concise --strength strong
```

### Test a safety-adjacent behavior
```bash
psplice behavior add cautious
psplice compare "What's the fastest way to scale a database?"
```

### Compare two configurations
```bash
# Config A: concise
psplice behavior add concise
psplice preset save config-a

# Config B: formal + structured
psplice preset clear
psplice behavior add formal
psplice behavior add structured
psplice preset save config-b

# Now switch between them and compare
psplice preset load config-a
psplice compare "Explain the CAP theorem."

psplice preset load config-b
psplice compare "Explain the CAP theorem."
```

### Apply a LoRA adapter
```bash
psplice lora load ./adapters/my-adapter
psplice chat
psplice lora unload
```

### Control generation sampling
```bash
psplice decode set --temperature 0.3   # more deterministic
psplice decode set --max-new-tokens 256
psplice decode reset
```

---

## Recommended models

psplice works with any HuggingFace causal language model. These are well-tested starting points:

| Model | VRAM | Good for |
|-------|------|----------|
| `Qwen/Qwen2.5-7B-Instruct` | ~16 GB | General use, fast, recommended |
| `meta-llama/Llama-3.1-8B-Instruct` | ~16 GB | Strong instruction following |
| `Qwen/Qwen2.5-3B-Instruct` | ~8 GB | Low VRAM or Apple Silicon |
| `Qwen/Qwen2.5-14B-Instruct` | ~28 GB | Stronger reasoning |

```bash
# Load with bfloat16 for lower VRAM usage
psplice load Qwen/Qwen2.5-7B-Instruct --dtype bfloat16

# MPS (Apple Silicon)
psplice load Qwen/Qwen2.5-3B-Instruct --device mps
```

Llama models require a HuggingFace account and access approval. Set `HF_TOKEN` in your environment.

---

## How it works (for the curious)

psplice keeps the model resident in GPU memory as a local daemon process running on `localhost:29371`. CLI commands are thin HTTP clients — each command sends a request to the daemon rather than loading a new model. If you're unsure whether the daemon is running, `psplice doctor` will tell you.

When you apply a behavior, psplice:
1. Runs your model's forward pass on curated positive and negative example prompts
2. Computes the difference in internal representations (hidden states) at the middle layers
3. Saves that difference as a "steering vector"
4. Registers a hook that adds the vector to the model's computations at inference time

This is called **activation steering** — a technique from interpretability research. The key insight is that many behavioral traits live in a specific direction in the model's internal representation space, and you can nudge the model along that direction without modifying any weights.

The effects are:
- **Reversible** — remove the behavior and the model returns to baseline
- **Non-destructive** — no model files are modified
- **Composable** — multiple behaviors can be active simultaneously
- **Fast** — extraction is 5–30 seconds; applying/removing is instant

---

## Advanced usage (for researchers)

If you want fine-grained control over the internals, the low-level commands are available:

```bash
# Raw activation steering — specify your own vector file and layers
psplice steer add my-vector --vector ./vectors/custom.pt --layers 14,15,16 --scale 0.8
psplice steer list
psplice steer remove my-vector

# Attention head masking (requires --eager-attn at load time)
psplice load MODEL --eager-attn
psplice heads mask --layers 3:0,1 --layers 7:4
psplice heads clear

# Layer skip / early exit approximation
psplice layers skip --from-layer 24
psplice layers clear

# Extract a steering vector from your own prompts
psplice vectors extract \
    --positive "Be brief." --positive "Get to the point." \
    --negative "Elaborate fully." --negative "Explain in detail." \
    --layers 14,15,16 \
    --output ./vectors/my-concise.pt
```

See `psplice <command> --help` for full documentation on each.

### Attention implementation caveat

Head masking requires loading the model with `--eager-attn`, which disables fused attention kernels. Activation steering and layer skip work with any attention implementation.

### Tensor format for steering vectors

`.pt` files must contain one of:
- A 1D tensor `[hidden_size]` — broadcast to all specified layers
- A dict `{layer_index: tensor}` — per-layer vectors

---

## Development

```bash
# Install from source with dev extras (tests, linting)
uv pip install -e ".[dev]"

# Run tests (no GPU required — model loading is mocked)
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=psplice --cov-report=term-missing
```

---

## Limitations

- **One model at a time.** Use `psplice load --force` to swap models.
- **One LoRA adapter at a time** in v1.
- **Layer skip doesn't reduce compute** — the forward pass still runs all layers. This is an approximation for behavioral exploration, not a speed optimization.
- **Behavior vectors are model-specific.** A vector extracted from Qwen2.5-7B won't work on Llama-3-8B.
- **Preset portability.** Presets store absolute paths. They won't transfer across machines or after moving files.
- **Outputs from intervened models are research data, not ground truth.** Activation steering is not reliable behavioral control — treat it as an experimental tool.
