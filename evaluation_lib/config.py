"""
evaluation_lib/config.py
========================
Single source of truth for all evaluation configuration:

  MODEL_REGISTRY          – short CLI keys → HuggingFace model IDs (base models)
  ALL_MODELS              – ordered list of all model keys (smallest → largest)
  SELECTED_MODELS_QUANT   – 4-model core set for quantization evaluation
  QWEN3_KEYS              – keys that need enable_thinking=False in chat template
  SYSTEM_PROMPT_*         – zero-shot and few-shot system prompts
"""

from __future__ import annotations

# ── Model registry ─────────────────────────────────────────────────────────────
# All models use base (pretrained-only) checkpoints — no -Instruct / -it suffix.
# Rationale: base models quantize more cleanly because their weights follow the
# near-normal distribution that NF4/INT4 was designed for; RLHF/instruction
# tuning shifts weight distributions away from that ideal.
#
# Qwen3 models are kept as-is: Alibaba released them as a unified base+chat
# checkpoint with no separate base-only variant.
MODEL_REGISTRY: dict[str, str] = {
    # ── Openly available base models ──────────────────────────────────────
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M",        # was SmolLM2-135M-Instruct
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M",        # was SmolLM2-360M-Instruct
    "smollm3":      "HuggingFaceTB/SmolLM3-3B",          # no separate base released
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",                 # was Qwen2.5-0.5B-Instruct
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",                 # was Qwen2.5-1.5B-Instruct
    "qwen3-0.6b":   "Qwen/Qwen3-0.6B",                   # unified base+chat model
    "qwen3-1.7b":   "Qwen/Qwen3-1.7B",                   # unified base+chat model
    "qwen3-4b":     "Qwen/Qwen3-4B",                     # unified base+chat model
    "tinyllama":    "TinyLlama/TinyLlama-1.1B",           # was TinyLlama-1.1B-Chat-v1.0
    # ── Gated models (require HF token + accepted licence) ────────────────
    "gemma3-270m":  "google/gemma-3-270m",                # was gemma-3-270m-it
    "gemma3-1b":    "google/gemma-3-1b",                  # was gemma-3-1b-it
    "llama3.2-3b":  "meta-llama/Llama-3.2-3B",           # was Llama-3.2-3B-Instruct
}

# Ordered smallest → largest (used as default run order in batch runners)
ALL_MODELS: list[str] = [
    "smollm2-135m",
    "gemma3-270m",
    "smollm2-360m",
    "qwen2.5-0.5b",
    "qwen3-0.6b",
    "tinyllama",
    "gemma3-1b",
    "qwen2.5-1.5b",
    "qwen3-1.7b",
    "smollm3",
    "llama3.2-3b",
    "qwen3-4b",
]

# ── 4-model core set for quantization evaluation ───────────────────────────────
# Covers a range of sizes and architectures from the baseline set.
SELECTED_MODELS_QUANT: dict[str, str] = {
    "qwen2.5-0.5b": MODEL_REGISTRY["qwen2.5-0.5b"],
    "qwen3-0.6b":   MODEL_REGISTRY["qwen3-0.6b"],
    "qwen2.5-1.5b": MODEL_REGISTRY["qwen2.5-1.5b"],
    "smollm3":      MODEL_REGISTRY["smollm3"],
}

# ── Qwen3 special handling ─────────────────────────────────────────────────────
# Qwen3 tokenizers support enable_thinking= in apply_chat_template.
# Always disable thinking for deterministic single-token routing output.
QWEN3_KEYS: frozenset[str] = frozenset({"qwen3-0.6b", "qwen3-1.7b", "qwen3-4b"})

# ── System prompts ─────────────────────────────────────────────────────────────
# Prompt order: system-prompt → available-tools → user-request.
# For chat models: system role contains SYSTEM_PROMPT; user role contains
# the tool list followed by the user request.
# For base models (no chat template): same order flattened into a completion prompt.

SYSTEM_PROMPT_ZERO_SHOT: str = (
    "You are a tool router.\n\n"
    "Rules:\n"
    "- Return only the tool name.\n"
    '- Return "none" if no tool matches.\n'
    "- Do not explain."
)

SYSTEM_PROMPT_FEW_SHOT: str = (
    "You are a tool router.\n\n"
    "Rules:\n"
    "- Return only the tool name.\n"
    '- Return "none" if no tool matches.\n'
    "- Do not explain.\n\n"
    "Examples:\n\n"
    # ── Example 1: clear match ──────────────────────────────────────────────
    "Available Tools:\n"
    "Name: nav_route_planner\n"
    "Description: Plans a driving route to a destination.\n"
    "\n"
    "Name: climate_control\n"
    "Description: Adjusts cabin temperature and fan settings.\n"
    "\n"
    "Name: media_player_ctrl\n"
    "Description: Controls music and audio playback.\n"
    "\n"
    'User Request: "Get me to the downtown office by 9 am."\n'
    "Selected Tool: nav_route_planner\n\n"
    # ── Example 2: clear match ──────────────────────────────────────────────
    "Available Tools:\n"
    "Name: call_handler\n"
    "Description: Makes and manages phone calls.\n"
    "\n"
    "Name: sms_messenger\n"
    "Description: Sends and reads SMS text messages.\n"
    "\n"
    "Name: vehicle_diagnostics\n"
    "Description: Reads and explains vehicle fault codes and sensor data.\n"
    "\n"
    "User Request: \"Send a quick text to the office: 'stuck in traffic'.\"\n"
    "Selected Tool: sms_messenger\n\n"
    # ── Example 3: no match → none ──────────────────────────────────────────
    "Available Tools:\n"
    "Name: ev_charging_scheduler\n"
    "Description: Schedules and manages EV charging sessions.\n"
    "\n"
    "Name: fuel_station_finder\n"
    "Description: Finds nearby fuel stations and current prices.\n"
    "\n"
    "Name: parking_locator\n"
    "Description: Finds and reserves parking spots.\n"
    "\n"
    'User Request: "How do I file my tax return online?"\n'
    "Selected Tool: none"
)

# Alias for code that imports just SYSTEM_PROMPT
SYSTEM_PROMPT: str = SYSTEM_PROMPT_ZERO_SHOT
