"""
evaluation_lib/config.py
========================
Single source of truth for all evaluation configuration:

  MODEL_REGISTRY  – short CLI keys → HuggingFace model IDs
  ALL_MODELS      – ordered list of all model keys (smallest → largest)
  QWEN3_KEYS      – keys that need enable_thinking=False in chat template
  SYSTEM_PROMPT_* – zero-shot and few-shot system prompts
"""

from __future__ import annotations

# ── Model registry ─────────────────────────────────────────────────────────────
# Instruct / chat checkpoints are preferred over base models throughout.
# Qwen3 and SmolLM3 ship as unified base+chat checkpoints with no separate
# -Instruct variant; their thinking mode is disabled at inference time via
# QWEN3_KEYS below.
# NOTE: granite3.3-2b uses thinking=False (not enable_thinking=) in
# apply_chat_template — see QWEN3_KEYS comment for context.
MODEL_REGISTRY: dict[str, str] = {
    # ── TINY (<300M) ──────────────────────────────────────────────────────
    "pythia-70m":     "EleutherAI/pythia-70m",                  # base only — no instruct variant exists
    "cerebras-111m":  "cerebras/Cerebras-GPT-111M",             # base only — no instruct variant exists
    "smollm2-135m":   "HuggingFaceTB/SmolLM2-135M-Instruct",   # instruct, open
    "gemma3-270m":    "google/gemma-3-270m-it",                 # instruct, gated (Google Gemma licence)
    # ── SMALL (<1B) ────────────────────────────────────────────────────────
    "smollm2-360m":   "HuggingFaceTB/SmolLM2-360M-Instruct",   # instruct, open
    "qwen2.5-0.5b":   "Qwen/Qwen2.5-0.5B-Instruct",            # instruct, open
    "qwen3-0.6b":     "Qwen/Qwen3-0.6B",                       # unified base+chat, open
    # ── MEDIUM (<2B) ───────────────────────────────────────────────────────
    "gemma3-1b":      "google/gemma-3-1b-it",                   # instruct, gated (Google Gemma licence)
    "llama3.2-1b":    "meta-llama/Llama-3.2-1B-Instruct",      # instruct, gated (Meta Llama 3.2 licence)
    "qwen3-1.7b":     "Qwen/Qwen3-1.7B",                       # unified base+chat, open
    "smollm2-1.7b":   "HuggingFaceTB/SmolLM2-1.7B-Instruct",  # instruct, open
    # ── LARGE (<=3B) ───────────────────────────────────────────────────────
    "granite3.3-2b":  "ibm-granite/granite-3.3-2b-instruct",   # instruct, open
    "gemma2-2b":      "google/gemma-2-2b-it",                   # instruct, gated (Google Gemma licence); Gemma 2 series
    "smollm3":        "HuggingFaceTB/SmolLM3-3B",              # unified base+chat, open
    "llama3.2-3b":    "meta-llama/Llama-3.2-3B-Instruct",      # instruct, gated (Meta Llama 3.2 licence)
}

# Ordered smallest → largest (used as default run order in batch runners)
ALL_MODELS: list[str] = [
    # TINY
    "pythia-70m",
    "cerebras-111m",
    "smollm2-135m",
    "gemma3-270m",
    # SMALL
    "smollm2-360m",
    "qwen2.5-0.5b",
    "qwen3-0.6b",
    # MEDIUM
    "gemma3-1b",
    "llama3.2-1b",
    "qwen3-1.7b",
    "smollm2-1.7b",
    # LARGE
    "granite3.3-2b",
    "gemma2-2b",
    "smollm3",
    "llama3.2-3b",
]

# ── Models that use enable_thinking= in apply_chat_template ──────────────────
# Qwen3 and SmolLM3 are unified base+chat checkpoints that default to thinking
# mode. Always disable for deterministic single-token routing output.
# NOTE: granite3.3-2b uses a *different* kwarg (thinking=False) rather than
# enable_thinking= — it is NOT listed here; callers must handle it separately.
QWEN3_KEYS: frozenset[str] = frozenset({"qwen3-0.6b", "qwen3-1.7b", "smollm3"})

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
