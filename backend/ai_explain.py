"""
AI layer (README: Features 3–4, 6–7).
- Intent detection (Feature 3)
- Plain English explanation (Feature 4)
- Risk signals (Feature 6)
- OpenRouter: cross-check and fallback (Feature 7)
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import google.generativeai as genai
import httpx

log = logging.getLogger("solana_tx_plain")

SECTION_LABELS = ("SUMMARY", "INTENT", "WALLET_IMPACT", "FEES", "PROGRAMS_USED", "RISK", "EXPLANATION")
SECTION_KEYS = {
    "SUMMARY": "summary",
    "INTENT": "intent",
    "WALLET_IMPACT": "wallet_impact",
    "FEES": "fees",
    "PROGRAMS_USED": "programs_used",
    "RISK": "risk",
    "EXPLANATION": "explanation",
}


def get_explanation(parsed: dict[str, Any], simple_mode: bool = True) -> dict[str, Any]:
    """
    Call Gemini (and OpenRouter when configured) for explanation.
    Returns: summary, intent, wallet_impact, fees, risk_flags, explanation, sections.
    When OPENROUTER_API_KEY is set, also returns openrouter_* for cross-check display.
    On error: { "error": "...", "message": "..." }.
    """
    gemini_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    openrouter_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not gemini_key and not openrouter_key:
        return _fallback("Set GEMINI_API_KEY or OPENROUTER_API_KEY in .env.")
    prompt = _build_prompt(parsed, simple_mode)

    gemini_result: dict[str, Any] | None = None
    openrouter_result: dict[str, Any] | None = None
    gemini_error: str | None = None
    openrouter_error: str | None = None

    def run_gemini() -> tuple[dict[str, Any] | None, str | None]:
        if not gemini_key:
            return None, None
        try:
            genai.configure(api_key=gemini_key)
            model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if not response.candidates:
                reason = getattr(response.prompt_feedback, "block_reason", None) or "no content"
                return None, f"Gemini: {reason}"
            text = (response.text or "").strip()
            if not text:
                return None, "Gemini returned empty response."
            out = _parse_response(text)
            _add_risk_flags(out)
            return out, None
        except Exception as e:
            return None, str(e)[:300]

    def run_openrouter() -> tuple[dict[str, Any] | None, str | None]:
        if not openrouter_key:
            return None, None
        result, err = _call_openrouter(prompt)
        if result:
            _add_risk_flags(result)
            return result, None
        return None, err or "OpenRouter failed"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        if gemini_key:
            futures["gemini"] = executor.submit(run_gemini)
        if openrouter_key:
            futures["openrouter"] = executor.submit(run_openrouter)
        future_to_name = {fut: name for name, fut in futures.items()}
        for fut in as_completed(futures.values()):
            name = future_to_name[fut]
            try:
                res, err = fut.result()
                if name == "gemini":
                    gemini_result, gemini_error = res, err
                else:
                    openrouter_result, openrouter_error = res, err
            except Exception as e:
                if name == "gemini":
                    gemini_error = str(e)[:300]
                else:
                    openrouter_error = str(e)[:300]

    # Primary: prefer Gemini; if Gemini failed (e.g. 429), use OpenRouter as fallback
    primary = gemini_result or openrouter_result
    if not primary:
        msg = gemini_error or openrouter_error or "No explanation generated."
        if "429" in msg or "quota" in msg.lower():
            return {"error": "quota", "message": msg}
        return {"error": "gemini", "message": msg}

    # Attach OpenRouter cross-check when we have both
    if gemini_result and openrouter_result:
        primary["openrouter_summary"] = openrouter_result.get("summary")
        primary["openrouter_explanation"] = openrouter_result.get("explanation")
        primary["openrouter_intent"] = openrouter_result.get("intent")
        primary["openrouter_risk"] = openrouter_result.get("risk")
        primary["openrouter_sections"] = openrouter_result.get("sections", {})
        log.info("Gemini + OpenRouter cross-check attached")
    elif openrouter_result and not gemini_result:
        primary["openrouter_summary"] = None
        primary["openrouter_explanation"] = None

    return primary


def _add_risk_flags(out: dict[str, Any]) -> None:
    risk_text = out.get("risk") or "No suspicious activity."
    out["risk_flags"] = (
        [risk_text]
        if risk_text and risk_text.lower() not in ("none.", "no suspicious activity.", "—")
        else []
    )


# —— Live activity: explain a group of txs (1–3s burst) — same sections as single-tx ——
LIVE_LABELS = (
    "SUMMARY",
    "INTENT",
    "WALLET_IMPACT",
    "FEES",
    "PROGRAMS_USED",
    "RISK",
    "WHY_MULTIPLE_TXS",  # only when multiple txs
    "EXPLANATION",
)


def explain_group(transactions: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Explain a group of transactions that occurred within 1–3 seconds (one user action).
    Returns the same shape as single-tx: summary, intent, wallet_impact, fees, programs_used, risk, explanation;
    plus why_multiple_txs when there are multiple transactions.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or ""
    if not api_key.strip():
        return _live_fallback("GEMINI_API_KEY not set.")
    genai.configure(api_key=api_key.strip())
    model = genai.GenerativeModel(os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"))
    prompt = _build_live_prompt(transactions)
    try:
        response = model.generate_content(prompt)
        if not response.candidates:
            return _live_fallback("No content from model.")
        text = (response.text or "").strip()
        if not text:
            return _live_fallback("Empty response.")
        return _parse_live_response(text)
    except Exception as e:
        log.warning("explain_group error: %s", e)
        return _live_fallback(str(e)[:200])


def _live_fallback(msg: str) -> dict[str, Any]:
    return {
        "summary": "Explanation unavailable.",
        "intent": "unknown",
        "wallet_impact": "—",
        "fees": "—",
        "programs_used": "—",
        "risk": "No suspicious activity.",
        "why_multiple_txs": "—",
        "explanation": msg,
        "error": msg,
    }


def _build_live_prompt(transactions: list[dict[str, Any]]) -> str:
    tx_summaries = []
    total_fee = 0.0
    for i, tx in enumerate(transactions[:10]):  # cap for token size
        t = {k: v for k, v in tx.items() if k not in ("_raw", "log_preview")}
        t["log_preview"] = (tx.get("log_preview") or "")[:400]
        total_fee += float(tx.get("fee_paid") or 0)
        try:
            tx_summaries.append(f"--- Tx {i+1} ---\n{json.dumps(t, default=str)}")
        except (TypeError, ValueError):
            tx_summaries.append(f"--- Tx {i+1} --- (parse error)")
    blocks = "\n\n".join(tx_summaries)
    multi = len(transactions) > 1
    why_section = (
        "\nWHY_MULTIPLE_TXS: [Why did multiple transactions occur? E.g. 'Solana executed several steps (approve, swap, settle) as separate txs in under 2 seconds.' One or two sentences.]"
        if multi
        else ""
    )
    return f"""You are explaining a burst of Solana transactions that just happened for one wallet (within 1–3 seconds). Use the same level of detail as a single-transaction explainer: full plain-English so the user fully understands what happened. Your job is to turn this burst into one human-readable story with the same sections as a hash-based explanation.

Reply with exactly these section headers and content. You may use multiple lines per section. Be thorough but clear.

SUMMARY: [Write 2–4 sentences. Describe what this activity did in plain English: who was involved, what moved (SOL or tokens), and what the outcome was. Add context so a non-technical reader fully understands.]
INTENT: [Exactly one of: SOL transfer, token swap, NFT mint, liquidity add/remove, staking, contract interaction, token transfer, unknown]
WALLET_IMPACT: [Describe what changed: SOL and/or token amounts per account. Be clear about sender vs receiver and any token names or amounts.]
FEES: [What was paid in SOL across these transactions and what it was for (e.g. "~0.0005 SOL total as network fees.").]
PROGRAMS_USED: [Which on-chain programs or apps were used. Name them in plain English and briefly what they did if relevant.]
RISK: [One or two sentences: anything risky, unusual, or worth double-checking. If nothing stands out, say "No suspicious activity."]{why_section}
EXPLANATION: [Write 3–6 sentences (or a short paragraph). Explain what happened step by step in plain English: what the user or contract did, how funds or tokens moved, why the fee or programs were involved, and what the user can infer. If multiple txs occurred, mention that Solana often executes many steps in under a second (fast finality, low fees) and that this is normal.]

Group of {len(transactions)} transaction(s) (total fee ~{total_fee:.6f} SOL):
{blocks}

Reply with only the sectioned response. Use the exact section labels above. Content for SUMMARY and EXPLANATION should be longer and more detailed."""


def _parse_live_response(text: str) -> dict[str, Any]:
    key_map = {
        "SUMMARY": "summary",
        "INTENT": "intent",
        "WALLET_IMPACT": "wallet_impact",
        "FEES": "fees",
        "PROGRAMS_USED": "programs_used",
        "RISK": "risk",
        "WHY_MULTIPLE_TXS": "why_multiple_txs",
        "EXPLANATION": "explanation",
    }
    out = {
        "summary": "No summary.",
        "intent": "unknown",
        "wallet_impact": "—",
        "fees": "—",
        "programs_used": "—",
        "risk": "No suspicious activity.",
        "why_multiple_txs": "—",
        "explanation": "—",
    }
    current_key: str | None = None
    lines: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            if current_key:
                lines.append("")
            continue
        matched = False
        for lbl in LIVE_LABELS:
            if line.upper().startswith(lbl + ":"):
                if current_key and current_key in out:
                    val = " ".join(lines).strip()
                    if val:
                        out[current_key] = val
                current_key = key_map.get(lbl, lbl.lower().replace(" ", "_"))
                rest = line[len(lbl) + 1:].strip()
                lines = [rest] if rest else []
                matched = True
                break
        if not matched and current_key:
            lines.append(line)
    if current_key and current_key in out:
        val = " ".join(lines).strip()
        if val:
            out[current_key] = val
    return out


def _call_openrouter(prompt: str) -> tuple[dict[str, Any] | None, str | None]:
    """
    Call OpenRouter with the given prompt. Returns (result_dict, error_message).
    If success: (result, None). If failure: (None, "reason").
    """
    api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        return None, None
    url = "https://openrouter.ai/api/v1/chat/completions"
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                url,
                json={
                    "model": os.environ.get("OPENROUTER_MODEL", "google/gemini-2.0-flash"),
                    "messages": [{"role": "user", "content": prompt}],
                },
                headers={"Authorization": f"Bearer {api_key}"},
            )
        body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        if resp.status_code != 200:
            err = body.get("error", {}).get("message") or body.get("message") or resp.text[:200] or f"HTTP {resp.status_code}"
            log.warning("OpenRouter HTTP %s: %s", resp.status_code, err)
            return None, f"HTTP {resp.status_code}: {err}"
        content = (body.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        if not content.strip():
            return None, "Empty response from OpenRouter"
        out = _parse_response(content.strip())
        return out, None
    except Exception as e:
        log.warning("OpenRouter failed: %s", e)
        return None, str(e)[:200]


def _fallback(msg: str) -> dict[str, Any]:
    return {
        "error": "config",
        "message": msg,
        "summary": "Explanation unavailable.",
        "intent": "unknown",
        "risk_flags": [],
        "explanation": msg,
    }


def _build_prompt(parsed: dict[str, Any], simple_mode: bool) -> str:
    mode = "Explain in simple terms for a beginner." if simple_mode else "Include program names and technical routing details."
    fee = parsed.get("fee_paid", 0)
    sol = parsed.get("sol_balance_change") or []
    tokens = parsed.get("token_balance_changes") or []
    programs = parsed.get("programs_used") or []
    instruction_types = parsed.get("instruction_types") or []
    log_preview = (parsed.get("log_preview") or "")[:1000]
    slot = parsed.get("slot")
    block_time = parsed.get("block_time")
    when = f"Slot: {slot}. Block time (Unix): {block_time}." if (slot is not None or block_time is not None) else ""

    return f"""You are a Solana transaction explainer. {mode}

From the transaction data below, reply with exactly these section headers and content. You may use multiple lines per section; start each section with the header in CAPS followed by a colon, then write the content. Be thorough but clear.

Required format:
SUMMARY: [Write 2–4 sentences. Describe what this transaction did in plain English: who was involved, what moved (SOL or tokens), and what the outcome was. Add a bit of context so a non-technical reader fully understands.]
INTENT: [Exactly one of: SOL transfer, token swap, NFT mint, liquidity add/remove, staking, contract interaction, token transfer, unknown]
WALLET_IMPACT: [Describe what changed: SOL and/or token amounts per account. Be clear about sender vs receiver and any token names or amounts. One or two sentences is fine.]
FEES: [What was paid in SOL and what it was for (e.g. "0.0005 SOL as the network transaction fee.").]
PROGRAMS_USED: [Which on-chain programs or apps were used. Name them in plain English (e.g. "Solana System Program" or "Jupiter Swap Router") and briefly what they did if relevant.]
RISK: [One or two sentences: anything risky, unusual, or worth double-checking (unknown programs, high fees, approvals). If nothing stands out, say "No suspicious activity." and optionally why it looks normal.]
EXPLANATION: [Write 3–6 sentences (or a short paragraph). Explain what happened step by step in plain English: what the user or contract did, how funds or tokens moved, why the fee or programs were involved, and what the user can infer from this transaction. Go into a bit more detail so the reader really understands.]

Transaction data:
- Fee (SOL): {fee}
- SOL balance changes: {sol}
- Token balance changes: {tokens}
- Programs: {programs[:10]}
- Instruction types: {instruction_types[:10]}
{f"- When: {when}" if when else ""}
- Log snippet:
{log_preview}

Reply with only the sectioned response. Use the exact section labels above. Content for SUMMARY and EXPLANATION should be longer and more detailed than one line."""


def _parse_response(text: str) -> dict[str, Any]:
    sections = {k: "" for k in SECTION_KEYS.values()}
    sections["summary"] = "No summary."
    sections["intent"] = "unknown"
    sections["risk"] = "No suspicious activity."
    sections["explanation"] = "—"

    current = None
    lines_acc = []

    def flush():
        nonlocal current, lines_acc
        if current and current in SECTION_KEYS:
            key = SECTION_KEYS[current]
            val = " ".join(lines_acc).strip()
            if val:
                sections[key] = val
        lines_acc = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            if current:
                lines_acc.append("")
            continue
        matched = False
        for label in SECTION_LABELS:
            if line.upper().startswith(label + ":"):
                flush()
                current = label
                rest = line[len(label) + 1:].strip()
                lines_acc = [rest] if rest else []
                matched = True
                break
        if not matched and current:
            lines_acc.append(line)
    flush()

    return {
        "sections": sections,
        "summary": sections.get("summary") or "No summary.",
        "intent": (sections.get("intent") or "unknown").strip().lower(),
        "wallet_impact": sections.get("wallet_impact") or "—",
        "fees": sections.get("fees") or "—",
        "programs_used": sections.get("programs_used") or "—",
        "risk": sections.get("risk") or "No suspicious activity.",
        "explanation": sections.get("explanation") or sections.get("summary") or "—",
        "risk_flags": [],
    }
