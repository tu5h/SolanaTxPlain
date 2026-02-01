"""
SolanaTxPlain API (README).
POST /explain → { tx_hash } → { summary, intent, wallet_changes, fees, risk_flags, explanation }
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from multiple locations; load backend/.env LAST with override so it wins
# (ensures OPENROUTER_API_KEY in backend/.env is used even if another .env exists)
_backend = Path(__file__).resolve().parent
_root = _backend.parent
load_dotenv(_root.parent / ".env")
load_dotenv(_root / ".env")
load_dotenv()
load_dotenv(_backend / ".env", override=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
log = logging.getLogger("solana_tx_plain")
if os.environ.get("GEMINI_API_KEY"):
    log.info("GEMINI_API_KEY loaded (use /debug to verify)")
else:
    log.warning("GEMINI_API_KEY not set — add to backend/.env")
if os.environ.get("OPENROUTER_API_KEY"):
    log.info("OPENROUTER_API_KEY loaded (fallback when Gemini 429)")
else:
    log.warning("OPENROUTER_API_KEY not set — add to backend/.env for fallback")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from backend.ai_explain import get_explanation
from backend.parser import parse_tx
from backend.solana_client import get_transaction

app = FastAPI(title="SolanaTxPlain", description="AI-powered Solana transaction explainer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(
        "<h1>SolanaTxPlain API</h1>"
        "<p><a href='/docs'>/docs</a> · <a href='/health'>/health</a> · <a href='/debug'>/debug</a></p>"
        "<p>POST /explain with body: { \"tx_hash\": \"...\" }</p>"
    )


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Avoid browser 404 for automatic favicon request."""
    return Response(status_code=204)

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/debug")
def debug():
    """Verify which API keys are loaded (masked). Use to confirm .env is read."""
    gemini = os.environ.get("GEMINI_API_KEY")
    openrouter = os.environ.get("OPENROUTER_API_KEY")
    return {
        "GEMINI_API_KEY": "set (" + (gemini[:8] + "..." + gemini[-4:] if gemini and len(gemini) > 12 else "***") + ")" if gemini else "not set",
        "GEMINI_MODEL": os.environ.get("GEMINI_MODEL", "(default: gemini-2.0-flash)"),
        "OPENROUTER_API_KEY": "set (" + (openrouter[:8] + "..." + openrouter[-4:] if openrouter and len(openrouter) > 12 else "***") + ")" if openrouter else "not set",
        "OPENROUTER_MODEL": os.environ.get("OPENROUTER_MODEL", "(default: google/gemini-2.0-flash)"),
    }


class ExplainRequest(BaseModel):
    tx_hash: str
    simple_mode: bool = True  # README Feature 8: Simple vs Technical mode


@app.post("/explain")
async def explain(req: ExplainRequest):
    tx_hash = (req.tx_hash or "").strip()
    if not tx_hash:
        raise HTTPException(status_code=400, detail="tx_hash is required")

    raw = await get_transaction(tx_hash)
    if not raw:
        raise HTTPException(status_code=404, detail="Transaction not found.")

    parsed = parse_tx(raw)
    ai = get_explanation(parsed, simple_mode=req.simple_mode)

    if ai.get("error"):
        msg = ai.get("message", "AI explanation failed.")
        if ai.get("error") == "quota":
            raise HTTPException(status_code=429, detail=msg)
        raise HTTPException(status_code=503, detail=msg)

    # README API output + openrouter cross-check when both Gemini and OpenRouter ran
    out = {
        "summary": ai["summary"],
        "intent": ai["intent"],
        "wallet_changes": {
            "sol_balance_change": parsed.get("sol_balance_change"),
            "token_balance_changes": parsed.get("token_balance_changes"),
            "wallet_impact_text": ai.get("wallet_impact"),
        },
        "fees": ai.get("fees") or f"{parsed.get('fee_paid', 0)} SOL",
        "risk_flags": ai.get("risk_flags", []),
        "explanation": ai["explanation"],
        "sections": ai.get("sections", {}),
        "slot": parsed.get("slot"),
        "block_time": parsed.get("block_time"),
    }
    if ai.get("openrouter_summary") is not None or ai.get("openrouter_explanation"):
        out["openrouter_summary"] = ai.get("openrouter_summary")
        out["openrouter_explanation"] = ai.get("openrouter_explanation")
        out["openrouter_intent"] = ai.get("openrouter_intent")
        out["openrouter_risk"] = ai.get("openrouter_risk")
        out["openrouter_sections"] = ai.get("openrouter_sections", {})
    return out
