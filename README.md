# SolanaTxPlain — AI-Powered Solana Transaction Explainer

https://devpost.com/software/solanatxplain

**Built for Royal Hackaway v9** · Sponsor Tracks: **Solana** · **Gemini** · **OpenRouter**

---

## ⚡ Try it locally ( — run in ~2 minutes)

1. **Clone and setup** (from project root):
   ```bash
   pip install -r backend/requirements.txt
   ```
   Copy `.env.example` to `backend/.env` and add your `GEMINI_API_KEY` ([get one](https://aistudio.google.com/apikey)).

2. **Terminal 1 — Backend:** `python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000`

3. **Terminal 2 — Frontend:** `python -m http.server 3000 --directory frontend`

4. Open **http://localhost:3000**. Paste any Solana tx hash (mainnet or devnet) → **Explain**. Or use **Live Activity** with a devnet wallet (see [DEV.md](DEV.md) for devnet setup).

Built locally due to easy testnet capability.

---

# 🚀 Overview

SolanaTxPlain is an AI-powered transaction explainer that converts complex Solana blockchain transactions into clear, plain English.

**Two ways to use it:**
- **Explain a single transaction** — Paste a tx hash (mainnet or devnet), get a full plain-English breakdown (summary, intent, wallet impact, fees, programs, risk, AI explanation).
- **Live Activity** — Enter a wallet address, click Start live; we listen for transactions and explain each burst in real time (mainnet or devnet; devnet uses polling so it works without a paid RPC).

Solana explorers show raw technical data (instruction logs, program calls, token changes, balance deltas). SolanaTxPlain acts like a **translator layer** between blockchain data and humans.

Think of it as:

> “Google Translate for Solana transactions.”

---

# 🎯 Problem This Project Solves

Blockchain transparency exists — but **human readability does not**.

Current Solana explorers are built for developers, not everyday users.

__Most users cannot answer__:
- Did I swap tokens or send them?
- Which app did I interact with?
- Did I approve something risky?
- What actually changed in my wallet?
- Was this expensive?
- Was this safe?

This project makes Solana understandable to non-technical users.

---

# 🧠 Core Idea

Convert raw Solana transaction data into:

- plain English summary
- intent classification
- wallet impact report
- token movement breakdown
- fee explanation
- AI reasoning
- optional voice explanation

---

# 🏆 Hackathon Alignment

This project intentionally aligns with multiple sponsor tracks:

## Solana Track
Uses real Solana on-chain transactions.

## AI Track
Uses LLM reasoning to interpret structured blockchain data.

## OpenRouter Track
Multi-model AI reasoning and fallback.

## ElevenLabs Track (Optional)
Voice explanation of transaction summary.

---

# ✨ What we built (highlights)

- **Single-tx explain** — Paste any Solana tx hash (mainnet or devnet); get summary, intent, wallet impact, fees, programs used, risk notes, and a full AI explanation. Optional OpenRouter cross-check.
- **Live Activity** — Enter a wallet; we listen for transactions and explain each burst in plain English. Instant “something happened” ping, then full AI summary when ready.
- **Mainnet + Devnet** — Switch network in the UI; devnet uses polling so it works with the public RPC (no paid WebSocket). Perfect for testing with free SOL.
- **Relative timestamps** — Live cards show “15s ago”, “1m ago” and update every second.
- **Listening countdown** — On devnet, “Next check in 2s” / “Checking…” so users see the app is still listening.
- **Two-phase notifications** — You see “Something just happened” immediately, then the same card fills in with the full explanation.

---

# 👤 Target Users

- wallet users
- crypto beginners
- DeFi users
- NFT traders
- support teams
- wallet providers
- compliance dashboards
- education tools

---

# 🔍 Example Output

Input:

Transaction Hash → paste here


Output:

Summary:
You swapped 0.42 SOL for 132 BONK using Jupiter aggregator.

Wallet Impact:
SOL: -0.42
BONK: +132

Fees:
0.0005 SOL

Programs Used:
Jupiter Swap Router

Risk Notes:
No suspicious contracts detected.

AI Explanation:
This transaction routed your swap across a liquidity pool to get the best price.


---

# 🧩 Feature Breakdown

---

# Feature 1 — Transaction Fetching

## What It Does
Fetches full transaction data using Solana RPC or explorer APIs.

## Why It Matters
Raw blockchain data is required before interpretation.

## Implementation
The backend calls:
- Solana RPC
- Solscan API
- Helius API (optional)

Data retrieved:
- instructions
- account keys
- program IDs
- logs
- token balance changes
- pre/post balances
- fees

---

# Feature 2 — Transaction Parser

## What It Does
Converts raw blockchain response into structured readable data.

## Why It Matters
AI models perform better on structured summaries than raw logs.

## Output Structure Example

{
sol_balance_change
token_balance_changes
programs_used
fee_paid
instruction_types
}


## Implementation Notes
You compute:
- wallet balance deltas
- token mint changes
- program labels
- transfer directions

---

# Feature 3 — Intent Detection (AI)

## What It Does
Classifies transaction purpose.

## Examples
- SOL transfer
- token swap
- NFT mint
- liquidity add/remove
- staking
- contract interaction
- unknown

## Why It Matters
Users care about **intent**, not instruction logs.

## AI Usage
LLM receives structured transaction summary and returns classification.

---

# Feature 4 — Plain English Explanation (AI)

## What It Does
Generates a human-readable explanation.

## Example

Instead of:
Instruction: spl-token transferChecked
User sees:
You transferred 25 USDC to wallet X.


## Why It Matters
This is the core value layer.

---

# Feature 5 — Wallet Impact Summary

## What It Does
Shows before vs after balances.

## Example

Before: 2.14 SOL
After: 1.72 SOL
Change: -0.42 SOL



## Why It Matters
Users understand outcomes quickly.

---

# Feature 6 — Risk Signals (AI + Rules)

## What It Does
Flags suspicious or notable patterns.

## Examples
- unknown program ID
- unusually high fee
- new contract interaction
- complex multi-hop swap
- approval-style transaction

## Why It Matters
Adds safety and trust value.

---

# Feature 7 — Multi-Model AI Reasoning (OpenRouter)

## What It Does
Uses multiple AI models to compare outputs.

## Flow

Model A → summary  
Model B → classification  
Model C → risk notes  

Compare + score agreement.

## Why It Matters
Demonstrates advanced AI orchestration.

---

# Feature 8 — Simple Mode vs Technical Mode

## What It Does

Two explanation styles:

### Simple Mode
Explain like a beginner.

### Technical Mode
Include program names and routing details.

## Why It Matters
Improves accessibility + judge appeal.

---

# Feature 9 — Voice Explanation (Optional — ElevenLabs)

## What It Does
Reads AI explanation aloud.

## Why It Matters
Shows multimodal AI usage.

---

# 🏗️ Architecture

Frontend Web App
↓
Backend API
↓
Solana Data Fetch Layer
↓
Transaction Parser
↓
AI Reasoning Layer
↓
Explanation Generator
↓
Response JSON



---

# ⚙️ Tech Stack

## Backend
- Python
- FastAPI
- Requests

## AI
- Gemini API
- OpenRouter models

## Blockchain Data
- Solana RPC
- Solscan / Helius

## Hosting
- See [DEPLOY.md](DEPLOY.md) for deployment options.

---

# 📦 API Design

## POST /explain

Input:
{ "tx_hash": "..." }


Output:
{
summary
intent
wallet_changes
fees
risk_flags
explanation
}


---

# 🖥️ Demo Flow

Judge demo steps:

1. Paste transaction hash
2. Click explain
3. See summary
4. See wallet changes
5. See risk flags
6. Toggle simple/technical mode
7. Play voice explanation
8. Show deployed endpoint (see DEPLOY.md)

---

# 🚀 Deployment

See [DEPLOY.md](DEPLOY.md) for deployment steps (e.g. push to GitHub, connect your hosting provider, deploy backend, use public URL in demo).

---

# 🧪 Testing Strategy

Test with:

- simple SOL transfers
- token swaps
- NFT mints
- DeFi interactions
- high-fee tx
- unknown program tx

---

# 📈 Upgrade Roadmap

## Upgrade — Data Layer
Add Helius enhanced transaction decoding.

## Upgrade — Parser
Add full instruction decoder per program.

## Upgrade — AI
Add fine-tuned prompt templates per tx type.

## Upgrade — Risk Engine
Add rule-based risk scoring + heuristics.

## Upgrade — UI
Add visual balance delta charts.

## Upgrade — Models
Use ensemble model voting with confidence score.

## Upgrade — Wallet Integration
Turn into embeddable widget for wallets.

## Upgrade — Caching
Store explained transactions for instant repeat lookups.

## Upgrade — Alerts
Flag dangerous transactions before signing.

## Upgrade — Multichain
Extend to Ethereum, Base, Polygon.

---

SolanaTxPlain turns raw chain data into human knowledge.


