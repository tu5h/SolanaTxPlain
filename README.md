# SolanaTxPlain — AI-Powered Solana Transaction Explainer

Built for Royal Hackaway v9  
Sponsor Tracks: Solana + Gemini + OpenRouter

---

# 🚀 Overview

SolanaTxPlain is an AI-powered transaction explainer that converts complex Solana blockchain transactions into clear, plain English.

Solana explorers show raw technical data:
- instruction logs
- program calls
- token account changes
- balance deltas
- contract interactions

This is difficult for normal users to understand.

SolanaTxPlain acts like a **translator layer** between blockchain data and humans.

Users paste a Solana transaction hash → the system fetches the transaction → parses the changes → uses AI to explain what actually happened.

Think of it as:

> “Google Translate for Solana transactions.”

---

# 🎯 Problem This Project Solves

Blockchain transparency exists — but **human readability does not**.

Current Solana explorers are built for developers, not everyday users.

Most users cannot answer:
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

## DigitalOcean Track
Backend API is deployed and live.

## OpenRouter Track
Multi-model AI reasoning and fallback.

## ElevenLabs Track (Optional)
Voice explanation of transaction summary.

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
DigitalOcean Hosted API
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
- DigitalOcean App Platform

## Optional
- ElevenLabs voice API

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
8. Show deployed DigitalOcean endpoint

---

# 🚀 Deployment (DigitalOcean)

Recommended: DigitalOcean App Platform

Steps:
1. Push repo to GitHub
2. Connect DigitalOcean
3. Deploy backend API
4. Get public URL
5. Use in demo

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


