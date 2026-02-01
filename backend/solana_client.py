"""Fetch Solana transaction data via public RPC (README: Feature 1 — Transaction Fetching)."""

import os

import httpx

SOLANA_MAINNET_RPC = os.environ.get("SOLANA_MAINNET_RPC") or "https://api.mainnet-beta.solana.com"
SOLANA_DEVNET_RPC = os.environ.get("SOLANA_DEVNET_RPC") or "https://api.devnet.solana.com"


def _rpc_url(network: str) -> str:
    return SOLANA_DEVNET_RPC if (network or "").strip().lower() == "devnet" else SOLANA_MAINNET_RPC


async def get_signatures_for_address(
    address: str, network: str = "mainnet", limit: int = 10, before: str | None = None
) -> list[dict]:
    """
    Fetch recent transaction signatures for an address (for polling-based live feed).
    Returns list of { signature, blockTime, err, ... }.
    """
    url = _rpc_url(network)
    params: list = [address, {"limit": limit}]
    if before:
        params[1]["before"] = before
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getSignaturesForAddress",
                "params": params,
            },
        )
        data = resp.json()
        if data.get("error"):
            return []
        return data.get("result") or []


async def get_transaction(tx_hash: str, network: str = "mainnet") -> dict | None:
    """
    Fetch full transaction by signature.
    network: "mainnet" (default) or "devnet".
    Returns RPC result: { meta, transaction } or None if not found.
    """
    url = _rpc_url(network)
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTransaction",
                "params": [
                    tx_hash,
                    {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0},
                ],
            },
        )
        data = resp.json()
        if data.get("error"):
            return None
        return data.get("result")
