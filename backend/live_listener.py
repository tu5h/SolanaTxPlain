"""
Live Solana transaction listener and grouper.
Subscribes to logs for a wallet via WebSocket, fetches txs, groups within 2–3s, explains via AI, emits to SSE.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any

import websockets

from backend.ai_explain import explain_group
from backend.parser import parse_tx
from backend.solana_client import get_signatures_for_address, get_transaction

log = logging.getLogger("solana_tx_plain")

SOLANA_MAINNET_WS = os.environ.get("SOLANA_MAINNET_WS") or "wss://api.mainnet-beta.solana.com"
SOLANA_DEVNET_WS = os.environ.get("SOLANA_DEVNET_WS") or "wss://api.devnet.solana.com"
GROUP_WINDOW_SEC = 2.5  # group txs that land within this many seconds
POLL_INTERVAL_SEC = 2.0  # devnet: poll getSignaturesForAddress every N seconds


def _ws_url(network: str) -> str:
    return SOLANA_DEVNET_WS if (network or "").strip().lower() == "devnet" else SOLANA_MAINNET_WS


def _is_devnet(network: str) -> bool:
    return (network or "").strip().lower() == "devnet"


async def fetch_and_parse(signature: str, network: str = "mainnet") -> tuple[str, dict[str, Any]] | None:
    """Fetch tx by signature and return (signature, parsed) or None."""
    raw = await get_transaction(signature, network=network)
    if not raw:
        return None
    parsed = parse_tx(raw)
    parsed["signature"] = signature
    parsed["_raw"] = raw
    return (signature, parsed)


async def run_listener(
    wallet: str,
    out_queue: asyncio.Queue,
    *,
    network: str = "mainnet",
    group_seconds: float = GROUP_WINDOW_SEC,
    stop: asyncio.Event | None = None,
) -> None:
    """
    Subscribe to Solana logs for wallet, buffer txs, group by time window, explain via AI, push to out_queue.
    Each item: {"type": "activity", "signatures": [...], "count": N, "wallet": wallet, "explanation": {...}, "just_happened": True}.
    """
    buffer: list[tuple[str, dict[str, Any], float]] = []
    stop = stop or asyncio.Event()
    last_flush = time.monotonic()
    loop = asyncio.get_event_loop()

    async def flush() -> None:
        nonlocal buffer, last_flush
        if not buffer:
            return
        group = buffer[:]
        buffer.clear()
        last_flush = time.monotonic()
        sigs = [s for s, _, _ in group]
        tx_list = [p for _, p, _ in group]
        try:
            explanation = await loop.run_in_executor(None, lambda: explain_group(tx_list))
            out_queue.put_nowait({
                "type": "activity",
                "signatures": sigs,
                "count": len(tx_list),
                "wallet": wallet,
                "explanation": explanation,
                "just_happened": True,
            })
        except asyncio.QueueFull:
            log.warning("Live out_queue full, dropping activity group")
        except Exception as e:
            log.warning("explain_group failed: %s", e)
            out_queue.put_nowait({
                "type": "activity",
                "signatures": sigs,
                "count": len(tx_list),
                "wallet": wallet,
                "explanation": {"summary": "Explanation failed.", "intent": "unknown", "wallet_impact": "—", "fees": "—", "programs_used": "—", "risk": "No suspicious activity.", "why_multiple_txs": "—", "explanation": str(e)[:200], "error": str(e)[:200]},
                "just_happened": True,
            })

    async def flush_loop() -> None:
        while not stop.is_set():
            await asyncio.sleep(0.5)
            if buffer and (time.monotonic() - last_flush) >= group_seconds:
                await flush()

    async def poll_loop() -> None:
        """Devnet: poll getSignaturesForAddress every N seconds (public devnet WS often doesn't deliver logsSubscribe)."""
        nonlocal buffer, last_flush
        seen_sigs: set[str] = set()
        max_seen = 200
        first_poll = True
        while not stop.is_set():
            try:
                sigs_result = await get_signatures_for_address(wallet, network=network, limit=10)
                if first_poll and sigs_result:
                    # Seed seen_sigs so we only explain txs that happen *after* we start
                    for item in sigs_result:
                        s = item.get("signature")
                        if s:
                            seen_sigs.add(s)
                    log.info("Devnet poll: seeded %s existing sigs (only new txs will be explained)", len(seen_sigs))
                    first_poll = False
                    await asyncio.sleep(POLL_INTERVAL_SEC)
                    continue
                first_poll = False
                for item in sigs_result or []:
                    sig = item.get("signature")
                    if not sig or sig in seen_sigs:
                        continue
                    seen_sigs.add(sig)
                    if len(seen_sigs) > max_seen:
                        seen_sigs = {s for s, _, _ in buffer} | {it.get("signature") for it in (sigs_result or [])[:20]}
                    fetched = await fetch_and_parse(sig, network=network)
                    if fetched:
                        _, parsed = fetched
                        buffer.append((sig, parsed, time.monotonic()))
                        log.info("Devnet poll: buffered tx %s (buffer size %s)", sig[:16], len(buffer))
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning("Devnet poll error: %s", e)
            await asyncio.sleep(POLL_INTERVAL_SEC)

    async def ws_loop() -> None:
        nonlocal buffer, last_flush
        while not stop.is_set():
            try:
                async with websockets.connect(
                    _ws_url(network),
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    # logsSubscribe: filter by wallet mention
                    await ws.send(json.dumps({
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "logsSubscribe",
                        "params": [
                            {"mentions": [wallet]},
                            {"commitment": "confirmed"},
                        ],
                    }))
                    # first response is subscription id
                    sub_resp = await ws.recv()
                    sub_data = json.loads(sub_resp)
                    if sub_data.get("error"):
                        log.warning("logsSubscribe error: %s", sub_data["error"])
                        await asyncio.sleep(5)
                        continue
                    log.info("Live listener subscribed for wallet %s... on %s (wait for txs from this wallet)", wallet[:12], network)
                    while not stop.is_set():
                        msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                        data = json.loads(msg)
                        method = data.get("method")
                        if method != "logsNotification":
                            continue
                        params = data.get("params") or {}
                        result = params.get("result") or {}
                        sig = result.get("signature")
                        err = result.get("err")
                        if not sig:
                            continue
                        log.info("logsNotification received for %s... (network=%s)", sig[:16], network)
                        if err:
                            log.debug("Tx %s failed on-chain: %s", sig[:16], err)
                        fetched = await fetch_and_parse(sig, network=network)
                        if fetched:
                            _, parsed = fetched
                            buffer.append((sig, parsed, time.monotonic()))
                            log.info("Buffered tx %s (buffer size %s)", sig[:16], len(buffer))
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning("Live listener ws error: %s", e)
                await asyncio.sleep(5)

    flush_task = asyncio.create_task(flush_loop())
    if _is_devnet(network):
        log.info("Live listener using POLLING for devnet (wallet %s...)", wallet[:12])
        poll_task = asyncio.create_task(poll_loop())
        try:
            await asyncio.gather(flush_task, poll_task)
        except asyncio.CancelledError:
            pass
        finally:
            flush_task.cancel()
            poll_task.cancel()
            await flush()
    else:
        ws_task = asyncio.create_task(ws_loop())
        try:
            await asyncio.gather(flush_task, ws_task)
        except asyncio.CancelledError:
            pass
        finally:
            flush_task.cancel()
            ws_task.cancel()
            await flush()
