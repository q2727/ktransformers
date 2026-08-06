#!/usr/bin/env python3
"""Colocated draft service backed by the official SSD DraftRunner.

This keeps the official drafter's paged KV cache, glue decode, outcome tree,
CUDA graphs, and tensor outcome cache in one dedicated MPS CUDA process.  The
KTransformers target exchanges only compact token/control messages over a
Unix-domain socket.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import torch

# The launcher puts only the speculative module directory (not the SGLang
# package root) on this process's PYTHONPATH.  That keeps the official SSD
# environment independent of SGLang's frontend/runtime dependencies.
from ssd_official_protocol import (
    FLAG_ERROR,
    FLAG_RESPONSE,
    BufferReader,
    BufferWriter,
    OfficialSSDOp,
    error_payload,
    recv_frame,
    send_frame,
)
from ssd.config import Config
from ssd.engine.draft_runner import DraftRunner
from ssd.engine.model_runner import ModelRunner
from ssd.utils.async_helpers.async_spec_helpers import make_glue_decode_input_ids
from ssd.utils.context import reset_context, set_context


logger = logging.getLogger("official-ssd-draft")


class ColocatedDraftRunner(DraftRunner):
    """Official asynchronous drafter without its mandatory two-rank NCCL loop."""

    def __init__(self, cfg: Config, gpu_memory_utilization: float):
        self.draft_cfg = self.create_draft_config(cfg)
        # Official async mode assumes a separate GPU and otherwise consumes
        # 80% of all free memory.  Only KV capacity is changed here; model,
        # paged-attention, glue/tree kernels, and CUDA graphs remain official.
        self.draft_cfg.gpu_memory_utilization = gpu_memory_utilization
        self.is_draft = True
        self.prev_num_tokens = None

        with (
            patch.object(torch.distributed, "init_process_group"),
            patch.object(torch.distributed, "new_group", return_value=None),
        ):
            ModelRunner.__init__(
                self,
                self.draft_cfg,
                rank=0,
                event=None,
                is_draft=True,
                num_tp_gpus=1,
                init_q=None,
            )

        self._reset_tree_cache_tensors()
        self._init_prealloc_buffers()
        self._draft_step_times = []


@dataclass
class DraftSession:
    rid: str
    seq_id: int
    block_table: torch.Tensor
    canonical_tokens: list[int]
    current_candidate: list[int]
    current_cache_hit: bool = False
    outcome_generation: int = 0


class OfficialSSDDraftEngine:
    def __init__(
        self,
        model: str,
        draft_length: int,
        fan_outs: list[int],
        fan_outs_miss: list[int],
        max_model_len: int,
        block_size: int,
        gpu_memory_utilization: float,
        verbose: bool,
    ) -> None:
        if len(fan_outs) != draft_length + 1:
            raise ValueError("fan_outs must contain exactly K+1 entries.")
        if len(fan_outs_miss) != draft_length + 1:
            raise ValueError("fan_outs_miss must contain exactly K+1 entries.")
        if any(value < 0 for value in [*fan_outs, *fan_outs_miss]):
            raise ValueError("fan-out entries must be non-negative.")
        if sum(fan_outs) == 0 or sum(fan_outs_miss) == 0:
            raise ValueError("fan-out lists must retain at least one branch.")
        if sum(fan_outs) != sum(fan_outs_miss):
            raise ValueError(
                "Official SSD currently requires equal hit/miss branch budgets."
            )
        if block_size < 2 * draft_length + 2:
            raise ValueError("block_size must be at least 2*K+2.")

        self.draft_length = draft_length
        self.fan_outs = fan_outs
        self.fan_outs_miss = fan_outs_miss
        self.branch_budget = sum(fan_outs)
        self.max_model_len = max_model_len
        self.block_size = block_size
        self._next_seq_id = 0
        self.session: DraftSession | None = None

        cfg = Config(
            model=model,
            draft=model,
            max_num_batched_tokens=max_model_len,
            max_num_seqs=1,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            num_gpus=1,
            enforce_eager=False,
            kvcache_block_size=block_size,
            speculate=True,
            speculate_k=draft_length,
            draft_async=True,
            async_fan_out=max(fan_outs),
            fan_out_list=fan_outs,
            fan_out_list_miss=fan_outs_miss,
            jit_speculate=True,
            verbose=verbose,
        )
        self.runner = ColocatedDraftRunner(cfg, gpu_memory_utilization)
        if self.runner.config.max_blocks > self.runner.config.num_kvcache_blocks:
            raise RuntimeError(
                "Official SSD allocated fewer KV blocks than one full request: "
                f"need={self.runner.config.max_blocks}, "
                f"have={self.runner.config.num_kvcache_blocks}."
            )
        logger.info(
            "official SSD runner ready model=%s K=%d fan_outs=%s "
            "fan_outs_miss=%s max_model_len=%d block_size=%d kv_blocks=%d",
            model,
            draft_length,
            fan_outs,
            fan_outs_miss,
            self.runner.config.max_model_len,
            block_size,
            self.runner.config.num_kvcache_blocks,
        )

    @property
    def device(self) -> torch.device:
        return self.runner.device

    def reset(self) -> None:
        self.runner._reset_tree_cache_tensors()
        self.session = None

    def _new_block_table(self) -> torch.Tensor:
        return torch.arange(
            self.runner.config.max_blocks,
            dtype=torch.int32,
            device=self.device,
        ).unsqueeze(0)

    @torch.inference_mode()
    def _prefill(self, prompt_tokens: list[int], block_table: torch.Tensor) -> None:
        if not prompt_tokens:
            raise ValueError("Official SSD requires at least one prompt token.")
        if len(prompt_tokens) >= self.runner.config.max_model_len:
            raise ValueError(
                f"Prompt has {len(prompt_tokens)} tokens but max_model_len is "
                f"{self.runner.config.max_model_len}."
            )
        counts = torch.tensor(
            [len(prompt_tokens)], dtype=torch.int64, device=self.device
        )
        context = self.runner.prepare_prefill_ctxt(counts, block_table)
        set_context(
            is_prefill=True,
            cu_seqlens_q=context["cu_seqlens_q"],
            cu_seqlens_k=context["cu_seqlens_k"],
            max_seqlen_q=context["max_seqlen_q"],
            max_seqlen_k=context["max_seqlen_k"],
            slot_mapping=context["slot_map"],
            context_lens=None,
        )
        try:
            self.runner.run_model(
                torch.tensor(
                    prompt_tokens, dtype=torch.int64, device=self.device
                ),
                context["positions"],
                is_prefill=True,
                last_only=True,
            )
        finally:
            reset_context()

    @torch.inference_mode()
    def _select_or_jit_candidate(
        self,
        session: DraftSession,
        *,
        accepted_length: int,
        recovery_token: int,
        num_tokens: int,
    ) -> tuple[list[int], bool]:
        request_keys = torch.tensor(
            [[session.seq_id, accepted_length, recovery_token]],
            dtype=torch.int64,
            device=self.device,
        )
        num_tokens_tensor = torch.tensor(
            [num_tokens], dtype=torch.int64, device=self.device
        )
        temperatures = torch.zeros(1, dtype=torch.float32, device=self.device)
        tokens, _logits, _glue_ids, cache_hits, _activations = (
            self.runner.hit_cache_and_respond(
                request_keys,
                1,
                self.draft_length,
                num_tokens_tensor,
                temperatures,
                session.block_table,
            )
        )
        torch.cuda.synchronize()
        return [int(token) for token in tokens[0].tolist()], bool(
            cache_hits[0].item()
        )

    @torch.inference_mode()
    def _commit_full_accept_tail(
        self, session: DraftSession, num_tokens: int
    ) -> None:
        """Write the final accepted draft token into persistent draft KV.

        ``jit_speculate`` runs K decode steps starting with the recovery token.
        It therefore writes the recovery token and draft tokens 0..K-2, while
        draft token K-1 is returned from logits but is not itself decoded.  The
        normal SSD BUILD/glue path writes that tail token speculatively.  A
        no-outcome-cache sequential-SD run skips BUILD, so a full K-token
        acceptance must commit the missing tail before decoding the new target
        recovery token.
        """
        if len(session.current_candidate) != self.draft_length:
            raise RuntimeError(
                "Cannot commit a full-accept tail without a complete candidate."
            )
        position_value = num_tokens - 2
        if position_value < 0:
            raise RuntimeError("Full-accept tail position is negative.")

        positions = torch.tensor(
            [position_value], dtype=torch.int64, device=self.device
        )
        block_indices = positions // self.block_size
        offsets = positions % self.block_size
        slot_mapping = (
            session.block_table[0, block_indices] * self.block_size + offsets
        ).to(torch.int32)
        context_lens = torch.tensor(
            [position_value + 1], dtype=torch.int32, device=self.device
        )
        set_context(
            is_prefill=False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=session.block_table,
            is_jit=True,
        )
        try:
            self.runner.run_model(
                torch.tensor(
                    [session.current_candidate[-1]],
                    dtype=torch.int64,
                    device=self.device,
                ),
                positions,
                is_prefill=False,
                last_only=True,
            )
        finally:
            reset_context()

    def init(self, rid: str, prefix: list[int]) -> tuple[list[int], bool, float]:
        if len(prefix) < 2:
            raise ValueError("INIT prefix must contain prompt plus recovery token.")
        if len(prefix) + self.draft_length >= self.runner.config.max_model_len:
            raise ValueError("INIT prefix leaves no room for a full draft.")

        begin = time.perf_counter()
        self.reset()
        block_table = self._new_block_table()
        self._prefill(prefix[:-1], block_table)
        session = DraftSession(
            rid=rid,
            seq_id=self._next_seq_id,
            block_table=block_table,
            canonical_tokens=list(prefix),
            current_candidate=[],
        )
        self._next_seq_id += 1
        candidate, cache_hit = self._select_or_jit_candidate(
            session,
            accepted_length=-1,
            recovery_token=prefix[-1],
            num_tokens=len(prefix),
        )
        session.current_candidate = candidate
        session.current_cache_hit = cache_hit
        self.session = session
        elapsed_ms = (time.perf_counter() - begin) * 1e3
        logger.info(
            "INIT rid=%s prefix=%d cache_hit=%s elapsed_ms=%.3f",
            rid,
            len(prefix),
            cache_hit,
            elapsed_ms,
        )
        return candidate, cache_hit, elapsed_ms

    def _require_session(self, rid: str) -> DraftSession:
        if self.session is None:
            raise RuntimeError("Official SSD has no active draft session.")
        if self.session.rid != rid:
            raise RuntimeError(
                f"Official SSD active rid is {self.session.rid}, not {rid}."
            )
        return self.session

    def jit(self, rid: str, prefix: list[int]) -> tuple[list[int], bool, float]:
        begin = time.perf_counter()
        session = self.session
        # Reconstruct after a service/session failure.  In the normal miss path
        # canonical KV is already valid, so retain it and only clear outcomes.
        if (
            session is None
            or session.rid != rid
            or session.canonical_tokens != prefix
        ):
            return self.init(rid, prefix)

        self.runner._reset_tree_cache_tensors()
        candidate, _ = self._select_or_jit_candidate(
            session,
            accepted_length=-1,
            recovery_token=prefix[-1],
            num_tokens=len(prefix),
        )
        session.current_candidate = candidate
        session.current_cache_hit = False
        elapsed_ms = (time.perf_counter() - begin) * 1e3
        logger.info("JIT rid=%s elapsed_ms=%.3f", rid, elapsed_ms)
        return candidate, False, elapsed_ms

    def select(
        self,
        rid: str,
        num_tokens: int,
        accepted_length: int,
        recovery_token: int,
        *,
        force_miss: bool = False,
    ) -> tuple[list[int], bool, float]:
        session = self._require_session(rid)
        if not 0 <= accepted_length <= self.draft_length:
            raise ValueError(
                f"accepted_length={accepted_length} is outside [0,{self.draft_length}]."
            )
        expected_tokens = [
            *session.canonical_tokens,
            *session.current_candidate[:accepted_length],
            recovery_token,
        ]
        if num_tokens != len(expected_tokens):
            raise RuntimeError(
                "Target/draft canonical length diverged: "
                f"target={num_tokens}, expected={len(expected_tokens)}."
            )

        begin = time.perf_counter()
        if force_miss:
            self.runner._reset_tree_cache_tensors()
            if accepted_length == self.draft_length:
                self._commit_full_accept_tail(session, num_tokens)
        candidate, cache_hit = self._select_or_jit_candidate(
            session,
            accepted_length=accepted_length,
            recovery_token=recovery_token,
            num_tokens=num_tokens,
        )
        session.canonical_tokens = expected_tokens
        session.current_candidate = candidate
        session.current_cache_hit = cache_hit
        elapsed_ms = (time.perf_counter() - begin) * 1e3
        logger.info(
            "%s rid=%s k=%d recovery=%d cache_hit=%s elapsed_ms=%.3f",
            "ADVANCE_JIT" if force_miss else "SELECT",
            rid,
            accepted_length,
            recovery_token,
            cache_hit,
            elapsed_ms,
        )
        if force_miss and cache_hit:
            raise RuntimeError("ADVANCE_JIT unexpectedly hit an outcome cache entry.")
        return candidate, cache_hit, elapsed_ms

    @torch.inference_mode()
    def build(
        self,
        rid: str,
        num_tokens: int,
        recovery_token: int,
        candidate_cache_hit: bool,
        candidate: list[int],
    ) -> tuple[float, float, float, float]:
        session = self._require_session(rid)
        if num_tokens != len(session.canonical_tokens):
            raise RuntimeError(
                f"BUILD length={num_tokens}, session={len(session.canonical_tokens)}."
            )
        if recovery_token != session.canonical_tokens[-1]:
            raise RuntimeError("BUILD recovery token does not match canonical state.")
        if candidate != session.current_candidate:
            raise RuntimeError("BUILD candidate does not match selected draft state.")
        if len(candidate) != self.draft_length:
            raise RuntimeError(
                f"BUILD candidate has {len(candidate)} tokens, "
                f"expected {self.draft_length}."
            )

        device = self.device
        returned_tokens = torch.tensor(
            [candidate], dtype=torch.int64, device=device
        )
        recovery = torch.tensor([recovery_token], dtype=torch.int64, device=device)
        glue_ids = make_glue_decode_input_ids(returned_tokens, recovery)
        cache_hits = torch.tensor(
            [int(candidate_cache_hit)], dtype=torch.int64, device=device
        )
        partial = {
            "num_tokens": torch.tensor(
                [num_tokens], dtype=torch.int64, device=device
            ),
            "seq_ids": torch.tensor(
                [session.seq_id], dtype=torch.int64, device=device
            ),
            "temperatures": torch.zeros(1, dtype=torch.float32, device=device),
            "dbt": session.block_table,
            "cache_hits": cache_hits,
            "returned_tokens": returned_tokens,
            "target_recovery_activations": None,
            "previous_activations": None,
            "extend_counts": None,
            "extend_eagle_acts": None,
            "extend_token_ids": None,
        }

        self.runner._reset_tree_cache_tensors()
        torch.cuda.synchronize()
        total_begin = time.perf_counter()
        with torch.cuda.nvtx.range("official_ssd_glue"):
            tree_args = self.runner._build_tree_batch(partial, glue_ids)
        torch.cuda.synchronize()
        glue_done = time.perf_counter()
        with torch.cuda.nvtx.range("official_ssd_tree_decode"):
            tokens, logits, activations = self.runner._decode_tree(tree_args)
        torch.cuda.synchronize()
        tree_done = time.perf_counter()
        self.runner._populate_tree_cache(
            tree_args,
            tokens,
            logits,
            tree_args["cache_hits"],
            activations,
        )
        torch.cuda.synchronize()
        populate_done = time.perf_counter()
        session.outcome_generation += 1

        glue_ms = (glue_done - total_begin) * 1e3
        tree_ms = (tree_done - glue_done) * 1e3
        populate_ms = (populate_done - tree_done) * 1e3
        total_ms = (populate_done - total_begin) * 1e3
        logger.info(
            "BUILD rid=%s generation=%d branches=%d glue_ms=%.3f "
            "tree_ms=%.3f populate_ms=%.3f total_ms=%.3f",
            rid,
            session.outcome_generation,
            self.branch_budget,
            glue_ms,
            tree_ms,
            populate_ms,
            total_ms,
        )
        return glue_ms, tree_ms, populate_ms, total_ms


class OfficialSSDServer:
    def __init__(self, socket_path: Path, engine: OfficialSSDDraftEngine):
        self.socket_path = socket_path
        self.engine = engine
        self.stopping = False
        self.listener: socket.socket | None = None

    def stop(self, *_args) -> None:
        self.stopping = True
        if self.listener is not None:
            try:
                self.listener.close()
            except OSError:
                pass

    @staticmethod
    def _candidate_payload(
        candidate: list[int], cache_hit: bool, elapsed_ms: float
    ) -> bytes:
        return (
            BufferWriter()
            .u8(cache_hit)
            .f64(elapsed_ms)
            .int_list(candidate)
            .finish()
        )

    def _handle(self, op: OfficialSSDOp, payload: bytes) -> bytes:
        reader = BufferReader(payload)
        if op == OfficialSSDOp.PING:
            reader.finish()
            return BufferWriter().text("pong").finish()
        if op == OfficialSSDOp.RESET:
            reader.finish()
            self.engine.reset()
            return b""
        if op == OfficialSSDOp.SHUTDOWN:
            reader.finish()
            self.stopping = True
            return b""
        if op == OfficialSSDOp.INIT:
            rid = reader.text()
            prefix = reader.int_list()
            reader.finish()
            candidate, cache_hit, elapsed_ms = self.engine.init(rid, prefix)
            return self._candidate_payload(candidate, cache_hit, elapsed_ms)
        if op == OfficialSSDOp.JIT:
            rid = reader.text()
            prefix = reader.int_list()
            reader.finish()
            candidate, cache_hit, elapsed_ms = self.engine.jit(rid, prefix)
            return self._candidate_payload(candidate, cache_hit, elapsed_ms)
        if op == OfficialSSDOp.SELECT:
            rid = reader.text()
            num_tokens = reader.i64()
            accepted_length = reader.i32()
            recovery_token = reader.i32()
            reader.finish()
            candidate, cache_hit, elapsed_ms = self.engine.select(
                rid, num_tokens, accepted_length, recovery_token
            )
            return self._candidate_payload(candidate, cache_hit, elapsed_ms)
        if op == OfficialSSDOp.ADVANCE_JIT:
            rid = reader.text()
            num_tokens = reader.i64()
            accepted_length = reader.i32()
            recovery_token = reader.i32()
            reader.finish()
            candidate, cache_hit, elapsed_ms = self.engine.select(
                rid,
                num_tokens,
                accepted_length,
                recovery_token,
                force_miss=True,
            )
            return self._candidate_payload(candidate, cache_hit, elapsed_ms)
        if op == OfficialSSDOp.BUILD:
            rid = reader.text()
            num_tokens = reader.i64()
            recovery_token = reader.i32()
            candidate_cache_hit = bool(reader.u8())
            candidate = reader.int_list()
            reader.finish()
            glue_ms, tree_ms, populate_ms, total_ms = self.engine.build(
                rid,
                num_tokens,
                recovery_token,
                candidate_cache_hit,
                candidate,
            )
            return (
                BufferWriter()
                .u32(self.engine.branch_budget)
                .f64(glue_ms)
                .f64(tree_ms)
                .f64(populate_ms)
                .f64(total_ms)
                .finish()
            )
        raise RuntimeError(f"Unsupported Official SSD operation {op}.")

    def _serve_connection(self, conn: socket.socket) -> None:
        with conn:
            while not self.stopping:
                try:
                    op, flags, payload = recv_frame(conn)
                except EOFError:
                    return
                if flags:
                    send_frame(
                        conn,
                        op,
                        error_payload("Requests must not set protocol flags."),
                        flags=FLAG_RESPONSE | FLAG_ERROR,
                    )
                    continue
                try:
                    response = self._handle(op, payload)
                    send_frame(conn, op, response, flags=FLAG_RESPONSE)
                except Exception as exc:  # keep the model process available
                    logger.error(
                        "Official SSD request %s failed: %s\n%s",
                        op.name,
                        exc,
                        traceback.format_exc(),
                    )
                    send_frame(
                        conn,
                        op,
                        error_payload(f"{type(exc).__name__}: {exc}"),
                        flags=FLAG_RESPONSE | FLAG_ERROR,
                    )

    def serve_forever(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener = listener
        listener.bind(str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        listener.listen(1)
        listener.settimeout(1.0)
        logger.info("listening on unix://%s", self.socket_path)
        try:
            while not self.stopping:
                try:
                    conn, _ = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self.stopping:
                        break
                    raise
                self._serve_connection(conn)
        finally:
            listener.close()
            try:
                self.socket_path.unlink()
            except FileNotFoundError:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--draft-length", type=int, default=5)
    parser.add_argument("--fan-outs", type=int, nargs="+", required=True)
    parser.add_argument("--fan-outs-miss", type=int, nargs="+")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.20)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    fan_outs_miss = args.fan_outs_miss or args.fan_outs
    engine = OfficialSSDDraftEngine(
        model=args.model,
        draft_length=args.draft_length,
        fan_outs=args.fan_outs,
        fan_outs_miss=fan_outs_miss,
        max_model_len=args.max_model_len,
        block_size=args.block_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        verbose=args.verbose,
    )
    server = OfficialSSDServer(args.socket, engine)
    signal.signal(signal.SIGTERM, server.stop)
    signal.signal(signal.SIGINT, server.stop)
    server.serve_forever()


if __name__ == "__main__":
    main()
