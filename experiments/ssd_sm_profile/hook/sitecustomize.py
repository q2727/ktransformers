"""Profiling-only NVTX annotations for SSD phase attribution.

Enable with SSD_PHASE_NVTX=1 and put this directory first on PYTHONPATH.
By default no synchronization is introduced.  The optional SSD_CORUN_* files are
used only by the controlled two-process co-run experiment.
"""

from __future__ import annotations

import functools
import os
import time


def _install() -> None:
    import torch

    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.speculative.eagle_worker import EAGLEWorker

    if getattr(EAGLEWorker, "_ssd_phase_nvtx_installed", False):
        return

    log_wall = os.environ.get("SSD_PHASE_LOG", "0") == "1"
    profile_role = os.environ.get("SSD_PROFILE_ROLE")
    corun_arm_file = os.environ.get("SSD_CORUN_ARM_FILE")
    corun_go_file = os.environ.get("SSD_CORUN_GO_FILE")
    corun_draft_ready_file = os.environ.get("SSD_CORUN_DRAFT_READY_FILE")

    def touch(path: str) -> None:
        with open(path, "a", encoding="utf-8"):
            pass

    def wrap_method(cls, method_name: str, range_name: str, skip_idle: bool = True):
        original = getattr(cls, method_name)

        @functools.wraps(original)
        def wrapped(self, *args, **kwargs):
            batch = args[0] if args else kwargs.get("batch")
            if skip_idle and batch is not None:
                forward_mode = getattr(batch, "forward_mode", None)
                if forward_mode is not None and forward_mode.is_idle():
                    return original(self, *args, **kwargs)
            active_range_name = range_name
            if (
                method_name == "verify"
                and profile_role == "target"
                and corun_arm_file
                and corun_go_file
                and os.path.exists(corun_arm_file)
                and not os.path.exists(corun_go_file)
            ):
                touch(corun_go_file)
                active_range_name = "ssd::target_verify_corun"
            torch.cuda.nvtx.range_push(active_range_name)
            begin = time.perf_counter_ns()
            try:
                return original(self, *args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter_ns() - begin) / 1e6
                torch.cuda.nvtx.range_pop()
                if log_wall:
                    print(
                        f"SSD_PHASE_WALL name={active_range_name} ms={elapsed_ms:.6f}",
                        flush=True,
                    )

        setattr(cls, method_name, wrapped)

    wrap_method(EAGLEWorker, "draft", "ssd::draft_tree")
    wrap_method(EAGLEWorker, "verify", "ssd::target_verify")
    wrap_method(
        EAGLEWorker,
        "forward_draft_extend_after_decode",
        "ssd::draft_reconcile",
    )
    EAGLEWorker._ssd_phase_nvtx_installed = True

    if profile_role == "draft":
        original_forward = TpModelWorker.forward_batch_generation

        @functools.wraps(original_forward)
        def wrapped_forward(self, model_worker_batch, *args, **kwargs):
            mode = getattr(model_worker_batch, "forward_mode", None)
            mode_name = getattr(mode, "name", str(mode)).lower()
            range_name = f"ssd::draft_server::{mode_name}"
            req_pool_indices = getattr(model_worker_batch, "req_pool_indices", None)
            batch_size = (
                int(req_pool_indices.numel()) if req_pool_indices is not None else 0
            )
            if (
                mode_name == "decode"
                and batch_size == 9
                and corun_arm_file
                and corun_go_file
                and corun_draft_ready_file
                and os.path.exists(corun_arm_file)
            ):
                range_name = "ssd::draft_server::decode_corun"
                if not os.path.exists(corun_draft_ready_file):
                    touch(corun_draft_ready_file)
                    deadline = time.monotonic() + float(
                        os.environ.get("SSD_CORUN_WAIT_TIMEOUT", "120")
                    )
                    while not os.path.exists(corun_go_file):
                        if time.monotonic() >= deadline:
                            raise RuntimeError("Timed out waiting for target VERIFY")
                        time.sleep(0.001)
            torch.cuda.nvtx.range_push(range_name)
            begin = time.perf_counter_ns()
            try:
                return original_forward(self, model_worker_batch, *args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter_ns() - begin) / 1e6
                torch.cuda.nvtx.range_pop()
                if log_wall:
                    print(
                        f"SSD_PHASE_WALL name={range_name} ms={elapsed_ms:.6f}",
                        flush=True,
                    )

        TpModelWorker.forward_batch_generation = wrapped_forward


if os.environ.get("SSD_PHASE_NVTX", "0") == "1":
    _install()
