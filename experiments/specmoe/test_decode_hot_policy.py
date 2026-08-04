#!/usr/bin/env python3
"""CPU-only tests for the KT decode-hot replacement policy."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.kt_decode_hot import (
    _INITIAL_RESIDENT,
    _LAYER_STATES,
    _is_promotion_step,
    _resident_slot_bytes,
    _select_layer_promotions,
    _validate_restored_residency,
)


def make_method(
    *, min_tokens: int = 4, hysteresis: float = 1.25, min_residency: int = 4
):
    return SimpleNamespace(
        kt_config=SimpleNamespace(
            layer_idx=900,
            kt_decode_hot_ema_decay=0.0,
            kt_decode_hot_min_residency=min_residency,
            kt_decode_hot_max_promotions=1,
            kt_decode_hot_min_tokens=min_tokens,
            kt_decode_hot_hysteresis=hysteresis,
        ),
        gpu_index_to_logical=torch.tensor([0, 1], dtype=torch.int32),
        global_num_experts=8,
        num_gpu_experts=2,
    )


class DecodeHotPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        _LAYER_STATES.clear()
        _INITIAL_RESIDENT.pop(901, None)

    def test_promotes_hottest_candidate_into_coldest_slot(self) -> None:
        method = make_method()
        promotions, resident = _select_layer_promotions(
            method, torch.tensor([0, 1, 0, 0, 0, 0, 5, 10])
        )
        self.assertEqual(len(promotions), 1)
        self.assertEqual((promotions[0].victim, promotions[0].candidate), (0, 7))
        self.assertEqual(resident.tolist(), [7, 1])

    def test_token_floor_and_hysteresis_block_marginal_candidate(self) -> None:
        method = make_method(min_tokens=4, hysteresis=1.5)
        promotions, _ = _select_layer_promotions(
            method, torch.tensor([8, 7, 9, 0, 0, 0, 0, 0])
        )
        self.assertEqual(promotions, [])

        _LAYER_STATES.clear()
        method = make_method(min_tokens=4, hysteresis=1.0)
        promotions, _ = _select_layer_promotions(
            method, torch.tensor([0, 0, 3, 0, 0, 0, 0, 0])
        )
        self.assertEqual(promotions, [])

    def test_newly_promoted_slots_respect_minimum_residency(self) -> None:
        method = make_method(min_residency=4)
        first, resident = _select_layer_promotions(
            method, torch.tensor([0, 0, 10, 0, 0, 0, 0, 0])
        )
        self.assertEqual(len(first), 1)
        method.gpu_index_to_logical.copy_(resident.to(torch.int32))

        second, resident = _select_layer_promotions(
            method, torch.tensor([0, 0, 10, 9, 0, 0, 0, 0])
        )
        self.assertEqual(len(second), 1)
        method.gpu_index_to_logical.copy_(resident.to(torch.int32))

        third, _ = _select_layer_promotions(
            method, torch.tensor([0, 0, 10, 9, 20, 0, 0, 0])
        )
        self.assertEqual(third, [])

    def test_refresh_gate_updates_ema_without_replacing_slot(self) -> None:
        method = make_method()
        promotions, resident = _select_layer_promotions(
            method,
            torch.tensor([0, 0, 0, 0, 0, 0, 0, 20]),
            allow_promotion=False,
        )
        self.assertEqual(promotions, [])
        self.assertEqual(resident.tolist(), [0, 1])
        self.assertGreater(float(_LAYER_STATES[900].ema[7]), 0.0)

    def test_promotion_schedule_fills_once_then_refreshes(self) -> None:
        self.assertTrue(_is_promotion_step(1, 16, 1, 16))
        self.assertTrue(_is_promotion_step(16, 16, 1, 16))
        self.assertFalse(_is_promotion_step(17, 16, 1, 16))
        self.assertTrue(_is_promotion_step(32, 16, 1, 16))

        self.assertTrue(_is_promotion_step(8, 16, 2, 16))
        self.assertFalse(_is_promotion_step(9, 16, 2, 16))

    def test_residency_restore_counts_one_slot_per_change(self) -> None:
        backup = (
            torch.empty((16, 4, 3), dtype=torch.bfloat16),
            torch.empty((16, 5, 2), dtype=torch.bfloat16),
        )
        self.assertEqual(_resident_slot_bytes(backup), (4 * 3 + 5 * 2) * 2)

    def test_residency_restore_validates_all_mapping_copies(self) -> None:
        initial = torch.tensor([0, 3], dtype=torch.int32)
        mask = torch.tensor([True, False, False, True])
        logical_to_gpu = torch.tensor([0, -1, -1, 1], dtype=torch.int32)
        method = SimpleNamespace(
            kt_config=SimpleNamespace(layer_idx=901),
            gpu_index_to_logical=initial.clone(),
            gpu_experts_mask=mask.clone(),
            logical_to_gpu_index=logical_to_gpu.clone(),
            gpu_experts_mask_cuda=mask.clone(),
            logical_to_gpu_index_cuda=logical_to_gpu.clone(),
            wrapper=SimpleNamespace(gpu_experts_mask=mask.clone()),
        )
        _INITIAL_RESIDENT[901] = initial
        _validate_restored_residency([method], [])

        method.logical_to_gpu_index_cuda[3] = -1
        with self.assertRaisesRegex(RuntimeError, "logical_to_gpu_index_cuda"):
            _validate_restored_residency([method], [])


if __name__ == "__main__":
    unittest.main()
