"""Experiment-specific Python startup module.

KT TBO now prepares local decode children inside the Qwen3.5 model, so no
global scheduler monkeypatch is required.  Keeping this module intentionally
empty also ensures ordinary TP batch semantics are preserved.
"""
