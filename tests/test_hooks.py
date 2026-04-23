"""Tests for HookManager lifecycle."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from psplice.runtime.hooks import HookManager


@pytest.fixture
def mgr():
    return HookManager()


@pytest.fixture
def linear():
    return nn.Linear(8, 8, bias=False)


class TestRegisterRemove:
    def test_register_post_hook(self, mgr, linear):
        called = []

        def hook(module, input, output):
            called.append(True)

        mgr.register("key1", linear, hook, hook_type="post")
        assert mgr.hook_count() == 1
        assert mgr.has_key("key1")

        x = torch.randn(1, 8)
        linear(x)
        assert len(called) == 1

    def test_register_pre_hook(self, mgr, linear):
        called = []

        def pre(module, args):
            called.append(True)
            return args

        mgr.register("pre_key", linear, pre, hook_type="pre")
        x = torch.randn(1, 8)
        linear(x)
        assert len(called) == 1

    def test_remove_clears_hook(self, mgr, linear):
        called = []

        def hook(module, input, output):
            called.append(True)

        mgr.register("key", linear, hook)
        mgr.remove("key")
        assert not mgr.has_key("key")

        x = torch.randn(1, 8)
        linear(x)
        assert len(called) == 0

    def test_remove_nonexistent_returns_false(self, mgr):
        assert mgr.remove("does_not_exist") is False

    def test_clear_removes_all(self, mgr, linear):
        for i in range(3):
            mgr.register(f"key{i}", linear, lambda m, i, o: None)
        assert mgr.hook_count() == 3
        mgr.clear()
        assert mgr.hook_count() == 0
        assert mgr.active_keys() == []


class TestMultipleHooksPerKey:
    def test_multiple_hooks_under_one_key(self, mgr):
        m1 = nn.Linear(4, 4)
        m2 = nn.Linear(4, 4)
        mgr.register("shared", m1, lambda m, i, o: None)
        mgr.register("shared", m2, lambda m, i, o: None)
        assert mgr.hook_count() == 2
        mgr.remove("shared")
        assert mgr.hook_count() == 0

    def test_remove_one_key_leaves_others(self, mgr):
        m = nn.Linear(4, 4)
        mgr.register("a", m, lambda m, i, o: None)
        mgr.register("b", m, lambda m, i, o: None)
        mgr.remove("a")
        assert mgr.has_key("b")
        assert not mgr.has_key("a")


class TestActiveKeys:
    def test_sorted(self, mgr, linear):
        mgr.register("z_key", linear, lambda m, i, o: None)
        mgr.register("a_key", linear, lambda m, i, o: None)
        assert mgr.active_keys() == ["a_key", "z_key"]
