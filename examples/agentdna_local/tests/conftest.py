"""Shared fixtures. Everything runs in a temporary local directory, and any
attempt to open a non-loopback socket fails the test."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

# examples/, whose agentdna_local/ directory is the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import agentdna_local

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    real_connect = socket.socket.connect

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in _LOOPBACK:
            raise AssertionError(f"test tried to reach the network: {address!r}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)


@pytest.fixture
def local_dir(tmp_path, monkeypatch) -> Path:
    """A fresh AGENTDNA_LOCAL_DIR with the patches installed."""
    directory = tmp_path / "agentdna-local"
    monkeypatch.setenv("AGENTDNA_LOCAL_DIR", str(directory))
    monkeypatch.delenv("AGENTDNA_LOCAL_WHITELIST", raising=False)
    agentdna_local.install(directory)
    return directory
