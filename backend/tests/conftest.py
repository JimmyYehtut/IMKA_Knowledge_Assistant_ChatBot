"""Shared test setup.

Unit tests never reach the network: OpenAI, Qdrant and Postgres are replaced
with fakes, and any outbound connection that slips through fails loudly
instead of costing money or depending on the dev stack. Tests that do need
the running stack are marked `live` and skipped unless IMKA_LIVE=1.
"""
import ipaddress
import os
import socket

import pytest

# Modules read these at import time; give them harmless values before any import.
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")
os.environ.setdefault("JWT_SECRET", "test-secret")
# No LangSmith tracing from tests: modules call load_dotenv(), which never overrides a
# variable that is already set, so these win over a developer .env that enables tracing.
# Otherwise every chain invoke queues a trace upload that the network guard blocks, and
# the tracer's background thread keeps the test process alive retrying it.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"


def _is_loopback(address) -> bool:
    host = address[0] if isinstance(address, tuple) else address
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


@pytest.fixture(autouse=True)
def _no_outbound_network(request, monkeypatch):
    if request.node.get_closest_marker("live"):
        return
    real_connect = socket.socket.connect

    def guarded_connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6) and not _is_loopback(address):
            raise RuntimeError(f"Unit test tried to open a network connection to {address!r}; mock it instead.")
        return real_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


def pytest_collection_modifyitems(config, items):
    if os.getenv("IMKA_LIVE") == "1":
        return
    skip_live = pytest.mark.skip(reason="needs the running stack; set IMKA_LIVE=1")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
