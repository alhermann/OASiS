"""A library banner written to fd 1 must never reach the JSON-RPC stream.

Importing KratosMultiphysics prints an ASCII logo from C code straight to
file descriptor 1 — which, over stdio transport, IS the protocol channel. One
such banner corrupted the stream mid-session and killed an agent's run. The
guard at the top of server.py redirects fd 1 to stderr while handing the real
channel to sys.stdout, which the MCP transport writes through.

The test reproduces the exact mechanism without Kratos: a subprocess applies
the guard, then writes to fd 1 the C-ish way (os.write) as a banner would,
and writes protocol bytes through sys.stdout. The parent asserts the captured
stdout contains ONLY protocol bytes and the banner landed on stderr.
"""
import subprocess
import sys

CHILD = r"""
import os, sys
_real = os.dup(1)
os.dup2(2, 1)
sys.stdout = os.fdopen(_real, "w", buffering=1)
os.write(1, b" |  /   KRATOS BANNER\n")          # what an import does at C level
print('{"jsonrpc": "2.0"}')                       # what the transport does
sys.stdout.flush()
"""


def test_banner_diverted_protocol_intact():
    r = subprocess.run([sys.executable, "-c", CHILD],
                       capture_output=True, text=True, timeout=30)
    assert '{"jsonrpc": "2.0"}' in r.stdout
    assert "KRATOS BANNER" not in r.stdout, (
        "banner reached the protocol channel — the stream would corrupt")
    assert "KRATOS BANNER" in r.stderr


def test_guard_is_first_in_server_py():
    """The guard must run before any import that could load a noisy library."""
    src = open("src/server.py").read()
    guard = src.index("os.dup2(2, 1)")
    first_heavy = src.index("from mcp.server.fastmcp")
    assert guard < first_heavy


if __name__ == "__main__":
    test_banner_diverted_protocol_intact()
    test_guard_is_first_in_server_py()
    print("fd guard: banner diverted, protocol intact — PASS")
