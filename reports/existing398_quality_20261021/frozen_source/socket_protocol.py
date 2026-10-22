#!/usr/bin/env python3
"""Small NumPy-over-TCP protocol shared by the two conda environments."""

from __future__ import annotations

import io
import socket
import struct
from collections.abc import Mapping

import numpy as np

HEADER = struct.Struct("!Q")
MAX_MESSAGE_BYTES = 256 * 1024 * 1024


def _receive_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Peer closed the socket")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_arrays(sock: socket.socket, arrays: Mapping[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    payload = buffer.getvalue()
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError(f"Message is too large: {len(payload)} bytes")
    sock.sendall(HEADER.pack(len(payload)))
    sock.sendall(payload)


def receive_arrays(sock: socket.socket) -> dict[str, np.ndarray]:
    (size,) = HEADER.unpack(_receive_exact(sock, HEADER.size))
    if size <= 0 or size > MAX_MESSAGE_BYTES:
        raise ValueError(f"Invalid message size: {size}")
    payload = _receive_exact(sock, size)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def command_message(command: str) -> dict[str, np.ndarray]:
    return {"command": np.asarray(command)}


def read_command(message: Mapping[str, np.ndarray]) -> str:
    if "command" not in message:
        raise ValueError("Request does not contain a command")
    return str(message["command"].item())

