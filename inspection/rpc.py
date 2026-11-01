"""Small synchronous MessagePack-RPC transport for AirSim 1.8.1.

Uses current msgpack instead of the old AirSim SDK's Tornado transport.
Only loopback connections are accepted. Binary image strings remain bytes.
"""
import socket
import time

import msgpack


def keys_to_text(value):
    if isinstance(value, dict):
        return {(k.decode() if isinstance(k, bytes) else k): keys_to_text(v)
                for k, v in value.items()}
    if isinstance(value, list):
        return [keys_to_text(v) for v in value]
    return value


class AirSimRPC:
    def __init__(self, port=41451, timeout=10):
        self.socket = socket.create_connection(('127.0.0.1', port), timeout)
        self.timeout = timeout
        self.unpacker = msgpack.Unpacker(raw=True, max_buffer_size=128 * 1024 * 1024)
        self.sequence = 0

    def close(self):
        self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def call(self, method, *params, timeout=None):
        self.sequence += 1
        request_id = self.sequence
        deadline = time.monotonic() + (timeout or self.timeout)
        self.socket.settimeout(max(0.01, deadline - time.monotonic()))
        self.socket.sendall(msgpack.packb([0, request_id, method, list(params)], use_bin_type=True))
        while True:
            for response in self.unpacker:
                if not isinstance(response, list) or len(response) != 4 or response[:2] != [1, request_id]:
                    raise RuntimeError('Unexpected AirSim RPC response; reconnect before retrying.')
                if response[2] is not None:
                    raise RuntimeError(f'{method}: {response[2]!r}')
                return keys_to_text(response[3])
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f'{method} exceeded its timeout; outcome is unknown.')
            self.socket.settimeout(remaining)
            chunk = self.socket.recv(65536)
            if not chunk:
                raise ConnectionError('AirSim closed the connection.')
            self.unpacker.feed(chunk)
