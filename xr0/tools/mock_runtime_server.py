#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import socket
import struct
import time

import numpy as np


def recv_all(conn: socket.socket, length: int) -> bytes | None:
    data = b""
    while len(data) < length:
        packet = conn.recv(length - len(data))
        if not packet:
            return None
        data += packet
    return data


def recv_message(conn: socket.socket):
    head = recv_all(conn, 4)
    if head is None:
        return None
    size = struct.unpack(">I", head)[0]
    body = recv_all(conn, size)
    if body is None:
        return None
    return pickle.loads(body)


def send_message(conn: socket.socket, payload) -> None:
    body = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack(">I", len(body)) + body)


def make_action(args: argparse.Namespace, request_index: int) -> np.ndarray:
    action = np.zeros((args.action_length, args.action_dim), dtype=np.float32)
    if args.mode == "pulse" and request_index % 2 == 1:
        action[0, args.pulse_dim] = args.pulse_value
    return action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XR0 runtime mock server for smoke testing.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10086)
    parser.add_argument("--action-length", type=int, default=30)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--mode", choices=("zeros", "pulse"), default="zeros")
    parser.add_argument("--pulse-dim", type=int, default=6, help="Dimension to pulse in `pulse` mode.")
    parser.add_argument("--pulse-value", type=float, default=0.0)
    parser.add_argument("--sleep-ms", type=float, default=0.0, help="Artificial response latency in milliseconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(f"Mock XR0 runtime server listening on {args.host}:{args.port}")

        while True:
            conn, addr = server.accept()
            request_index = 0
            print(f"Accepted connection from {addr[0]}:{addr[1]}")
            with conn:
                while True:
                    request = recv_message(conn)
                    if request is None:
                        print("Client disconnected.")
                        break

                    request_index += 1
                    state = request.get("state")
                    state_shape = tuple(state.shape) if hasattr(state, "shape") else None
                    input_ids = request.get("input_ids")
                    token_shape = tuple(input_ids.shape) if hasattr(input_ids, "shape") else None
                    print(
                        f"[req {request_index:03d}] keys={sorted(request.keys())} "
                        f"state_shape={state_shape} token_shape={token_shape}"
                    )

                    if args.sleep_ms > 0:
                        time.sleep(args.sleep_ms / 1000.0)

                    send_message(conn, make_action(args, request_index))


if __name__ == "__main__":
    main()
