"""Run a ConTrack checkpoint against the independent h2r_bench evaluator."""

import argparse
from pathlib import Path
import socket
import time

import torch

from .controller import Controller
from .protocol import receive, send


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--socket', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()
    if not 0 < args.timeout < float('inf'):
        parser.error('--timeout must be positive and finite')
    torch.set_num_threads(1)
    controller = Controller(args.prepared, args.checkpoint, args.device)
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(args.timeout)
        deadline = time.monotonic()+args.timeout
        while True:
            try:
                connection.connect(args.socket)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Evaluator socket unavailable')
                time.sleep(0.1)
        send(connection, controller.start(receive(connection)))
        while True:
            response = controller.step(receive(connection))
            if response is None:
                break
            send(connection, response)


if __name__ == '__main__':
    main()
