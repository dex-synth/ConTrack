"""Length-prefixed h2r-evaluation-v1 JSON over a persistent Unix socket."""

import json
import struct

MAX_BYTES = 32*1024*1024


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f'Invalid JSON number: {value}')


def receive(connection):
    def exact(size):
        blocks = bytearray()
        while len(blocks) < size:
            block = connection.recv(size-len(blocks))
            if not block:
                raise EOFError('Evaluator disconnected')
            blocks.extend(block)
        return bytes(blocks)
    size = struct.unpack('!I', exact(4))[0]
    if not 0 < size <= MAX_BYTES:
        raise ValueError('Invalid protocol message size')
    return json.loads(exact(size), object_pairs_hook=_unique, parse_constant=_constant)


def send(connection, message):
    encoded = json.dumps(message, allow_nan=False, separators=(',', ':')).encode()
    if len(encoded) > MAX_BYTES:
        raise ValueError('Protocol message too large')
    connection.sendall(struct.pack('!I', len(encoded))+encoded)
