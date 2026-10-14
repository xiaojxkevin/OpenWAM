"""OpenPI-compatible NumPy MessagePack encoding (no pickle fallback).

Wire format mirrors openpi_client.msgpack_numpy, adapted from msgpack-numpy.
"""

import msgpack
import numpy as np


def _pack_array(value):
    if isinstance(value, (np.ndarray, np.generic)):
        if value.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype: {value.dtype}")
        if isinstance(value, np.ndarray):
            return {
                b"__ndarray__": True,
                b"data": value.tobytes(),
                b"dtype": value.dtype.str,
                b"shape": value.shape,
            }
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(f"Cannot encode {type(value).__name__}")


def _unpack_array(value):
    if b"__ndarray__" in value or b"__npgeneric__" in value:
        dtype = np.dtype(value[b"dtype"])
        if dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype: {dtype}")
        if b"__ndarray__" in value:
            return np.ndarray(buffer=value[b"data"], dtype=dtype, shape=value[b"shape"])
        return dtype.type(value[b"data"])
    return value


def packb(value):
    return msgpack.packb(value, default=_pack_array)


def unpackb(value):
    return msgpack.unpackb(value, object_hook=_unpack_array)
