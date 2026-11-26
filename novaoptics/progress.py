"""Progress reporting and cooperative cancellation for long pipeline runs.

Front ends (the GUI) register listeners that receive every recorded snapshot, and can
request cancellation; the optimizer checks for it between iterations, so a run stops
within one Levenberg-Marquardt step on every worker thread.
"""

from __future__ import annotations

import threading

_LISTENERS: list = []
_CANCEL = threading.Event()


class Cancelled(Exception):
    """Raised inside the pipeline when a cancel was requested."""


def add_listener(fn):
    """fn(recorder, snapshot) is called (from worker threads) for every snapshot."""
    _LISTENERS.append(fn)


def remove_listener(fn):
    if fn in _LISTENERS:
        _LISTENERS.remove(fn)


def emit(rec, snap):
    for fn in list(_LISTENERS):
        fn(rec, snap)


def request_cancel():
    _CANCEL.set()


def reset_cancel():
    _CANCEL.clear()


def check():
    if _CANCEL.is_set():
        raise Cancelled()
