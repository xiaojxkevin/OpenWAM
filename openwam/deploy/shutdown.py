"""Bounded signal handling for the standalone GPU serving CLI.

Only the CLI uses this helper. Library servers retain their normal lifecycle.
After an operator requests exit, let asyncio close sockets, then exit without
running native-extension destructors during Python interpreter finalization.
"""

import logging
import os
import signal
import sys
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)
SHUTDOWN_TIMEOUT_S = 5.0


def run_serving_cli(target: Callable[[], None], timeout_s: float = SHUTDOWN_TIMEOUT_S) -> None:
    """Handle SIGINT/SIGTERM, bounding executor/socket cleanup by *timeout_s*.

    asyncio's default executor cannot cancel a running CUDA inference thread.
    A daemon watchdog bounds that wait; a second signal exits immediately.
    os._exit is intentional here: this serving process owns GPU resources and
    must not re-enter interpreter finalization after a requested shutdown.
    """
    exit_code = None
    watchdog = None
    previous = {}

    def force_exit():
        os._exit(exit_code)

    def request_exit(signum, _frame):
        nonlocal exit_code, watchdog
        if exit_code is not None:
            force_exit()
        exit_code = 128 + signum
        # Arm before touching logging or beginning asyncio cancellation.
        watchdog = threading.Timer(timeout_s, force_exit)
        watchdog.daemon = True
        watchdog.start()
        logger.info(
            "Received %s; stopping server (maximum %.1fs, signal again to exit immediately)",
            signal.Signals(signum).name,
            timeout_s,
        )
        raise KeyboardInterrupt

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, request_exit)
        try:
            target()
        except KeyboardInterrupt:
            if exit_code is None:
                # Also cover a programmatically raised KeyboardInterrupt.
                exit_code = 130
                watchdog = threading.Timer(timeout_s, force_exit)
                watchdog.daemon = True
                watchdog.start()
    finally:
        if exit_code is not None:
            logger.info("Server stopped; exiting with code %d", exit_code)
            # Preserve terminal output but keep the watchdog armed if flushing
            # or logging blocks. The OS releases sockets/CUDA memory on exit.
            sys.stdout.flush()
            sys.stderr.flush()
            force_exit()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
