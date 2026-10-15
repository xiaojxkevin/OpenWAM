"""Process-level signal regressions; no model, GPU or robot is required."""

import select
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path

SHUTDOWN_MODULE = Path(__file__).resolve().parents[1] / "openwam/deploy/shutdown.py"
CHILD = f"""
import asyncio
import atexit
import logging
import runpy
import signal
import sys
import threading

run_cli = runpy.run_path({str(SHUTDOWN_MODULE)!r})['run_serving_cli']
logging.basicConfig(level=logging.INFO)
atexit.register(lambda: print('INTERPRETER_FINALIZATION', flush=True))
mode = sys.argv[1]
before = {{s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}}

async def serve():
    if mode == 'busy':
        started = threading.Event()
        def infer():
            started.set()
            threading.Event().wait()
        asyncio.create_task(asyncio.to_thread(infer))
        while not started.is_set():
            await asyncio.sleep(0.001)
    print('READY', flush=True)
    try:
        await asyncio.Future()
    finally:
        print('LOOP_CANCELLED', flush=True)

if mode == 'normal':
    run_cli(lambda: None)
    assert all(signal.getsignal(s) == h for s, h in before.items())
    print('NORMAL_RETURN', flush=True)
else:
    run_cli(lambda: asyncio.run(serve()), timeout_s=float(sys.argv[2]))
"""


class ServingShutdownTests(unittest.TestCase):
    def start_child(self, mode="idle", timeout=5):
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", CHILD, mode, str(timeout)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        def cleanup():
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=3)

        self.addCleanup(cleanup)
        if mode != "normal":
            self.assert_line(process, "READY")
        return process

    def assert_line(self, process, expected):
        ready, _, _ = select.select([process.stdout], [], [], 3)
        self.assertTrue(ready, f"Child did not emit {expected!r}")
        self.assertEqual(process.stdout.readline().strip(), expected)

    def test_sigint_and_sigterm_close_loop_without_interpreter_finalization(self):
        for signum in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=signum):
                process = self.start_child()
                process.send_signal(signum)
                out, err = process.communicate(timeout=3)
                self.assertEqual(process.returncode, 128 + signum, err)
                self.assertIn("LOOP_CANCELLED", out)
                self.assertNotIn("INTERPRETER_FINALIZATION", out)
                self.assertNotIn("Traceback", err)

    def test_running_inference_thread_cannot_block_exit_indefinitely(self):
        process = self.start_child("busy", timeout=0.2)
        started = time.monotonic()
        process.send_signal(signal.SIGINT)
        out, err = process.communicate(timeout=3)
        self.assertEqual(process.returncode, 130, err)
        self.assertLess(time.monotonic() - started, 3)
        self.assertIn("LOOP_CANCELLED", out)
        self.assertNotIn("INTERPRETER_FINALIZATION", out)
        self.assertNotIn("Traceback", err)

    def test_second_signal_exits_immediately_while_inference_is_running(self):
        process = self.start_child("busy", timeout=30)
        process.send_signal(signal.SIGINT)
        self.assert_line(process, "LOOP_CANCELLED")
        process.send_signal(signal.SIGINT)
        _, err = process.communicate(timeout=3)
        self.assertEqual(process.returncode, 130, err)

    def test_normal_return_restores_signal_handlers_and_finalizes_normally(self):
        process = self.start_child("normal")
        out, err = process.communicate(timeout=3)
        self.assertEqual(process.returncode, 0, err)
        self.assertIn("NORMAL_RETURN", out)
        self.assertIn("INTERPRETER_FINALIZATION", out)


if __name__ == "__main__":
    unittest.main()
