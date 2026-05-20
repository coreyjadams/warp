# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for thread-safe stream access.

Validates per-thread stream isolation (TLS), concurrent ScopedStream /
ScopedDevice usage, kernel launches from multiple host threads, explicit
stream= cross-thread enqueue, stream create/destroy races, concurrent
module loading, and mixed scoped-context composition.

Tests are parametrized over (num_threads, num_streams) to cover:
- M=N  (1:1 -- each thread owns one stream)
- M>N  (multiple threads share streams)
- M<N  (fewer threads than streams)
"""

import threading
import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

# Array size for kernel tests -- large enough to keep the GPU busy
# but small enough to avoid OOM when many threads allocate simultaneously.
ARRAY_N = 256 * 1024


@wp.kernel
def inc(a: wp.array[float]):
    tid = wp.tid()
    a[tid] = a[tid] + 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_threaded(worker_fn, num_threads, barrier=None):
    """Spawn *num_threads* threads running *worker_fn(worker_id)*.

    Returns (results, errors) where both are lists collected from each
    thread via thread-safe append.  *barrier* is passed through to the
    worker if not None (created automatically when omitted).
    """
    if barrier is None:
        barrier = threading.Barrier(num_threads)

    results = []
    errors = []
    results_lock = threading.Lock()
    errors_lock = threading.Lock()

    def _target(wid):
        try:
            val = worker_fn(wid, barrier)
            with results_lock:
                results.append((wid, val))
        except Exception as exc:
            with errors_lock:
                errors.append((wid, exc))

    threads = [threading.Thread(target=_target, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return results, errors


def _assert_no_errors(test, errors):
    """Fail the test case if any worker thread raised an exception."""
    if errors:
        msgs = [f"  worker {wid}: {type(exc).__name__}: {exc}" for wid, exc in errors]
        test.fail("Worker thread(s) raised exceptions:\n" + "\n".join(msgs))


# ---------------------------------------------------------------------------
# 1. Per-thread stream isolation
# ---------------------------------------------------------------------------


def test_thread_local_stream_isolation(test, device, num_threads, num_streams):
    device = wp.get_device(device)
    streams = [wp.Stream(device) for _ in range(num_streams)]
    main_stream = device.stream

    def worker(wid, barrier):
        s = streams[wid % num_streams]
        barrier.wait()
        device.set_stream(s)
        observed = device.stream
        return observed is s

    results, errors = _run_threaded(worker, num_threads)
    _assert_no_errors(test, errors)

    # Every worker must have seen its own stream
    for wid, ok in results:
        test.assertTrue(ok, f"Worker {wid} did not observe its assigned stream")

    # Main thread's stream must be unchanged
    test.assertIs(device.stream, main_stream, "Main thread stream was mutated")


# ---------------------------------------------------------------------------
# 2. Concurrent ScopedStream
# ---------------------------------------------------------------------------


def test_concurrent_scoped_stream(test, device, num_threads, num_streams):
    device = wp.get_device(device)
    streams = [wp.Stream(device) for _ in range(num_streams)]
    main_stream = device.stream
    num_iters = 10

    # One array per thread so there's no data race on array contents
    arrays = [wp.zeros(ARRAY_N, dtype=float, device=device) for _ in range(num_threads)]

    def worker(wid, barrier):
        s = streams[wid % num_streams]
        a = arrays[wid]
        barrier.wait()
        with wp.ScopedStream(s):
            for _ in range(num_iters):
                wp.launch(inc, dim=a.size, inputs=[a], stream=s)
        wp.synchronize_stream(s)

    results, errors = _run_threaded(worker, num_threads)
    _assert_no_errors(test, errors)

    for wid in range(num_threads):
        expected = np.full(ARRAY_N, float(num_iters), dtype=np.float32)
        np.testing.assert_allclose(
            arrays[wid].numpy(), expected, err_msg=f"Worker {wid} array mismatch"
        )

    test.assertIs(device.stream, main_stream, "Main thread stream was mutated")


# ---------------------------------------------------------------------------
# 3. Concurrent ScopedDevice
# ---------------------------------------------------------------------------


def test_concurrent_scoped_device(test, device, num_threads):
    device = wp.get_device(device)
    main_default = wp._src.context.runtime.default_device

    def worker(wid, barrier):
        barrier.wait()
        with wp.ScopedDevice(device):
            observed = wp._src.context.runtime.default_device
            return observed == device

    results, errors = _run_threaded(worker, num_threads)
    _assert_no_errors(test, errors)

    for wid, ok in results:
        test.assertTrue(ok, f"Worker {wid} did not see the correct default device")

    test.assertEqual(
        wp._src.context.runtime.default_device,
        main_default,
        "Main thread default device was mutated",
    )


# ---------------------------------------------------------------------------
# 4. Concurrent kernel launches
# ---------------------------------------------------------------------------


def test_concurrent_kernel_launch(test, device, num_threads, num_streams):
    device = wp.get_device(device)
    streams = [wp.Stream(device) for _ in range(num_streams)]
    num_iters = 10

    arrays = [wp.zeros(ARRAY_N, dtype=float, device=device) for _ in range(num_threads)]

    def worker(wid, barrier):
        s = streams[wid % num_streams]
        a = arrays[wid]
        barrier.wait()
        with wp.ScopedStream(s):
            for _ in range(num_iters):
                wp.launch(inc, dim=a.size, inputs=[a])
        wp.synchronize_stream(s)

    results, errors = _run_threaded(worker, num_threads)
    _assert_no_errors(test, errors)

    expected = np.full(ARRAY_N, float(num_iters), dtype=np.float32)
    for wid in range(num_threads):
        np.testing.assert_allclose(
            arrays[wid].numpy(), expected, err_msg=f"Worker {wid} array mismatch"
        )


# ---------------------------------------------------------------------------
# 5. Explicit stream= cross-thread launch
# ---------------------------------------------------------------------------


def test_explicit_stream_cross_thread(test, device, num_threads, num_streams):
    device = wp.get_device(device)
    streams = [wp.Stream(device) for _ in range(num_streams)]
    num_iters = 10

    # One array per *stream* -- multiple threads may target the same array
    arrays = [wp.zeros(ARRAY_N, dtype=float, device=device) for _ in range(num_streams)]

    def worker(wid, barrier):
        sid = wid % num_streams
        s = streams[sid]
        a = arrays[sid]
        barrier.wait()
        for _ in range(num_iters):
            wp.launch(inc, dim=a.size, inputs=[a], stream=s)

    results, errors = _run_threaded(worker, num_threads)
    _assert_no_errors(test, errors)

    # Synchronize all streams before reading back
    for s in streams:
        wp.synchronize_stream(s)

    # Each stream's array was incremented by (threads_per_stream * num_iters)
    for sid in range(num_streams):
        threads_on_stream = sum(1 for t in range(num_threads) if t % num_streams == sid)
        expected_val = float(threads_on_stream * num_iters)
        expected = np.full(ARRAY_N, expected_val, dtype=np.float32)
        np.testing.assert_allclose(
            arrays[sid].numpy(), expected, err_msg=f"Stream {sid} array mismatch"
        )


# ---------------------------------------------------------------------------
# 6. Stream create/destroy under concurrency
# ---------------------------------------------------------------------------


def test_concurrent_stream_create_destroy(test, device, num_threads, streams_per_thread):
    device = wp.get_device(device)

    def worker(wid, barrier):
        barrier.wait()
        for _ in range(streams_per_thread):
            s = wp.Stream(device)
            del s

    results, errors = _run_threaded(worker, num_threads)
    _assert_no_errors(test, errors)


# ---------------------------------------------------------------------------
# 7. Concurrent module loading
# ---------------------------------------------------------------------------


def test_concurrent_module_load(test, device, num_threads):
    device = wp.get_device(device)

    # Use a unique-module kernel so that each call to load_module must
    # actually compile on the first request.
    @wp.kernel(module="unique")
    def _load_test_kernel(x: wp.array[float]):
        tid = wp.tid()
        x[tid] = float(tid)

    module = _load_test_kernel.module

    def worker(wid, barrier):
        barrier.wait()
        module.load(device)
        return True

    results, errors = _run_threaded(worker, num_threads)
    _assert_no_errors(test, errors)

    for wid, ok in results:
        test.assertTrue(ok, f"Worker {wid} failed to load module")


# ---------------------------------------------------------------------------
# 8. Mixed ScopedDevice + ScopedStream
# ---------------------------------------------------------------------------


def test_mixed_scoped_contexts(test, device, num_threads, num_streams):
    device = wp.get_device(device)
    streams = [wp.Stream(device) for _ in range(num_streams)]
    num_iters = 10
    main_stream = device.stream
    main_default = wp._src.context.runtime.default_device

    arrays = [wp.zeros(ARRAY_N, dtype=float, device=device) for _ in range(num_threads)]

    def worker(wid, barrier):
        s = streams[wid % num_streams]
        a = arrays[wid]
        barrier.wait()
        with wp.ScopedDevice(device):
            with wp.ScopedStream(s):
                for _ in range(num_iters):
                    wp.launch(inc, dim=a.size, inputs=[a])
        wp.synchronize_stream(s)

    results, errors = _run_threaded(worker, num_threads)
    _assert_no_errors(test, errors)

    expected = np.full(ARRAY_N, float(num_iters), dtype=np.float32)
    for wid in range(num_threads):
        np.testing.assert_allclose(
            arrays[wid].numpy(), expected, err_msg=f"Worker {wid} array mismatch"
        )

    test.assertIs(device.stream, main_stream, "Main thread stream was mutated")
    test.assertEqual(
        wp._src.context.runtime.default_device,
        main_default,
        "Main thread default device was mutated",
    )


# ===========================================================================
# Test registration
# ===========================================================================

devices = get_selected_cuda_test_devices()


class TestStreamThreads(unittest.TestCase):
    pass


# 1. Per-thread stream isolation -- (M, N) configs
for m, n in [(4, 4), (4, 2), (2, 4)]:
    add_function_test(
        TestStreamThreads,
        f"test_thread_local_stream_isolation_{m}t_{n}s",
        test_thread_local_stream_isolation,
        devices=devices,
        num_threads=m,
        num_streams=n,
    )

# 2. Concurrent ScopedStream
for m, n in [(4, 4), (4, 1)]:
    add_function_test(
        TestStreamThreads,
        f"test_concurrent_scoped_stream_{m}t_{n}s",
        test_concurrent_scoped_stream,
        devices=devices,
        num_threads=m,
        num_streams=n,
    )

# 3. Concurrent ScopedDevice
for m in [4, 8]:
    add_function_test(
        TestStreamThreads,
        f"test_concurrent_scoped_device_{m}t",
        test_concurrent_scoped_device,
        devices=devices,
        num_threads=m,
    )

# 4. Concurrent kernel launches
for m, n in [(4, 4), (8, 2), (2, 4)]:
    add_function_test(
        TestStreamThreads,
        f"test_concurrent_kernel_launch_{m}t_{n}s",
        test_concurrent_kernel_launch,
        devices=devices,
        num_threads=m,
        num_streams=n,
    )

# 5. Explicit stream= cross-thread launch
for m, n in [(4, 2), (4, 4)]:
    add_function_test(
        TestStreamThreads,
        f"test_explicit_stream_cross_thread_{m}t_{n}s",
        test_explicit_stream_cross_thread,
        devices=devices,
        num_threads=m,
        num_streams=n,
    )

# 6. Stream create/destroy under concurrency
add_function_test(
    TestStreamThreads,
    "test_concurrent_stream_create_destroy_8t_20k",
    test_concurrent_stream_create_destroy,
    devices=devices,
    num_threads=8,
    streams_per_thread=20,
)

# 7. Concurrent module loading
add_function_test(
    TestStreamThreads,
    "test_concurrent_module_load_4t",
    test_concurrent_module_load,
    devices=devices,
    num_threads=4,
)

# 8. Mixed ScopedDevice + ScopedStream
for m, n in [(4, 4), (4, 2)]:
    add_function_test(
        TestStreamThreads,
        f"test_mixed_scoped_contexts_{m}t_{n}s",
        test_mixed_scoped_contexts,
        devices=devices,
        num_threads=m,
        num_streams=n,
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
