#!/usr/bin/env python3
"""Standalone reproducer for the ``wp.Mesh`` destructor / BVH allocator
race under a prefetch-shaped multithreaded SDF workload.

Strips out every bit of physicsnemo machinery -- the only imports are
``torch`` and ``warp``.  The only Warp call site is a verbatim copy of
``physicsnemo.nn.functional.geometry.sdf.signed_distance_field_impl``'s
body, *minus* the process-wide ``threading.Lock`` that currently masks
the bug in production.

Worker model is the same shape as
``physicsnemo.datapipes.DataLoader._iter_prefetch``:

- A ``ThreadPoolExecutor`` with ``--threads`` workers (default 8, the
  recipe's ``num_workers``).
- A shared pool of ``--num-streams`` ``torch.cuda.Stream`` objects
  (default 4, the recipe's ``num_streams``).
- Each task picks a stream by round-robin index, enters
  ``with torch.cuda.stream(s):``, then runs the SDF cycle on that
  stream.  Multiple workers can therefore be inside
  ``torch.cuda.stream(s)`` on the *same* ``s`` concurrently -- that's
  the production race surface.

Mitigations the SDF body carries (matching physicsnemo HEAD):

- ``warp_stream_guard``-equivalent: ``stream_from_torch`` +
  ``ScopedStream`` + exit-time ``record_event`` / ``wait_event`` on
  the warp internal stream.
- ``mesh.record_stream(stream)`` after the ``wp.launch`` (records a
  CUDA event so ``Mesh.__del__`` waits for the BVH kernel to drain
  before freeing the BVH GPU memory).
- Explicit ``del mesh, wp_*`` inside the ``ScopedStream`` block so
  ``Mesh.__del__`` runs while the warp launch stream is still scoped.

Mitigation removed:

- The process-wide ``threading.Lock`` (``_sdf_lock`` in
  ``physicsnemo``).

If this reproducer crashes, the warp-side fix is the only path to a
clean ``_sdf_lock``-less recipe.

Usage:

    # default config (8 threads, 4 streams, ~5k-tri sphere, 200k queries):
    python repro_sdf_thread_race.py

    # crank it up to provoke faster:
    python repro_sdf_thread_race.py --threads 16 --iters 4000

    # control: serialise with the same process-wide lock physicsnemo uses
    # -- expected to pass (this is the lock we want to make redundant):
    python repro_sdf_thread_race.py --use-lock

    # control: serialise to one stream -- expected to pass (no inter-
    # stream race surface):
    python repro_sdf_thread_race.py --num-streams 1

    # control: single thread -- expected to pass (no concurrent
    # destructors / allocator calls):
    python repro_sdf_thread_race.py --threads 1

Exit code: 0 if every worker completed cleanly, 1 if any worker raised
(race reproduced), 2 if no CUDA device.

Dependencies: ``torch``, ``warp-lang``.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext

import torch
import warp as wp
wp.init()


### Kernel body mirrors physicsnemo.nn.functional.geometry.sdf
### ._bvh_query_distance verbatim.  Registered once at module scope so
### the per-thread first-touch path doesn't race ``Module.load`` (warp
### upstream already locks ``Module.load`` since commit 53019f98, but
### we want to isolate the destructor / allocator race, not the
### module-load race).
@wp.kernel
def _bvh_query_distance(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3f),
    max_dist: wp.float32,
    sdf: wp.array(dtype=wp.float32),
    sdf_hit_point: wp.array(dtype=wp.vec3f),
    use_sign_winding_number: bool = False,
):
    tid = wp.tid()

    if use_sign_winding_number:
        res = wp.mesh_query_point_sign_winding_number(mesh_id, points[tid], max_dist)
    else:
        res = wp.mesh_query_point_sign_normal(mesh_id, points[tid], max_dist)

    mesh = wp.mesh_get(mesh_id)

    p0 = mesh.points[mesh.indices[3 * res.face + 0]]
    p1 = mesh.points[mesh.indices[3 * res.face + 1]]
    p2 = mesh.points[mesh.indices[3 * res.face + 2]]

    p_closest = res.u * p0 + res.v * p1 + (1.0 - res.u - res.v) * p2

    sdf[tid] = res.sign * wp.abs(wp.length(points[tid] - p_closest))
    sdf_hit_point[tid] = p_closest


def make_uv_sphere(
    n_rings: int, n_segments: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a UV-sphere mesh, vectorised.

    Same construction recipe used by physicsnemo's
    ``SignedDistanceField.make_inputs_forward`` benchmark generator -- a
    realistic-ish closed mesh sized by ``n_rings * n_segments``.  At the
    defaults (32, 80) you get ~2.5k vertices / ~5k triangles, which
    pushes a non-trivial BVH build through warp every call.
    """
    phi = torch.linspace(0, torch.pi, n_rings + 2, device=device)[1:-1]
    theta = torch.linspace(0, 2 * torch.pi, n_segments + 1, device=device)[:-1]
    phi_g, theta_g = torch.meshgrid(phi, theta, indexing="ij")

    sin_phi = phi_g.sin()
    ring_points = torch.stack(
        [sin_phi * theta_g.cos(), sin_phi * theta_g.sin(), phi_g.cos()],
        dim=-1,
    ).reshape(-1, 3)

    vertices = torch.cat(
        [
            torch.tensor([[0.0, 0.0, 1.0]], device=device),
            ring_points,
            torch.tensor([[0.0, 0.0, -1.0]], device=device),
        ]
    ).to(torch.float32)

    south_idx = n_rings * n_segments + 1
    j = torch.arange(n_segments, device=device)
    j_next = (j + 1) % n_segments

    north_fan = torch.stack([torch.zeros_like(j), 1 + j, 1 + j_next], dim=1)

    r = torch.arange(n_rings - 1, device=device).unsqueeze(1)
    base = 1 + r * n_segments
    p00, p01 = base + j, base + j_next
    p10, p11 = base + n_segments + j, base + n_segments + j_next
    body_tris = torch.stack(
        [
            torch.stack([p00, p10, p11], dim=-1),
            torch.stack([p00, p11, p01], dim=-1),
        ],
        dim=2,
    ).reshape(-1, 3)

    last = south_idx - n_segments
    south_fan = torch.stack(
        [last + j, torch.full_like(j, south_idx), last + j_next], dim=1
    )

    indices = (
        torch.cat([north_fan, body_tris, south_fan]).to(torch.int32).reshape(-1)
    )
    return vertices, indices


def signed_distance_field_no_lock(
    mesh_vertices: torch.Tensor,
    mesh_indices: torch.Tensor,
    input_points: torch.Tensor,
    *,
    use_sign_winding_number: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Verbatim copy of physicsnemo's ``signed_distance_field_impl`` body
    minus the ``with _sdf_lock:`` wrap.

    Every other mitigation physicsnemo carries today is kept:
    ``warp_stream_guard``-equivalent (inlined here so this file has no
    physicsnemo import), ``mesh.record_stream``, explicit
    ``del mesh, wp_*`` inside the ``ScopedStream`` scope.
    """
    if input_points.shape[-1] != 3:
        raise ValueError("input_points must have last dimension of size 3")

    if mesh_indices.ndim == 2:
        if mesh_indices.shape[-1] != 3:
            raise ValueError("mesh_indices 2D must be (n_faces, 3)")
        mesh_indices = mesh_indices.reshape(-1)
    elif mesh_indices.ndim != 1:
        raise ValueError("mesh_indices must be 1D or (n_faces, 3) 2D")

    input_shape = input_points.shape
    input_points = input_points.reshape(-1, 3)
    N = len(input_points)

    ### Outputs allocated on the calling thread's current torch CUDA
    ### stream -- identical to physicsnemo.
    sdf = torch.zeros(N, dtype=torch.float32, device=input_points.device)
    sdf_hit_point = torch.zeros(
        N, 3, dtype=torch.float32, device=input_points.device
    )

    mesh_vertices_f32 = mesh_vertices.to(torch.float32)
    mesh_indices_i32 = mesh_indices.to(torch.int32).contiguous()
    input_points_f32 = input_points.to(torch.float32)

    ### --- warp_stream_guard, inlined ---
    ### Pull the worker thread's current torch stream, wrap as a warp
    ### stream, drive warp on it under ScopedStream.  On exit, record a
    ### CUDA event on the (possibly-different) warp internal stream and
    ### make the torch stream wait on it -- the GPU-side ordering hook
    ### that physicsnemo's FunctionSpec.warp_stream_guard installs.
    torch_stream = torch.cuda.current_stream(input_points.device)
    wp_launch_stream = wp.stream_from_torch(torch_stream)

    with wp.ScopedStream(wp_launch_stream):
        wp.init()

        wp_vertices = wp.from_torch(mesh_vertices_f32, dtype=wp.vec3)
        wp_indices = wp.from_torch(mesh_indices_i32, dtype=wp.int32)
        wp_input_points = wp.from_torch(input_points_f32, dtype=wp.vec3)

        wp_sdf = wp.from_torch(sdf, dtype=wp.float32)
        wp_sdf_hit_point = wp.from_torch(sdf_hit_point, dtype=wp.vec3f)

        mesh = wp.Mesh(
            points=wp_vertices,
            indices=wp_indices,
            support_winding_number=use_sign_winding_number,
        )

        wp.launch(
            kernel=_bvh_query_distance,
            dim=N,
            inputs=[
                mesh.id,
                wp_input_points,
                wp.float32(1.0e8),
                wp_sdf,
                wp_sdf_hit_point,
                use_sign_winding_number,
            ],
            stream=wp_launch_stream,
        )

        ### Per-stream event-sync hook in Mesh.__del__ before the BVH
        ### frees.  Same name and semantic as torch.Tensor.record_stream.
        mesh.record_stream(wp_launch_stream)

        ### Explicit destruction inside the ScopedStream so Mesh.__del__
        ### runs while the warp launch stream is still in scope; the
        ### per-stream event sync inside __del__ drains the launch
        ### stream before the BVH free.
        del mesh, wp_vertices, wp_indices, wp_input_points, wp_sdf, wp_sdf_hit_point

    ### --- warp_stream_guard exit ---
    warp_torch_stream = torch.cuda.ExternalStream(wp_launch_stream.cuda_stream)
    event = warp_torch_stream.record_event()
    torch.cuda.current_stream(input_points.device).wait_event(event)

    return sdf.reshape(input_shape[:-1]), sdf_hit_point.reshape(input_shape)


### Optional process-wide lock used by the ``--use-lock`` control
### invocation.  This is the lock physicsnemo's
### ``signed_distance_field_impl`` carries today.
_sdf_lock = threading.Lock()


def _worker(
    worker_id: int,
    barrier: threading.Barrier,
    iters: int,
    *,
    mesh_vertices: torch.Tensor,
    mesh_indices: torch.Tensor,
    num_query_points: int,
    use_winding_number: bool,
    stream_pool: "list[torch.cuda.Stream]",
    stream_counter: "itertools.count[int]",
    use_lock: bool,
) -> tuple[int, BaseException | None]:
    """One worker thread.

    Holds at the barrier so every worker hits its first SDF call as
    close to simultaneously as Python allows, then loops ``iters`` SDF
    calls, each picking the next stream from the shared pool by
    round-robin index.

    Returns ``(worker_id, exception_or_None)``.  The first exception on
    each worker is propagated; everything else is masked because the
    CUDA context is poisoned by then and every subsequent warp / torch
    call would just resurface the same error.
    """
    device = mesh_vertices.device
    num_streams = len(stream_pool)
    lock_ctx = _sdf_lock if use_lock else nullcontext()

    try:
        barrier.wait(timeout=30.0)

        for _ in range(iters):
            ### Round-robin stream pick.  ``itertools.count`` is
            ### thread-safe under the GIL because ``next()`` is a single
            ### bytecode -- matching the recipe's stream_idx behaviour.
            stream_idx = next(stream_counter) % num_streams
            stream = stream_pool[stream_idx]

            with torch.cuda.stream(stream):
                ### Fresh query-point tensor every iter (mirrors a
                ### prefetch worker producing a fresh sample each call).
                qpoints = (
                    3.0
                    * torch.rand(
                        num_query_points, 3, device=device, dtype=torch.float32
                    )
                    - 1.5
                )

                with lock_ctx:
                    signed_distance_field_no_lock(
                        mesh_vertices,
                        mesh_indices,
                        qpoints,
                        use_sign_winding_number=use_winding_number,
                    )

        ### One sync at the end (not per-iter) so we don't drain the
        ### launch queue between iters -- we want destructor /
        ### allocator races to overlap in-flight kernels.
        torch.cuda.synchronize(device)

    except BaseException as exc:  # noqa: BLE001 -- want every error
        return worker_id, exc

    return worker_id, None


def _print_env() -> None:
    print(f"python = {sys.version.split()[0]}")
    print(f"torch  = {torch.__version__}")
    try:
        print(f"warp   = {wp.__version__}")
    except AttributeError:
        ### Some warp builds expose this via wp.config.version instead.
        try:
            print(f"warp   = {wp.config.version}")
        except AttributeError:
            print("warp   = <unknown version>")
    print(
        f"cuda   = {torch.version.cuda}  available={torch.cuda.is_available()}"
    )
    if torch.cuda.is_available():
        print(f"device = {torch.cuda.get_device_name(0)}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--threads",
        type=int,
        default=8,
        help=(
            "worker thread count.  Default 8 matches physicsnemo's "
            "DataLoader num_workers default."
        ),
    )
    p.add_argument(
        "--num-streams",
        type=int,
        default=4,
        help=(
            "shared torch.cuda.Stream pool size.  Default 4 matches "
            "physicsnemo's DataLoader num_streams default."
        ),
    )
    p.add_argument(
        "--iters",
        type=int,
        default=1000,
        help="SDF calls per worker thread",
    )
    p.add_argument(
        "--n-rings",
        type=int,
        default=32,
        help="UV-sphere ring count (mesh latitude resolution)",
    )
    p.add_argument(
        "--n-segments",
        type=int,
        default=80,
        help="UV-sphere segment count (mesh longitude resolution)",
    )
    p.add_argument(
        "--num-points",
        type=int,
        default=200_000,
        help=(
            "BVH query points per SDF call.  Default 200k matches the "
            "recipe's typical interior-mesh size."
        ),
    )
    p.add_argument(
        "--use-winding-number",
        dest="use_winding_number",
        action="store_true",
        default=True,
        help=(
            "use winding-number sign computation (default).  Matches "
            "the recipe's `use_winding_number: true`."
        ),
    )
    p.add_argument(
        "--no-winding-number",
        dest="use_winding_number",
        action="store_false",
        help="use the cheaper normal-sign computation instead",
    )
    p.add_argument(
        "--warmup",
        action="store_true",
        help=(
            "one synchronous main-thread SDF call before spawning "
            "workers, so worker first-touch doesn't race the PTX "
            "module-load.  Module.load is internally locked in warp "
            "(commit 53019f98) so this is a defensive control, not a "
            "fix for the destructor race."
        ),
    )
    p.add_argument(
        "--use-lock",
        action="store_true",
        help=(
            "wrap every SDF call in the same process-wide threading.Lock "
            "physicsnemo's signed_distance_field_impl uses today.  "
            "Expected to pass.  This is the workaround we want to make "
            "redundant with a warp-side fix."
        ),
    )
    args = p.parse_args()

    _print_env()

    if not torch.cuda.is_available():
        print("CUDA not available; this reproducer needs a GPU.", file=sys.stderr)
        return 2

    device = torch.device("cuda:0")

    ### Build the mesh once on the main thread.  All workers read the
    ### same torch tensor objects but each worker constructs its own
    ### ``wp.Mesh`` from them on its own stream -- that's the race
    ### surface (per-worker BVH construct + destroy concurrently).
    mesh_vertices, mesh_indices = make_uv_sphere(args.n_rings, args.n_segments, device)
    n_tris = mesh_indices.numel() // 3

    print(
        f"[main] threads={args.threads} num_streams={args.num_streams} "
        f"iters={args.iters} num_points={args.num_points} "
        f"use_winding_number={args.use_winding_number} use_lock={args.use_lock} "
        f"n_vertices={mesh_vertices.shape[0]} n_triangles={n_tris} "
        f"warmup={args.warmup}"
    )

    if args.warmup:
        warm_qpts = (
            3.0
            * torch.rand(args.num_points, 3, device=device, dtype=torch.float32)
            - 1.5
        )
        signed_distance_field_no_lock(
            mesh_vertices,
            mesh_indices,
            warm_qpts,
            use_sign_winding_number=args.use_winding_number,
        )
        torch.cuda.synchronize(device)
        del warm_qpts
        print("[main] warmup complete")

    stream_pool = [
        torch.cuda.Stream(device=device) for _ in range(args.num_streams)
    ]
    stream_counter = itertools.count()
    barrier = threading.Barrier(args.threads)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(
        max_workers=args.threads, thread_name_prefix="sdf_repro"
    ) as ex:
        futs = [
            ex.submit(
                _worker,
                wid,
                barrier,
                args.iters,
                mesh_vertices=mesh_vertices,
                mesh_indices=mesh_indices,
                num_query_points=args.num_points,
                use_winding_number=args.use_winding_number,
                stream_pool=stream_pool,
                stream_counter=stream_counter,
                use_lock=args.use_lock,
            )
            for wid in range(args.threads)
        ]
        results: list[tuple[int, BaseException | None]] = []
        for f in as_completed(futs):
            results.append(f.result())
    dt = time.perf_counter() - t0

    failures = [(wid, exc) for wid, exc in results if exc is not None]
    print(
        f"[main] done in {dt:.2f}s  ok={len(results) - len(failures)}  "
        f"failed={len(failures)}"
    )

    for wid, exc in sorted(failures, key=lambda t: t[0]):
        print(f"\n[worker {wid}] {type(exc).__name__}: {exc}")
        traceback.print_exception(type(exc), exc, exc.__traceback__)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
