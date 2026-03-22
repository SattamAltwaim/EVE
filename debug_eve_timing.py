#!/usr/bin/env python3
"""
debug_eve_timing.py — Full-fidelity EVE timing diagnostic.

Runs two scenarios back-to-back on identical hardware:

  A) ISOLATED  — EVE phases run from a clean allocator (no prior backward).
                 Matches what Experiment 5 in the notebook measures.

  B) REAL      — EVE phases run after a genuine forward+backward pass,
                 exactly as in training.  Reveals allocator-fragmentation cost.

Both scenarios time the same internal phases with torch.cuda.synchronize()
and log CUDA memory stats so the fragmentation hypothesis can be confirmed
or ruled out.
"""

from __future__ import annotations
import importlib, math, os, subprocess, sys, time
from typing import Dict, List, Optional, Tuple

# ── 0. Repo / version info ────────────────────────────────────────────────────

def _git(cmd: List[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return "(unavailable)"


def print_header() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    git_hash   = _git(["git", "-C", script_dir, "log", "--oneline", "-3"])
    git_branch = _git(["git", "-C", script_dir, "branch", "--show-current"])
    eve_file   = None
    for p in sys.path:
        candidate = os.path.join(p, "eve_optimizer", "eve.py")
        if os.path.exists(candidate):
            eve_file = candidate
            break

    print("=" * 72)
    print("  EVE TIMING DEBUG SCRIPT")
    print("=" * 72)
    print(f"  Python      : {sys.version.split()[0]}")
    print(f"  Script dir  : {script_dir}")
    print(f"  eve.py path : {eve_file or 'NOT FOUND on sys.path'}")
    print(f"  Git branch  : {git_branch}")
    print(f"  Recent commits:")
    for line in git_hash.splitlines():
        print(f"    {line}")
    print("=" * 72)


# ── 1. Imports (after path is set up) ────────────────────────────────────────

print_header()

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

if "eve_optimizer" in sys.modules:
    importlib.reload(sys.modules["eve_optimizer"])
from eve_optimizer import EVE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n  Device      : {DEVICE}")
if DEVICE == "cuda":
    print(f"  GPU         : {torch.cuda.get_device_name(0)}")
print()

# ── 2. Helpers ────────────────────────────────────────────────────────────────

def cuda_sync() -> None:
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def mem_mb() -> Tuple[float, float]:
    """Returns (allocated_MB, reserved_MB)."""
    if DEVICE != "cuda":
        return 0.0, 0.0
    return (torch.cuda.memory_allocated() / 1024**2,
            torch.cuda.memory_reserved()  / 1024**2)


def alloc_retries() -> int:
    if DEVICE != "cuda":
        return 0
    stats = torch.cuda.memory_stats()
    return stats.get("num_alloc_retries", 0)


# ── 3. Model + data ───────────────────────────────────────────────────────────

def make_model() -> nn.Module:
    m = torchvision.models.resnet18(weights=None, num_classes=10)
    return m.to(DEVICE)


def make_loader(n_samples: int = 5000, batch_size: int = 128) -> torch.utils.data.DataLoader:
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    ds = torchvision.datasets.CIFAR10(
        root="/tmp/cifar10", train=True, download=True, transform=tf
    )
    subset = torch.utils.data.Subset(ds, range(min(n_samples, len(ds))))
    return torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=(DEVICE == "cuda"),
    )


# ── 4. Per-phase timed EVE step (shared by both scenarios) ───────────────────

PHASE_NAMES = [
    "P1     moments (m, m_slow, v, v_max)",
    "P2     param backup clone",
    "P3a    L_base forward",
    "P3b    K in-place probe cycles",
    "P4     fitness + softmax + alpha_eff",
    "P5     final update",
]


def timed_eve_step(
    opt: EVE,
    model: nn.Module,
    loss_fn: nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    current_loss: float,
    timing: Dict[str, List[float]],
    mem_log: Dict[str, List[Tuple[float, float]]],
) -> None:
    """
    Replicates the K>1 path of EVE.step() with CUDA-synced timing and memory
    logging injected between every phase.  Uses the actual optimizer state
    so results are equivalent to calling opt.step() for real.
    """
    K = opt.defaults["K"]
    inp = xb
    tgt = yb

    # ── P1: moments ───────────────────────────────────────────────────────────
    m_hat_map: Dict[int, torch.Tensor] = {}
    m_hat_slow_map: Dict[int, torch.Tensor] = {}
    denom_map: Dict[int, torch.Tensor] = {}
    params_with_grad: List[Tuple[Dict, torch.Tensor]] = []

    cuda_sync(); t1_s = time.perf_counter()
    for group in opt.param_groups:
        beta1, beta2 = group["betas"]
        beta1_slow = group["beta1_slow"]
        for p in group["params"]:
            if p.grad is None:
                continue
            state = opt.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["m"] = torch.zeros_like(p)
                state["m_slow"] = torch.zeros_like(p)
                state["v"] = torch.zeros_like(p)
                state["v_max"] = torch.zeros_like(p)
            state["step"] += 1
            grad = p.grad
            m, m_slow = state["m"], state["m_slow"]
            v, v_max = state["v"], state["v_max"]

            m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            m_slow.mul_(beta1_slow).add_(grad, alpha=1.0 - beta1_slow)
            v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            step = state["step"]
            bc1 = 1.0 - beta1 ** step
            bc1_slow = 1.0 - beta1_slow ** step
            bc2_sqrt = math.sqrt(1.0 - beta2 ** step)

            torch.maximum(v_max, v, out=v_max)

            ptr = p.data_ptr()
            m_hat_map[ptr] = m / bc1
            m_hat_slow_map[ptr] = m_slow / bc1_slow
            denom_map[ptr] = (v_max.sqrt() / bc2_sqrt).add_(group["eps"])

            params_with_grad.append((group, p))
    cuda_sync(); t1 = time.perf_counter()
    timing["P1     moments (m, m_slow, v, v_max)"].append((t1 - t1_s) * 1e3)
    mem_log["after_P1"].append(mem_mb())

    # ── P2: alpha grid + param backup ─────────────────────────────────────────
    alphas = [k / (K - 1) for k in range(K)]

    cuda_sync(); t2_s = time.perf_counter()
    saved: Dict[int, torch.Tensor] = {
        p.data_ptr(): p.data.clone() for _, p in params_with_grad
    }
    cuda_sync(); t2 = time.perf_counter()
    timing["P2     param backup clone"].append((t2 - t2_s) * 1e3)
    mem_log["after_P2"].append(mem_mb())

    # ── P3a: L_base forward ───────────────────────────────────────────────────
    was_training = model.training
    model.eval()

    cuda_sync(); t3a_s = time.perf_counter()
    with torch.no_grad():
        L_base = loss_fn(model(inp), tgt)
    cuda_sync(); t3a = time.perf_counter()
    timing["P3a    L_base forward"].append((t3a - t3a_s) * 1e3)
    mem_log["after_P3a"].append(mem_mb())

    # ── P3b: K in-place probe cycles ──────────────────────────────────────────
    probe_losses: List[torch.Tensor] = []

    cuda_sync(); t3b_s = time.perf_counter()
    with torch.no_grad():
        for k in range(K):
            a_k = alphas[k]
            for group, p in params_with_grad:
                ptr = p.data_ptr()
                d_k = (
                    m_hat_map[ptr].mul(-(1.0 - a_k))
                    .add_(m_hat_slow_map[ptr], alpha=-a_k)
                ).div_(denom_map[ptr])
                p.data.add_(d_k, alpha=group["lr"])
            probe_losses.append(loss_fn(model(inp), tgt))
            for _, p in params_with_grad:
                p.data.copy_(saved[p.data_ptr()])
    losses = torch.stack(probe_losses)
    cuda_sync(); t3b = time.perf_counter()
    timing["P3b    K in-place probe cycles"].append((t3b - t3b_s) * 1e3)
    mem_log["after_P3b"].append(mem_mb())

    model.train(was_training)

    # ── P4: fitness + softmax + alpha_eff ─────────────────────────────────────
    cuda_sync(); t4_s = time.perf_counter()
    fitness = L_base - losses
    beta_sel = opt.defaults["beta_sel"]
    weights = torch.softmax(beta_sel * fitness, dim=0)
    alphas_t = torch.tensor(alphas, device=weights.device, dtype=weights.dtype)
    alpha_eff = (weights * alphas_t).sum().item()
    cuda_sync(); t4 = time.perf_counter()
    timing["P4     fitness + softmax + alpha_eff"].append((t4 - t4_s) * 1e3)
    mem_log["after_P4"].append(mem_mb())

    # ── P5: final update ──────────────────────────────────────────────────────
    cuda_sync(); t5_s = time.perf_counter()
    for group, p in params_with_grad:
        ptr = p.data_ptr()
        lr, wd = group["lr"], group["weight_decay"]
        d_final = (
            m_hat_map[ptr].mul(-(1.0 - alpha_eff))
            .add_(m_hat_slow_map[ptr], alpha=-alpha_eff)
        ).div_(denom_map[ptr])
        with torch.no_grad():
            if wd != 0.0:
                p.data.mul_(1.0 - lr * wd)
            p.data.add_(d_final, alpha=lr)
    cuda_sync(); t5 = time.perf_counter()
    timing["P5     final update"].append((t5 - t5_s) * 1e3)
    mem_log["after_P5"].append(mem_mb())


# ── 5. Print helpers ──────────────────────────────────────────────────────────

def print_timing_table(
    label: str,
    timing: Dict[str, List[float]],
    mem_log: Dict[str, List[Tuple[float, float]]],
    retries_before: int,
    retries_after: int,
) -> None:
    total = sum(np.mean(v) for v in timing.values())
    print(f"\n{'─'*72}")
    print(f"  {label}")
    print(f"{'─'*72}")
    print(f"  {'Phase':<42s} {'Mean ms':>8s} {'p50':>8s} {'p95':>8s} {'%':>6s}  {'Alloc MB':>9s}")
    print(f"  {'─'*42} {'─'*8} {'─'*8} {'─'*8} {'─'*6}  {'─'*9}")

    mem_keys = list(mem_log.keys())
    for i, (phase, vals) in enumerate(timing.items()):
        a = np.array(vals)
        mkey = mem_keys[i] if i < len(mem_keys) else None
        alloc_mb = np.mean([m[0] for m in mem_log[mkey]]) if mkey else 0.0
        print(f"  {phase:<42s} {a.mean():8.2f} {np.median(a):8.2f} "
              f"{np.percentile(a,95):8.2f} {100*a.mean()/total:5.1f}%  "
              f"{alloc_mb:8.1f}")

    print(f"  {'─'*42} {'─'*8}")
    print(f"  {'TOTAL EVE phases':<42s} {total:8.2f}")
    print(f"\n  CUDA alloc retries  before EVE: {retries_before}")
    print(f"  CUDA alloc retries  after EVE : {retries_after}")
    print(f"  New retries during EVE        : {retries_after - retries_before}")


# ── 6. Scenario A: ISOLATED (clean allocator, no prior backward) ──────────────

def run_isolated(loader, loss_fn, n_warmup=5, n_measure=20):
    print("\n" + "="*72)
    print("  SCENARIO A: ISOLATED (no prior backward — matches Exp5)")
    print("="*72)

    torch.manual_seed(42)
    model = make_model()
    opt   = EVE(model.parameters(), lr=1e-3, K=4,
                record_diagnostics=False)

    timing  = {k: [] for k in PHASE_NAMES}
    mem_log = {f"after_{k.split()[0]}": [] for k in PHASE_NAMES}

    data_iter = iter(loader)
    retries_total_before = 0
    retries_total_after  = 0

    for step_idx in range(n_warmup + n_measure):
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            xb, yb = next(data_iter)
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        model.train()
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        current_loss = loss.item()

        if step_idx < n_warmup:
            opt.step(model=model, loss_fn=loss_fn, data=(xb, yb),
                     current_loss=current_loss)
            continue

        torch.cuda.empty_cache()

        rb = alloc_retries()
        timed_eve_step(opt, model, loss_fn, xb, yb, current_loss, timing, mem_log)
        ra = alloc_retries()
        retries_total_before += rb
        retries_total_after  += ra

    print_timing_table(
        "ISOLATED — fresh cache before each step",
        timing, mem_log,
        retries_total_before, retries_total_after,
    )
    return timing


# ── 7. Scenario B: REAL (EVE step immediately after backward) ────────────────

def run_real(loader, loss_fn, n_warmup=5, n_measure=20):
    print("\n" + "="*72)
    print("  SCENARIO B: REAL (EVE runs right after backward — actual training)")
    print("="*72)

    torch.manual_seed(42)
    model = make_model()
    opt   = EVE(model.parameters(), lr=1e-3, K=4,
                record_diagnostics=False)

    timing  = {k: [] for k in PHASE_NAMES}
    mem_log = {f"after_{k.split()[0]}": [] for k in PHASE_NAMES}

    batch_wall_times: List[float] = []
    data_iter = iter(loader)
    retries_total_before = 0
    retries_total_after  = 0

    for step_idx in range(n_warmup + n_measure):
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            xb, yb = next(data_iter)
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        cuda_sync(); t_batch_start = time.perf_counter()

        model.train()
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        current_loss = loss.item()

        if step_idx < n_warmup:
            opt.step(model=model, loss_fn=loss_fn, data=(xb, yb),
                     current_loss=current_loss)
            cuda_sync()
            continue

        alloc_mb_before, reserved_mb_before = mem_mb()
        rb = alloc_retries()

        timed_eve_step(opt, model, loss_fn, xb, yb, current_loss, timing, mem_log)

        ra = alloc_retries()
        alloc_mb_after, reserved_mb_after = mem_mb()
        retries_total_before += rb
        retries_total_after  += ra

        cuda_sync(); t_batch_end = time.perf_counter()
        batch_wall_times.append((t_batch_end - t_batch_start) * 1e3)

    bwt = np.array(batch_wall_times)
    print(f"\n  Full-batch wall time (fwd+bwd+EVE):")
    print(f"    Mean={bwt.mean():.1f}ms  Median={np.median(bwt):.1f}ms  "
          f"p95={np.percentile(bwt,95):.1f}ms")
    print(f"\n  CUDA memory just BEFORE EVE step (post-backward):")
    print(f"    Allocated={alloc_mb_before:.1f} MB  Reserved={reserved_mb_before:.1f} MB")
    print(f"  CUDA memory just AFTER EVE step:")
    print(f"    Allocated={alloc_mb_after:.1f} MB  Reserved={reserved_mb_after:.1f} MB")

    print_timing_table(
        "REAL — EVE runs immediately after backward (no cache clear)",
        timing, mem_log,
        retries_total_before, retries_total_after,
    )
    return timing, bwt


# ── 8. Scenarios C & D: actual opt.step() end-to-end ─────────────────────────

def run_actual_step(loader, loss_fn, record_diagnostics: bool,
                    n_warmup=5, n_measure=20):
    """
    Calls the REAL opt.step() (not timed_eve_step) in a genuine training loop
    with CUDA-synchronized batch timing.
    """
    label_c_d = "C" if not record_diagnostics else "D"
    diag_str  = "record_diagnostics=False" if not record_diagnostics else "record_diagnostics=True"

    print("\n" + "="*72)
    print(f"  SCENARIO {label_c_d}: ACTUAL opt.step()  [{diag_str}]")
    print("="*72)

    torch.manual_seed(42)
    model = make_model()
    opt   = EVE(model.parameters(), lr=1e-3, K=4,
                record_diagnostics=record_diagnostics)

    batch_wall_times: List[float] = []
    data_iter = iter(loader)
    retries_before_total = alloc_retries()

    for step_idx in range(n_warmup + n_measure):
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            xb, yb = next(data_iter)
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        model.train()

        cuda_sync(); t0 = time.perf_counter()

        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step(model=model, loss_fn=loss_fn, data=(xb, yb),
                 current_loss=loss.item())

        cuda_sync(); batch_ms = (time.perf_counter() - t0) * 1e3

        if step_idx >= n_warmup:
            batch_wall_times.append(batch_ms)

    retries_after_total = alloc_retries()
    bwt = np.array(batch_wall_times)

    alloc_mb, reserved_mb = mem_mb()
    print(f"\n  Full-batch wall time (fwd + bwd + opt.step):")
    print(f"    Mean={bwt.mean():.1f}ms  Median={np.median(bwt):.1f}ms  "
          f"p95={np.percentile(bwt,95):.1f}ms  Max={bwt.max():.1f}ms")
    print(f"\n  CUDA memory at end:")
    print(f"    Allocated={alloc_mb:.1f} MB  Reserved={reserved_mb:.1f} MB")
    print(f"\n  CUDA alloc retries (cumulative over all steps):")
    print(f"    Before scenario : {retries_before_total}")
    print(f"    After scenario  : {retries_after_total}")
    print(f"    New retries     : {retries_after_total - retries_before_total}")

    return bwt


# ── 9. Comparison summary ─────────────────────────────────────────────────────

def print_comparison(iso_timing, real_timing, real_bwt):
    print("\n" + "="*72)
    print("  COMPARISON SUMMARY")
    print("="*72)
    print(f"  {'Phase':<42s} {'Isolated ms':>12s}  {'Real ms':>10s}  {'Slowdown':>9s}")
    print(f"  {'─'*42} {'─'*12}  {'─'*10}  {'─'*9}")
    iso_total  = sum(np.mean(v) for v in iso_timing.values())
    real_total = sum(np.mean(v) for v in real_timing.values())
    for phase in PHASE_NAMES:
        iso_ms  = np.mean(iso_timing[phase])
        real_ms = np.mean(real_timing[phase])
        sx      = real_ms / iso_ms if iso_ms > 0 else float("inf")
        print(f"  {phase:<42s} {iso_ms:12.2f}  {real_ms:10.2f}  {sx:8.2f}x")
    print(f"  {'─'*42} {'─'*12}  {'─'*10}  {'─'*9}")
    print(f"  {'TOTAL EVE phases':<42s} {iso_total:12.2f}  {real_total:10.2f}  "
          f"{real_total/iso_total:8.2f}x")

    eve_overhead_ms = real_total - iso_total
    fwd_bwd_ms      = np.mean(real_bwt) - real_total
    print(f"\n  Full batch wall time (mean)  : {np.mean(real_bwt):.1f} ms")
    print(f"  Fwd+bwd portion (estimated) : {fwd_bwd_ms:.1f} ms")
    print(f"  EVE instrumented phases     : {real_total:.1f} ms")
    print(f"  Unexplained gap             : {np.mean(real_bwt) - real_total - fwd_bwd_ms:.1f} ms")
    print(f"\n  KEY QUESTION — does the slowdown correlate with a specific phase?")
    slowdowns = {
        phase: np.mean(real_timing[phase]) / max(np.mean(iso_timing[phase]), 0.01)
        for phase in PHASE_NAMES
    }
    worst = max(slowdowns, key=slowdowns.get)
    print(f"  Most-slowed phase: [{worst}]  {slowdowns[worst]:.2f}x slower in real training")


# ── 10. Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    N_SAMPLES   = 5000
    BATCH_SIZE  = 128
    N_WARMUP    = 5
    N_MEASURE   = 20

    print(f"\n  Config: N_SAMPLES={N_SAMPLES}  BATCH_SIZE={BATCH_SIZE}  "
          f"K=4  n_warmup={N_WARMUP}  n_measure={N_MEASURE}\n")

    loss_fn = nn.CrossEntropyLoss()
    loader  = make_loader(N_SAMPLES, BATCH_SIZE)

    iso_timing = run_isolated(loader, loss_fn, N_WARMUP, N_MEASURE)

    real_timing, real_bwt = run_real(loader, loss_fn, N_WARMUP, N_MEASURE)

    print_comparison(iso_timing, real_timing, real_bwt)

    bwt_c = run_actual_step(loader, loss_fn,
                            record_diagnostics=False,
                            n_warmup=N_WARMUP, n_measure=N_MEASURE)

    bwt_d = run_actual_step(loader, loss_fn,
                            record_diagnostics=True,
                            n_warmup=N_WARMUP, n_measure=N_MEASURE)

    print("\n" + "="*72)
    print("  FINAL SUMMARY — all scenarios")
    print("="*72)
    print(f"  {'Scenario':<52s} {'Mean ms':>8s}  {'Median':>8s}  {'p95':>8s}")
    print(f"  {'─'*52} {'─'*8}  {'─'*8}  {'─'*8}")

    def row(label, arr):
        a = np.array(arr)
        print(f"  {label:<52s} {a.mean():8.1f}  {np.median(a):8.1f}  "
              f"{np.percentile(a,95):8.1f}")

    iso_total_per_step = [
        sum(iso_timing[p][i] for p in PHASE_NAMES)
        for i in range(N_MEASURE)
    ]
    real_total_per_step = [
        sum(real_timing[p][i] for p in PHASE_NAMES)
        for i in range(N_MEASURE)
    ]
    row("A  Isolated  — TOTAL EVE phases (ms)", iso_total_per_step)
    row("B  Real      — TOTAL EVE phases (ms, after backward)", real_total_per_step)
    row("B  Real      — full batch incl. fwd+bwd (ms)", real_bwt)
    row("C  opt.step  — full batch, record_diagnostics=False", bwt_c)
    row("D  opt.step  — full batch, record_diagnostics=True ", bwt_d)

    print(f"  {'─'*52} {'─'*8}  {'─'*8}  {'─'*8}")
    diag_overhead = np.mean(bwt_d) - np.mean(bwt_c)
    print(f"\n  _record_step_diagnostics overhead (D - C): {diag_overhead:+.1f} ms/step")
    if np.mean(bwt_c) > 0:
        print(f"  Overhead vs AdamW (~11ms):  C={np.mean(bwt_c)/11*100-100:.0f}%  "
              f"D={np.mean(bwt_d)/11*100-100:.0f}%")

    print("\n" + "="*72)
    print("  DONE")
    print("="*72 + "\n")
