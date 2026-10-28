#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/plot.py — vẽ biểu đồ từ các file CSV benchmark (TV2 · S6)

Schema CSV (đóng băng cùng common.h, plan §3.7):
  algo,model,n,P,T,run,t_total,t_compute,t_comm,verified,rel_err,cutoff,host,commit

Cách dùng
  python3 scripts/plot.py                       # đọc results/shared và results/cluster
  python3 scripts/plot.py results/shared/x.csv  # hoặc chỉ định file / thư mục
  python3 scripts/plot.py --make-sample         # sinh CSV GIẢ LẬP để thử script
  python3 scripts/plot.py results/sample --out /tmp/figs --note "g++ 13.3 -O3 -march=native -fopenmp"

Biểu đồ (mỗi máy/host một bộ, lưu vào --out, mặc định report/figures):
  time_<host>            thời gian (median, thanh sai số min–max), cấu hình tốt nhất cho mỗi n
  gflops_<host>          GFLOPS hiệu dụng = 2n^3 / t (Strassen cũng dùng 2n^3)
  speedup_<host>         speedup theo số thread, mỗi n quét thread một panel
  efficiency_<host>      hiệu suất = speedup / p
  strong_<host>          strong scaling MPI/hybrid theo số lõi P*T
  comm_<host>_n<N>       t_compute vs t_comm của MPI/hybrid ở n = N
  cutoff_<host>          ảnh hưởng của cutoff Strassen (S3)
  strassen_vs_naive_<host>  tỉ lệ t_naive / t_strassen (> 1: Strassen thắng)
  weak_<host>            weak scaling (chỉ khi truyền --weak-n0)

Lưu ý khi đọc biểu đồ (plan §7): mỗi hình ghi máy, commit, số lần chạy, thống kê;
compiler/flags truyền bằng --note.
"""
import argparse
import glob
import math
import os
import sys
import textwrap

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLUMNS = ["algo", "model", "n", "P", "T", "run", "t_total", "t_compute",
           "t_comm", "verified", "rel_err", "cutoff", "host", "commit"]
NUMERIC = ["n", "P", "T", "run", "t_total", "t_compute", "t_comm", "rel_err", "cutoff"]
SHARED_MODELS = {"omp", "shared"}          # shared-memory đa luồng (blas ghi là "shared")
DIST_MODELS = {"mpi", "hybrid"}

MODEL_LABEL = {"seq": "seq", "omp": "OpenMP", "mpi": "MPI", "hybrid": "MPI+OpenMP",
               "shared": "đa luồng"}


# ------------------------------------------------------------------ nạp dữ liệu
def warn(msg):
    print(f"[plot] {msg}", file=sys.stderr)


def load(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += glob.glob(os.path.join(p, "**", "*.csv"), recursive=True)
        else:
            files += glob.glob(p)
    files = sorted(set(files))
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, dtype=str)
        except Exception as e:  # file hỏng/rỗng
            warn(f"bỏ qua {f}: {e}")
            continue
        if not set(COLUMNS).issubset(df.columns):
            warn(f"bỏ qua {f}: thiếu cột {sorted(set(COLUMNS) - set(df.columns))}")
            continue
        df = df[df["algo"] != "algo"]        # header bị lặp do chạy tay nối tiếp
        df["source"] = os.path.basename(f)
        frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["verified"] = df["verified"].astype(str).str.lower().eq("true")
    df = df.dropna(subset=["n", "P", "T", "t_total"])
    for c in ["n", "P", "T", "run", "cutoff"]:
        df[c] = df[c].astype(int)
    return df


def aggregate(df):
    g = df.groupby(["host", "algo", "model", "n", "P", "T", "cutoff"])
    out = g.agg(t_med=("t_total", "median"), t_min=("t_total", "min"),
                t_max=("t_total", "max"), comp=("t_compute", "median"),
                comm=("t_comm", "median"), runs=("t_total", "count"),
                err=("rel_err", "max"), ok=("verified", "all"),
                commit=("commit", lambda s: ",".join(sorted(set(s))))).reset_index()
    out["cores"] = out["P"] * out["T"]
    out["gf_med"] = 2.0 * out["n"] ** 3 / out["t_med"] / 1e9
    out["gf_lo"] = 2.0 * out["n"] ** 3 / out["t_max"] / 1e9
    out["gf_hi"] = 2.0 * out["n"] ** 3 / out["t_min"] / 1e9
    return out


def series_name(algo, model):
    return f"{algo} ({MODEL_LABEL.get(model, model)})"


def is_strassen(algo):
    return algo.startswith("strassen")


def pick_cutoff(a, cutoff):
    """Strassen: chỉ lấy cutoff mặc định; các thuật toán khác không có cutoff."""
    return a[(~a["algo"].map(is_strassen)) | (a["cutoff"] == cutoff)]


# ------------------------------------------------------------------- baseline
def baseline_time(a, host, algo, n, force_algo=None):
    """Thời gian tuần tự của CÙNG thuật toán (hoặc của force_algo), median."""
    algo = force_algo or algo
    h = a[(a["host"] == host) & (a["algo"] == algo) & (a["n"] == n)]
    seq = h[h["model"] == "seq"]
    if seq.empty:
        seq = h[(h["P"] == 1) & (h["T"] == 1)]      # blas: lấy T=1
    if seq.empty:
        return None
    return seq["t_med"].min()


# ---------------------------------------------------------------------- tiện ích
def footer(fig, host, raw, note):
    r = raw[raw["host"] == host]
    runs = r.groupby(["algo", "model", "n", "P", "T", "cutoff"]).size()
    rng = f"{runs.min()}" if runs.min() == runs.max() else f"{runs.min()}–{runs.max()} (tuỳ n, theo §3.9)"
    txt = (f"Máy: {host} · commit: {', '.join(sorted(set(r['commit'])))} · "
           f"số lần chạy mỗi cấu hình: {rng} · "
           f"thống kê: median, thanh sai số = min–max")
    if note:
        txt += f" · {note}"
    fig.text(0.01, 0.005, textwrap.fill(txt, max(40, int(fig.get_figwidth() * 17))),
             fontsize=7, va="bottom", ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))


def suptitle(fig, text, size=9):
    fig.suptitle(textwrap.fill(text, max(30, int(fig.get_figwidth() * 11))), fontsize=size)


def save(fig, out, name, fmts):
    os.makedirs(out, exist_ok=True)
    for ext in fmts:
        path = os.path.join(out, f"{name}.{ext}")
        fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {os.path.join(out, name)}.{'/'.join(fmts)}")


def errbars(med, lo, hi):
    return [np.maximum(np.array(med) - np.array(lo), 0),
            np.maximum(np.array(hi) - np.array(med), 0)]


def best_per(a, keys):
    """Với mỗi tổ hợp `keys`, lấy cấu hình (P,T) có t_med nhỏ nhất."""
    idx = a.groupby(keys)["t_med"].idxmin()
    return a.loc[idx]


# ------------------------------------------------------------------ các biểu đồ
def plot_time_gflops(a, host, raw, out, fmts, note):
    h = a[a["host"] == host]
    best = best_per(h, ["algo", "model", "n"])
    for metric, ylabel, fname, logy in [("time", "Thời gian t_total (s)", "time", True),
                                        ("gflops", "GFLOPS hiệu dụng (2n³/t)", "gflops", False)]:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for (algo, model), s in best.groupby(["algo", "model"]):
            s = s.sort_values("n")
            if metric == "time":
                y, lo, hi = s["t_med"], s["t_min"], s["t_max"]
            else:
                y, lo, hi = s["gf_med"], s["gf_lo"], s["gf_hi"]
            ax.errorbar(s["n"], y, yerr=errbars(y, lo, hi), marker="o", ms=4, capsize=2,
                        label=series_name(algo, model))
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("Kích thước ma trận n")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel.split(' (')[0]} — cấu hình tốt nhất cho mỗi n ({host})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        footer(fig, host, raw, note)
        save(fig, out, f"{fname}_{host}", fmts)


def plot_scaling(a, host, raw, out, fmts, note, metric, baseline, sweep_ns):
    h = a[(a["host"] == host) & (a["model"].isin(SHARED_MODELS))]
    ns = [n for n in sorted(h["n"].unique())
          if h[h["n"] == n]["T"].nunique() >= 3 and (not sweep_ns or n in sweep_ns)]
    if not ns:
        warn(f"{host}: chưa có dữ liệu quét thread (>= 3 giá trị T) cho shared-memory")
        return
    force = "naive_ikj" if baseline == "naive" else None
    fig, axes = plt.subplots(1, len(ns), figsize=(4.6 * len(ns), 4.4), squeeze=False)
    for ax, n in zip(axes[0], ns):
        sub = h[h["n"] == n]
        for (algo, model), s in sub.groupby(["algo", "model"]):
            base = baseline_time(a, host, algo, n, force)
            if base is None:
                continue
            s = s.sort_values("T")
            sp, lo, hi = base / s["t_med"], base / s["t_max"], base / s["t_min"]
            y, ylo, yhi = (sp, lo, hi) if metric == "speedup" else (sp / s["T"], lo / s["T"], hi / s["T"])
            ax.errorbar(s["T"], y, yerr=errbars(y, ylo, yhi), marker="o", ms=4, capsize=2,
                        label=series_name(algo, model))
        tmax = sub["T"].max()
        ts = np.array(sorted(sub["T"].unique()))
        ax.plot(ts, ts if metric == "speedup" else np.ones_like(ts), "k--", lw=1, label="lý tưởng")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ts)
        ax.set_xticklabels([str(t) for t in ts])
        if metric == "speedup":
            ax.set_yscale("log", base=2)
        ax.set_title(f"n = {n}")
        ax.set_xlabel("Số thread T")
        ax.set_ylabel("Speedup" if metric == "speedup" else "Hiệu suất = speedup / T")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    what = "cùng thuật toán, bản tuần tự" if baseline == "same" else "naive_ikj tuần tự"
    suptitle(fig, f"{'Speedup' if metric == 'speedup' else 'Hiệu suất'} so với {what} ({host})", 10)
    footer(fig, host, raw, note)
    save(fig, out, f"{metric}_{host}" + ("" if baseline == "same" else "_vs_naive"), fmts)


def plot_strong(a, host, raw, out, fmts, note):
    h = a[(a["host"] == host) & (a["model"].isin(DIST_MODELS))]
    ns = [n for n in sorted(h["n"].unique()) if h[h["n"] == n]["cores"].nunique() >= 2]
    if not ns:
        warn(f"{host}: chưa có dữ liệu MPI/hybrid cho strong scaling")
        return
    fig, axes = plt.subplots(1, len(ns), figsize=(4.6 * len(ns), 4.4), squeeze=False)
    for ax, n in zip(axes[0], ns):
        sub = h[h["n"] == n]
        for (algo, model), s in sub.groupby(["algo", "model"]):
            base = baseline_time(a, host, algo, n)
            if base is None:
                continue
            s = best_per(s, ["cores"]).sort_values("cores")
            y = base / s["t_med"]
            ax.errorbar(s["cores"], y, yerr=errbars(y, base / s["t_max"], base / s["t_min"]),
                        marker="o", ms=4, capsize=2, label=series_name(algo, model))
        cs = np.array(sorted(sub["cores"].unique()))
        ax.plot(cs, cs, "k--", lw=1, label="lý tưởng")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks(cs)
        ax.set_xticklabels([str(c) for c in cs])
        ax.set_title(f"n = {n}")
        ax.set_xlabel("Số lõi dùng P×T")
        ax.set_ylabel("Speedup so với bản tuần tự")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    suptitle(fig, f"Strong scaling MPI / hybrid ({host})", 10)
    footer(fig, host, raw, note)
    save(fig, out, f"strong_{host}", fmts)


def plot_comm(a, host, raw, out, fmts, note):
    h = a[(a["host"] == host) & (a["model"].isin(DIST_MODELS))]
    for n in sorted(h["n"].unique()):
        sub = h[h["n"] == n]
        groups = [(k, s) for k, s in sub.groupby(["algo", "model"]) if len(s) >= 2]
        if not groups:
            continue
        fig, axes = plt.subplots(1, len(groups), figsize=(4.8 * len(groups), 4.2), squeeze=False)
        for ax, ((algo, model), s) in zip(axes[0], groups):
            s = s.sort_values(["cores", "P"])
            x = np.arange(len(s))
            w = 0.38
            ax.bar(x - w / 2, s["comp"], w, label="t_compute (max các rank)")
            ax.bar(x + w / 2, s["comm"], w, label="t_comm (max các rank)")
            ax.plot(x, s["t_med"], "ko-", ms=4, label="t_total")
            ax.set_xticks(x)
            ax.set_xticklabels([f"P={p}\nT={t}" for p, t in zip(s["P"], s["T"])], fontsize=7)
            ax.set_ylabel("Thời gian (s)")
            ax.set_title(f"{series_name(algo, model)} · n = {n}", fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend(fontsize=7)
        suptitle(fig, f"Tách t_compute / t_comm ({host}) — hai giá trị không nhất thiết cộng đúng bằng t_total", 9)
        footer(fig, host, raw, note)
        save(fig, out, f"comm_{host}_n{n}", fmts)


def plot_cutoff(a, host, raw, out, fmts, note):
    h = a[(a["host"] == host) & a["algo"].map(is_strassen)]
    groups = [(k, s) for k, s in h.groupby(["algo", "model", "T", "P"]) if s["cutoff"].nunique() >= 2]
    if not groups:
        warn(f"{host}: chưa có dữ liệu quét cutoff (chạy: scripts/run_shared.sh --cutoff-sweep)")
        return
    fig, axes = plt.subplots(1, len(groups), figsize=(4.8 * len(groups), 4.3), squeeze=False)
    for ax, ((algo, model, T, P), s) in zip(axes[0], groups):
        for n, sn in s.groupby("n"):
            if sn["cutoff"].nunique() < 2:      # n không có quét cutoff
                continue
            sn = sn.sort_values("cutoff")
            ax.errorbar(sn["cutoff"], sn["t_med"], yerr=errbars(sn["t_med"], sn["t_min"], sn["t_max"]),
                        marker="o", ms=4, capsize=2, label=f"n = {n}")
            best = sn.loc[sn["t_med"].idxmin()]
            ax.plot([best["cutoff"]], [best["t_med"]], "k*", ms=11)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        cs = sorted(s["cutoff"].unique())
        ax.set_xticks(cs)
        ax.set_xticklabels([str(c) for c in cs])
        ax.set_xlabel("cutoff")
        ax.set_ylabel("Thời gian t_total (s)")
        ax.set_title(f"{series_name(algo, model)}, T = {T}", fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    suptitle(fig, f"Ảnh hưởng của cutoff Strassen ({host}); ★ = cutoff tốt nhất", 10)
    footer(fig, host, raw, note)
    save(fig, out, f"cutoff_{host}", fmts)


def plot_strassen_vs_naive(a, host, raw, out, fmts, note):
    h = a[a["host"] == host]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    drawn = False
    for model in ("seq", "omp"):
        pairs = {}
        for n in sorted(h["n"].unique()):
            nv = h[(h["algo"] == "naive_ikj") & (h["model"] == model) & (h["n"] == n)]
            st = h[(h["algo"] == "strassen") & (h["model"] == model) & (h["n"] == n)]
            common = set(nv["T"]) & set(st["T"])
            if common:
                pairs[n] = (nv, st, common)
        if not pairs:
            continue
        # Dùng MỘT giá trị T cố định cho mọi n (lớn nhất có ở mọi n) để đường không bị nhảy.
        t_fixed = max(set.intersection(*[c for _, _, c in pairs.values()]) or {1})
        xs, ys = [], []
        for n, (nv, st, common) in pairs.items():
            if t_fixed in common:
                xs.append(n)
                ys.append(nv[nv["T"] == t_fixed]["t_med"].iloc[0] / st[st["T"] == t_fixed]["t_med"].iloc[0])
        if xs:
            ax.plot(xs, ys, marker="o", label=f"{MODEL_LABEL[model]} (T = {t_fixed})")
            drawn = True
    if not drawn:
        plt.close(fig)
        warn(f"{host}: chưa có cặp naive_ikj / strassen cùng model để so sánh")
        return
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Kích thước ma trận n")
    ax.set_ylabel("t_naive / t_strassen   (> 1: Strassen nhanh hơn)")
    ax.set_title(f"Khi nào Strassen thắng Naive? ({host})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    footer(fig, host, raw, note)
    save(fig, out, f"strassen_vs_naive_{host}", fmts)


def plot_weak(a, host, raw, out, fmts, note, n0, rule):
    h = a[(a["host"] == host) & (a["model"].isin(DIST_MODELS))]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    drawn = False
    for (algo, model), s in h.groupby(["algo", "model"]):
        s = s[s["T"] == s["T"].min()]
        base = s[(s["P"] == 1) & (s["n"] == n0)]
        if base.empty:
            continue
        t1 = base["t_med"].iloc[0]
        xs, ys = [], []
        for p in sorted(s["P"].unique()):
            target = n0 * (p ** (1 / 3) if rule == "cbrt" else math.sqrt(p))
            cand = s[s["P"] == p]
            if cand.empty:
                continue
            row = cand.iloc[(cand["n"] - target).abs().argsort().iloc[0]]
            if abs(row["n"] - target) / target > 0.05:
                continue
            xs.append(p)
            ys.append(t1 / row["t_med"])
        if xs:
            ax.plot(xs, ys, marker="o", label=series_name(algo, model))
            drawn = True
    if not drawn:
        plt.close(fig)
        warn(f"{host}: không có dữ liệu weak scaling khớp n0={n0}, rule={rule}")
        return
    ax.axhline(1.0, color="k", ls="--", lw=1, label="lý tưởng")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Số tiến trình P")
    ax.set_ylabel(f"Hiệu suất weak scaling = t(P=1, n0) / t(P, n(P))")
    rule_txt = "n ∝ ∛P (giữ nguyên số phép tính/tiến trình)" if rule == "cbrt" else "n ∝ √P (giữ nguyên bộ nhớ/tiến trình)"
    ax.set_title(f"Weak scaling, n0 = {n0}, {rule_txt} ({host})", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    footer(fig, host, raw, note)
    save(fig, out, f"weak_{host}", fmts)


# ------------------------------------------------------------ dữ liệu giả lập
def make_sample(path):
    """CSV GIẢ LẬP (host = sample-host) — chỉ để thử script, KHÔNG phải số đo thật."""
    rng = np.random.default_rng(1)
    rows = []

    def add(algo, model, n, P, T, t, comp=None, comm=0.0, cutoff=0, runs=5, err=1e-16):
        for r in range(1, runs + 1):
            tt = t * (1 + rng.normal(0, 0.02))
            cc = (comp if comp is not None else t) * (1 + rng.normal(0, 0.02))
            rows.append([algo, model, n, P, T, r, f"{tt:.9g}", f"{cc:.9g}", f"{comm:.9g}",
                         "true", f"{err:.4g}", cutoff, "sample-host", "sample"])

    g_ikj, g_ijk, g_blas = 4e9, 0.9e9, 28e9

    def s_omp(T, a):           # Amdahl đơn giản, 4 lõi vật lý
        eff = min(T, 4) + (0.15 * (T - 4) if T > 4 else 0)
        return eff / (1 + a * (min(T, 4) - 1))

    def strassen_ratio(n):     # t_strassen / t_naive
        return 1.15 if n <= 1000 else 0.95 if n <= 2000 else 0.85 if n <= 4000 else 0.78

    for n in [100, 500, 1000, 2000, 4000, 6000, 8000, 10000]:
        flops = 2.0 * n ** 3
        t_ikj, t_blas = flops / g_ikj, flops / g_blas
        runs = 5 if n <= 4000 else 3
        runs1 = 1 if n >= 8000 else runs
        sweep = [1, 2, 4, 8] if n in (2000, 4000, 8000) else [1, 4]
        add("naive_ikj", "seq", n, 1, 1, t_ikj, runs=runs1)
        if n <= 2000:
            add("naive_ijk", "seq", n, 1, 1, flops / g_ijk, runs=runs)
        add("strassen", "seq", n, 1, 1, t_ikj * strassen_ratio(n), cutoff=128, runs=runs1, err=2e-12)
        for T in sweep:
            r = runs1 if T == 1 else runs
            add("naive_ikj", "omp", n, 1, T, t_ikj / s_omp(T, 0.04), runs=r)
            add("strassen", "omp", n, 1, T, t_ikj * strassen_ratio(n) / s_omp(T, 0.10),
                cutoff=128, runs=r, err=2e-12)
            add("blas", "shared", n, 1, T, t_blas / s_omp(T, 0.06), runs=r)

    for n in (1000, 2000, 4000):                    # quét cutoff (U-shape)
        t0 = 2.0 * n ** 3 / g_ikj * strassen_ratio(n)
        for c in (32, 64, 128, 256, 512):
            f = 1 + 0.12 * abs(math.log2(c / 128))
            add("strassen", "seq", n, 1, 1, t0 * f, cutoff=c, err=2e-12)

    for n in (2000, 4000):                          # MPI / hybrid (strong)
        t_seq = 2.0 * n ** 3 / g_ikj
        for P in (1, 2, 4):
            comm = 0.0 if P == 1 else n * n * 8 * (P - 1) / 1.5e9
            comp = t_seq / P * 1.05
            add("naive_ikj", "mpi", n, P, 1, comp + comm, comp=comp, comm=comm)
        comp = t_seq / 4 * 1.05
        comm = n * n * 8 / 1.5e9
        add("naive_ikj", "hybrid", n, 2, 2, comp + comm, comp=comp, comm=comm)

    n0 = 2000                                       # weak scaling, n ∝ ∛P
    for P in (1, 2, 4, 8):
        n = int(round(n0 * P ** (1 / 3)))
        comp = 2.0 * n ** 3 / g_ikj / P * 1.05
        comm = 0.0 if P == 1 else n * n * 8 * (P - 1) / 1.5e9
        add("naive_ikj", "mpi", n, P, 1, comp + comm, comp=comp, comm=comm)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
    print(f"[plot] đã ghi dữ liệu GIẢ LẬP: {path}")
    print("[plot] thử:  python3 scripts/plot.py results/sample --out /tmp/figs --weak-n0 2000")


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Vẽ biểu đồ từ CSV benchmark ma trận.")
    ap.add_argument("paths", nargs="*", help="file/thư mục CSV (mặc định: results/shared, results/cluster)")
    ap.add_argument("--out", default="report/figures", help="thư mục xuất hình")
    ap.add_argument("--fmt", default="png", help="png, pdf hoặc png,pdf")
    ap.add_argument("--host", help="chỉ vẽ cho host này")
    ap.add_argument("--cutoff", type=int, default=128, help="cutoff Strassen dùng cho các hình chính (mặc định 128)")
    ap.add_argument("--note", default="", help="ghi chú compiler/flags in ở chân hình")
    ap.add_argument("--baseline", choices=["same", "naive"], default="same",
                    help="speedup so với bản tuần tự cùng thuật toán (same) hoặc naive_ikj tuần tự (naive)")
    ap.add_argument("--sweep-n", default="", help='chỉ vẽ scaling cho các n này, ví dụ "2000 4000 8000"')
    ap.add_argument("--weak-n0", type=int, help="n0 (P=1) để vẽ weak scaling")
    ap.add_argument("--weak-rule", choices=["cbrt", "sqrt"], default="cbrt")
    ap.add_argument("--strict", action="store_true", help="bỏ các dòng verified=false")
    ap.add_argument("--make-sample", action="store_true", help="sinh CSV giả lập vào results/sample/")
    args = ap.parse_args()

    if args.make_sample:
        make_sample(os.path.join("results", "sample", "sample-host_20260101.csv"))
        return

    paths = args.paths or [p for p in ("results/shared", "results/cluster") if os.path.isdir(p)]
    if not paths:
        sys.exit("Không có đường dẫn nào. Truyền file/thư mục CSV hoặc chạy từ gốc repo.")
    raw = load(paths)
    if raw is None or raw.empty:
        sys.exit("Không đọc được dòng dữ liệu nào từ: " + ", ".join(paths))

    bad = (~raw["verified"]).sum()
    if bad:
        warn(f"{bad}/{len(raw)} dòng có verified=false (chưa chạy --verify hoặc kết quả sai)")
        if args.strict:
            raw = raw[raw["verified"]]
    if args.host:
        raw = raw[raw["host"] == args.host]
    if raw.empty:
        sys.exit("Không còn dữ liệu sau khi lọc.")

    a = pick_cutoff(aggregate(raw), args.cutoff)
    fmts = [f.strip() for f in args.fmt.split(",") if f.strip()]
    sweep = [int(x) for x in args.sweep_n.split()] if args.sweep_n else []

    for host in sorted(raw["host"].unique()):
        print(f"[plot] === {host}: {len(raw[raw['host'] == host])} dòng ===")
        plot_time_gflops(a, host, raw, args.out, fmts, args.note)
        for metric in ("speedup", "efficiency"):
            plot_scaling(a, host, raw, args.out, fmts, args.note, metric, args.baseline, sweep)
        plot_strong(a, host, raw, args.out, fmts, args.note)
        plot_comm(a, host, raw, args.out, fmts, args.note)
        plot_cutoff(aggregate(raw), host, raw, args.out, fmts, args.note)   # cần mọi cutoff
        plot_strassen_vs_naive(a, host, raw, args.out, fmts, args.note)
        if args.weak_n0:
            plot_weak(a, host, raw, args.out, fmts, args.note, args.weak_n0, args.weak_rule)


if __name__ == "__main__":
    main()
