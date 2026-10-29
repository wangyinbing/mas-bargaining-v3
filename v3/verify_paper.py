"""Recompute every number quoted in the paper from the shipped data.

Usage: python v3/verify_paper.py
Prints claimed vs computed; exits non-zero if any check fails.
"""
import sys, math
import pandas as pd, numpy as np
from scipy import stats
import itertools

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent      # works in the working tree and in the archive
MAIN = ROOT / "output_v3" / "main"
XV30 = ROOT / "output_v3" / "xvendor30"

FAIL = []


def chk(name, claimed, computed, tol=0.0015):
    if isinstance(claimed, (int, float)):
        if claimed >= 100:
            tol = max(tol, 1.0)
        elif 0.5 <= claimed <= 1.5:
            tol = max(tol, 0.005)
        ok = abs(claimed - computed) <= tol
    else:
        ok = claimed == computed
    if not ok:
        FAIL.append(name)
    print(f"{'OK ' if ok else 'FAIL'}  {name:52s} claimed={claimed!r:>12}  computed={computed:.4f}")


def pie(r, delta=0.9):
    return int(round(10 * delta ** (r - 1)))


def load_main():
    d = pd.read_csv(MAIN / "episodes.csv").drop_duplicates(subset=["model", "protocol", "trial", "episode"])
    for c in ["deal", "rounds", "welfare", "tokens_in", "tokens_out", "demand_a_mean", "fallback_offered", "confirm_failed"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["tok"] = d.tokens_in + d.tokens_out
    d["eff"] = d.welfare / 10
    return d


def load_judge():
    f = pd.read_csv(MAIN / "audit" / "judge_first.csv")
    j = pd.read_csv(MAIN / "audit" / "judge.csv")
    return f, j


def main():
    d = load_main()
    f, j = load_judge()
    ft = d[d.protocol == "free_text"]

    # ── free dialogue conventions ───────────────────────────────
    first = f[f.first_round > 0]
    chk("free dialogue: first-agreement rate", 0.997, len(first) / len(f))
    chk("free dialogue: mean first round", 2.1, first.first_round.mean(), tol=0.06)
    chk("free dialogue: share agreeing in round 1", 0.72, (first.first_round == 1).mean(), tol=0.005)

    eff_first = np.where(f.first_round > 0, f.first_round.apply(pie) / 10, 0.0).mean()
    chk("free dialogue: first-agreement efficiency", 0.905, eff_first, tol=0.002)

    late = j[j.deal == 1]
    chk("free dialogue: last-agreement rate", 0.95, j.deal.mean(), tol=0.002)
    chk("free dialogue: mean last round", 9.8, late["round"].mean(), tol=0.06)
    chk("free dialogue: last-agreement efficiency", 0.388,
        (np.where(j.deal == 1, j["round"].apply(pie) / 10, 0.0)).mean(), tol=0.002)
    chk("free dialogue: deals missed by late window", 13, int(((f.first_round > 0) & (j.deal == 0)).sum()))

    # ── hook arms ───────────────────────────────────────────────
    for key, path, deal_c, eff_c, rnd_c in [
        ("strict", str(ROOT / "output_v3" / "free_exec" / "episodes.csv"), 0.42, 0.262, 6.3),
        ("forgiving", str(ROOT / "output_v3" / "free_loose" / "episodes.csv"), 0.70, 0.567, 3.4),
    ]:
        h = pd.read_csv(path).drop_duplicates(subset=["model", "protocol", "trial", "episode"])
        h["welfare"] = pd.to_numeric(h.welfare, errors="coerce")
        h["rounds"] = pd.to_numeric(h.rounds, errors="coerce")
        chk(f"{key} hook: deal rate", deal_c, h.deal.mean(), tol=0.005)
        chk(f"{key} hook: efficiency", eff_c, (h.welfare / 10).mean(), tol=0.005)
        chk(f"{key} hook: mean round", rnd_c, h[h.deal == 1].rounds.mean(), tol=0.06)

    # ── Table 1 ─────────────────────────────────────────────────
    want = {
        ("flash", "sealed_bid"): (1.000, 0.935), ("flash", "sealed_compromise"): (1.000, 1.000),
        ("flash", "sealed_compromise_verify"): (0.933, 0.871), ("flash", "alternating_offers"): (0.873, 0.805),
        ("pro", "sealed_bid"): (0.993, 0.972), ("pro", "sealed_compromise"): (1.000, 0.998),
        ("pro", "sealed_compromise_verify"): (1.000, 0.960), ("pro", "alternating_offers"): (1.000, 0.865),
    }
    for (m, p), (dc, ec) in want.items():
        s = d[(d.model == m) & (d.protocol == p)]
        chk(f"T1 {m}/{p} deal", dc, s.deal.mean())
        chk(f"T1 {m}/{p} efficiency", ec, s.eff.mean())

    # ── commitment cost / demands / tokens ──────────────────────
    for m in ["flash", "pro"]:
        a = d[(d.model == m) & (d.protocol == "sealed_compromise")].eff.mean()
        b = d[(d.model == m) & (d.protocol == "sealed_compromise_verify")].eff.mean()
        print(f"     commitment cost {m}: {(a-b)*100:.1f} points")
    chk("demand SPB flash", 6.23, d[(d.model == "flash") & (d.protocol == "sealed_compromise")].demand_a_mean.mean(), tol=0.01)
    chk("demand SPB+ flash", 7.57, d[(d.model == "flash") & (d.protocol == "sealed_compromise_verify")].demand_a_mean.mean(), tol=0.01)
    chk("tokens free dialogue flash", 52559, d[(d.model == "flash") & (d.protocol == "free_text")].tok.mean(), tol=1)
    chk("tokens free dialogue pro", 19310, d[(d.model == "pro") & (d.protocol == "free_text")].tok.mean(), tol=1)
    chk("tokens SPB flash", 4558, d[(d.model == "flash") & (d.protocol == "sealed_compromise")].tok.mean(), tol=1)
    chk("tokens sealed flash", 4206, d[(d.model == "flash") & (d.protocol == "sealed_bid")].tok.mean(), tol=1)

    # ── statistics ──────────────────────────────────────────────
    protos = ["free_text", "alternating_offers", "sealed_bid", "sealed_compromise", "sealed_compromise_verify"]
    for m, hc, pc in [("flash", 325.9, 2.8e-69), ("pro", 465.4, 2.0e-99)]:
        s = d[d.model == m]
        H, p = stats.kruskal(*[s[s.protocol == q].eff for q in protos])
        chk(f"KW {m} H", hc, H, tol=0.15)
        print(f"     KW {m} p = {p:.1e} (paper {pc:.1e})")
    for m, pc in [("flash", 3.9e-18), ("pro", 2.1e-4)]:
        s = d[d.model == m]
        p = stats.mannwhitneyu(s[s.protocol == "sealed_bid"].eff, s[s.protocol == "sealed_compromise"].eff, alternative="two-sided").pvalue
        print(f"     SPB vs sealed {m}: p={p:.2e} (paper {pc:.1e})")
    for m, pc in [("flash", 3.7e-10), ("pro", 8.1e-8)]:
        s = d[d.model == m]
        p = stats.mannwhitneyu(s[s.protocol == "sealed_compromise"].eff, s[s.protocol == "sealed_compromise_verify"].eff, alternative="two-sided").pvalue
        print(f"     SPB+ vs SPB {m}: p={p:.2e} (paper {pc:.1e})")

    # ── delta sensitivity ───────────────────────────────────────
    print("\n     delta sensitivity (mean efficiency):")
    for delta in [0.80, 0.90, 0.95]:
        row = []
        for q in ["sealed_compromise", "sealed_bid", "alternating_offers"]:
            s = d[d.protocol == q]
            e = np.where(s.deal == 1, s.rounds.apply(lambda r: pie(r, delta)) / 10, 0.0).mean()
            row.append(f"{q}={e:.3f}")
        print(f"       delta={delta}: " + "  ".join(row))

    # ── slope ───────────────────────────────────────────────────
    print("\n     learning slopes (per episode within trial):")
    for m, q, want_s in [("pro", "alternating_offers", 0.123), ("flash", "sealed_compromise_verify", 0.082),
                         ("flash", "sealed_bid", 0.036), ("pro", "sealed_bid", 0.028)]:
        s = d[(d.model == m) & (d.protocol == q)]
        r = stats.linregress(s.episode.astype(float), s.eff)
        chk(f"slope {m}/{q}", want_s, r.slope, tol=0.002)
        print(f"       R2={r.rvalue**2:.2f} p={r.pvalue:.1e}")

    # ── Rubinstein ──────────────────────────────────────────────
    print("\n     Rubinstein benchmark (proposer share from the accepted proposal):")
    import glob, json, tarfile
    shares = {}
    d = MAIN / "transcripts"
    if d.is_dir() and any(d.glob("*.json")):
        recs = [json.load(open(f, encoding="utf-8")) for f in d.glob("*alternating_offers*.json")]
    else:                                   # archive ships transcripts as a tarball
        with tarfile.open(MAIN / "transcripts.tar.gz") as tf:
            recs = [json.loads(tf.extractfile(m).read().decode("utf-8"))
                    for m in tf.getmembers()
                    if "alternating_offers" in m.name and m.name.endswith(".json")]
    for rec in recs:
        r = rec["row"]
        if r.get("status") != "ok" or int(r.get("deal", 0)) != 1:
            continue
        props = rec.get("proposals") or []
        if not props:
            continue
        _, mine, _ = props[-1]
        rnd = int(r["rounds"])
        shares.setdefault(r["model"], []).append(mine / max(1, pie(rnd)))
    for m, want in [("flash", 0.624), ("pro", 0.509)]:
        chk(f"Rubinstein proposer share {m}", want, float(np.mean(shares[m])), tol=0.002)

    # ── cross-vendor ────────────────────────────────────────────
    x = pd.read_csv(XV30 / "episodes.csv")
    x = x[x.status == "ok"].drop_duplicates(subset=["model", "protocol", "trial", "episode"])
    x["eff"] = pd.to_numeric(x.welfare, errors="coerce") / 10
    print("\n     cross-vendor:")
    for (m, p), want in {
        ("glm45air", "sealed_bid"): 1.000, ("glm45air", "sealed_compromise"): 1.000,
        ("glm45air", "sealed_compromise_verify"): 0.933,
        ("qwen32b", "sealed_bid"): 0.753, ("qwen32b", "sealed_compromise"): 0.776,
        ("qwen32b", "sealed_compromise_verify"): 0.981,
    }.items():
        s = x[(x.model == m) & (x.protocol == p)]
        chk(f"XV {m}/{p}", want, s.eff.mean(), tol=0.0015)
        print(f"       n={len(s)}")

    print("\n" + ("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} FAILED:\n  " + "\n  ".join(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
