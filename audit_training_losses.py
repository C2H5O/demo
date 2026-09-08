"""Audit logged loss magnitudes, resume duplicates and small regularizer shares."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

COMPONENTS = ("depth", "camera", "highlight", "smooth", "attention")


def audit(path):
    content = Path(path).read_bytes()
    rows = []
    for number, line in enumerate(content.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("phase") == "train" and "loss/total" in row:
            if not all(math.isfinite(float(row["loss/" + k])) for k in
                       ("total", "highlight", "smooth", *(name + "_weighted" for name in COMPONENTS))):
                raise ValueError("Non-finite loss at line {}".format(number))
            if row["loss/total"] <= 0:
                raise ValueError("Cannot report shares for a nonpositive total at line {}".format(number))
            rows.append(row)
    if not rows:
        raise ValueError("No training loss records")
    latest = {}
    resets, conflicts = [], 0
    previous = None
    for row in rows:
        step = row["global_step"]
        if previous is not None and step <= previous:
            resets.append({"from_step": previous, "to_step": step})
        if step in latest and row != latest[step]:
            conflicts += 1
        latest[step] = row
        previous = step
    selected = [latest[step] for step in sorted(latest)]
    epochs = defaultdict(list)
    for row in selected:
        epochs[row["epoch"]].append(row)
    typical_count = Counter(len(rs) for rs in epochs.values()).most_common(1)[0][0]
    table = []
    for epoch, rs in sorted(epochs.items()):
        total_sum = sum(r["loss/total"] for r in rs)
        item = {"epoch_zero_based": epoch, "record_count": len(rs),
                "count_matches_typical_epoch": len(rs) == typical_count,
                "total_mean": total_sum / len(rs),
                "highlight_raw_mean": sum(r["loss/highlight"] for r in rs) / len(rs),
                "smooth_raw_mean": sum(r["loss/smooth"] for r in rs) / len(rs)}
        for name in COMPONENTS:
            key = "loss/" + name + "_weighted"
            item[name + "_weighted_mean"] = sum(r[key] for r in rs) / len(rs)
            item[name + "_share_percent_ratio_of_sums"] = 100 * sum(r[key] for r in rs) / total_sum
            item[name + "_share_percent_mean_of_ratios"] = 100 * sum(r[key]/r["loss/total"] for r in rs) / len(rs)
        table.append(item)
    steps = sorted(latest)
    return {
        "source": str(Path(path).resolve()), "sha256": hashlib.sha256(content).hexdigest(),
        "raw_training_records": len(rows), "unique_steps": len(selected),
        "duplicate_records": len(rows) - len(selected), "conflicting_duplicate_records": conflicts,
        "step_rewinds": resets, "selection": "last occurrence per global_step, matching user plot_loss.py; inferred resumed trajectory, not proven checkpoint lineage",
        "missing_steps_in_observed_range": steps[-1] - steps[0] + 1 - len(steps),
        "first_step": steps[0], "last_step": steps[-1],
        "typical_logged_records_per_epoch": typical_count,
        "max_total_conservation_residual": max(abs(r["loss/total"] - sum(r["loss/"+n+"_weighted"] for n in COMPONENTS)) for r in selected),
        "zero_counts": {n: sum(r["loss/"+n] == 0 for r in selected) for n in ("highlight", "smooth")},
        "all_record_max_total_conservation_residual": max(abs(r["loss/total"] - sum(r["loss/"+n+"_weighted"] for n in COMPONENTS)) for r in rows),
        "epochs": table,
        "limits": ["Scalar loss shares do not measure parameter-gradient shares.",
                   "Legacy logs contain no highlight mask coverage or per-loss gradients.",
                   "Epoch completeness is inferred from record counts, not checkpoints."]}


def save_report(result, output, plots=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "epochs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["epochs"][0]))
        writer.writeheader()
        writer.writerows(result["epochs"])
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        table = result["epochs"]
        x = [r["epoch_zero_based"] + 1 for r in table]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for name, color in (("highlight", "#23855b"), ("smooth", "#bd4939")):
            axes[0].plot(x, [r[name+"_raw_mean"] for r in table], marker="o", label=name, color=color)
            axes[1].plot(x, [r[name+"_share_percent_ratio_of_sums"] for r in table], marker="o", label=name, color=color)
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Mean raw loss (log scale)")
        axes[1].set_ylabel("Weighted share of total (%)")
        axes[1].set_ylim(bottom=0)
        for ax in axes:
            ax.set_xlabel("Training epoch (one-based)")
            ax.grid(alpha=0.2)
            ax.legend()
        fig.suptitle("Regularizers remain nonzero; last epoch is partial")
        fig.tight_layout()
        fig.savefig(output / "regularizers.png", dpi=180)
        fig.savefig(output / "regularizers.svg")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    result = audit(args.metrics)
    save_report(result, args.output_dir, args.plots)
    print(json.dumps({k: v for k, v in result.items() if k != "epochs"}, indent=2))


if __name__ == "__main__":
    main()
