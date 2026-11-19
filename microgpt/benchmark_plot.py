import math
from pathlib import Path


def save_benchmark_plot(medians: dict[str, float], path: Path) -> None:
    from matplotlib import rc_context
    from matplotlib.figure import Figure
    from matplotlib.transforms import blended_transform_factory

    labels = {"exojit": "ExoJIT", "jax": "JAX", "numpy": "NumPy", "torch": "PyTorch", "plain": "Optimized Python", "original": "Original Python"}
    order = sorted(medians, key=medians.get)
    times = [medians[name] for name in order]
    lo = math.floor(math.log10(min(times)))
    if 10**lo == min(times):
        lo -= 1
    hi = max(lo + 1, math.ceil(math.log10(max(times))))
    ticks = [10**exponent for exponent in range(lo, hi + 1)]
    left = ticks[0]

    with rc_context({"font.family": "cmr10", "font.size": 7.5, "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "svg.fonttype": "path", "svg.hashsalt": "microgpt-benchmark"}):
        width, height = 360, 126
        fig = Figure(figsize=(width / 72, height / 72), facecolor="white")
        ax = fig.add_axes((96 / width, 29 / height, 201 / width, 81 / height))
        ax.set_xscale("log")
        ax.barh(range(len(order)), [time - left for time in times], left=left, height=0.48, color="#2459b5")
        ax.set_xlim(left, ticks[-1])
        ax.set_xticks(ticks, [f"{tick / 1000:g} ms" if tick >= 1000 else "$" + rf"{tick:g}\,\mu\mathrm{{s}}" + "$" for tick in ticks])
        ax.set_yticks(range(len(order)), [labels[name] for name in order])
        ax.set_ylim(len(order) - 0.45, -0.55)
        ax.set_xlabel("median time per training step (log scale, lower is better)", fontsize=7, labelpad=6)
        ax.tick_params(axis="both", which="both", length=0, pad=4)
        ax.set_axisbelow(True)
        ax.grid(axis="x", which="major", color="#d9d9d9", linewidth=0.3)
        for spine in ax.spines.values():
            spine.set_visible(False)

        row_text = blended_transform_factory(fig.transFigure, ax.transData)
        fig.text(348 / width, 1 - 10.5 / height, "median", ha="right", va="center")
        for row, us in enumerate(times):
            elapsed = "$" + (rf"{us:.0f}\,\mu\mathrm{{s}}" if us < 1000 else rf"{us / 1000:.1f}\,\mathrm{{ms}}") + "$"
            ax.text(348 / width, row, elapsed, transform=row_text, ha="right", va="center")

        description = "Horizontal bars show median runtime per training step on a logarithmic scale. Each bar starts at the smallest labeled tick, not zero. Shorter is faster."
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=216, metadata={"Title": "MicroGPT training time", "Description": description, "Date": None})
        if path.suffix.lower() == ".svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
