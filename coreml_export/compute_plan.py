#!/usr/bin/env python3
"""Where does Core ML run each op of an .mlpackage? (macOS 14.4+ and coremltools 8+ only.)

    python coreml_export/compute_plan.py model.mlpackage [--units ane|gpu|all|cpu] [--top 30]

Prints, for the chosen compute-unit setting, the estimated-cost share and op count per device the
plan assigns, then the ops the Neural Engine cannot take (type, name, cost share), largest first.
This is the Xcode performance report as text: the tool that tells whether an "ANE" number measured
by the Bench is the Neural Engine or a CPU fallback with sync boundaries.
"""
import argparse, collections, sys

import coremltools as ct

try:   # the native bridge: absent when pip could only install the pure-Python part (no wheel for this Python)
    from coremltools import libcoremlpython  # noqa: F401
except ImportError:
    sys.exit(f"coremltools {ct.__version__} on Python {sys.version.split()[0]} has no native bridge "
             "(libcoremlpython): pip installed the pure-Python package only, so nothing can be compiled or "
             "planned. Use a Python that coremltools ships binaries for (3.11 or 3.12), e.g.\n"
             "  python3.12 -m venv ~/ctplan && ~/ctplan/bin/pip install -U coremltools\n"
             "  ~/ctplan/bin/python coreml_export/compute_plan.py model.mlpackage --units ane")
from coremltools.models.compute_plan import MLComputePlan

UNITS = {"ane": ct.ComputeUnit.CPU_AND_NE, "gpu": ct.ComputeUnit.CPU_AND_GPU,
         "all": ct.ComputeUnit.ALL, "cpu": ct.ComputeUnit.CPU_ONLY}


def device_name(dev) -> str:
    n = type(dev).__name__
    return {"MLNeuralEngineComputeDevice": "ANE", "MLGPUComputeDevice": "GPU", "MLCPUComputeDevice": "CPU"}.get(n, n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--units", default="ane", choices=sorted(UNITS))
    ap.add_argument("--top", type=int, default=30, help="how many unsupported ops to list")
    a = ap.parse_args()

    compiled = ct.utils.compile_model(a.model) if not a.model.endswith(".mlmodelc") else a.model
    plan = MLComputePlan.load_from_path(path=compiled, compute_units=UNITS[a.units])
    program = plan.model_structure.program
    if program is None:
        print("not an ML program; nothing to report"); return 1
    ops = list(program.functions["main"].block.operations)

    by_dev_cost, by_dev_n = collections.Counter(), collections.Counter()
    off_ane = []
    total = 0.0
    for op in ops:
        if op.operator_name == "const":
            continue
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        cost = plan.get_estimated_cost_for_mlprogram_operation(op)
        w = float(cost.weight) if cost is not None else 0.0
        total += w
        dev = device_name(usage.preferred_compute_device) if usage is not None else "?"
        by_dev_cost[dev] += w; by_dev_n[dev] += 1
        supported = {device_name(d) for d in (usage.supported_compute_devices if usage is not None else [])}
        if "ANE" not in supported:
            name = op.outputs[0].name if op.outputs else "?"
            off_ane.append((w, op.operator_name, name, ",".join(sorted(supported)) or "?"))

    print(f"{a.model}  units={a.units}  ops={len(ops)}")
    for dev in sorted(by_dev_cost, key=lambda d: -by_dev_cost[d]):
        share = 100 * by_dev_cost[dev] / total if total else 0
        print(f"  {dev:>4}: {by_dev_n[dev]:5d} ops  {share:5.1f}% of estimated cost")
    off_ane.sort(reverse=True)
    off_cost = sum(w for w, *_ in off_ane)
    print(f"ops the ANE cannot run: {len(off_ane)} ({100 * off_cost / total if total else 0:.1f}% of estimated cost)")
    kinds = collections.Counter(t for _, t, _, _ in off_ane)
    print("  by type: " + ", ".join(f"{t}:{n}" for t, n in kinds.most_common()))
    for w, t, name, sup in off_ane[: a.top]:
        print(f"  {100 * w / total if total else 0:5.2f}%  {t:<24} {name:<40} runs on {sup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
