"""Local package commands: verify, test, tables, compile."""
from pathlib import Path
import hashlib
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PAPER = ROOT/"paper/paper1"
mode = sys.argv[1] if len(sys.argv)>1 else "verify"
if mode == "verify":
    rows = (ROOT/"MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
    for row in rows:
        expected, relative = row.split("  ", 1)
        target = (ROOT/relative).resolve()
        if not target.is_relative_to(ROOT):
            raise RuntimeError("Manifest path outside package")
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Hash mismatch: "+relative)
    print(f"Verified {len(rows)} packaged files")
elif mode == "test":
    subprocess.run([sys.executable, "-m", "pytest", "experiments/spectral-risk-calibration/tests", "-q"],
                   cwd=ROOT, check=True)
elif mode == "tables":
    subprocess.run([sys.executable, str(PAPER/"notes/calibration-integration/generate_tables.py")], check=True)
elif mode == "figures":
    subprocess.run([sys.executable, str(PAPER/"figures/draw_operator_law_map.py")], check=True)
    experiment = ROOT/"experiments/spectral-risk-calibration"
    subprocess.run([sys.executable, str(experiment/"scripts/s16_visual_alarm_report.py"),
                    "--results", str(experiment/"results/visual_alarm_v1")], check=True)
    subprocess.run([sys.executable, str(experiment/"scripts/s15c_equal_fpr.py"),
                    "--plot-only"], check=True)
elif mode == "compile":
    for args in (["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                 ["bibtex", "main"],
                 ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                 ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]):
        subprocess.run(args, cwd=PAPER, check=True)
else:
    raise SystemExit("Choose verify, test, figures, tables, or compile")
