"""Table-scoped audit: every number in main.tex tables vs saved JSONs."""
import json
import re
import io
import sys

sys.stdout.reconfigure(encoding="utf-8")
TEX = "C:/Users/Peng1/OneDrive/文档/ChatGPT/hypergraph/paper/paper1/main.tex"
R = "C:/Users/Peng1/OneDrive/文档/ChatGPT/hypergraph/paper/paper1/experiments/prob2-relation-error/results/"
U = "results/"
tex = io.open(TEX, encoding="utf-8").read()
fails = []


def table(label):
    for m in re.finditer(r"\\begin\{table\}.*?\\end\{table\}", tex, re.S):
        if ("\\label{" + label + "}") in m.group(0):
            return m.group(0)
    return ""


def nums(line):
    line = line.replace("$", "").replace("\\%", "").replace(";", " ")
    return [float(x) for x in re.findall(r"-?\d+\.?\d*", line)]


sv = json.load(open(U + "structure_validation.json", encoding="utf-8"))
tA = table("tab:familyA")
for s in sv["structure"]:
    if s["family"] != "A":
        continue
    row = [l for l in tA.splitlines() if l.strip().startswith(str(s["d"]) + " &")]
    v = nums(row[0])
    # v: d, Phi, L_A, U_B, betaA, ci_lo, ci_hi, betaB, sep  (CI adds 2)
    wants = [(1, round(s["theory_corrected"]["Phi_hub"], 0)),
             (2, round(s["theory_corrected"]["L_A"], 1)),
             (3, round(s["theory_corrected"]["U_B"], 1)),
             (4, round(s["mean_betaA"], 1)),
             (7, round(s["mean_betaB"], 2)),
             (8, round(100 * s["sep_freq"], 1))]
    for idx, want in wants:
        got = v[idx] if idx < len(v) else None
        if got is None or abs(got - want) > 0.06 * max(1, abs(want)):
            fails.append(("famA", s["d"], idx, got, want))

tB = table("tab:familyB")
for s in sv["structure"]:
    if s["family"] != "B":
        continue
    row = [l for l in tB.splitlines()
           if l.strip().startswith("%.2f &" % s["gamma"])]
    v = nums(row[0])
    wants = [round(s["Phi_witness"], 1), round(s["mean_betaA"], 2),
             round(s["mean_betaB"], 2),
             round(s["mean_betaA"] / s["mean_betaB"], 2),
             round(100 * s["sep_freq"], 1)]
    for idx, want in enumerate(wants, start=1):
        got = v[idx] if idx < len(v) else None
        if got is None or abs(got - want) > 0.06 * max(1, abs(want)):
            fails.append(("famB", s["gamma"], idx, got, want))

tJ = table("tab:joint-mc")
for row in json.load(open(R + "joint_mc_verification.json",
                          encoding="utf-8"))["rows"]:
    line = [l for l in tJ.splitlines()
            if l.strip().startswith("%.2f & %d &" % (row["p"], row["d"]))]
    v = nums(line[0])
    # v: p, d, L_A, U_B, nocenter, 5000, afail, bfail, sepfail, ci lo, ci hi
    wants = [(2, round(row["L_A"], 2), 0.01),
             (3, round(row["U_B"], 2), 0.01),
             (4, row["counts"]["no_wrong_center"], 0.01),
             (6, 0.0, 0.01),
             (7, 0.0, 0.01),
             (8, row["counts"]["sep_fail"], 0.01)]
    for idx, want, tol in wants:
        got = v[idx] if idx < len(v) else None
        if got is None or abs(got - want) > tol:
            fails.append(("jointMC", row["p"], row["d"], idx, got, want))

tC = table("tab:cyclic-verify")
for s in json.load(open(R + "cyclic_overlap.json", encoding="utf-8")):
    line = [l for l in tC.splitlines()
            if l.strip().startswith(str(s["d"]) + " &")]
    v = nums(line[0])
    # v: d, Q, H, L_A, U_B, emp%, cenA, cenB
    wants = [(3, round(s["L_A"], 2)), (4, round(s["U_B"], 2)),
             (6, round(s["cenA"], 2)), (7, round(s["cenB"], 2))]
    for idx, want in wants:
        got = v[idx] if idx < len(v) else None
        if got is None or abs(got - want) > 0.06 * max(1, abs(want)):
            fails.append(("cyclic", s["d"], idx, got, want))

# prose numbers: native + subspace
nv = json.load(open(U + "native_validation.json", encoding="utf-8"))
for s in nv["settings"]:
    for val in [s["mean_betaA"], s["mean_betaB"], abs(s["Phi_witness"]),
                s["witness_degree"]]:
        if ("%.2f" % val) not in tex and ("%d" % round(val)) not in tex:
            fails.append(("native-prose-missing", s["tag"], s["p"], val))

sub = json.load(open(U + "subspace_validation.json", encoding="utf-8"))
for s in sub["settings"]:
    for val in [s["mean_distA"], s["mean_distB"], s["diff"],
                s["eigengap"]]:
        ok = any(("%.3f" % val) in tex or ("%.2f" % val) in tex
                 or ("%.4g" % val) in tex or ("%.3f" % val).rstrip("0")
                 in tex for _ in [0])
        if not ok:
            fails.append(("subspace-prose-missing", s["tag"], s["p"], val))

# real-main + bootstrap + batches + attribution + mixed (spot)
gcn = json.load(open(R + "gcn_fixed.json", encoding="utf-8"))
if "%.2f" % gcn["ratio"] not in tex:
    fails.append(("gcn-ratio", gcn["ratio"], None, None, None))
# the appendix summarizes the 12-cell bootstrap table in prose; check the
# summary endpoints instead of removed per-cell values
fb = json.load(open(R + "full_bootstrap.json", encoding="utf-8"))
los = [res["ci"][0] for setting in fb for res in setting["results"]]
his = [res["ci"][1] for setting in fb for res in setting["results"]]
prose = io.open(TEX, encoding="utf-8").read()
m = re.search(r"range[^0-9]*([0-9.]+)[^0-9]*--[^0-9]*([0-9.]+)", prose)
if not m or abs(float(m.group(1)) - round(min(los), 3)) > 0.001 \
        or abs(float(m.group(2)) - round(max(his), 3)) > 0.001:
    fails.append(("bootstrap-summary", min(los), max(his),
                  m.groups() if m else None, None))
att = json.load(open(R + "attribution_fixed.json", encoding="utf-8"))
for tag, vals in att.items():
    if "%.2f" % vals["cen_real"] not in tex:
        fails.append(("attribution", tag, vals["cen_real"], None, None))

print("checks complete; mismatches:", len(fails))
for f in fails:
    print("  ", f)
if not fails:
    print("ALL SCOPED TABLE/PROSE CHECKS PASS")
