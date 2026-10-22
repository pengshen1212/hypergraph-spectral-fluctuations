"""Rebuild the approved Figure 1 operator/law schematic."""
from pathlib import Path
import json
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",
    "font.size": 8.5,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.unicode_minus": False,
})

C = {"text": "#253343", "muted": "#566477", "line": "#8D9BAC",
     "operator": "#F5F7FA", "law": "#F0F5FA", "law_line": "#8AA3B9",
     "accent": "#245E76", "score": "#EAF3F6", "rule": "#DCE3EA"}

fig = plt.figure(figsize=(183/25.4, 115/25.4), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, 183), ylim=(0, 115))
ax.set_axis_off()
text_records = []
containment = []


def text(x, y, label, size=8.5, weight="normal", color=None,
         ha="center", va="center", linespacing=1.3):
    artist = ax.text(x, y, label, fontsize=size, fontweight=weight,
                     color=color or C["text"], ha=ha, va=va,
                     linespacing=linespacing)
    text_records.append(artist)
    return artist


def box(x, y, w, h, fill, edge, lw=0.85):
    patch = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0,rounding_size=1.6",
                          linewidth=lw, facecolor=fill, edgecolor=edge)
    ax.add_patch(patch)
    return patch


def arrow(start, end, color=None, lw=1.0):
    patch = FancyArrowPatch(start, end, arrowstyle="-|>",
                            mutation_scale=8, linewidth=lw,
                            color=color or C["muted"],
                            shrinkA=0, shrinkB=0)
    ax.add_patch(patch)
    return patch


def heading(y, letter, title):
    text(7, y, letter, size=10, weight="bold", ha="left")
    text(13, y, title, size=10, weight="bold", ha="left")


heading(108, "a", "Operator comparison")
operators = [
    (7, r"$B(a)$", "Raw relation operator"),
    (68, r"$S_0(a)$", "Fixed expected degrees"),
    (129, r"$S(a)$", "Degrees recomputed\nfor every draw"),
]
for x, symbol, desc in operators:
    patch = box(x, 79, 47, 22, C["operator"], C["line"])
    artists = [text(x+23.5, 94.3, symbol, size=13),
               text(x+23.5, 85.1, desc, size=8.2)]
    containment.extend((artist, patch) for artist in artists)
arrow((54, 90), (68, 90))
arrow((115, 90), (129, 90))
text(61, 74.7, "Static degree weighting", size=7.5, color=C["muted"])
text(122, 74.7, "Dynamic degree recomputation", size=7.5, color=C["muted"])
ax.plot([7, 176], [69, 69], color=C["rule"], lw=0.6)

heading(62, "b", "Prediction-law comparison")
laws = [
    (7, "L1", "Specified node-product law", "Jointly induced relations"),
    (65, "L2", "Fitted node-product law", "Node marginals retained"),
    (123, "L3", "Complete training runs", "Run-level dependence retained"),
]
for x, key, name, detail in laws:
    patch = box(x, 32.5, 53, 23, C["law"], C["law_line"])
    artists = [text(x+26.5, 50.1, key, size=10, weight="bold", color=C["accent"]),
               text(x+26.5, 43.4, name, size=8.05),
               text(x+26.5, 37.5, detail, size=7.75, color=C["muted"])]
    containment.extend((artist, patch) for artist in artists)
    arrow((x+26.5, 32.5), (x+26.5, 23.7), color=C["accent"])

sink = box(7, 4.5, 169, 19.2, C["score"], C["accent"], lw=0.95)
artists = [
    text(91.5, 18.1,
         "Jointly construct relations  ·  Recompute degrees  ·  Apply frozen scoring",
         size=8.05, color=C["accent"]),
    text(91.5, 10.5,
         r"$Y\;\longrightarrow\;a(Y)\;\longrightarrow\;D(a)"
         r"\;\longrightarrow\;T(Y)=\Vert S(a(Y))-\widehat{M}\Vert_2$",
         size=11.2),
]
containment.extend((artist, sink) for artist in artists)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
issues = []
for artist in text_records:
    bounds = artist.get_window_extent(renderer)
    if not fig.bbox.contains(bounds.x0, bounds.y0) or not fig.bbox.contains(bounds.x1, bounds.y1):
        issues.append("Text outside canvas: " + artist.get_text())
for artist, patch in containment:
    bounds = artist.get_window_extent(renderer)
    container = patch.get_window_extent(renderer)
    if (bounds.x0 < container.x0+1 or bounds.x1 > container.x1-1 or
            bounds.y0 < container.y0+1 or bounds.y1 > container.y1-1):
        issues.append("Text outside its box: " + artist.get_text())
if issues:
    raise RuntimeError("\n".join(issues))

for ext in ("pdf", "svg", "png"):
    kwargs = {"dpi": 300} if ext == "png" else {}
    if ext == "pdf":
        kwargs["metadata"] = {"CreationDate": None, "ModDate": None}
    fig.savefig(OUT / ("operator_law_map." + ext), facecolor="white", **kwargs)
(OUT / "operator_law_map_qa.json").write_text(json.dumps({
    "status": "PASS", "size_mm": [183, 115], "backend": "python/matplotlib",
    "text_artists": len(text_records), "text_box_checks": len(containment),
    "bounds_issues": issues, "quantitative_data": False,
    "usage": "approved manuscript Figure 1",
}, indent=2)+"\n", encoding="utf-8")
plt.close(fig)
print("Exported standalone PNG, SVG and PDF; all text containment checks passed")
