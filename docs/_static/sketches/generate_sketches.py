"""Generate EnerGNN documentation sketches as SVG (light/dark variants).

Each figure is authored once as a function of the "ink" (foreground) colour so
that the ``*_black`` (light-theme) and ``*_white`` (dark-theme) variants can
never drift apart. Accent colours are chosen to read on both a white and a
near-black background, so they stay constant across variants.

Run ``python docs/_static/sketches/generate_sketches.py`` from anywhere to
(re)generate the SVGs into ``docs/_static``. No third-party dependency needed.
"""

import os

# Emit next to the docs/_static directory (this script lives in _static/sketches/).
STATIC = os.environ.get("STATIC_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
MONO = "'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace"

# Accent palette (constant across light/dark).
TEAL = "#0fa3a3"
BLUE = "#4c8bf5"
PURPLE = "#9b6dff"
ORANGE = "#e2892b"
RED = "#e5484d"
GREEN = "#2ea043"
PINK = "#e26aa5"


def variants(ink, muted, faint):
    return dict(ink=ink, muted=muted, faint=faint)


LIGHT = variants(ink="#1b1b1b", muted="#5b6470", faint="#c7ccd4")
DARK = variants(ink="#ececec", muted="#a7b0bd", faint="#4a515c")


def box(x, y, w, h, *, stroke, fill="none", rx=10, sw=2, dash=None, opacity=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d} opacity="{opacity}"/>'
    )


def text(x, y, s, *, fill, size=17, weight=400, anchor="middle", family=FONT, style="normal", spacing=None):
    ls = f' letter-spacing="{spacing}"' if spacing else ""
    return (
        f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" font-style="{style}" fill="{fill}" '
        f'text-anchor="{anchor}"{ls}>{s}</text>'
    )


def arrow(x1, y1, x2, y2, *, stroke, sw=2.2, marker="arrow", dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
        f'stroke-width="{sw}" marker-end="url(#{marker})"{d}/>'
    )


def defs(c):
    def m(name, color):
        return (
            f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0 0 L10 5 L0 10 z" fill="{color}"/></marker>'
        )
    return (
        "<defs>"
        + m("arrow", c["ink"])
        + m("arrowMuted", c["muted"])
        + m("arrowTeal", TEAL)
        + m("arrowBlue", BLUE)
        + m("arrowPurple", PURPLE)
        + m("arrowOrange", ORANGE)
        + m("arrowRed", RED)
        + "</defs>"
    )


def svg(w, h, body, c):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}" font-family="{FONT}">'
        + defs(c)
        + body
        + "</svg>"
    )


def write(name, w, h, build):
    for suffix, c in (("black", LIGHT), ("white", DARK)):
        content = svg(w, h, build(c), c)
        path = os.path.join(STATIC, f"{name}_{suffix}.svg")
        with open(path, "w") as fh:
            fh.write(content)
        print("wrote", path)


# ---------------------------------------------------------------------------
# Figure 1 : GNN architecture pipeline
# ---------------------------------------------------------------------------

def fig_gnn_pipeline(c):
    W = 1080
    b = []
    modules = [
        ("Normalizer", "rescale input features", TEAL, "arrowTeal"),
        ("Encoder", "features → latent", BLUE, "arrowBlue"),
        ("Coupler", "message passing × N", PURPLE, "arrowPurple"),
        ("Decoder", "latent → decision", ORANGE, "arrowOrange"),
    ]
    bw, bh = 190, 88
    gap = 42
    x0 = 92
    top = 152
    # input chip
    b.append(text(46, top + bh / 2 - 4, "x", fill=c["ink"], size=30, style="italic", family="Georgia, serif"))
    b.append(text(46, top + bh / 2 + 22, "context", fill=c["muted"], size=13))
    b.append(arrow(70, top + bh / 2, x0 - 8, top + bh / 2, stroke=c["ink"]))

    # data-type labels flowing between modules
    flow_labels = ["Graph", "Graph", "h : A×d", None]

    xs = []
    for i, (title, sub, accent, amark) in enumerate(modules):
        x = x0 + i * (bw + gap)
        xs.append(x)
        b.append(box(x, top, bw, bh, stroke=accent, fill=accent, rx=12, sw=2.4, opacity=0.10))
        b.append(box(x, top, bw, bh, stroke=accent, fill="none", rx=12, sw=2.4))
        b.append(text(x + bw / 2, top + 37, title, fill=c["ink"], size=21, weight=600))
        b.append(text(x + bw / 2, top + 63, sub, fill=c["muted"], size=13.5))
        # arrow to next
        if i < len(modules) - 1:
            ax1 = x + bw
            ax2 = x + bw + gap
            b.append(arrow(ax1 + 4, top + bh / 2, ax2 - 4, top + bh / 2, stroke=c["ink"]))
            lbl = flow_labels[i]
            if lbl:
                b.append(text((ax1 + ax2) / 2, top - 14, lbl, fill=c["muted"], size=13, family=MONO))

    # output chip
    lastx = xs[-1] + bw
    b.append(arrow(lastx + 4, top + bh / 2, lastx + gap + 4, top + bh / 2, stroke=c["ink"]))
    ox = lastx + gap + 20
    b.append(text(ox + 14, top + bh / 2 - 4, "ŷ", fill=c["ink"], size=30, style="italic", family="Georgia, serif"))
    b.append(text(ox + 14, top + bh / 2 + 22, "decision", fill=c["muted"], size=13))

    # theta (trainable weights) fan into the three learnable modules
    tx = xs[2] + bw / 2  # above the coupler (central learnable module)
    ty = top - 56
    b.append(text(tx, ty, "θ", fill=RED, size=22, style="italic", family="Georgia, serif"))
    b.append(text(tx + 20, ty - 1, "trainable weights", fill=c["muted"], size=12.5, anchor="start"))
    for i in (1, 2, 3):
        cx = xs[i] + bw / 2
        b.append(arrow(tx, ty + 8, cx, top - 3, stroke=RED, sw=1.5, marker="arrowRed", dash="4 3"))

    # bracket under encoder..decoder : "GNN core (vmapped over the batch)"
    by = top + bh + 26
    bx1 = xs[1] - 6
    bx2 = xs[3] + bw + 6
    b.append(
        f'<path d="M{bx1} {by} L{bx1} {by+8} L{bx2} {by+8} L{bx2} {by}" '
        f'fill="none" stroke={chr(34)}{c["muted"]}{chr(34)} stroke-width="1.6"/>'
    )
    b.append(text((bx1 + bx2) / 2, by + 30, "GNN core — encoder · coupler · decoder, vmapped over the batch",
                  fill=c["muted"], size=14))
    # normalizer note
    b.append(text(xs[0] + bw / 2, by + 30, "runs once", fill=c["muted"], size=13, style="italic"))

    # title
    b.append(text(W / 2, 40, "GNN forward pass", fill=c["ink"], size=22, weight=600))
    b.append(text(W / 2, 62, "energnn.model.GNN  ·  each module subclasses a Flax nnx.Module interface",
                  fill=c["muted"], size=13.5, family=MONO))
    return "".join(b)


# ---------------------------------------------------------------------------
# Figure 2 : H2MG data structure (code level)
# ---------------------------------------------------------------------------

def fig_data_structure(c):
    W, H = 1080, 560
    b = []
    b.append(text(W / 2, 38, "Anatomy of an H2MG Graph", fill=c["ink"], size=22, weight=600))
    b.append(text(W / 2, 60, "energnn.graph.Graph  —  a dict of hyper-edge sets over a shared pool of addresses",
                  fill=c["muted"], size=13.5, family=MONO))

    # Graph container (left)
    gx, gy, gw, gh = 40, 96, 300, 250
    b.append(box(gx, gy, gw, gh, stroke=c["ink"], fill=c["ink"], rx=14, sw=2, opacity=0.04))
    b.append(box(gx, gy, gw, gh, stroke=c["ink"], fill="none", rx=14, sw=2))
    b.append(text(gx + 18, gy + 30, "Graph", fill=c["ink"], size=19, weight=600, anchor="start"))
    fields = [
        ("hyper_edge_sets", "dict[str, HyperEdgeSet]", BLUE),
        ("non_fictitious_addresses", "mask over the address pool", ORANGE),
        ("true_shape / current_shape", "GraphShape (padding-aware)", c["muted"]),
    ]
    fy = gy + 62
    for name, desc, col in fields:
        b.append(f'<rect x="{gx+16}" y="{fy-18}" width="8" height="8" rx="2" fill="{col}"/>')
        b.append(text(gx + 34, fy - 10, name, fill=c["ink"], size=14.5, weight=600, anchor="start", family=MONO))
        b.append(text(gx + 34, fy + 9, desc, fill=c["muted"], size=12.5, anchor="start"))
        fy += 54

    # Two HyperEdgeSet cards (middle)
    hx = 400
    cards = [
        ("\"lines\"", [("port_dict", "{ bus1, bus2 } → addr", PURPLE),
                        ("feature_array", "[n × f]  (r, x, …)", GREEN),
                        ("non_fictitious", "[n]  real vs padded", ORANGE)], 96),
        ("\"generators\"", [("port_dict", "{ bus } → addr", PURPLE),
                             ("feature_array", "[m × f]  (p0, q0)", GREEN),
                             ("non_fictitious", "[m]", ORANGE)], 300),
    ]
    cw, ch = 300, 176
    for label, rows, cy in cards:
        b.append(box(hx, cy, cw, ch, stroke=BLUE, fill=BLUE, rx=12, sw=2, opacity=0.08))
        b.append(box(hx, cy, cw, ch, stroke=BLUE, fill="none", rx=12, sw=2))
        b.append(text(hx + 16, cy + 28, "HyperEdgeSet", fill=BLUE, size=13, weight=600, anchor="start", family=MONO))
        b.append(text(hx + cw - 14, cy + 28, label, fill=c["ink"], size=16, weight=600, anchor="end", family=MONO))
        ry = cy + 60
        for name, desc, col in rows:
            b.append(f'<rect x="{hx+16}" y="{ry-15}" width="8" height="8" rx="2" fill="{col}"/>')
            b.append(text(hx + 34, cy_off := ry - 7, name, fill=c["ink"], size=13.5, weight=600, anchor="start", family=MONO))
            b.append(text(hx + 34, ry + 11, desc, fill=c["muted"], size=12, anchor="start"))
            ry += 40
    # arrow Graph -> hyper_edge_sets cards
    b.append(arrow(gx + gw + 4, gy + 78, hx - 6, 96 + 20, stroke=BLUE, marker="arrowBlue", sw=1.8))
    b.append(arrow(gx + gw + 4, gy + 84, hx - 6, 300 + 20, stroke=BLUE, marker="arrowBlue", sw=1.8))

    # Address pool (right)
    ax, ay, aw, ah = 760, 150, 280, 300
    b.append(box(ax, ay, aw, ah, stroke=ORANGE, fill=ORANGE, rx=12, sw=2, opacity=0.07))
    b.append(box(ax, ay, aw, ah, stroke=ORANGE, fill="none", rx=12, sw=2))
    b.append(text(ax + aw / 2, ay + 30, "Address pool", fill=ORANGE, size=17, weight=600))
    b.append(text(ax + aw / 2, ay + 50, "shared interface between hyper-edges", fill=c["muted"], size=11.5))
    b.append(text(ax + aw / 2, ay + 68, "no features — only connectivity", fill=c["muted"], size=11.5, style="italic"))
    # squares 0..7
    per = 4
    sq = 44
    sgap = 18
    grid_w = per * sq + (per - 1) * sgap
    sx0 = ax + (aw - grid_w) / 2
    sy0 = ay + 92
    for i in range(8):
        r, cc = divmod(i, per)
        sx = sx0 + cc * (sq + sgap)
        sy = sy0 + r * (sq + sgap + 14)
        b.append(box(sx, sy, sq, sq, stroke=c["ink"], fill=c["ink"], rx=7, sw=1.6, opacity=0.06))
        b.append(box(sx, sy, sq, sq, stroke=c["ink"], fill="none", rx=7, sw=1.6))
        b.append(text(sx + sq / 2, sy + sq / 2 + 6, str(i), fill=c["ink"], size=16, weight=600, family=MONO))
    b.append(text(ax + aw / 2, sy0 + 2 * (sq + sgap + 14) + 6, "a ∈ { 0, 1, …, A−1 }", fill=c["muted"], size=12.5, family=MONO))

    # arrows from port_dict rows into pool
    b.append(arrow(hx + cw + 4, 96 + 66, ax - 6, ay + 118, stroke=PURPLE, marker="arrowPurple", sw=1.8, dash="5 4"))
    b.append(arrow(hx + cw + 4, 300 + 66, ax - 6, ay + 176, stroke=PURPLE, marker="arrowPurple", sw=1.8, dash="5 4"))
    b.append(text((hx + cw + ax) / 2, 250, "ports index", fill=PURPLE, size=12.5, family=MONO))
    b.append(text((hx + cw + ax) / 2, 266, "addresses", fill=PURPLE, size=12.5, family=MONO))

    b.append(text(W / 2, H - 16,
                  "A JaxBackend makes every array a JAX array — differentiable and jit / vmap-friendly.",
                  fill=c["muted"], size=13))
    return "".join(b)


# ---------------------------------------------------------------------------
# Figure 3 : One recurrent message-passing step
# ---------------------------------------------------------------------------

def fig_message_passing(c):
    W, H = 1080, 400
    b = []
    b.append(text(W / 2, 38, "One message-passing step (RecurrentCoupler)", fill=c["ink"], size=22, weight=600))
    b.append(text(W / 2, 60, "repeated N times as an explicit Euler update  h ← h + Δt · φ(ψ¹, …, ψⁿ)",
                  fill=c["muted"], size=13.5, family=MONO))

    stages = [
        ("gather", "read coordinates h at each\nport address of every edge", BLUE),
        ("ξ  per class / port MLP", "apply edge & port-specific\nMLP to  [ h_ports , x_e ]", PURPLE),
        ("scatter-add → σ", "sum incoming messages\nback onto each address a", ORANGE),
        ("φ  ·  Euler step", "h ← h + Δt · φ(messages)\nloop for N steps", TEAL),
    ]
    bw, bh = 224, 130
    gap = 226 - 224 + 24
    x0 = 40
    top = 150
    xs = []
    for i, (title, sub, accent) in enumerate(stages):
        x = x0 + i * (bw + 24)
        xs.append(x)
        b.append(box(x, top, bw, bh, stroke=accent, fill=accent, rx=12, sw=2.2, opacity=0.10))
        b.append(box(x, top, bw, bh, stroke=accent, fill="none", rx=12, sw=2.2))
        b.append(text(x + bw / 2, top + 34, str(i + 1), fill=accent, size=13, weight=700))
        b.append(text(x + bw / 2, top + 60, title, fill=c["ink"], size=17, weight=600))
        for j, line in enumerate(sub.split("\n")):
            b.append(text(x + bw / 2, top + 86 + j * 19, line, fill=c["muted"], size=13))
        if i < len(stages) - 1:
            b.append(arrow(x + bw + 3, top + bh / 2, x + bw + 24 - 3, top + bh / 2, stroke=c["ink"]))

    # feedback loop arrow from last box back to first (h updated)
    y_lo = top + bh + 46
    x_last = xs[-1] + bw / 2
    x_first = xs[0] + bw / 2
    b.append(
        f'<path d="M{x_last} {top+bh} L{x_last} {y_lo} L{x_first} {y_lo} L{x_first} {top+bh}" '
        f'fill="none" stroke="{PURPLE}" stroke-width="2" stroke-dasharray="6 4" marker-end="url(#arrowPurple)"/>'
    )
    b.append(text(W / 2, y_lo + 22, "updated coordinates  h  feed the next step", fill=PURPLE, size=14))

    # top math strip
    b.append(text(W / 2, 108,
                  "ψ(h, x)_a  =  σ ( Σ  ξᶜʳᵒ ( h_e , x_e ) )   over all edges incident to address a",
                  fill=c["ink"], size=15, family=MONO))

    b.append(text(W / 2, H - 18,
                  "Equivariant by construction: it uses only gather / MLP / scatter-add — no address ordering.",
                  fill=c["muted"], size=13))
    return "".join(b)


if __name__ == "__main__":
    write("energnn_gnn_pipeline", 1080, 330, fig_gnn_pipeline)
    write("energnn_data_structure", 1080, 560, fig_data_structure)
    write("energnn_message_passing", 1080, 400, fig_message_passing)
    print("done")
