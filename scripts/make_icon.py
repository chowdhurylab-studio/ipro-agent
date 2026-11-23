#!/usr/bin/env python3
"""Draw the plugin's listing icon.

The mark is the problem the tool solves, not an abstraction of it: the ring is
the 25 OmpF constriction positions of the shipped example, the faint circle
through them is the wild-type eyelet at 8.07 A (docs/DESIGN.md section 4), and
the filled disc is the sub-4 A bottleneck PoreDesigner asked for. The two radii
are in the measured proportion, 3.99 / 8.07, rather than chosen by eye.

Rendered at 4x and downsampled, because PIL's ellipse has no antialiasing.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 512
SCALE = 4
BG = (13, 17, 23)        # near-black, reads on both light and dark listings
DOT = (201, 209, 217)    # the design positions
RING = (48, 62, 74)      # the wild-type eyelet, faint
PORE = (79, 179, 164)    # the bottleneck that passes the filter

WILD_TYPE_A = 8.07       # docs/DESIGN.md section 4
DESIGNED_A = 3.99        # docs/DESIGN.md section 7o
N_POSITIONS = 25         # examples/ompf_pore/spec_full.yaml


def draw(path: Path) -> None:
    s = SIZE * SCALE
    img = Image.new("RGB", (s, s), BG)
    d = ImageDraw.Draw(img)
    c = s / 2

    ring_r = s * 0.30                                   # where the CA ring sits
    eyelet_r = ring_r * 0.74                            # the wild-type opening
    pore_r = eyelet_r * (DESIGNED_A / WILD_TYPE_A)      # measured, not eyeballed

    d.ellipse([c - eyelet_r, c - eyelet_r, c + eyelet_r, c + eyelet_r],
              outline=RING, width=int(s * 0.006))
    d.ellipse([c - pore_r, c - pore_r, c + pore_r, c + pore_r], fill=PORE)

    dot_r = s * 0.026
    for i in range(N_POSITIONS):
        a = 2 * math.pi * i / N_POSITIONS - math.pi / 2
        x, y = c + ring_r * math.cos(a), c + ring_r * math.sin(a)
        d.ellipse([x - dot_r, y - dot_r, x + dot_r, y + dot_r], fill=DOT)

    img.resize((SIZE, SIZE), Image.LANCZOS).save(path, "PNG", optimize=True)


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "icon.png"
    draw(out)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB, {SIZE}x{SIZE})")
