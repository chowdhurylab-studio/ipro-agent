"""Render a terminal-session GIF from a real captured transcript.

Every output line here was produced by running the commands shown; nothing is
staged. Lines are wrapped at the terminal width and trimmed with a visible
ellipsis where a note is longer than the frame.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

COLS, ROWS = 92, 26
FONT = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 15)
BOLD = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 15, index=1)
PAD, TITLE_H = 14, 30
CW = int(FONT.getlength("M"))
LH = 19
W = COLS * CW + 2 * PAD
H = TITLE_H + ROWS * LH + 2 * PAD

BG, FG = (24, 26, 32), (205, 212, 222)
DIM, GREEN, CYAN, YELLOW = (120, 128, 142), (126, 199, 134), (104, 182, 224), (226, 184, 96)
TITLE_BG = (34, 37, 45)


def colour(line: str) -> tuple:
    s = line.strip()
    if line.startswith("$ "):
        return GREEN
    if s.startswith("- ") or s.startswith("notes:"):
        return YELLOW
    if "->" in s and s.split("->")[0].strip().isdigit():
        return CYAN
    if s.startswith(("mode:", "score:", "mutations", "spec is usable", "design space")):
        return CYAN
    if s.startswith(("structure:", "spec:", "goal:", "design")):
        return FG
    return FG


def frame(lines: list[str], cursor: bool = False) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, TITLE_H], fill=TITLE_BG)
    for i, c in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
        d.ellipse([14 + i * 18, 10, 24 + i * 18, 20], fill=c)
    d.text((W // 2 - 110, 7), "ipro-agent  —  inspect, validate, run", font=FONT, fill=DIM)
    y = TITLE_H + PAD
    for ln in lines[-ROWS:]:
        f = BOLD if ln.startswith("$ ") else FONT
        d.text((PAD, y), ln[:COLS], font=f, fill=colour(ln))
        y += LH
    if cursor and len(lines) <= ROWS:
        x = PAD + CW * len(lines[-1][:COLS]) if lines else PAD
        d.rectangle([x, y - LH + 2, x + CW - 2, y - 3], fill=(205, 212, 222))
    return img


def wrap(text: str, width: int, indent: str = "    ") -> list[str]:
    out, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = indent + word
        else:
            # No .strip(): it ate the continuation indent.
            line = f"{line} {word}" if line else word
    if line:
        out.append(line)
    return out


def build(script: list[tuple], out: Path) -> None:
    frames, delays = [], []
    shown: list[str] = []
    for kind, payload in script:
        if kind == "cmd":
            shown.append("$ ")
            for i in range(1, len(payload) + 1, 3):
                shown[-1] = "$ " + payload[:i]
                frames.append(frame(shown, cursor=True))
                delays.append(45)
            shown[-1] = "$ " + payload
            frames.append(frame(shown, cursor=True))
            delays.append(400)
        elif kind == "out":
            for ln in payload:
                shown.append(ln)
                frames.append(frame(shown))
                delays.append(90)
        elif kind == "hold":
            frames.append(frame(shown))
            delays.append(payload)
        elif kind == "blank":
            shown.append("")
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=64) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=delays,
                loop=0, optimize=True, disposal=2)
    print(f"{out}  {out.stat().st_size / 1024:.0f} KB, {len(frames)} frames")


if __name__ == "__main__":
    out = Path(sys.argv[1])
    S = [
        ("cmd", "python3 -m ipro_agent inspect 2omf_A.pdb --positions 18,20,38,40"),
        ("out", [
            "structure: 2omf_A.pdb",
            "  chain A: 340 residues, A1..A340",
            "  18 -> A18 is VAL",
            "      - {position: A18, allowed: [standard]}",
            "  20 -> A20 is LEU",
            "  38 -> A38 is MET",
            "  40 -> A40 is TYR",
        ]),
        ("hold", 1700),
        ("blank", None),
        ("cmd", "python3 -m ipro_agent validate my_case.yaml"),
        ("out", [
            "design positions (4):",
            "  A18: VAL->9 allowed (A,I,L,M,F,P,W,Y,V)",
            "  A20: LEU->9 allowed (A,I,L,M,F,P,W,Y,V)",
            "design space: 6561 sequences",
            "spec is usable",
        ]),
        ("hold", 1500),
        ("blank", None),
        ("cmd", "python3 -m ipro_agent run my_case.yaml --run-dir runs/case1 --iterations 2"),
        ("out", [
            "mode: fixed-backbone",
            "iterations: 2 (accepted 2)",
            "score: 1.5283 -> 1.5283 (change +0.0000)",
            "mutations (0): (none -- native retained)",
            "notes:",
        ] + wrap("- these allowed residues are NOT reachable, because every rotamer of "
                 "them clashes with the current template beyond the 40.0 kcal/mol pruning "
                 "window -- A18: TRP, TYR; A20: PRO; A38: PHE. ...", COLS - 2, "    ")
          + wrap("- THE PORE RADIUS HERE IS NOT A PORE RADIUS. ... the axis origin clears "
                 "the nearest surface by only 0.30 A -- less than the 1.40 A a water "
                 "needs ...", COLS - 2, "    ")),
        ("hold", 3800),
    ]
    build(S, out)
