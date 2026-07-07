"""Generate committed dino frame PNGs from reviewed pixel matrices.

Run: uv run --extra gen python scripts/gen_dino_assets.py
Frames are 24x22, drawn in dark gray on transparent, matching the Chrome T-rex.
Re-run after editing a matrix; commit the resulting PNGs.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

BODY = (60, 60, 66, 255)
TRANSPARENT = (0, 0, 0, 0)
SCALE = 1

_UPPER = [
    "             #########  ",
    "            ##########  ",
    "            ##o#######  ",
    "            ##########  ",
    "            #######     ",
    "            ########    ",
    "  #        ########     ",
    "  ##      #########     ",
    "  ###    ##########     ",
    "  #####  ##########     ",
    "  #################     ",
    "   #################    ",
    "    ############# #     ",
    "     ############       ",
    "      ##########        ",
    "       ########         ",
]

_LEGS_STAND = [
    "      ####  ###       ",
    "      ###    ##       ",
    "      ##     ##       ",
    "      ##     ##       ",
    "      ##     ##       ",
    "      ##     ##       ",
]

_LEGS_RUN1 = [
    "      ####  ###       ",
    "      ###    ##       ",
    "      ##     ##       ",
    "      ##     #        ",
    "      ##              ",
    "      ###             ",
]

_LEGS_RUN2 = [
    "      ####  ###       ",
    "      ###    ##       ",
    "       #     ##       ",
    "             ##       ",
    "             ###      ",
    "            ###       ",
]

_CACTUS = [
    "  #  ",
    "  #  ",
    "# #  ",
    "# # #",
    "### #",
    "  # #",
    "  ###",
    "  #  ",
    "  #  ",
    "  #  ",
]


def _pad(rows: list[str], width: int) -> list[str]:
    return [row.ljust(width) for row in rows]


def _frame(legs: list[str], *, dead: bool = False) -> list[str]:
    rows = list(_UPPER) + legs
    if dead:
        rows = [row.replace("o", "x") for row in rows]
    return rows


def _render(rows: list[str], path: Path) -> None:
    rows = _pad(rows, max(len(row) for row in rows))
    height, width = len(rows), len(rows[0])
    img = Image.new("RGBA", (width * SCALE, height * SCALE), TRANSPARENT)
    px = img.load()
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch != "#":
                continue
            for dy in range(SCALE):
                for dx in range(SCALE):
                    px[x * SCALE + dx, y * SCALE + dy] = BODY
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main() -> None:
    out = (
        Path(__file__).resolve().parent.parent
        / "anon_proxy"
        / "assets"
        / "dino"
        / "classic"
    )
    _render(_frame(_LEGS_STAND), out / "stand.png")
    _render(_frame(_LEGS_RUN1), out / "run1.png")
    _render(_frame(_LEGS_RUN2), out / "run2.png")
    _render(_frame(_LEGS_STAND, dead=True), out / "dead.png")
    _render(_CACTUS, out / "cactus.png")
    print(f"wrote frames to {out}")


if __name__ == "__main__":
    main()
