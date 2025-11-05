#!/usr/bin/env python3
import re, sys, pathlib

VERSION_PATH = pathlib.Path(__file__).resolve().parent.parent / "VERSION"
PARTS = ("major", "minor", "patch")

def parse(v: str):
    m = re.match(r"^\s*(\d+)\.(\d+)\.(\d+)\s*$", v)
    if not m:
        raise SystemExit(f"Invalid VERSION '{v}'. Expected X.Y.Z")
    return [int(m.group(1)), int(m.group(2)), int(m.group(3))]

def formatv(p):
    return f"{p[0]}.{p[1]}.{p[2]}"

def bump(parts, which):
    i = PARTS.index(which)
    parts[i] += 1
    # limit to 0..99 for minor/patch; cascade up
    if which == "patch" and parts[2] > 99:
        parts[2] = 0
        parts[1] += 1
    if (which in ("patch", "minor")) and parts[1] > 99:
        parts[1] = 0
        parts[0] += 1
    if which == "major":
        # on major bump, reset minor/patch
        parts[1] = 0
        parts[2] = 0
    return parts

def main():
    which = "patch"
    if len(sys.argv) > 1:
        which = sys.argv[1].lower()
        if which not in PARTS:
            raise SystemExit(f"Use one of: {PARTS}")
    if not VERSION_PATH.exists():
        VERSION_PATH.write_text("1.0.0", encoding="utf-8")
    cur = VERSION_PATH.read_text(encoding="utf-8").strip() or "1.0.0"
    parts = parse(cur)
    new = formatv(bump(parts, which))
    VERSION_PATH.write_text(new + "\n", encoding="utf-8")
    print(new)

if __name__ == "__main__":
    main()
