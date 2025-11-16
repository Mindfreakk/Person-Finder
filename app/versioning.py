# versioning.py
import pathlib

VERSION_FILE = pathlib.Path(__file__).with_name("APP_VERSION.txt")


def get_app_version() -> str:
    """Read current app version from a file, e.g. '1.2.3'."""
    if VERSION_FILE.exists():
        return VERSION_FILE.read_text(encoding="utf-8").strip()
    return "v1.0.0"  # default/fallback


def bump_patch_version() -> str:
    """
    Simple example: bump patch number (x.y.Z → x.y.(Z+1)),
    return new version string and write it back.
    """
    current = get_app_version().lstrip("v")
    parts = (current or "1.0.0").split(".")
    while len(parts) < 3:
        parts.append("0")
    major, minor, patch = map(int, parts[:3])
    patch += 1
    new_version = f"v{major}.{minor}.{patch}"
    VERSION_FILE.write_text(new_version, encoding="utf-8")
    return new_version
