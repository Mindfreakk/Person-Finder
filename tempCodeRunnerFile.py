# ----------------- Version Control -----------------
VERSION_FILE = "VERSION"

def get_version():
    """Read the version from the VERSION file, or initialize it."""
    if not os.path.exists(VERSION_FILE):
        with open(VERSION_FILE, "w") as f:
            f.write("1.0.0.0")
    with open(VERSION_FILE, "r") as f:
        version = f.read().strip()
        parts = version.split(".")
        while len(parts) < 4:
            parts.append("0")
        return ".".join(parts)

def bump_version():
    """Automatically increment version with cascading: build -> patch -> minor -> major."""
    version = get_version()
    major, minor, patch, build = map(int, version.split("."))

    build += 1
    if build > 9:
        build = 0
        patch += 1
        if patch > 9:
            patch = 0
            minor += 1
            if minor > 9:
                minor = 0
                major += 1

    new_version = f"{major}.{minor}.{patch}.{build}"
    with open(VERSION_FILE, "w") as f:
        f.write(new_version)
    return new_version

# ----------------- Auto Bump on Actual Server Start -----------------
if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or os.environ.get("FLASK_RUN_FROM_CLI") == "true":
    APP_VERSION = bump_version()
else:
    APP_VERSION = get_version()