#!/usr/bin/env python3
# scripts/bump_version.py
import sys
import json
from datetime import datetime
from pathlib import Path
import subprocess

VERSION_FILE = Path(__file__).resolve().parents[1] / 'static' / 'version.json'

def read_version():
    if not VERSION_FILE.exists():
        return {"version": "0.0.0"}
    return json.loads(VERSION_FILE.read_text(encoding='utf-8'))

def write_version(data):
    VERSION_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding='utf-8')

def bump_semver(vstr, kind):
    # simple semantic version bump
    parts = vstr.split('-', 1)
    base = parts[0]
    pre = '-' + parts[1] if len(parts) > 1 else ''
    major, minor, patch = (base.split('.') + ['0','0','0'])[:3]
    major = int(major); minor = int(minor); patch = int(patch)
    if kind == 'major':
        major += 1; minor = 0; patch = 0
    elif kind == 'minor':
        minor += 1; patch = 0
    elif kind == 'patch':
        patch += 1
    else:
        raise ValueError("kind must be one of: major, minor, patch")
    return f"{major}.{minor}.{patch}{pre}"

def git_commit_new_version(new_version):
    try:
        subprocess.run(['git', 'add', str(VERSION_FILE)], check=True)
        subprocess.run(['git', 'commit', '-m', f'Bump version to {new_version}'], check=True)
    except Exception:
        # ignore if git not available / no repo
        pass

def main():
    if len(sys.argv) < 2:
        print("Usage: bump_version.py [patch|minor|major]")
        sys.exit(1)
    kind = sys.argv[1]
    data = read_version()
    old = data.get('version', '0.0.0')
    new = bump_semver(old, kind)
    data['version'] = new
    data['build'] = datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'
    write_version(data)
    print(f"Version bumped: {old} -> {new}")
    # optionally commit
    git_commit_new_version(new)
    print("Wrote", VERSION_FILE)

if __name__ == '__main__':
    main()
