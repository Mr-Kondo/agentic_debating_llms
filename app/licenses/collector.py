"""Collect package license information from uv.lock and dist-info metadata."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from re import sub


@dataclass(slots=True, frozen=True)
class PackageInfo:
    """License and project metadata for one installed package."""

    name: str
    version: str
    license: str
    home_url: str


def _normalize_pkg_name(name: str) -> str:
    """Convert package name to canonical form used in dist-info directory names."""
    return sub(r"[-_.]+", "_", name).lower()


def _find_dist_info(site_packages: Path, name: str, version: str) -> Path | None:
    """Locate the dist-info directory for a given package name and version."""
    normalized = _normalize_pkg_name(name)
    # Exact match first
    exact = site_packages / f"{normalized}-{version}.dist-info"
    if exact.is_dir():
        return exact
    # Case-insensitive glob fallback (handles mixed capitalisation)
    for candidate in site_packages.glob(f"{normalized}-*.dist-info"):
        if candidate.is_dir():
            return candidate
    return None


def _extract_license(metadata_text: str) -> str:
    """Extract license identifier from METADATA text.

    Priority order:
      1. License-Expression header (SPDX, modern packaging standard)
      2. License header (legacy field)
      3. License :: ... classifiers (oldest fallback)
    Returns "UNKNOWN" when no usable information is found.
    """
    license_expression: str | None = None
    license_field: str | None = None
    license_classifiers: list[str] = []

    for line in metadata_text.splitlines():
        if line.startswith("License-Expression:"):
            license_expression = line.split(":", 1)[1].strip()
        elif line.startswith("License:") and not line.startswith("License-Expression:"):
            value = line.split(":", 1)[1].strip()
            if value and value.upper() != "UNKNOWN":
                license_field = value
        elif line.startswith("Classifier: License ::"):
            parts = line.split(" :: ")
            if len(parts) >= 3:
                # Skip generic "OSI Approved" without a specific name
                label = parts[-1].strip()
                if label != "OSI Approved":
                    license_classifiers.append(label)

    if license_expression:
        return license_expression
    if license_field:
        return license_field
    if license_classifiers:
        return " / ".join(license_classifiers)
    return "UNKNOWN"


def _extract_home_url(metadata_text: str) -> str:
    """Extract a representative project homepage URL from METADATA text."""
    home_page: str = ""
    project_urls: dict[str, str] = {}

    for line in metadata_text.splitlines():
        if line.startswith("Home-page:"):
            value = line.split(":", 1)[1].strip()
            if value and value.upper() != "UNKNOWN":
                home_page = value
        elif line.startswith("Project-URL:"):
            rest = line.split(":", 1)[1].strip()
            if "," in rest:
                label, url = rest.split(",", 1)
                project_urls[label.strip().lower()] = url.strip()

    for label in ("homepage", "source", "repository", "documentation"):
        if label in project_urls:
            return project_urls[label]
    return home_page


def _resolve_runtime_names(lock_data: dict) -> set[str]:
    """BFS through uv.lock to collect all runtime package names.

    Starts from the root project's direct runtime dependencies (non-dev)
    and follows each package's own ``dependencies`` transitively.
    """
    all_packages: dict[str, dict] = {pkg["name"]: pkg for pkg in lock_data["package"]}

    root = next(
        pkg for pkg in lock_data["package"] if pkg.get("source", {}).get("virtual") == "."
    )

    # Seed with direct runtime deps only (not optional-dependencies.dev)
    queue: list[str] = [dep["name"] for dep in root.get("dependencies", [])]
    visited: set[str] = set()

    while queue:
        name = queue.pop()
        if name in visited:
            continue
        visited.add(name)
        pkg = all_packages.get(name, {})
        for dep in pkg.get("dependencies", []):
            dep_name = dep["name"]
            if dep_name not in visited:
                queue.append(dep_name)

    return visited


def collect(lock_path: Path, site_packages: Path) -> list[PackageInfo]:
    """Return license info for all runtime packages listed in *lock_path*.

    Reads ``uv.lock`` to determine the runtime dependency closure, then
    reads each package's distribution metadata from *site_packages*.
    Results are sorted alphabetically by package name.
    """
    with lock_path.open("rb") as fh:
        lock_data = tomllib.load(fh)

    all_packages: dict[str, dict] = {pkg["name"]: pkg for pkg in lock_data["package"]}
    runtime_names = _resolve_runtime_names(lock_data)

    results: list[PackageInfo] = []
    for name in sorted(runtime_names):
        version = all_packages[name]["version"]
        dist_info = _find_dist_info(site_packages, name, version)

        if dist_info is None:
            # Package is in the lock file but not installed on this platform
            # (e.g. Windows-only conditional dependency). Skip it.
            continue

        license_str = "UNKNOWN"
        home_url = ""

        for meta_name in ("METADATA", "PKG-INFO"):
            meta_file = dist_info / meta_name
            if meta_file.exists():
                text = meta_file.read_text(encoding="utf-8", errors="replace")
                license_str = _extract_license(text)
                home_url = _extract_home_url(text)
                break

        results.append(PackageInfo(name=name, version=version, license=license_str, home_url=home_url))

    return results
