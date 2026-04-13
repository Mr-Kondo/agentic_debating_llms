"""Tests for app.licenses – collector, normalizer helpers, and renderer."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from app.licenses.collector import (
    PackageInfo,
    _extract_home_url,
    _extract_license,
    _find_dist_info,
    _normalize_pkg_name,
    _resolve_runtime_names,
    collect,
)
from app.licenses.renderer import render_markdown, write_markdown


# ---------------------------------------------------------------------------
# _normalize_pkg_name
# ---------------------------------------------------------------------------


def test_normalize_pkg_name_hyphens() -> None:
    assert _normalize_pkg_name("my-Package") == "my_package"


def test_normalize_pkg_name_dots() -> None:
    assert _normalize_pkg_name("my.pkg.name") == "my_pkg_name"


def test_normalize_pkg_name_mixed() -> None:
    assert _normalize_pkg_name("My_Package-1.0") == "my_package_1_0"


# ---------------------------------------------------------------------------
# _extract_license
# ---------------------------------------------------------------------------


def test_extract_license_expression_wins() -> None:
    metadata = textwrap.dedent("""\
        Metadata-Version: 2.4
        Name: foo
        Version: 1.0
        License-Expression: MIT
        License: Apache-2.0
        Classifier: License :: OSI Approved :: Apache Software License
    """)
    assert _extract_license(metadata) == "MIT"


def test_extract_license_field_fallback() -> None:
    metadata = textwrap.dedent("""\
        Metadata-Version: 2.3
        Name: bar
        Version: 0.5
        License: BSD-3-Clause
    """)
    assert _extract_license(metadata) == "BSD-3-Clause"


def test_extract_license_classifier_fallback() -> None:
    metadata = textwrap.dedent("""\
        Metadata-Version: 2.1
        Name: baz
        Classifier: License :: OSI Approved :: MIT License
    """)
    assert _extract_license(metadata) == "MIT License"


def test_extract_license_skips_generic_osi_approved() -> None:
    metadata = textwrap.dedent("""\
        Metadata-Version: 2.1
        Name: baz
        Classifier: License :: OSI Approved
    """)
    assert _extract_license(metadata) == "UNKNOWN"


def test_extract_license_unknown_when_missing() -> None:
    metadata = "Metadata-Version: 2.4\nName: qux\nVersion: 0.1\n"
    assert _extract_license(metadata) == "UNKNOWN"


def test_extract_license_ignores_unknown_field_value() -> None:
    metadata = textwrap.dedent("""\
        Metadata-Version: 2.3
        License: UNKNOWN
    """)
    assert _extract_license(metadata) == "UNKNOWN"


# ---------------------------------------------------------------------------
# _extract_home_url
# ---------------------------------------------------------------------------


def test_extract_home_url_project_url_homepage() -> None:
    metadata = textwrap.dedent("""\
        Project-URL: Homepage, https://example.com
        Project-URL: Source, https://github.com/foo/bar
    """)
    assert _extract_home_url(metadata) == "https://example.com"


def test_extract_home_url_source_fallback() -> None:
    metadata = textwrap.dedent("""\
        Project-URL: Source, https://github.com/foo/bar
    """)
    assert _extract_home_url(metadata) == "https://github.com/foo/bar"


def test_extract_home_url_legacy_home_page() -> None:
    metadata = "Home-page: https://legacy.example.com\n"
    assert _extract_home_url(metadata) == "https://legacy.example.com"


def test_extract_home_url_empty_when_missing() -> None:
    metadata = "Metadata-Version: 2.4\n"
    assert _extract_home_url(metadata) == ""


def test_extract_home_url_ignores_unknown_home_page() -> None:
    metadata = "Home-page: UNKNOWN\n"
    assert _extract_home_url(metadata) == ""


# ---------------------------------------------------------------------------
# _find_dist_info
# ---------------------------------------------------------------------------


def test_find_dist_info_exact(tmp_path: Path) -> None:
    dist_info = tmp_path / "my_pkg-1.2.3.dist-info"
    dist_info.mkdir()
    result = _find_dist_info(tmp_path, "my-pkg", "1.2.3")
    assert result == dist_info


def test_find_dist_info_returns_none_when_missing(tmp_path: Path) -> None:
    result = _find_dist_info(tmp_path, "nonexistent", "0.0.1")
    assert result is None


def test_find_dist_info_glob_fallback(tmp_path: Path) -> None:
    # Version doesn't match exactly but glob should still find it
    dist_info = tmp_path / "somelib-9.9.9.dist-info"
    dist_info.mkdir()
    result = _find_dist_info(tmp_path, "somelib", "1.0.0")
    assert result == dist_info


# ---------------------------------------------------------------------------
# _resolve_runtime_names (using a minimal synthetic lock structure)
# ---------------------------------------------------------------------------


def _make_lock_data(
    runtime_deps: list[str],
    dev_deps: list[str],
    packages: dict[str, list[str]],
) -> dict:
    """Build a minimal uv.lock data structure for testing."""
    all_packages = [
        {
            "name": "my-app",
            "version": "1.0",
            "source": {"virtual": "."},
            "dependencies": [{"name": n} for n in runtime_deps],
            "optional-dependencies": {"dev": [{"name": n} for n in dev_deps]},
        }
    ]
    for name, deps in packages.items():
        all_packages.append(
            {
                "name": name,
                "version": "0.1",
                "source": {"registry": "https://pypi.org/simple"},
                "dependencies": [{"name": d} for d in deps],
            }
        )
    return {"package": all_packages}


def test_resolve_runtime_excludes_dev_deps() -> None:
    data = _make_lock_data(
        runtime_deps=["requests"],
        dev_deps=["pytest"],
        packages={"requests": [], "pytest": ["pluggy"], "pluggy": []},
    )
    result = _resolve_runtime_names(data)
    assert "requests" in result
    assert "pytest" not in result
    assert "pluggy" not in result


def test_resolve_runtime_includes_transitive() -> None:
    data = _make_lock_data(
        runtime_deps=["a"],
        dev_deps=[],
        packages={"a": ["b"], "b": ["c"], "c": []},
    )
    result = _resolve_runtime_names(data)
    assert result == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# collect (integration-style with a fake venv layout)
# ---------------------------------------------------------------------------


def _make_fake_venv(tmp_path: Path, packages: dict[str, dict]) -> Path:
    """Create minimal dist-info METADATA files and return site-packages path."""
    site = tmp_path / "site-packages"
    site.mkdir()
    for name, meta in packages.items():
        normalized = name.replace("-", "_")
        version = meta["version"]
        dist_info = site / f"{normalized}-{version}.dist-info"
        dist_info.mkdir()
        lines = [
            f"Metadata-Version: 2.4",
            f"Name: {name}",
            f"Version: {version}",
        ]
        if "license" in meta:
            lines.append(f"License-Expression: {meta['license']}")
        if "url" in meta:
            lines.append(f"Project-URL: Homepage, {meta['url']}")
        (dist_info / "METADATA").write_text("\n".join(lines), encoding="utf-8")
    return site


def _make_fake_lock(tmp_path: Path, packages: dict[str, str]) -> Path:
    """Write a minimal TOML uv.lock file and return its path.

    Uses inline tables for ``source`` so that ``dependencies`` stays at the
    ``[[package]]`` level, matching the real uv.lock format.
    """
    runtime_deps = list(packages.keys())
    dep_entries = ", ".join(f'{{ name = "{n}" }}' for n in runtime_deps)
    blocks = [
        'version = 1',
        '',
        '[[package]]',
        'name = "test-app"',
        'version = "0.1.0"',
        'source = { virtual = "." }',
        f'dependencies = [{dep_entries}]',
        '',
    ]
    for name, version in packages.items():
        blocks += [
            '[[package]]',
            f'name = "{name}"',
            f'version = "{version}"',
            'source = { registry = "https://pypi.org/simple" }',
            '',
        ]
    lock_path = tmp_path / "uv.lock"
    lock_path.write_text("\n".join(blocks), encoding="utf-8")
    return lock_path


def test_collect_returns_installed_packages(tmp_path: Path) -> None:
    site = _make_fake_venv(tmp_path, {"alpha": {"version": "1.0", "license": "MIT", "url": "https://alpha.example.com"}})
    lock = _make_fake_lock(tmp_path, {"alpha": "1.0"})
    result = collect(lock, site)
    assert len(result) == 1
    assert result[0].name == "alpha"
    assert result[0].version == "1.0"
    assert result[0].license == "MIT"
    assert result[0].home_url == "https://alpha.example.com"


def test_collect_skips_uninstalled_packages(tmp_path: Path) -> None:
    # 'beta' in lockfile but no dist-info (platform-conditional, not installed)
    site = _make_fake_venv(tmp_path, {"alpha": {"version": "1.0", "license": "MIT"}})
    lock = _make_fake_lock(tmp_path, {"alpha": "1.0", "beta": "2.0"})
    result = collect(lock, site)
    names = [p.name for p in result]
    assert "alpha" in names
    assert "beta" not in names


def test_collect_sorted_by_name(tmp_path: Path) -> None:
    pkgs = {"zeta": {"version": "1.0", "license": "MIT"}, "alpha": {"version": "2.0", "license": "MIT"}}
    site = _make_fake_venv(tmp_path, pkgs)
    lock = _make_fake_lock(tmp_path, {name: meta["version"] for name, meta in pkgs.items()})
    result = collect(lock, site)
    assert [p.name for p in result] == ["alpha", "zeta"]


# ---------------------------------------------------------------------------
# renderer
# ---------------------------------------------------------------------------


def test_render_markdown_contains_package_info() -> None:
    packages = [
        PackageInfo(name="alpha", version="1.0", license="MIT", home_url="https://example.com"),
        PackageInfo(name="beta", version="2.0", license="Apache-2.0", home_url=""),
    ]
    md = render_markdown(packages)
    assert "alpha" in md
    assert "1.0" in md
    assert "`MIT`" in md
    assert "[link](https://example.com)" in md
    assert "beta" in md
    assert "`Apache-2.0`" in md
    assert "—" in md  # no URL for beta


def test_render_markdown_has_table_header() -> None:
    md = render_markdown([])
    assert "| Package | Version | License | Project |" in md


def test_write_markdown_creates_file(tmp_path: Path) -> None:
    packages = [PackageInfo(name="x", version="0.1", license="MIT", home_url="")]
    out = tmp_path / "sub" / "licenses.md"
    write_markdown(packages, out)
    assert out.exists()
    assert "x" in out.read_text(encoding="utf-8")
