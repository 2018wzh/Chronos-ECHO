from pathlib import Path

import nbformat


NOTEBOOK = Path("notebooks/timemmd_colab_benchmark.ipynb")


def _notebook_source() -> str:
    return "\n".join(cell.get("source", "") for cell in nbformat.read(NOTEBOOK, as_version=4).cells)


def test_timemmd_colab_notebook_is_valid_and_standalone():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)

    source = _notebook_source()
    assert "DATA_ROOT" in source
    assert "MODELS" in source
    assert "AURORA_REFERENCE" in source
    assert "MANIFEST_CSV" in source
    assert "make_archive" in source
    assert "chronos2-echo[timemmd] @ git+https://github.com/2018wzh/Chronos-ECHO.git" in source
    assert "from scripts.timemmd" not in source
    assert "scripts/timemmd" not in source
    assert "D:\\" not in source


def test_timemmd_colab_notebook_has_expected_reader_flow():
    headings = [
        cell.source.strip().splitlines()[0]
        for cell in nbformat.read(NOTEBOOK, as_version=4).cells
        if cell.cell_type == "markdown" and cell.source.strip().startswith("## ")
    ]

    assert headings[:6] == [
        "## Goal",
        "## Setup",
        "## Parameters",
        "## Benchmark Helpers",
        "## Run",
        "## Results",
    ]
