"""Convert and verify report diaries using the MilenSem2D converter."""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Final

import nbformat
from nbclient import NotebookClient

from scripts.project_paths import PROJECT_ROOT

DIARIES: Final = ("dev_1_1_dataset_preprocessing", "dev_1_2_smoothing_pilot", "dev_1_3_point_source", "dev_1_4_server_preparation")


def semantic_tree(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            node.value.value = node.value.value.strip()
    return ast.dump(tree)


def build(stem: str, execute: bool) -> None:
    source = PROJECT_ROOT / f"{stem}.py"
    output = source.with_suffix(".ipynb")
    subprocess.run([sys.executable, "-m", "py2jupyter", str(source), str(output)], check=True)
    notebook = nbformat.read(output, as_version=4)
    notebook.nbformat_minor = 5
    notebook.metadata.kernelspec = {"display_name": "SeismicDataset (.venv)", "language": "python", "name": "seismic-dataset"}
    notebook.metadata.language_info = {"name": "python", "version": sys.version.split()[0]}
    for index, cell in enumerate(notebook.cells):
        cell["id"] = f"{stem}-{index:02d}"
        if cell.cell_type == "code":
            ast.parse(cell.source)
    nbformat.validate(notebook)
    nbformat.write(notebook, output)
    with tempfile.TemporaryDirectory(prefix="notebook-roundtrip-", dir=PROJECT_ROOT) as temporary:
        recovered = Path(temporary) / source.name
        subprocess.run([sys.executable, "-m", "py2jupyter", str(output), str(recovered)], check=True)
        if semantic_tree(source) != semantic_tree(recovered):
            raise ValueError(f"Notebook roundtrip changed the diary: {stem}")
    if execute:
        os.environ["JUPYTER_PATH"] = str(PROJECT_ROOT / ".venv/share/jupyter")
        os.environ["MPLBACKEND"] = "Agg"
        NotebookClient(notebook, timeout=30, kernel_name="seismic-dataset",
                       resources={"metadata": {"path": str(PROJECT_ROOT)}}).execute()
        nbformat.write(notebook, output)
    print(f"{stem}: {len(notebook.cells)} cells; syntax, format and roundtrip OK; executed={execute}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run only report diaries, no solver or mass processing")
    parser.add_argument("--diary", choices=DIARIES, action="append", help="Convert only the selected diary; repeat for several")
    args = parser.parse_args()
    for stem in args.diary or DIARIES:
        build(stem, args.execute)


if __name__ == "__main__":
    main()
