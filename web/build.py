"""Build web/index.html from a saved training run.

Usage (from the repo root):
    python web/build.py logs/20261118-0910_64-32-32-10.json
"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python web/build.py <path to logs/*.json>")
    run = json.loads(pathlib.Path(sys.argv[1]).read_text())

    # The page's JavaScript assumes this exact architecture.
    if run["nin"] != 64 or run["nouts"] != [32, 32, 10]:
        sys.exit(f"page expects 64 -> [32, 32, 10], got {run['nin']} -> {run['nouts']}")

    weights = ",".join(f"{w:.5f}".rstrip("0").rstrip(".") for w in run["weights"])
    html = (HERE / "template.html").read_text(encoding="utf-8")
    html = html.replace("/*__WEIGHTS__*/", f"const WEIGHTS=[{weights}];")
    html = html.replace("__TEST_ACC__", f"{run['test_acc'] * 100:.1f}%")

    out = HERE / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({len(run['weights'])} params, test acc {run['test_acc']:.3f})")


if __name__ == "__main__":
    main()
