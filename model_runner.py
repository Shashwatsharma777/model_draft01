# model_runner.py — wrapper to call predict.py with sensible defaults
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    # Resolve paths relative to this script
    here = Path(__file__).resolve().parent
    predict_py = (here / "predict.py").as_posix()

    # Import config to get defaults
    sys.path.insert(0, here.as_posix())
    try:
        import config
    except Exception as e:
        print(f"ERROR: Could not import config.py from {here}\n{e}", file=sys.stderr)
        sys.exit(1)

    # Build parser with defaults from config
    ap = argparse.ArgumentParser(description="Run predict.py with defaults from config.py")
    ap.add_argument("--input", help="Path to input CSV (defaults to config.DEFAULT_INPUT_CSV)")
    ap.add_argument("--output", help="Path to output CSV (defaults to config.DEFAULT_OUTPUT_CSV)")
    ap.add_argument("--output_proba", help="Path to probabilities CSV (defaults to config.DEFAULT_PROBA_CSV; only used when --proba is set)")
    ap.add_argument("--proba", action="store_true", help="Also generate probabilities CSV")
    args = ap.parse_args()

    # Resolve defaults
    input_path = Path(args.input) if args.input else Path(config.DEFAULT_INPUT_CSV)
    output_path = Path(args.output) if args.output else Path(config.DEFAULT_OUTPUT_CSV)
    proba_path = Path(args.output_proba) if args.output_proba else Path(getattr(config, "DEFAULT_PROBA_CSV", output_path.with_name("predictions_proba.csv")))

    # Basic checks
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.proba:
        proba_path.parent.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        sys.executable,
        predict_py,
        "--input", input_path.as_posix(),
        "--output", output_path.as_posix(),
    ]
    if args.proba:
        # We’ll pass --proba and also the explicit output path if user provided one
        cmd.append("--proba")
        if args.output_proba:
            cmd.extend(["--output_proba", proba_path.as_posix()])

    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)

    # Pipe through stdout/stderr from predict.py
    if res.stdout:
        print(res.stdout.rstrip())
    if res.returncode != 0:
        if res.stderr:
            print(res.stderr.rstrip(), file=sys.stderr)
        sys.exit(res.returncode)

if __name__ == "__main__":
    main()
