#!/usr/bin/env python
"""
Script to combine two USD files of the T1 robot.
Takes the upper body from the first USD and the lower body from the second USD.

Usage:
    python combine_usd.py <upper_body_usd> <lower_body_usd> <output_usd>
"""

import argparse
import sys
from pathlib import Path

try:
    from pxr import Usd
except ImportError:
    print("Error: USD Python bindings not found.")
    sys.exit(1)


def combine_usd_files(upper_body_usd: Path, lower_body_usd: Path, output_usd: Path):
    """Simply load both USDs into the same file as references."""

    # Create new stage
    output_stage = Usd.Stage.CreateNew(str(output_usd))

    # Add both USD files as sublayers
    root_layer = output_stage.GetRootLayer()
    root_layer.subLayerPaths.append(str(upper_body_usd))
    root_layer.subLayerPaths.append(str(lower_body_usd))

    # Save
    output_stage.GetRootLayer().Save()
    print(f"Combined USD saved to: {output_usd}")
    print(f"  - Upper body USD: {upper_body_usd}")
    print(f"  - Lower body USD: {lower_body_usd}")


def main():
    parser = argparse.ArgumentParser(description="Combine two T1 robot USD files")
    parser.add_argument("upper_body_usd", type=str)
    parser.add_argument("lower_body_usd", type=str)
    parser.add_argument("output_usd", type=str)
    args = parser.parse_args()

    upper_body_path = Path(args.upper_body_usd)
    lower_body_path = Path(args.lower_body_usd)
    output_path = Path(args.output_usd)

    if not upper_body_path.exists():
        print(f"Error: {upper_body_path} not found")
        sys.exit(1)

    if not lower_body_path.exists():
        print(f"Error: {lower_body_path} not found")
        sys.exit(1)

    combine_usd_files(upper_body_path, lower_body_path, output_path)


if __name__ == "__main__":
    main()
