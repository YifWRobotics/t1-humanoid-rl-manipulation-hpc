"""
Generate FK dataset by running both far and close generators, then merge the results.

This script:
1. Runs fk-generator_far.py to generate far-reach samples
2. Runs fk-generator_close.py to generate close-reach samples
3. Merges both datasets into final output files
"""

import subprocess
import numpy as np
import shutil
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parents[3]
FAR_GENERATOR = REPO_ROOT / "scripts/data_collect/Feasible_manipulation_poses/fk-generator_far.py"
CLOSE_GENERATOR = REPO_ROOT / "scripts/data_collect/Feasible_manipulation_poses/fk-generator_close.py"

DEFAULT_HAND_POSES = REPO_ROOT / "source/isaaclab_assets/data/hand_pose_commands.npz"
DEFAULT_DATASET = REPO_ROOT / "source/isaaclab_assets/data/feasible_poses_dataset.npz"

FAR_HAND_POSES = REPO_ROOT / "source/isaaclab_assets/data/hand_pose_commands_far.npz"
FAR_DATASET = REPO_ROOT / "source/isaaclab_assets/data/feasible_poses_dataset_far.npz"
CLOSE_HAND_POSES = REPO_ROOT / "source/isaaclab_assets/data/hand_pose_commands_close.npz"
CLOSE_DATASET = REPO_ROOT / "source/isaaclab_assets/data/feasible_poses_dataset_close.npz"

MERGED_HAND_POSES = REPO_ROOT / "source/isaaclab_assets/data/hand_pose_commands.npz"
MERGED_DATASET = REPO_ROOT / "source/isaaclab_assets/data/feasible_poses_dataset.npz"


def run_generator(script_path, label):
    """Run a generator script and return success status."""
    print(f"\n{'='*80}")
    print(f"Running {label} generator: {script_path.name}")
    print(f"{'='*80}\n")

    isaaclab_sh = REPO_ROOT / "isaaclab.sh"
    cmd = [str(isaaclab_sh), "-p", str(script_path)]

    try:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
        print(f"\n✓ {label} generator completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {label} generator failed with error code {e.returncode}")
        return False


def move_outputs(src_poses, src_dataset, dst_poses, dst_dataset, label):
    """Move generator outputs to temporary storage."""
    print(f"\nMoving {label} outputs to temporary storage...")

    if src_poses.exists():
        shutil.move(str(src_poses), str(dst_poses))
        print(f"  ✓ Moved {src_poses.name} → {dst_poses.name}")
    else:
        print(f"  ✗ Warning: {src_poses.name} not found")
        return False

    if src_dataset.exists():
        shutil.move(str(src_dataset), str(dst_dataset))
        print(f"  ✓ Moved {src_dataset.name} → {dst_dataset.name}")
    else:
        print(f"  ✗ Warning: {src_dataset.name} not found")
        return False

    return True


def merge_datasets():
    """Merge far and close datasets into final outputs."""
    print(f"\n{'='*80}")
    print("Merging datasets")
    print(f"{'='*80}\n")

    print("Loading hand pose commands...")
    far_poses_data = np.load(FAR_HAND_POSES)
    close_poses_data = np.load(CLOSE_HAND_POSES)

    far_poses = far_poses_data['poses']
    close_poses = close_poses_data['poses']

    print(f"  Far poses shape:   {far_poses.shape}")
    print(f"  Close poses shape: {close_poses.shape}")

    merged_poses = np.concatenate([far_poses, close_poses], axis=0)
    print(f"  Merged poses shape: {merged_poses.shape}")

    print("\nLoading dataset files...")
    far_dataset = np.load(FAR_DATASET)
    close_dataset = np.load(CLOSE_DATASET)

    merged_qref = np.concatenate([far_dataset['qref'], close_dataset['qref']], axis=0)
    merged_BLH = np.concatenate([far_dataset['xyzwxyz_BLH'], close_dataset['xyzwxyz_BLH']], axis=0)
    merged_BRH = np.concatenate([far_dataset['xyzwxyz_BRH'], close_dataset['xyzwxyz_BRH']], axis=0)

    far_samples = len(far_dataset['qref'])
    close_samples = len(close_dataset['qref'])
    total_samples = far_samples + close_samples

    merged_starting_steps = np.arange(total_samples)
    merged_ending_steps = np.arange(total_samples)

    # dt should be identical in both datasets
    merged_dt = far_dataset['dt']

    print(f"  qref shape: {merged_qref.shape}")
    print(f"  xyzwxyz_BLH shape: {merged_BLH.shape}")
    print(f"  xyzwxyz_BRH shape: {merged_BRH.shape}")
    print(f"  Total samples: {total_samples} (far: {far_samples}, close: {close_samples})")

    print("\nSaving merged datasets...")
    np.savez(MERGED_HAND_POSES, poses=merged_poses)
    print(f"  ✓ Saved {MERGED_HAND_POSES}")

    np.savez(
        MERGED_DATASET,
        qref=merged_qref,
        xyzwxyz_BLH=merged_BLH,
        xyzwxyz_BRH=merged_BRH,
        starting_steps=merged_starting_steps,
        ending_steps=merged_ending_steps,
        dt=merged_dt
    )
    print(f"  ✓ Saved {MERGED_DATASET}")

    print(f"\n{'='*80}")
    print("Merge Summary")
    print(f"{'='*80}")
    print(f"Total samples: {total_samples}")
    print(f"  - Far reach samples:   {far_samples} ({100*far_samples/total_samples:.1f}%)")
    print(f"  - Close reach samples: {close_samples} ({100*close_samples/total_samples:.1f}%)")
    print("\nOutput files:")
    print(f"  - {MERGED_HAND_POSES}")
    print(f"  - {MERGED_DATASET}")


def cleanup_temp_files():
    """Remove temporary files."""
    print("\nCleaning up temporary files...")
    temp_files = [FAR_HAND_POSES, FAR_DATASET, CLOSE_HAND_POSES, CLOSE_DATASET]
    for temp_file in temp_files:
        if temp_file.exists():
            temp_file.unlink()
            print(f"  ✓ Removed {temp_file.name}")


def main():
    print(f"\n{'='*80}")
    print("FK Dataset Generation and Merge Pipeline")
    print(f"{'='*80}")
    print(f"Repository root: {REPO_ROOT}")
    print(f"Far generator:   {FAR_GENERATOR.relative_to(REPO_ROOT)}")
    print(f"Close generator: {CLOSE_GENERATOR.relative_to(REPO_ROOT)}")

    if not FAR_GENERATOR.exists():
        print(f"\n✗ Error: Far generator not found at {FAR_GENERATOR}")
        return 1
    if not CLOSE_GENERATOR.exists():
        print(f"\n✗ Error: Close generator not found at {CLOSE_GENERATOR}")
        return 1

    if not run_generator(FAR_GENERATOR, "FAR"):
        print("\n✗ Pipeline failed: Far generator error")
        return 1

    if not move_outputs(DEFAULT_HAND_POSES, DEFAULT_DATASET,
                        FAR_HAND_POSES, FAR_DATASET, "FAR"):
        print("\n✗ Pipeline failed: Could not move far generator outputs")
        return 1

    if not run_generator(CLOSE_GENERATOR, "CLOSE"):
        print("\n✗ Pipeline failed: Close generator error")
        return 1

    if not move_outputs(DEFAULT_HAND_POSES, DEFAULT_DATASET,
                        CLOSE_HAND_POSES, CLOSE_DATASET, "CLOSE"):
        print("\n✗ Pipeline failed: Could not move close generator outputs")
        return 1

    try:
        merge_datasets()
    except Exception as e:
        print(f"\n✗ Pipeline failed: Merge error - {e}")
        import traceback
        traceback.print_exc()
        return 1

    cleanup_temp_files()

    print(f"\n{'='*80}")
    print("✓ Pipeline completed successfully!")
    print(f"{'='*80}\n")

    return 0


if __name__ == "__main__":
    exit(main())
