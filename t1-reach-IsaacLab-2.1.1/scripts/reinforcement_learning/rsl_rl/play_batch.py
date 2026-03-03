# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and compare multiple checkpoints from Weights & Biases in IsaacLab.

This tool provides an interactive workflow to:
1. Select an IsaacLab task
2. Query and select W&B runs
3. Download checkpoints
4. Play policies in IsaacLab with interactive UI switching
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.prompt import IntPrompt, Prompt
from rich.table import Table

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play multiple RL checkpoints from W&B with interactive switching.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_known_args()[0]

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]]

# ===== Data Structures =====


@dataclass
class BatchPlayConfig:
    """Configuration for batch play session."""

    task_name: str
    num_envs: int
    entity_name: str
    project_name: str
    time_window_str: str


@dataclass
class RunInfo:
    """Information about a W&B run."""

    run_id: str
    run_path: str  # "entity/project/run_id"
    name: str
    created_at: str  # ISO format string from wandb API


@dataclass
class PolicyInfo:
    """Information about a downloaded policy checkpoint."""

    run_id: str
    run_name: str
    checkpoint_path: str
    run_path: str
    notes: str = ""
    tags: list = field(default_factory=list)


# ===== Phase 1: TUI Setup Manager =====

CACHE_FILE = Path.cwd() / ".playbatch.cache"


def load_cache() -> Optional[dict]:
    """Load cached configuration from .playbatch.cache."""
    if not CACHE_FILE.exists():
        return None
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def save_cache(config: BatchPlayConfig):
    """Save configuration to .playbatch.cache."""
    cache_data = {
        "task_name": config.task_name,
        "num_envs": config.num_envs,
        "entity_name": config.entity_name,
        "project_name": config.project_name,
        "time_window": config.time_window_str,
    }
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache_data, f, indent=2)
    except IOError as e:
        print(f"[WARNING] Could not save cache: {e}")


def prompt_user_inputs(available_tasks: List[str]) -> BatchPlayConfig:
    """Interactive TUI to gather configuration.

    Args:
        available_tasks: List of task IDs from gym.registry (LocoManip tasks, Play prioritized)
    """
    console = Console()
    cache = load_cache()
    if cache:
        console.print("\n[dim]Loaded cached configuration from .playbatch.cache[/dim]")

    console.print("\n[bold cyan]===== IsaacLab Batch Play - Configuration =====[/bold cyan]\n")

    # Show available tasks
    table = Table(title="Available Tasks")
    table.add_column("Index", style="cyan", justify="right")
    table.add_column("Task Name", style="green")
    for idx, task in enumerate(available_tasks):
        table.add_row(str(idx), task)
    console.print(table)

    # Task selection
    default_task_idx = None
    if cache and cache.get("task_name"):
        try:
            default_task_idx = available_tasks.index(cache["task_name"])
        except ValueError:
            pass

    if default_task_idx is not None:
        task_idx = IntPrompt.ask("\nSelect task index", default=default_task_idx)
    else:
        task_idx = IntPrompt.ask("\nSelect task index")

    if task_idx < 0 or task_idx >= len(available_tasks):
        console.print("[red]Invalid task index![/red]")
        sys.exit(1)

    task_name = available_tasks[task_idx]

    # Other config
    num_envs = IntPrompt.ask("Number of environments", default=cache.get("num_envs", 32) if cache else 32)
    entity_name = (
        Prompt.ask("Wandb entity name", default=cache.get("entity_name", ""))
        if cache and cache.get("entity_name")
        else Prompt.ask("Wandb entity name")
    )
    project_name = (
        Prompt.ask("Wandb project name", default=cache.get("project_name", ""))
        if cache and cache.get("project_name")
        else Prompt.ask("Wandb project name")
    )
    time_window = Prompt.ask(
        "Time to look back (e.g., 24h, 7d)", default=cache.get("time_window", "24h") if cache else "24h"
    )

    console.print(f"\n[green]Task: {task_name}[/green]")
    console.print(f"[green]Num envs: {num_envs}[/green]")
    console.print(f"[green]W&B: {entity_name}/{project_name}[/green]")
    console.print(f"[green]Time window: {time_window}[/green]\n")

    config = BatchPlayConfig(task_name, num_envs, entity_name, project_name, time_window)
    save_cache(config)
    return config


# ===== Phase 2: Wandb Run Query =====


def parse_time_window(time_str: str) -> timedelta:
    """Parse '24h', '7d', '2w' to timedelta."""
    match = re.match(r"^(\d+)([hdw])$", time_str.lower())
    if not match:
        raise ValueError(f"Invalid time format: {time_str}. Use format like '24h', '7d', '2w'")

    value = int(match.group(1))
    unit = match.group(2)

    if unit == "h":
        return timedelta(hours=value)
    elif unit == "d":
        return timedelta(days=value)
    elif unit == "w":
        return timedelta(weeks=value)


def ensure_wandb_authenticated():
    """Check if wandb is authenticated."""
    try:
        import wandb
    except ImportError:
        print("[ERROR] wandb is not installed!")
        print("[INFO] Please install it with: pip install wandb")
        sys.exit(1)

    if not wandb.api.api_key:
        print("[ERROR] Wandb not authenticated!")
        print("[INFO] Please run: wandb login")
        print("[INFO] Or set WANDB_API_KEY environment variable")
        sys.exit(1)


def query_wandb_runs(config: BatchPlayConfig) -> List[RunInfo]:
    """Query wandb API for runs within time window - NO TASK FILTERING."""
    import wandb

    console = Console()
    console.print(f"\n[bold cyan]Querying W&B runs from {config.entity_name}/{config.project_name}...[/bold cyan]")

    api = wandb.Api()

    cutoff = datetime.now() - parse_time_window(config.time_window_str)
    runs = api.runs(
        f"{config.entity_name}/{config.project_name}",
        filters={"created_at": {"$gte": cutoff.isoformat()}},
    )

    run_infos = []
    for run in runs:
        run_infos.append(
            RunInfo(
                run_id=run.id, run_path=f"{run.entity}/{run.project}/{run.id}", name=run.name, created_at=run.created_at
            )
        )

    console.print(f"[green]Found {len(run_infos)} runs[/green]\n")
    return run_infos


def prompt_run_selection(runs: List[RunInfo]) -> List[RunInfo]:
    """Display runs and let user select (default: all)."""
    console = Console()
    table = Table(title="Available Runs")
    table.add_column("Index", style="cyan", justify="right")
    table.add_column("Run Name", style="green")
    table.add_column("Created At", style="yellow")

    for idx, run in enumerate(runs):
        # created_at is already a formatted string from wandb
        created_str = run.created_at[:16] if len(run.created_at) >= 16 else run.created_at  # Show YYYY-MM-DD HH:MM
        table.add_row(str(idx), run.name, created_str)
    console.print(table)

    selection = Prompt.ask("\nSelect runs (comma-separated indices, or 'all')", default="all")

    if selection.lower() == "all":
        console.print(f"[green]Selected all {len(runs)} runs[/green]\n")
        return runs

    try:
        indices = [int(i.strip()) for i in selection.split(",")]
        selected_runs = [runs[i] for i in indices]
        console.print(f"[green]Selected {len(selected_runs)} runs[/green]\n")
        return selected_runs
    except (ValueError, IndexError) as e:
        console.print(f"[red]Invalid selection: {e}[/red]")
        sys.exit(1)


# ===== Phase 3: Checkpoint Download =====


def download_all_checkpoints(runs: List[RunInfo], config: BatchPlayConfig) -> List[PolicyInfo]:
    """Download checkpoints in parallel to wandb_downloads/."""
    console = Console()
    console.print(f"\n[bold cyan]Downloading {len(runs)} checkpoints...[/bold cyan]")

    download_root = Path.cwd() / "wandb_downloads" / config.project_name
    download_root.mkdir(parents=True, exist_ok=True)

    policies = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_run = {executor.submit(_download_single, run, download_root): run for run in runs}

        for future in as_completed(future_to_run):
            try:
                policy_info = future.result()
                policies.append(policy_info)
                console.print(f"[green]✓ Downloaded: {policy_info.run_name}[/green]")
            except Exception as e:
                run = future_to_run[future]
                console.print(f"[red]✗ Failed to download {run.name}: {e}[/red]")

    console.print(f"\n[bold green]Successfully downloaded {len(policies)}/{len(runs)} checkpoints[/bold green]\n")
    return policies


def _download_single(run: RunInfo, download_root: Path) -> PolicyInfo:
    """Download single checkpoint using existing wandb_utils."""
    import wandb

    # Local import to avoid issues before Isaac Sim launch
    sys.path.insert(0, str(Path(__file__).parent))
    from wandb_utils import get_wandb_checkpoint_path

    # Download to wandb_downloads/{project}/
    checkpoint_path = get_wandb_checkpoint_path(str(download_root), run.run_path)

    # Fetch notes and tags from W&B
    api = wandb.Api()
    wandb_run = api.run(run.run_path)
    notes = wandb_run.notes or ""
    tags = list(wandb_run.tags) if wandb_run.tags else []

    return PolicyInfo(
        run_id=run.run_id, run_name=run.name, checkpoint_path=checkpoint_path, run_path=run.run_path, notes=notes, tags=tags
    )


# ===== Phase 4: IsaacLab Playback =====


class SinglePolicyManager:
    """Manages policy loading/unloading - only ONE policy in memory at a time."""

    def __init__(self, policies: List[PolicyInfo], env, agent_cfg, device: str):
        self.policies = policies
        self.env = env
        self.agent_cfg = agent_cfg
        self.device = device

        self.active_idx = 0
        self.current_runner = None
        self.current_policy = None
        self.current_policy_nn = None

        # Load first policy
        self.load_policy(0)

    def load_policy(self, idx: int):
        """Load policy at index, unload previous if exists."""
        import torch

        if idx == self.active_idx and self.current_policy is not None:
            return  # Already loaded

        # Unload previous policy (free memory)
        if self.current_runner is not None:
            del self.current_runner
            del self.current_policy
            del self.current_policy_nn
            torch.cuda.empty_cache()

        # Load new policy
        policy_info = self.policies[idx]
        print(f"[INFO] Loading policy: {policy_info.run_name}")

        # Create runner and load checkpoint
        from rsl_rl.runners import OnPolicyRunner
        runner = OnPolicyRunner(self.env, self.agent_cfg.to_dict(), log_dir=None, device=self.device)
        runner.load(policy_info.checkpoint_path)

        # Extract policy for inference
        policy = runner.get_inference_policy(device=self.device)

        # Extract neural network (for reset)
        try:
            policy_nn = runner.alg.policy  # RSL-RL v2.3+
        except AttributeError:
            policy_nn = runner.alg.actor_critic  # RSL-RL v2.2

        self.current_runner = runner
        self.current_policy = policy
        self.current_policy_nn = policy_nn
        self.active_idx = idx

        print(f"[INFO] Policy loaded: {policy_info.run_name}")

    def switch_policy(self, new_idx: int, env):
        """Switch to new policy and RESET environment."""
        if new_idx < 0 or new_idx >= len(self.policies):
            print(f"[WARNING] Invalid policy index: {new_idx}")
            return None

        # Load new policy (unloads old one)
        self.load_policy(new_idx)

        # Reset environment and get observations
        env.reset()
        obs, _ = env.get_observations()
        print(f"[INFO] Switched to: {self.policies[new_idx].run_name} (environment reset)")

        return obs

    def get_active_policy(self):
        """Return current policy and policy_nn."""
        return self.current_policy, self.current_policy_nn

    def get_policy_names(self) -> List[str]:
        """Return list of all policy names for UI."""
        return [p.run_name for p in self.policies]


def create_policy_switcher_ui(policy_manager: SinglePolicyManager, env, project_name: str):
    """Create UI tab inside IsaacLab with policy management controls."""
    import asyncio

    import omni.ui
    from isaaclab_rl.rsl_rl import export_policy_as_onnx

    # Create window as a docked tab
    window = omni.ui.Window(
        "Policy Switcher", width=600, height=400, visible=True, dock_preference=omni.ui.DockPreference.LEFT_BOTTOM
    )

    # Dock the window asynchronously
    async def dock_window():
        """Docks the Policy Switcher window to the IsaacLab window."""
        # Wait for the window to be created
        for _ in range(5):
            if omni.ui.Workspace.get_window("Policy Switcher"):
                break
            await asyncio.sleep(0.1)

        # Dock next to IsaacLab or Property window
        custom_window = omni.ui.Workspace.get_window("Policy Switcher")
        # Try to dock next to IsaacLab window first, then Property window
        target_window = omni.ui.Workspace.get_window("IsaacLab") or omni.ui.Workspace.get_window("Property")
        if custom_window and target_window:
            custom_window.dock_in(target_window, omni.ui.DockPosition.SAME, 0.5)

    asyncio.ensure_future(dock_window())

    obs_holder = {"obs": None}  # Mutable holder for observation updates
    ui_elements = {"status_label": None, "policy_rows": []}
    feedback_data = {"comment": "", "status": None}  # None, "SimOK", or "SimFail"

    def load_feedback_for_active_policy():
        """Load existing feedback data for the currently active policy."""
        active_policy = policy_manager.policies[policy_manager.active_idx]

        # Extract SimOK/SimFail status from tags
        status = None
        if "SimOK" in active_policy.tags:
            status = "SimOK"
        elif "SimFail" in active_policy.tags:
            status = "SimFail"

        # Don't overwrite user's current input - only load if feedback is empty
        if not feedback_data.get("comment") and feedback_data.get("status") is None:
            feedback_data["status"] = status
            # Optionally load last comment from notes
            # feedback_data["comment"] = ""  # Keep empty for new input

    def rebuild_ui():
        """Rebuild the UI with current policies."""
        window.frame.clear()
        ui_elements["policy_rows"].clear()

        # Load feedback for the currently active policy
        load_feedback_for_active_policy()

        with window.frame:
            with omni.ui.VStack(spacing=4, style={"margin": 5}):
                # Header - Active policy
                active_policy = policy_manager.policies[policy_manager.active_idx]
                existing_status = None
                if "SimOK" in active_policy.tags:
                    existing_status = "SimOK"
                elif "SimFail" in active_policy.tags:
                    existing_status = "SimFail"

                with omni.ui.VStack(spacing=2, style={"background_color": 0xFF2A2A2A}):
                    omni.ui.Label(
                        f"Active: {active_policy.run_name}",
                        style={"font_size": 14, "color": 0xFF00FF00, "margin": 5},
                        word_wrap=True,
                    )
                    if existing_status:
                        status_color = 0xFF00AA00 if existing_status == "SimOK" else 0xFFAA0000
                        omni.ui.Label(
                            f"Existing Tag: {existing_status}",
                            height=20,
                            style={"font_size": 11, "color": status_color, "margin_left": 5, "margin_bottom": 5},
                        )

                omni.ui.Separator(height=1)

                # Feedback section
                with omni.ui.VStack(spacing=3, style={"background_color": 0xFF1F1F1F, "margin": 2}):
                    # Comment input
                    omni.ui.Label("Comment:", height=20, style={"font_size": 11, "color": 0xFFAAAAAA})
                    comment_field = omni.ui.StringField(height=22)
                    comment_field.model.set_value(feedback_data.get("comment", ""))

                    def on_comment_changed(model):
                        feedback_data["comment"] = model.get_value_as_string()

                    comment_field.model.add_value_changed_fn(on_comment_changed)

                    # Status buttons row
                    with omni.ui.HStack(height=30, spacing=5):
                        def make_status_callback(status):
                            def on_click():
                                feedback_data["status"] = status
                                rebuild_ui()

                            return on_click

                        # SimOK button
                        simok_selected = feedback_data.get("status") == "SimOK"
                        omni.ui.Button(
                            "SimOK",
                            clicked_fn=make_status_callback("SimOK"),
                            width=80,
                            height=26,
                            style={
                                "background_color": 0xFF00AA00 if simok_selected else 0xFF003300,
                                "font_size": 12,
                            },
                        )

                        # SimFail button
                        simfail_selected = feedback_data.get("status") == "SimFail"
                        omni.ui.Button(
                            "SimFail",
                            clicked_fn=make_status_callback("SimFail"),
                            width=80,
                            height=26,
                            style={
                                "background_color": 0xFFAA0000 if simfail_selected else 0xFF330000,
                                "font_size": 12,
                            },
                        )

                        # Spacer
                        omni.ui.Spacer(width=omni.ui.Fraction(1))

                        # Save button
                        def on_save():
                            import wandb

                            active_policy = policy_manager.policies[policy_manager.active_idx]
                            comment = feedback_data.get("comment", "")
                            status = feedback_data.get("status")

                            if not comment and not status:
                                print("[WARNING] No comment or status selected to save")
                                return

                            print(f"[INFO] Saving feedback for run: {active_policy.run_name}")

                            try:
                                # Get the run
                                api = wandb.Api()
                                run = api.run(active_policy.run_path)

                                # Add comment to notes
                                if comment:
                                    existing_notes = run.notes or ""
                                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    new_note = f"[{timestamp}] {comment}"
                                    run.notes = f"{existing_notes}\n{new_note}".strip()

                                # Update tags - remove old SimOK/SimFail, add new one
                                if status:
                                    tags = list(run.tags)
                                    # Remove existing SimOK/SimFail tags
                                    tags = [t for t in tags if t not in ["SimOK", "SimFail"]]
                                    # Add new status tag
                                    tags.append(status)
                                    run.tags = tags

                                run.update()
                                print(f"[INFO] ✓ Saved feedback to W&B run {active_policy.run_name}")

                                # Clear feedback after save
                                feedback_data["comment"] = ""
                                feedback_data["status"] = None
                                rebuild_ui()

                            except Exception as e:
                                print(f"[ERROR] Failed to save feedback: {e}")

                        omni.ui.Button(
                            "Save",
                            clicked_fn=on_save,
                            width=80,
                            height=26,
                            style={"background_color": 0xFF0066AA, "font_size": 12},
                        )

                omni.ui.Separator(height=2)

                # Table header
                with omni.ui.HStack(height=25, spacing=5, style={"background_color": 0xFF1A1A1A, "margin": 2}):
                    omni.ui.Label(
                        "Experiment Name",
                        width=250,
                        style={"font_size": 12, "color": 0xFFAAAAAA, "font_weight": "bold"},
                    )
                    omni.ui.Label("Actions", width=260, style={"font_size": 12, "color": 0xFFAAAAAA, "font_weight": "bold"})

                omni.ui.Separator(height=1)

                # Policy rows with scrolling
                with omni.ui.ScrollingFrame(
                    height=omni.ui.Percent(100),
                    horizontal_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                    vertical_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                ):
                    with omni.ui.VStack(spacing=2):
                        for idx, policy_info in enumerate(policy_manager.policies):
                            is_active = idx == policy_manager.active_idx
                            name_color = 0xFF00FF00 if is_active else 0xFFCCCCCC
                            row_bg = 0xFF2A2A2A if is_active else 0xFF1F1F1F

                            with omni.ui.HStack(height=0, spacing=5, style={"background_color": row_bg, "margin": 2}):
                                # Run name with word wrap
                                omni.ui.Label(
                                    policy_info.run_name,
                                    width=250,
                                    style={"font_size": 12, "color": name_color},
                                    word_wrap=True,
                                )

                                # Action buttons - fixed width container
                                with omni.ui.HStack(width=260, spacing=4):

                                    def make_activate_callback(i):
                                        def on_activate():
                                            new_obs = policy_manager.switch_policy(i, env)
                                            if new_obs is not None:
                                                obs_holder["obs"] = new_obs
                                                # Clear feedback to load new policy's feedback
                                                feedback_data["comment"] = ""
                                                feedback_data["status"] = None
                                                rebuild_ui()

                                        return on_activate

                                    def make_remove_callback(i):
                                        def on_remove():
                                            if len(policy_manager.policies) <= 1:
                                                print("[WARNING] Cannot remove the last policy!")
                                                return

                                            removed_policy = policy_manager.policies[i]
                                            print(f"[INFO] Removing policy: {removed_policy.run_name}")

                                            # If removing active policy, switch to another one first
                                            if i == policy_manager.active_idx:
                                                new_idx = 0 if i > 0 else 1
                                                policy_manager.switch_policy(new_idx, env)

                                            # Remove the policy
                                            del policy_manager.policies[i]

                                            # Update active_idx if needed
                                            if policy_manager.active_idx > i:
                                                policy_manager.active_idx -= 1
                                            elif policy_manager.active_idx >= len(policy_manager.policies):
                                                policy_manager.active_idx = len(policy_manager.policies) - 1

                                            rebuild_ui()

                                        return on_remove

                                    def make_export_callback(i):
                                        def on_export():
                                            policy_info = policy_manager.policies[i]
                                            export_dir = Path.cwd() / "export" / f"export_{project_name}"
                                            export_dir.mkdir(parents=True, exist_ok=True)
                                            filename = f"{policy_info.run_id}__{policy_info.run_name}.onnx"
                                            export_path = export_dir / filename

                                            print(f"[INFO] Exporting {policy_info.run_name} to {export_path}...")

                                            # Load the policy if not active
                                            if i != policy_manager.active_idx:
                                                policy_manager.load_policy(i)

                                            # Get policy_nn and normalizer
                                            policy_nn = policy_manager.current_policy_nn

                                            # Extract normalizer
                                            if hasattr(policy_nn, "actor_obs_normalizer"):
                                                normalizer = policy_nn.actor_obs_normalizer
                                            elif hasattr(policy_nn, "student_obs_normalizer"):
                                                normalizer = policy_nn.student_obs_normalizer
                                            else:
                                                normalizer = None

                                            # Export to ONNX
                                            export_policy_as_onnx(
                                                policy_nn,
                                                normalizer=normalizer,
                                                path=str(export_dir),
                                                filename=filename,
                                            )

                                            print(f"[INFO] ✓ Exported to: {export_path}")

                                        return on_export

                                    # Activate button
                                    omni.ui.Button(
                                        "Activate" if not is_active else "Active",
                                        clicked_fn=make_activate_callback(idx),
                                        width=80,
                                        height=28,
                                        style={
                                            "background_color": 0xFF00AA00 if is_active else 0xFF005500,
                                            "font_size": 14,
                                        },
                                        enabled=not is_active,
                                    )

                                    # Remove button
                                    omni.ui.Button(
                                        "Remove",
                                        clicked_fn=make_remove_callback(idx),
                                        width=80,
                                        height=28,
                                        style={"background_color": 0xFFAA0000, "font_size": 14},
                                    )

                                    # Export button
                                    omni.ui.Button(
                                        "Export",
                                        clicked_fn=make_export_callback(idx),
                                        width=80,
                                        height=28,
                                        style={"background_color": 0xFF0066AA, "font_size": 14},
                                    )

    # Initial build
    rebuild_ui()

    return window, obs_holder


# ===== Main Entry Point =====


def main():
    """Main function orchestrating all phases."""
    console = Console()

    # ===== Launch Isaac Sim =====
    console.print("\n[bold cyan]Launching Isaac Sim...[/bold cyan]")
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # ===== Import Isaac Sim dependencies =====
    import gymnasium as gym

    import cli_args  # isort: skip
    import isaaclab_tasks  # noqa: F401

    # ===== Get available tasks from gym.registry =====
    all_tasks = [spec.id for spec in gym.registry.values()]
    play_tasks = sorted([t for t in all_tasks if "Play" in t])
    other_tasks = sorted([t for t in all_tasks if "Play" not in t])
    available_tasks = play_tasks + other_tasks

    console.print(f"[green]Found {len(available_tasks)} LocoManip tasks[/green]")

    # ===== Gather configuration =====
    config = prompt_user_inputs(available_tasks)

    # ===== Query Wandb =====
    ensure_wandb_authenticated()
    runs = query_wandb_runs(config)

    if not runs:
        console.print("[red]No runs found in the specified time window![/red]")
        simulation_app.close()
        return

    selected_runs = prompt_run_selection(runs)

    # ===== Phase 3: Download Checkpoints =====
    policies = download_all_checkpoints(selected_runs, config)

    if not policies:
        console.print("[red]No policies downloaded successfully![/red]")
        simulation_app.close()
        return

    # ===== Phase 4: IsaacLab Playback =====
    import torch

    from isaaclab.envs import (
        DirectMARLEnv,
        DirectMARLEnvCfg,
        DirectRLEnvCfg,
        ManagerBasedRLEnvCfg,
        multi_agent_to_single_agent,
    )
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.hydra import hydra_task_config

    # Use the config from Phase 1
    task_name = config.task_name
    num_envs = config.num_envs

    @hydra_task_config(task_name, "rsl_rl_cfg_entry_point")
    def setup_env(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
        """Setup environment and run playback loop."""
        # Override num_envs
        env_cfg.scene.num_envs = num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # Create environment
        console.print(f"[bold cyan]Creating environment: {task_name}[/bold cyan]")
        env = gym.make(task_name, cfg=env_cfg, render_mode=None)

        # Wrap for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # Create policy manager (loads first policy)
        console.print(f"[bold cyan]Initializing policy manager with {len(policies)} policies[/bold cyan]")
        policy_manager = SinglePolicyManager(policies, env, agent_cfg, args_cli.device or "cuda:0")

        # Create UI window in IsaacLab
        console.print("[bold cyan]Creating UI window in IsaacLab...[/bold cyan]")
        ui_window, obs_holder = create_policy_switcher_ui(policy_manager, env, config.project_name)

        # Initial observation
        obs, _ = env.get_observations()
        obs_holder["obs"] = obs

        console.print("\n[bold green]===== Playback Started! Use UI to switch policies =====[/bold green]\n")

        # Inference loop
        while simulation_app.is_running():
            with torch.inference_mode():
                # Check if observation was updated by UI (policy switch)
                if obs_holder["obs"] is not None:
                    obs = obs_holder["obs"]
                    obs_holder["obs"] = None  # Clear

                # Get current policy
                policy, policy_nn = policy_manager.get_active_policy()

                # Run inference
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)

                # Reset policy states for terminated episodes
                policy_nn.reset(dones)

        env.close()

    setup_env()
    simulation_app.close()


if __name__ == "__main__":
    main()
