from __future__ import annotations

from dataclasses import MISSING
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, SPHERE_MARKER_CFG
from isaaclab.utils import configclass

from .commands import UniformVelocityCommandPls, PhaseCommand, FKCommand


@configclass
class UniformVelocityCommandPlsCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = UniformVelocityCommandPls

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    heading_command: bool = False
    """Whether to use heading command or angular velocity command. Defaults to False.

    If True, the angular velocity command is computed from the heading error, where the
    target heading is sampled uniformly from provided range. Otherwise, the angular velocity
    command is sampled uniformly from provided range.
    """

    heading_control_stiffness: float = 1.0
    """Scale factor to convert the heading error to angular velocity command. Defaults to 1.0."""

    rel_standing_envs: float = 0.0
    """The sampled probability of environments that should be standing still. Defaults to 0.0."""

    rel_rotonly_envs: float = 0.0
    """The sampled probability of environments that should be rotation only. Defaults to 0.0."""

    rel_heading_envs: float = 1.0
    """The sampled probability of environments where the robots follow the heading-based angular velocity command
    (the others follow the sampled angular velocity command). Defaults to 1.0.

    This parameter is only used if :attr:`heading_command` is True.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        lin_vel_x: tuple[float, float] = MISSING
        """Range for the linear-x velocity command (in m/s)."""

        lin_vel_y: tuple[float, float] = MISSING
        """Range for the linear-y velocity command (in m/s)."""

        ang_vel_z: tuple[float, float] = MISSING
        """Range for the angular-z velocity command (in rad/s)."""

        heading: tuple[float, float] | None = None
        """Range for the heading command (in rad). Defaults to None.

        This parameter is only used if :attr:`~UniformVelocityCommandCfg.heading_command` is True.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""

    goal_linvel_viz_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_goal")
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""
    goal_angvel_viz_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/angvel_goal")
    """The configuration for the goal angvel visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_linvel_viz_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_current")
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""
    current_angvel_viz_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/angvel_current")
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_linvel_viz_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    goal_angvel_viz_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_linvel_viz_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_angvel_viz_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)


@configclass
class PhaseCommandCfg(CommandTermCfg):
    class_type: type = PhaseCommand
    phase_frequency_hz_range: tuple[float, float] = (1.0, 2.0)


@configclass
class FKCommandCfg(CommandTermCfg):
    """Configuration for FK command generator.

    The FK command loads hand pose data from npz files. The npz file should contain
    a matrix of shape (N, 7, 2) where:
    - N: number of hand poses
    - 7: 3 position + 4 quaternion (xyzw format)
    - 2: left hand (index 0) and right hand (index 1)

    The command samples two indices independently for left and right hands and outputs
    position and quaternion for both hands. Similar to GMRReferenceCommand, it supports
    default pose tracking and randomly sampled hand poses.
    """

    class_type: type = FKCommand

    asset_name: str = MISSING
    """Name of the robot asset in the scene."""

    fk_data_path: str = MISSING
    """Path to FK hand pose data. Should be a single .npz file containing a matrix
    of shape (N, 7, 2) where:
    - N: number of hand poses
    - 7: [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]
    - 2: [left_hand, right_hand]
    """

    rel_default: float = 0.5
    """The proportion of environments that track the default hand pose.
    Set to 0.0 to disable default pose tracking."""

    rel_fk: float = 0.4
    """The proportion of environments that track FK-sampled hand poses from the dataset.
    Set to 0.0 to disable FK sampling."""

    rel_random: float = 0.1
    """The proportion of environments that track purely random hand poses (sampled in extended workspace).
    This helps the policy generalize beyond FK-feasible poses. Set to 0.0 to disable random sampling."""

    rel_DProllout: float = 0.0
    """The proportion of environments that track diffusion policy rollout trajectories from HDF5 file.
    These environments will follow hand pose trajectories at 10Hz (updating every 5 timesteps at 50Hz policy rate).
    Set to 0.0 to disable DP rollout tracking."""

    dp_data_path: str = ""
    """Path to HDF5 file containing DP rollout trajectories. Required if rel_DProllout > 0.
    The HDF5 file should have structure: data/demo_X/teleop/Command/hand/{left_pose_(Trunk), right_pose_(Trunk)}
    where each pose array has shape (N, 7) with [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]."""

    reference_body_name: str = "Trunk"
    """Body name to use as the reference frame for the output commands.
    All hand positions and rotations are relative to this body.
    Example: 'Trunk' or 'base_link' to use robot base as reference frame.
    The body name should match a body in the robot articulation."""

    viz_robot_frames: list[str] = []
    """List of robot body names to visualize as coordinate frames.
    These should match body names in the robot articulation.
    Example: ['left_hand_ee', 'right_hand_ee'] to visualize current robot hand frames.
    """

    fk_viz_cfg: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Command/fk_env_viz")
    """Visualization marker configuration for environments using FK poses.
    A green sphere will appear above the robot in environments using FK sampling."""

    random_viz_cfg: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Command/random_env_viz")
    """Visualization marker configuration for environments using random poses.
    A red sphere will appear above the robot in environments using random sampling."""

    dp_rollout_viz_cfg: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Command/dp_rollout_env_viz")
    """Visualization marker configuration for environments using DP rollout trajectories.
    A purple sphere will appear above the robot in environments using DP rollout tracking."""

    # Safety limits for workspace clamping
    enable_workspace_limits: bool = False
    """Whether to enable workspace position limits. If True, sampled hand positions
    will be clamped to the specified workspace bounds."""

    @configclass
    class WorkspaceLimits:
        """Position limits for hand workspace (in base/Trunk frame)."""
        # Symmetric limits (X/Z identical, Y mirrored)
        left_x: tuple[float, float] = (0.08, 0.4)
        left_y: tuple[float, float] = (-0.05, 0.4)
        left_z: tuple[float, float] = (-0.2, 0.4)
        right_x: tuple[float, float] = (0.08, 0.4)
        right_y: tuple[float, float] = (-0.4, 0.05)
        right_z: tuple[float, float] = (-0.2, 0.4)

    workspace_limits: WorkspaceLimits = WorkspaceLimits()
    """Workspace position limits for left and right hands."""

    # Safety limits for orientation
    enable_orientation_limits: bool = False
    """Whether to enable orientation limits. If True, sampled hand orientations
    that deviate too much from default will be clamped."""

    max_orientation_deviation_deg: float = 45.0
    """Maximum allowed orientation deviation from default pose (in degrees).
    Only used if enable_orientation_limits is True."""

    # Random pose sampling configuration (for rel_random environments)
    @configclass
    class RandomPoseRanges:
        """Position and orientation ranges for purely random pose sampling.
        These ranges define a larger workspace than FK-feasible poses for robustness training."""
        # Left hand ranges (in base/Trunk frame)
        left_x: tuple[float, float] = (0.05, 0.5)
        left_y: tuple[float, float] = (-0.1, 0.5)
        left_z: tuple[float, float] = (-0.3, 0.5)
        # Right hand ranges (in base/Trunk frame)
        right_x: tuple[float, float] = (0.05, 0.5)
        right_y: tuple[float, float] = (-0.5, 0.1)
        right_z: tuple[float, float] = (-0.3, 0.5)
        # Orientation ranges (in radians, relative to default orientation)
        left_roll: tuple[float, float] = (-0.5, 0.5)
        left_pitch: tuple[float, float] = (-0.5, 0.5)
        left_yaw: tuple[float, float] = (-0.5, 0.5)
        right_roll: tuple[float, float] = (-0.5, 0.5)
        right_pitch: tuple[float, float] = (-0.5, 0.5)
        right_yaw: tuple[float, float] = (-0.5, 0.5)

    random_pose_ranges: RandomPoseRanges = RandomPoseRanges()
    """Workspace ranges for purely random pose sampling (used for rel_random environments)."""
