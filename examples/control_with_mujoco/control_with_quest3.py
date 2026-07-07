"""Cartesian-space teleoperation of an i2rt arm using a Meta Quest3 controller.

Bridges ``quest3_reader.QuestReader`` into ``MujocoControlInterface``: the Right
Hand pose drives the end-effector target (both position and orientation) and
the Right Trigger drives the gripper. Right Grip is a clutch — hold it to
track your hand, release it to freeze the target and reposition your hand
freely (like lifting a mouse).

The viewer still starts in VIS mode; press SPACE, or press the Quest3's A
Button, to enter CONTROL mode, same as ``control_with_mujoco.py``. The A
Button lets you toggle modes without keyboard/display access to the viewer
(e.g. when the control host is only reachable over SSH). The clutch only has
an effect once CONTROL mode is active.

``QUEST_TO_ROBOT_ROTATION`` below maps the Quest3 app's world-frame axis
convention to the robot's MuJoCo world frame. Verify it empirically (move the
controller along a known robot axis and confirm the arm's motion direction
matches) before trusting absolute directions.

Usage:
    python examples/control_with_mujoco/control_with_quest3.py --sim
    python examples/control_with_mujoco/control_with_quest3.py --channel can0 --quest-ip 192.168.1.23
"""

import argparse
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logging.basicConfig(level=logging.INFO)

GRIP_ENGAGE_THRESHOLD = 0.5
QUEST_POLL_HZ = 100.0

# Robot world frame expressed in the Quest3 frame: Quest3 +X (right) is robot
# -Y, Quest3 +Y (up) is robot +Z, Quest3 +Z (backward) is robot +X.
QUEST_TO_ROBOT_ROTATION = np.array(
    [
        [0, 0, 1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

def _reexpress_position_in_robot_frame(pos_delta: np.ndarray) -> np.ndarray:
    """Reexpress a Quest3-frame world-frame position delta in the robot's world frame."""
    #print(QUEST_TO_ROBOT_ROTATION)
    return QUEST_TO_ROBOT_ROTATION @ pos_delta


def _rotation_delta_in_robot_frame(hand: np.ndarray, quest_pose_engage: np.ndarray) -> np.ndarray:
    """World-frame hand rotation delta since engage, remapped into the robot's world frame.

    Mirrors ``_reexpress_position_in_robot_frame``: the Quest3-frame delta
    rotation (from engage orientation to current orientation, expressed in
    the Quest3 world axes) is conjugated by ``QUEST_TO_ROBOT_ROTATION`` to
    express the same physical rotation in the robot's world axes.
    """
    rot_delta_quest = hand[:3, :3] @ quest_pose_engage[:3, :3].T
    return QUEST_TO_ROBOT_ROTATION @ rot_delta_quest @ QUEST_TO_ROBOT_ROTATION.T


class Quest3Clutch:
    """Delta + clutch mapping: Right Grip engages, Right Hand pose drives the EE target."""

    def __init__(self, iface: Any, reader: Any) -> None:
        self._iface = iface
        self._reader = reader
        self._engaged = False
        self._quest_pose_engage: Optional[np.ndarray] = None
        self._ee_pose_engage: Optional[np.ndarray] = None
        self._a_button_prev = False

    def step(self) -> None:
        data: Dict[str, Any] = self._reader.get_data()
        if not data:
            return

        a_button = bool(data.get("A Button", False))
        if a_button and not self._a_button_prev:
            self._iface.toggle_mode()
        self._a_button_prev = a_button

        trigger = data.get("Right Trigger")
        if trigger is not None:
            self._iface.set_gripper(1.0 - float(trigger))

        hand = data.get("Right Hand")
        grip = data.get("Right Grip", 0.0)
        if hand is None:
            return

        #print(f"[quest3] Right Hand translation: {hand[:3, 3]}")

        engaged_now = grip > GRIP_ENGAGE_THRESHOLD
        if engaged_now and not self._engaged:
            self._quest_pose_engage = hand.copy()
            self._ee_pose_engage = self._iface.get_ee_pose()
            self._engaged = True
        elif not engaged_now and self._engaged:
            self._engaged = False

        if not self._engaged or self._quest_pose_engage is None:
            return

        # World-frame position/rotation delta, computed relative to the quest world
        # axes (not the hand's own local axes), then remapped into the robot frame.
        pos_delta = hand[:3, 3] - self._quest_pose_engage[:3, 3]
        # print(f"[quest3] Right Hand delta: {pos_delta}")
        pos_delta_robot = _reexpress_position_in_robot_frame(pos_delta)
        # print(f"[quest3] Right Hand delta in robot frame: {pos_delta_robot}")
        rot_delta_robot = _rotation_delta_in_robot_frame(hand, self._quest_pose_engage)
        # print(f"[quest3] Right Hand rotation delta in robot frame: {rot_delta_robot}")

        target = np.eye(4)
        target[:3, :3] = rot_delta_robot @ self._ee_pose_engage[:3, :3]
        target[:3, 3] = self._ee_pose_engage[:3, 3] + pos_delta_robot
        self._iface.set_target_pose(target)


def _quest3_loop(iface: Any, reader: Any, stop_event: threading.Event) -> None:
    iface.wait_until_ready()
    clutch = Quest3Clutch(iface, reader)
    dt = 1.0 / QUEST_POLL_HZ
    while not stop_event.is_set():
        clutch.step()
        time.sleep(dt)


def _parse_args() -> argparse.Namespace:
    from i2rt.robots.utils import ArmType, GripperType

    parser = argparse.ArgumentParser(description="Quest3 -> MuJoCo Cartesian teleoperation")
    parser.add_argument("--arm", type=str, default="yam", choices=[a.value for a in ArmType])
    parser.add_argument("--gripper", type=str, default="linear_4310", choices=[g.value for g in GripperType])
    parser.add_argument("--channel", type=str, default="can0", help="CAN channel")
    parser.add_argument("--sim", action="store_true", help="Use SimRobot")
    parser.add_argument("--site", type=str, default=None, help="EE site name (auto-detected if omitted)")
    parser.add_argument("--dt", type=float, default=0.02, help="Viewer loop timestep (s)")
    parser.add_argument("--quest-ip", type=str, default=None, help="Quest3 IP (omit to use ADB)")
    parser.add_argument("--quest-port", type=int, default=12345)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import ArmType, GripperType
    from i2rt.utils.mujoco_control_interface import MujocoControlInterface
    from quest3_reader import QuestReader

    arm = ArmType.from_string_name(args.arm)
    gripper = GripperType.from_string_name(args.gripper)
    if arm == ArmType.NO_ARM and gripper == GripperType.NO_GRIPPER:
        raise SystemExit("--gripper cannot be 'no_gripper' when --arm is 'no_arm'")

    robot = get_yam_robot(channel=args.channel, arm_type=arm, gripper_type=gripper, sim=args.sim)
    if args.sim and hasattr(robot, "start_server"):
        robot.start_server()

    if args.site is not None:
        site = args.site
    elif gripper == GripperType.YAM_TEACHING_HANDLE:
        site = "tcp_site"
    else:
        site = "grasp_site"

    iface = MujocoControlInterface.from_robot(robot, ee_site=site, dt=args.dt)

    reader = QuestReader(args.quest_ip, args.quest_port)
    stop_event = threading.Event()
    quest_thread = threading.Thread(target=_quest3_loop, args=(iface, reader, stop_event), daemon=True)
    quest_thread.start()

    print("[quest3] Press SPACE in the viewer, or the Quest3 A Button, to enter CONTROL mode.")
    print("[quest3] Hold Right Grip to drive the arm with your right hand; release to reposition freely.")
    try:
        iface.run()
    finally:
        stop_event.set()
        reader.close()


if __name__ == "__main__":
    main()
