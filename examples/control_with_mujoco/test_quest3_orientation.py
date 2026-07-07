"""Diagnostic script: check the translation/orientation assumptions behind control_with_quest3.py.

Three independent checks, printed side by side:

  1. Right Hand translation. Since the first frame, this prints the hand's
     position delta both in the raw Quest3 world frame and remapped into the
     robot's world frame via the same QUEST_TO_ROBOT_ROTATION /
     _reexpress_in_robot_frame used by control_with_quest3.py. Move the
     controller along one Quest3 axis at a time (e.g. straight up) and check
     that exactly one robot-frame column changes, in the direction you'd
     expect from QUEST_TO_ROBOT_ROTATION -- do this before trusting absolute
     directions during teleop.

  2. Quest3 Head vs Right Hand orientation. Point the controller the same way
     you're looking (e.g. hold it against your temple) and watch the
     "Hand-in-Head" column -- if head and hand were rigidly aligned it would
     stay constant. In practice the Quest3 controller and headset don't share
     a common "forward" convention, so expect an offset; this just lets you
     see how large/stable it is.

  3. Robot EE-site orientation vs the MuJoCo world frame, using FK on the
     robot's current joint positions (no viewer needed). This is the
     "robot world vs ee" alignment that QUEST_TO_ROBOT_ROTATION in
     control_with_quest3.py has to account for -- if the EE frame isn't
     identity-aligned with world at a neutral pose, a raw world-frame
     rotation delta from the controller won't map onto EE rotation the way
     you'd naively expect.

Usage:
    python examples/control_with_mujoco/test_quest3_orientation.py
    python examples/control_with_mujoco/test_quest3_orientation.py --sim
    python examples/control_with_mujoco/test_quest3_orientation.py --channel can0
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def euler_str(rot: np.ndarray) -> str:
    deg = Rotation.from_matrix(rot).as_euler("xyz", degrees=True)
    return f"[{deg[0]:7.2f}, {deg[1]:7.2f}, {deg[2]:7.2f}]"


def pos_str(pos: np.ndarray) -> str:
    return f"[{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}]"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quest3 / robot orientation alignment check")
    parser.add_argument("--quest-ip", type=str, default=None, help="Quest3 IP (omit to use ADB)")
    parser.add_argument("--quest-port", type=int, default=12345)
    parser.add_argument("--sim", action="store_true", help="Use a sim robot for the EE-vs-world check")
    parser.add_argument("--channel", type=str, default="can0", help="CAN channel (ignored if --sim)")
    parser.add_argument("--arm", type=str, default="yam")
    parser.add_argument("--gripper", type=str, default="linear_4310")
    parser.add_argument("--site", type=str, default=None, help="EE site name (auto-detected if omitted)")
    parser.add_argument("--no-robot", action="store_true", help="Skip the EE-vs-world check entirely")
    parser.add_argument("--hz", type=float, default=2.0, help="Print rate")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    from control_with_quest3 import QUEST_TO_ROBOT_ROTATION
    from quest3_reader import QuestReader

    reader = QuestReader(args.quest_ip, args.quest_port)

    iface = None
    if not args.no_robot:
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import ArmType, GripperType
        from i2rt.utils.mujoco_control_interface import MujocoControlInterface

        arm = ArmType.from_string_name(args.arm)
        gripper = GripperType.from_string_name(args.gripper)
        robot = get_yam_robot(channel=args.channel, arm_type=arm, gripper_type=gripper, sim=args.sim)
        if args.sim and hasattr(robot, "start_server"):
            robot.start_server()
        site = args.site or ("tcp_site" if gripper == GripperType.YAM_TEACHING_HANDLE else "grasp_site")
        iface = MujocoControlInterface.from_robot(robot, ee_site=site)

    header = (
        f"{'R.Hand dPos quest (m)':<26}{'R.Hand dPos robot (m)':<26}"
        f"{'Head (xyz deg)':<26}{'R.Hand (xyz deg)':<26}{'Hand-in-Head (xyz deg)':<26}"
    )
    if iface is not None:
        header += f"{'EE-vs-world (xyz deg)':<26}"
    print("Move your right hand along one axis at a time; Ctrl+C to quit.")
    print(header)

    hand_origin: Optional[np.ndarray] = None
    dt = 1.0 / args.hz
    try:
        while True:
            data = reader.get_data()
            head = data.get("Head")
            hand = data.get("Right Hand")
            if hand is not None:
                if hand_origin is None:
                    hand_origin = hand[:3, 3].copy()
                pos_delta_quest = hand[:3, 3] - hand_origin
                pos_delta_robot = QUEST_TO_ROBOT_ROTATION @ pos_delta_quest
                line = f"{pos_str(pos_delta_quest):<26}{pos_str(pos_delta_robot):<26}"
            else:
                line = f"{'--':<26}{'--':<26}"
            if head is not None and hand is not None:
                rel = head[:3, :3].T @ hand[:3, :3]
                line += f"{euler_str(head[:3, :3]):<26}{euler_str(hand[:3, :3]):<26}{euler_str(rel):<26}"
            else:
                line += "waiting for quest3 data..."
            if iface is not None:
                ee_pose = iface.get_ee_pose()
                line += f"{euler_str(ee_pose[:3, :3]):<26}"
            print(line)
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()


if __name__ == "__main__":
    main()
