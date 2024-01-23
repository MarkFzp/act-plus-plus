from interbotix_xs_modules.arm import InterbotixManipulatorXS
from aloha_scripts.robot_utils import move_arms, torque_on, move_grippers
from constants import PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
import argparse
import numpy as np

# for calibrating head cam and arms being symmetrical

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--all', action='store_true', default=False)
    args = argparser.parse_args()

    puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_left', init_node=True)
    puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_right', init_node=False)

    all_bots = [puppet_bot_left, puppet_bot_right]
    for bot in all_bots:
        torque_on(bot)
    
    multiplier = np.array([-1, 1, 1, -1, 1, 1])
    puppet_sleep_position_left = np.array([-0.8, -0.5, 0.5, 0, 0.65, 0])
    puppet_sleep_position_right = puppet_sleep_position_left * multiplier
    all_positions = [puppet_sleep_position_left, puppet_sleep_position_right]
    move_arms(all_bots, all_positions, move_time=2)

    # move_grippers(all_bots, [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=1)  # open


if __name__ == '__main__':
    main()
