#!/bin/bash
# Writes the ROS 2 environment to .vscode/ros.env so the VS Code debugger
# (launch.json "envFile") can import rclpy and the robot's message packages.
# Run automatically as the preLaunchTask before every F5.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"

source /opt/ros/humble/setup.bash
source /home/ubuntu/ros2_ws/install/local_setup.bash
source /home/ubuntu/third_party_ros2/third_party_ws/install/local_setup.bash

# Let scripts anywhere in the project do "from interfaces.clawpiinterface import ClawPi".
export PYTHONPATH="$(dirname "$HERE"):$PYTHONPATH"

env | grep -E '^(PATH|PYTHONPATH|LD_LIBRARY_PATH|AMENT_PREFIX_PATH|COLCON_PREFIX_PATH|CMAKE_PREFIX_PATH|PKG_CONFIG_PATH|ROS_[A-Z_]*|RMW_[A-Z_]*)=' \
    > "$HERE/ros.env"

# Use UDP only: VS Code runs as root here, and Fast DDS shared memory
# between root and the ubuntu-user bringup nodes fails.
echo "FASTRTPS_DEFAULT_PROFILES_FILE=$HERE/fastdds_udp.xml" >> "$HERE/ros.env"
echo "RCUTILS_COLORIZED_OUTPUT=1" >> "$HERE/ros.env"
