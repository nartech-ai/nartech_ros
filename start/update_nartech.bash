#!/bin/bash

# This updater script pull the src code for
# NACE, ONA, MeTTa-NARS, MeTTa-Morph, NARS-GPT
# as well as builds packages for ROS workspace

# Check if the local branch is behind the remote branch

if zenity --question --text="Do you want to update NARTECH?" --title="Update NARTECH" --icon="/home/nartech/Documents/logo.png"; then
    cd ~/nartech_ws/src
    rm -rf nartech_moveit_config
    cd ~
    grep -qxF "alias metta='python3 ~/hyperon-experimental/python/hyperon/metta.py'" ~/.bashrc || echo "alias metta='python3 ~/hyperon-experimental/python/hyperon/metta.py'" >> ~/.bashrc
    grep -q '^export OPENAI_API_KEY=' ~/.bashrc || { key=$(zenity --entry --title="Set OPENAI_API_KEY" --text="Enter your OpenAI API Key:"); [ -n "$key" ] && echo "export OPENAI_API_KEY=\"$key\"" >> ~/.bashrc; }
    cd ~
    cd PLN
    git remote set-url origin https://github.com/trueagi-io/PLN
    git pull
    cd ~
    ls metta-motto || git clone https://github.com/zarqa-ai/metta-motto
    cp -r ~/metta-motto/motto /home/nartech/nartech_ws/src/nartech_ros/
    mv /home/nartech/Desktop/logo*.png /home/nartech/Documents/
    cd ~/nartech_ws/src/nartech_ros/
    git pull
    cp ~/nartech_ws/src/nartech_ros/manipulation/config/open_manipulator_controllers.yaml ~/nartech_ws/config/open_manipulator_controllers.yaml
    cp -r ~/nartech_ws/src/nartech_ros/manipulation/tb4_openmanipulator_moveit ~/nartech_ws/src/
    cp -r ~/nartech_ws/src/nartech_ros/manipulation/open_manipulator_x_description ~/nartech_ws/src/
    cp ~/nartech_ws/src/nartech_ros/manipulation/nav2_minimal_tb4_description/urdf/standard/turtlebot4.urdf.xacro /opt/ros/jazzy/share/nav2_minimal_tb4_description/urdf/standard/
    cd ~/AniNAL
    git pull
    cd ~/OpenNARS-for-Applications
    git pull
    sh build.sh
    cd ~/NACE
    git pull
    cd ~/metta-morph
    git pull
    cd ~/metta-morph/metta-nars
    git pull
    cd ~/metta-nars
    git pull
    cd ~/NARS-GPT
    git pull
    # User selected Yes, proceed with updates
    cd ~/nartech_ws
    colcon build
    cd ~/AniNAL
    sh ./build.sh
fi

# Common actions regardless of the choice
zenity --info --text "NARTECH ready! Press 'OK', then 'Terminal', then enter './start_nartech.bash metta demo_with_nars.metta'!  " --title="Update Status" --icon="/home/nartech/Documents/logo.png"
gnome-terminal -- bash -i -c "geany /home/nartech/NACE/input.metta &"

