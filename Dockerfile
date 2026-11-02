ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base

ARG ROS_DISTRO
ENV ROS_DISTRO=${ROS_DISTRO}
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-c"]

# Install system dependencies & ROS 2 packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-std-msgs \
    ros-${ROS_DISTRO}-sensor-msgs \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install heavy PyTorch wheels FIRST before copying any project files so layer cache is never invalidated by code changes
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Runtime user matching the host user (uid/gid 1000) so FastDDS shared memory
# transport works between host processes and this container (ipc: host).
RUN groupadd -g 1000 ros && useradd -m -u 1000 -g 1000 -s /bin/bash ros

WORKDIR /app

# Copy requirements and install python packages
# (--ignore-requires-python: diffusers main declares >=3.11 but runs on humble's 3.10)
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt --ignore-requires-python

# diffusers main also enforces python>=3.11 at import time via its dependency
# table, although the code itself is 3.10-compatible (verified with compileall).
# Relax the runtime guard for humble's Python 3.10.
RUN sed -i 's/"python": "python>=3.11.0"/"python": "python>=3.10.0"/' \
    /usr/local/lib/python3.10/dist-packages/diffusers/dependency_versions_table.py

# Copy package source files
COPY . /app/ros2_ws/src/bob_flux3k

# Build ROS 2 workspace
WORKDIR /app/ros2_ws
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && colcon build --packages-select bob_flux3k

WORKDIR /app

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "run", "bob_flux3k", "tti", "--ros-args", "--params-file", "/app/ros2_ws/src/bob_flux3k/config/tti.yaml"]
