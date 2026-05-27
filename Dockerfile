FROM osrf/ros:humble-desktop

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set up workspace
WORKDIR /ros2_cross_predictor

# Install Python dependencies
COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN pip install --upgrade Pillow

# Copy ROS2 packages
COPY my_msgs/ ./my_msgs/
COPY src/ ./src/

# Copy your models
COPY models/ ./models/

# Copy YOLO model weights from project root
COPY *.pt ./

# Build the workspace
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install"

# Source the workspace on every shell launch
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /ros2_cross_predictor/install/setup.bash" >> ~/.bashrc

# Default entrypoint
CMD ["/bin/bash"]