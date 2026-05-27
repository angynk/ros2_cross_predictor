# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os
import queue
import threading

import cv2
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image

PRELOAD_QUEUE_SIZE = 10


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')

        with open('src/cross_predictor/cross_predictor/config.yaml') as f:
            _cfg = yaml.safe_load(f)
        self.declare_parameter('image_folder', _cfg.get('CAMERA_FOLDER', ''))
        self.declare_parameter('topic_name', '/image_raw')
        self.declare_parameter('publish_period', 0.025)
        self.image_folder = self.get_parameter('image_folder').value
        self.topic_name = self.get_parameter('topic_name').value
        self.publish_period = self.get_parameter('publish_period').value

        qos = QoSProfile(depth=10)
        self.publisher_ = self.create_publisher(Image, self.topic_name, qos)
        self.seq = 0

        self.image_files = sorted(
            glob.glob(os.path.join(self.image_folder, '*.jpg'))
        )

        if not self.image_files:
            self.get_logger().error(f'No images found in {self.image_folder}')
            return

        self.bridge = CvBridge()
        self.index = 0

        # Background thread preloads frames so timer_callback never blocks on disk
        self._frame_queue = queue.Queue(maxsize=PRELOAD_QUEUE_SIZE)
        self._preload_thread = threading.Thread(target=self._preload_loop, daemon=True)
        self._preload_thread.start()

        self.timer = self.create_timer(self.publish_period, self.timer_callback)

        self.get_logger().info(
            f'Publishing {len(self.image_files)} images from {self.image_folder} '
            f'every {self.publish_period}s'
        )

    def _preload_loop(self):
        idx = 0
        while rclpy.ok():
            path = self.image_files[idx]
            image = cv2.imread(path)
            if image is not None:
                self._frame_queue.put((path, image))
            idx = (idx + 1) % len(self.image_files)

    def timer_callback(self):
        try:
            image_path, image = self._frame_queue.get_nowait()
        except queue.Empty:
            self.get_logger().warn('Frame queue empty — preloader lagging behind publish rate')
            return

        msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = os.path.basename(image_path)
        self.seq += 1

        self.publisher_.publish(msg)
        self.get_logger().info(f'Published: {os.path.basename(image_path)}')


def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    executor = MultiThreadedExecutor()
    executor.add_node(minimal_publisher)
    try:
        executor.spin()
    finally:
        minimal_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
