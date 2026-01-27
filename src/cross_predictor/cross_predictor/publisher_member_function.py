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

import rclpy
import glob
import os
import cv2

import rclpy

from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')

        self.declare_parameter('image_folder', '/home/angie-melo/Documents/DataSets/JAAD/images/video_0190')
        self.declare_parameter('topic_name', '/image_raw')
        self.declare_parameter('publish_period', 0.5)
        self.image_folder = self.get_parameter('image_folder').value
        self.topic_name = self.get_parameter('topic_name').value
        self.publish_period = self.get_parameter('publish_period').value
        self.publisher_ = self.create_publisher(Image, self.topic_name, 10)

        self.timer = self.create_timer(self.publish_period, self.timer_callback)
        
        self.image_files = sorted(
            glob.glob(os.path.join(self.image_folder, '*'))
        )

        if not self.image_files:
            self.get_logger().error(f'No images found in {self.image_folder}')
            return

        self.bridge = CvBridge()
        self.index = 0

        self.get_logger().info(
            f'Publishing {len(self.image_files)} images from {self.image_folder} '
            f'every {self.publish_period}s'
        )


    def timer_callback(self):
        image_path = self.image_files[self.index]
        image = cv2.imread(image_path)

        if image is None:
            self.get_logger().warn(f'Failed to read image: {image_path}')
            return

        msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = image_path 

        self.publisher_.publish(msg)
        self.get_logger().info(f'Published: {os.path.basename(image_path)}')

        # Move to next image (loop)
        self.index = (self.index + 1) % len(self.image_files)

        '''msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1'''


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
