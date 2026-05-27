import os
import glob
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


class VelodyneFilePublisher(Node):
    def __init__(self):
        super().__init__('velodyne_file_publisher')
        with open('src/cross_predictor/cross_predictor/config.yaml') as f:
            _cfg = yaml.safe_load(f)
        self.declare_parameter('topic_name', '/velodyne_points')
        self.declare_parameter('folder_path', _cfg.get('RADAR_FOLDER', ''))
        self.declare_parameter('publish_period', 0.1)

        self.topic_name = self.get_parameter('topic_name').value
        self.folder_path = self.get_parameter('folder_path').value
        self.publish_period = self.get_parameter('publish_period').value

        self.bin_files = sorted(glob.glob(os.path.join(self.folder_path, '*.bin')))
        if not self.bin_files:
            self.get_logger().error(f"No .bin files found in '{self.folder_path}'")
            raise RuntimeError('No .bin files found')

        self.get_logger().info(f"Found {len(self.bin_files)} .bin files in '{self.folder_path}'")

        qos = QoSProfile(depth=50)
        self.publisher_ = self.create_publisher(PointCloud2, self.topic_name, qos)
        self.timer = self.create_timer(self.publish_period, self.timer_callback)
        self.frame_idx = 0

    def timer_callback(self):
        bin_path = self.bin_files[self.frame_idx]
        cloud_msg = self._bin_to_pointcloud2(bin_path)
        self.publisher_.publish(cloud_msg)
        self.get_logger().info(
            f"Published frame {self.frame_idx} ({os.path.basename(bin_path)})"
            f" — {cloud_msg.width} points"
        )
        self.frame_idx = (self.frame_idx + 1) % len(self.bin_files)

    def _bin_to_pointcloud2(self, bin_path: str) -> PointCloud2:
        # KITTI/Velodyne binary: flat array of float32 [x, y, z, intensity, ...]
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

        fields = [
            PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'velodyne'

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = points.shape[0]
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 16          # 4 fields × 4 bytes
        msg.row_step = msg.point_step * msg.width
        msg.data = points.tobytes()
        msg.is_dense = True
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = VelodyneFilePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()