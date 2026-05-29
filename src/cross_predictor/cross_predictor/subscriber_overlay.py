import ast
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


CROSSING_COLOR = (0, 0, 255)      # red  (BGR)
NO_CROSSING_COLOR = (0, 200, 0)   # green
UNKNOWN_COLOR = (200, 200, 0)     # yellow

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
THICKNESS = 2
LINE_H = 22   # pixels between text lines
MARGIN = 10   # pixels from left edge


def _color_for(prediction: str):
    if prediction == 'crossing':
        return CROSSING_COLOR
    if prediction == 'not-crossing':
        return NO_CROSSING_COLOR
    return UNKNOWN_COLOR


class OverlayNode(Node):
    """Draws /cross_predictor/final predictions over /image_raw frames."""

    def __init__(self):
        super().__init__('prediction_overlay')
        self.bridge = CvBridge()
        self._latest_prediction = None

        # ── Video recording parameters ────────────────────────────────────
        self.declare_parameter('save_video', False)
        self.declare_parameter('video_path', 'overlay_output.mp4')
        self.declare_parameter('video_fps',  10.0)

        self._save_video = self.get_parameter('save_video').value
        self._video_path = self.get_parameter('video_path').value
        self._video_fps  = self.get_parameter('video_fps').value
        self._writer: cv2.VideoWriter | None = None   # opened lazily on first frame

        if self._save_video:
            self.get_logger().info(
                f'Video recording enabled → {self._video_path}  ({self._video_fps} fps)')

        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(
            String,
            '/cross_predictor/final',
            self._prediction_callback,
            10,
        )
        self.create_subscription(
            Image,
            '/image_raw',
            self._image_callback,
            qos_best_effort,
        )
        self._pub = self.create_publisher(Image, '/cross_predictor/image_overlay', 10)
        self.get_logger().info('OverlayNode ready — subscribing to /cross_predictor/final and /image_raw')

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _prediction_callback(self, msg: String):
        try:
            self._latest_prediction = ast.literal_eval(msg.data)
        except Exception as e:
            self.get_logger().warn(f'Could not parse prediction message: {e}')

    def _image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return

        frame = self._draw_predictions(frame)

        # Publish annotated frame
        out_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        out_msg.header = msg.header
        self._pub.publish(out_msg)

        # Write to video file if enabled
        if self._save_video:
            self._write_frame(frame)

    # ── Video writer ──────────────────────────────────────────────────────

    def _write_frame(self, frame):
        if self._writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self._writer = cv2.VideoWriter(
                self._video_path, fourcc, self._video_fps, (w, h))
            if not self._writer.isOpened():
                self.get_logger().error(
                    f'Could not open VideoWriter for {self._video_path}')
                self._writer = None
                self._save_video = False   # disable to avoid repeated errors
                return
            self.get_logger().info(
                f'VideoWriter opened: {w}x{h} @ {self._video_fps} fps → {self._video_path}')
        self._writer.write(frame)

    def destroy_node(self):
        if self._writer is not None and self._writer.isOpened():
            self._writer.release()
            self.get_logger().info(f'Video saved to {self._video_path}')
        super().destroy_node()

    # ── Drawing ───────────────────────────────────────────────────────────

    def _draw_predictions(self, frame):
        pred = self._latest_prediction
        if pred is None:
            return frame

        frame_id = pred.get('frame_id', '')
        pedestrians = pred.get('pedestrians', {})

        cv2.putText(frame, f'Frame: {frame_id}', (MARGIN, LINE_H),
                    FONT, FONT_SCALE, (255, 255, 255), THICKNESS, cv2.LINE_AA)

        if not pedestrians:
            cv2.putText(frame, 'No pedestrians detected', (MARGIN, LINE_H * 2),
                        FONT, FONT_SCALE, UNKNOWN_COLOR, THICKNESS, cv2.LINE_AA)
            return frame

        y = LINE_H * 2
        for ped_id, data in pedestrians.items():
            prediction = data.get('prediction', 'unknown')
            prob = data.get('crossing_probability', 0.0)
            features = data.get('features', {})
            color = _color_for(prediction)

            lines = [
                f'Ped {ped_id} | {prediction.upper()}  ({prob:.1%})',
                f'  action={features.get("action", "?")}  '
                f'proximity={features.get("proximity", "?")}',
                f'  attention={features.get("attention", "?")}  '
                f'orientation={features.get("orientation", "?")}',
                f'  distance={features.get("distance", "?")}',
            ]
            for line in lines:
                cv2.putText(frame, line, (MARGIN, y),
                            FONT, FONT_SCALE, color, THICKNESS, cv2.LINE_AA)
                y += LINE_H
            y += 6

        return frame


def main(args=None):
    rclpy.init(args=args)
    node = OverlayNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
