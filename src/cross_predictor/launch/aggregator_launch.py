from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. The Camera / Source
        Node(
            package='cross_predictor',
            executable='talker',
            name='camera'
        ),
        # 2. Action Processor
        Node(
            package='cross_predictor',
            executable='action',
            name='action_worker'
        ),
        # 3. Attention Processor
        Node(
            package='cross_predictor',
            executable='attention',
            name='attention_worker'
        ),
        # 4. Orientation Processor
        Node(
            package='cross_predictor',
            executable='orientation',
            name='orientation_worker'
        ),
        # 5. The Aggregator (The one with the Manual Map)
        Node(
            package='cross_predictor',
            executable='aggregator',
            name='final_aggregator',
            output='screen' # Ensures you see the sync logs
        )
    ])