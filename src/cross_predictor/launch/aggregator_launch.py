from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. Proximity Processor
        Node(
            package='cross_predictor',
            executable='proximity',
            name='proximity_worker'
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
        # 4. Distance Processor
        #Node(
        #    package='cross_predictor',
        #    executable='distance',
        #    name='distance_worker'
        #),
        # 5. The Aggregator (The one with the Manual Map)
        Node(
            package='cross_predictor',
            executable='aggregator',
            name='final_aggregator',
            output='screen' # Ensures you see the sync logs
        )
    ])