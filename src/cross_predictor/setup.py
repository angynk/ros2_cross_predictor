from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'cross_predictor'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', 'cross_predictor', 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='angie-melo',
    maintainer_email='angynk17@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'predictor = cross_predictor.predictor:main',
            'talker = cross_predictor.publisher_member_function:main',
            'orientation = cross_predictor.subscriber_orientation_function:main',
            'attention = cross_predictor.subscriber_attention_function:main',
            'action = cross_predictor.subscriber_action_function:main',
            'proximity = cross_predictor.subscriber_proximity_function:main',
            'aggregator = cross_predictor.aggregator_predictor_function:main',
        ],
    },
)
