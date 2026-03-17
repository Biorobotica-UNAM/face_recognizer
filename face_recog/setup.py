from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'face_recog'

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return paths

qwen_files = package_files('weights/qwen2vl')

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/face_recog/weights/qwen2vl', qwen_files),
    ],
    install_requires=[
        'setuptools', 'opencv-python', 'numpy', 'ultralytics',
        'torch', 'transformers', 'Pillow', 'cv-bridge'
    ],
    zip_safe=True,
    maintainer='rusanrod',
    maintainer_email='rsandovr@gmail.com',
    description='Vision package',
    license='MIT',
    entry_points={
        'console_scripts': [
            'face = face_recog.face_recog:main',
            'facev2 = face_recog.face_recogv2:main',
            'faceFlorence = face_recog.face_recogFlorence2:main'
            
        ],
    },
)

