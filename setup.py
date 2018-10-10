from setuptools import setup

setup(
    name='agent',
    version='0.0.1',
    packages=['tensorboard/plugins/agent'],
    package_data={'agent': ['resources/*']},
    url='https://github.com/andrewschreiber/agent',
    author='Andrew Schreiber',
    install_requires=[
      'futures',
      'grpcio'
    ],
)
