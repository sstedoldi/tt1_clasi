from setuptools import setup

setup(
    name='gcp_task',
    version='0.1',
    py_modules=['gcp_task', 'distilbert_utils'], 
    include_package_data=True,
    description='Training package for Vertex AI'
)