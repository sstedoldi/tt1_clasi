# Unused setup.py for GCP Vertex AI custom training job
from setuptools import setup

setup(
    name='gcp_task',
    version='0.1',
    # IMPORTANTE: Usamos py_modules porque tus scripts están en la raíz, 
    # no en una carpeta con __init__.py
    py_modules=['gcp_task', 'distilbert_utils'],
    
    install_requires=[
        # --- Core ML ---
        'transformers>=4.51.3',
        'scikit-learn>=1.6.1', 
        
        # --- Data & Utilities ---
        'pandas>=2.2.3',
        'numpy',              
        'tqdm>=4.67.1',
        'python-json-logger>=4.0.0',
        
        # --- GCP Integration ---
        'google-cloud-storage>=3.7.0'
    ],
    
    include_package_data=True,
    description='Training package for Vertex AI custom job'
)