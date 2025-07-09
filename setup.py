from setuptools import setup, find_packages
import os

setup(
    name='contextchain',
    version='0.1.0',  # Beta version for PyPI (use 0.1.0b0 if pre-release)
    description='Pipeline-based execution framework for contextual task chains',
    long_description=open('README.md').read() if os.path.exists('README.md') else 'A framework for orchestrating AI and full-stack workflows.',
    long_description_content_type='text/markdown' if os.path.exists('README.md') else 'text/plain',
    author='Nihal Nazeer',
    author_email='Nhaal160@gmail.com',
    url='https://github.com/yourusername/contextchain',  # Replace with your repo URL
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'pymongo>=4.13.2',
        'requests>=2.32.4',
        'pydantic>=2.11.7',
        'urllib3<2',
        'python-dotenv>=1.0.0',
    ],
    entry_points={
        'console_scripts': [
            'contextchain=app.cli:main'
        ]
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',  # Will be overridden by pyproject.toml license
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)