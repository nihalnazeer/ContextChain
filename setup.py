from setuptools import setup, find_packages

setup(
    name='contextchain',
    version='1.0.0',
    description='Pipeline-based execution framework for contextual task chains',
    author='Nihal Nazeer',
    author_email='youremail@example.com',
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'requests',
        'pydantic',
        'urllib3',
        'python-dotenv',
        # Add any others your project uses
    ],
    entry_points={
        'console_scripts': [
            'contextchain=app.cli:main'  # If you define a CLI
        ]
    },
    python_requires='>=3.8',
)
