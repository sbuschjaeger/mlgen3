from setuptools import setup, find_packages

setup(name='mlgen3',
    version='1.0',
    description='mlgen3',
    author='tbd',
    author_email='tbd',
    url='tbd',
    python_requires='>=3.7',
    packages=find_packages('.'),
    install_requires=[
        'scikit-learn', 'numpy', 'pandas', 'tqdm'
    ]
)