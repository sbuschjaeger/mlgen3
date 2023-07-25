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
        'numpy', 'pandas', 'tqdm', 'importlib_resources'
    ],
    extras_require = {
        'cpp': ["astyle-py"],
        'pruning': ["PyPruning @ git+ssh://git@github.com/sbuschjaeger/PyPruning.git"],
        'neuralnet': ["torch", "torchvision", "torchaudio", "torchinfo"],
        'weka': ["python-javabridge", "python-weka-wrapper3"]
    }
)