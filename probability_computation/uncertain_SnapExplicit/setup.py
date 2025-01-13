from setuptools import find_packages, setup

setup(
    name='uncertain',
    version='0.4.0',
    packages=find_packages(),
    install_requires=['torch>=1.7.0', 'numpy', 'scipy', 'pytorch-lightning', 'h5py', 'pandas', 'requests', 'matplotlib',
                      'scikit-learn', 'tqdm'],
    license='MIT',
    classifiers=['Development Status :: 1 - Beta',
                 'License :: OSI Approved :: MIT License',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
)
