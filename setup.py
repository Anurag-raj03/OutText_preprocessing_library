from setuptools import setup, find_packages

setup(
    name='OutText_preprocessing',
    version='0.1.0',
    description='A package for outlier removal and text preprocessing',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Anurag Raj',
    author_email='anuragraj4483@gmail.com',
    url="https://github.com/Anurag-raj03/OutText_preprocessing_library.git",
    packages=find_packages(),  # Automatically discover all packages
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "scikit-learn",
        "nltk",
        "emoji",
        "autocorrect",
        "pytest",  # Add any required libraries
    ],
    python_requires='>=3.6',
    test_suite='nose.collector',  # Use `nose` for tests or `unittest`
    tests_require=['nose'],  # Add necessary testing dependencies
)
