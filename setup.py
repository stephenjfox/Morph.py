from distutils.core import setup
from setuptools import find_packages

setup(
    name='morph-py',  # How you named your package folder (MyLib)
    packages=find_packages(),  # Chose the same as "name"
    # Start with a small number and increase it with every change you make
    version='0.0.6-beta',
    # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    license='GNU',
    # Give a short description about your library
    description='Automate the refinement of your neural network architecture',
    author='Stephen Fox',  # Type in your name
    author_email='stevethecoder34@gmail.com',  # Type in your E-Mail
    # Provide either the link to your github or to your website
    url='https://github.com/stephenjfox/Morph.py',
    # I explain this later on
    download_url='https://github.com/stephenjfox/Morph.py/archive/v0.0.6.tar.gz',
    keywords=[
        'machine learning', 'deep learning', 'nas', 'architecture',
        'neural networks'
    ],
    install_requires=[  # I get to this in a second
        'torch',
        'numpy',
        'torchvision',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',  # either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',  # Define that your audience are developers
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',  # Again, pick a license
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
