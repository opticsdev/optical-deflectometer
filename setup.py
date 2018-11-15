"""
setup.py file for optical-deflectometer


"""
from setuptools import setup, find_packages
setup(
    name="optical-deflectometer",
    version="0.1a",
    packages=find_packages(),
    scripts=['say_hello.py'],

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=['docutils>=0.3'],

    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.rst'],
        # And include any *.msg files found in the 'hello' package, too:
        'hello': ['*.msg'],
    },

    # metadata to display on PyPI
    author="J. Johnson",
    author_email="opticsdeveloper@gmail.com",
    description="Open Source Optical Deflectometry collection, processing, and reconstruction",
    license="LGPL v3",
    keywords="optics deflectometer SCOTS lens",
    url="https://github.com/opticsdev/optical-deflectometer",   # project home page, if any
    project_urls={
        "Source Code": "https://github.com/opticsdev/optical-deflectometer",
    }

)
