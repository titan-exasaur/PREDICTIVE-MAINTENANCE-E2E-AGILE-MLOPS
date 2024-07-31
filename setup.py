from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    This will return a list of requirements, excluding -e . if present.
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = [req.strip() for req in file_obj.readlines()]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements

setup(
    name="industrial_predictive_maintenance",
    version="0.0.1",
    author="kumar",
    author_email="tinytachyon14341@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
