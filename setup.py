from setuptools import find_packages,setup
from typing import List
HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''THis function will create the requirements list !'''
    requirements=[]
    with open(file_path) as fp:
        requirements = fp.readlines()
        requirements = [req.replace('\n',' ') for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)


setup(
    name='ModelForge',
    version='0.0.1',
    author='Aryan',
    author_email='infiniteknowledge425@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)
