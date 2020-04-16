import yaml
from setuptools import setup

with open("environment.yml", 'r') as stream:
    out = yaml.load(stream)
    requirements = out['dependencies'][1:]  # we do not return python

setup(
    name='static_optim',
    version='0.1.0',
    description="Reimplementation of the Opensim static optimization",
    author="Benjamin Michaud, Romain Martinez & Mickael Begon",
    author_email='martinez.staps@gmail.com',
    url='https://github.com/pyomeca/',
    license='Apache 2.0',
    packages=['static_optim'],
    install_requires=requirements,
    keywords='pyomeca',
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
