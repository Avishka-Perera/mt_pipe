"""Setup file"""

from setuptools import setup, find_packages

with open("requirements/requirements.txt", encoding="utf-8") as handler:
    requirements = handler.readlines()
with open("requirements/dev.txt", encoding="utf-8") as handler:
    dev_requirements = handler.readlines()

setup(
    name="mt_pipe",
    version="0.1.0",
    author="Avishka Perera",
    author_email="pereramat2000@gmail.com",
    description="A Training, Validation, and Testing pipeline focused on multitask setups",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={"dev": dev_requirements},
    entry_points={
        "console_scripts": [
            "mt_pipe=mt_pipe.main:main",
        ],
    },
)
