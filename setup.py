from pathlib import Path

from setuptools import find_packages, setup

here = Path(__file__).parent

requirements_path = here / "requirements" / "prod.txt"


def read_requirements(path):
    try:
        with path.open(mode="rt", encoding="utf-8") as fp:
            return list(filter(None, (line.split("#")[0].strip() for line in fp)))
    except IndexError:
        raise RuntimeError(f"{path} is broken")


setup(
    name="lazytext",
    python_requires=">=3.7.0",
    setup_requires=["setuptools_scm"],
    install_requires=read_requirements(requirements_path),
    use_scm_version={
        "version_scheme": "guess-next-dev",
        "local_scheme": "dirty-tag",
        "write_to": "src/lazytext/_repo_version.py",
        "write_to_template": 'version = "{version}"\n',
        "relative_to": __file__,
    },
    include_package_data=True,
    package_data={},
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={"console_scripts": ["lazytext = lazytext.cli:entrypoint"]},
    author="lazytext",
    author_email="jay.vala@msn.com",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering",
        "Topic :: Text Processing :: General",
    ],
    url="https://github.com/jdvala/lazytext",
)
