"""Setup lcvn_ac installation."""

from os import path as op
import re
from setuptools import find_packages, setup


def _read(f):
    return open(op.join(op.dirname(__file__), f)).read() if op.exists(f) else ""


_meta = _read("lcvn_ac/__init__.py")


def find_meta(_meta, string):
    l_match = re.search(r"^" + string + r'\s*=\s*"(.*)"', _meta, re.M)
    if l_match:
        return l_match.group(1)
    raise RuntimeError(f"Unable to find {string} string.")


install_requires = [
    l for l in _read("requirements.txt").split("\n") if l and not l.startswith("#") and not l.startswith("-")
]

meta = dict(
    name=find_meta(_meta, "__project__"),
    version=find_meta(_meta, "__version__"),
    license=find_meta(_meta, "__license__"),
    description="Lcvn-AC: Language-Conditioned Imitation Learning with World Models",
    platforms=("Any"),
    zip_safe=False,
    keywords="pytorch lcvn-ac".split(),
    author=find_meta(_meta, "__author__"),
    author_email=find_meta(_meta, "__email__"),
    url="",
    packages=find_packages(exclude=["tests"]),
    install_requires=install_requires,
)

if __name__ == "__main__":
    print("find_package", find_packages(exclude=["tests"]))
    setup(**meta)
