"""
Publishing:
    either:
        $make publish
    or manually:
        run tests
        remember to change the version in setup(... ) below
        $ git commit -am "<my message>"
        $ git push
        $ git tag -a v<my.version.id> -m "<comment to my version>"  # tag version
        $ git push origin v<my.version.id>  # explicitly push tag to the shared server
        $ python setup.py sdist
        $ twine upload dist/*

"""
from setuptools import setup, find_packages
setup(
    packages=find_packages(),

)


# in the terminal:
# pip install .
