#!/user/bin/env python
"""
w7x option starter
"""
import sys
import unittest
import doctest
import pathlib
import argparse
from .__init__ import *


def run_doctests():
    """
    Find all doctests and execute them
    """
    this_dir = pathlib.Path(__file__).parent.resolve()
    for f in list(this_dir.glob('**/*.py')):
        doctest.testfile(str(f.resolve()),
                         module_relative=False)  # , verbose=True, optionflags=doctest.ELLIPSIS)


def load_unittests(loader=None, suite=None):
    if loader is None:
        loader = unittest.TestLoader()
    if suite is None:
        suite = unittest.TestSuite()
    parent = pathlib.Path(__file__).parent.parent
    for f in list(parent.glob('*/test*.py')):
        sys.path.insert(0, str(f.parent))
        mod = __import__(f.name[:-3])
        for test in loader.loadTestsFromModule(mod):
            suite.addTests(test)
        sys.path.remove(str(f.parent))
    return suite


def run_unittests(arg):
    run_doctests()
    # unittest.main(defaultTest='load_unittests')


def parse_args(args):
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='tf_neiss app')
    parser.add_argument('--version', action='version', version='v' + __version__,
                        help="Show program's version number and exit")

    # subparsers
    subparsers = parser.add_subparsers(help='sub-command help')

    # create the parser for the "test" command
    parserExtend = subparsers.add_parser('test', help='test help')
    parserExtend.set_defaults(func=run_unittests)

    # If no arguments were used, print the base-level help which lists possible commands.
    if len(args) == 0:
        parser.print_help(file=sys.stderr)
        sys.exit(1)

    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
