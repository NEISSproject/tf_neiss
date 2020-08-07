#!/user/bin/env python
"""
w7x option starter
"""
import argparse
import sys

import tf_neiss


class SomeAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        print("Example action invoked by manage in namespace: %r with values %r"
              " and option string %r" % (namespace, values, option_string))
        setattr(namespace, self.dest, values)


def manage(args):
    print("Managing!")
    print(args.x * args.y)


def parse_args(args):
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='tf_neiss app')
    parser.add_argument('--version', action='version',
                        version='v' + tf_neiss.__version__,
                        help="Show program's version number and exit")
    parser = argparse.ArgumentParser(prog='tf_neiss app')

    # subparsers
    subparsers = parser.add_subparsers(help='sub-command help')

    # create the parser for the "test" command
    example_sub_parser = subparsers.add_parser('manage', help='manage something')
    example_sub_parser.add_argument('-x', type=int, default=1)
    example_sub_parser.add_argument('-y', type=float, default=42.)
    example_sub_parser.set_defaults(func=manage)

    # If no arguments were used, print base-level help with possible commands.
    if len(args) == 0:
        parser.print_help(file=sys.stderr)
        sys.exit(1)

    args = parser.parse_args(args)
    # let argparse do the job of calling the appropriate function after
    # argument parsing is complete
    args.func(args)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
