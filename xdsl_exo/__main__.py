import os
import sys
from argparse import ArgumentParser
from pathlib import Path

from xdsl_exo import CompilerOptions, compile_path


def main():
    parser = ArgumentParser(prog=Path(sys.argv[0]).name, description="Compile an Exo library to MLIR.")
    parser.add_argument("source", type=str, help="Source file to compile")
    parser.add_argument("-o", "--output", help="Output file. Defaults to stdout.")
    parser.add_argument("--target", default="llvm", choices=["llvm", "exo", "builtin", "lowered", "scf"])
    parser.add_argument("--prefix", help="Prefix to prepend to all procedure names.")
    args = parser.parse_args()

    opts = CompilerOptions()
    opts.target = args.target
    opts.prefix = args.prefix

    dest = None
    if args.output and args.output != "-":
        dest = Path(args.output)
        os.makedirs(dest.parent, exist_ok=True)

    compile_path(Path(args.source), dest, opts)


if __name__ == "__main__":
    main()
