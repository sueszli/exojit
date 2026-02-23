import os
import sys
from argparse import ArgumentParser
from pathlib import Path

from xdsl_exo import compile_path
from xdsl_exo.compiler import CompilerOptions

def main():
    parser = ArgumentParser(prog=Path(sys.argv[0]).name, description="Compile an Exo library to MLIR.")
    parser.add_argument("source", type=str, nargs="+", help="Source file(s) to compile")
    parser.add_argument("-o", "--output", help="The output target. For single source files, this is the output file. For multiple source files, this is the output directory.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    parser.add_argument("--target", default="llvm", choices=["llvm", "exo", "builtin", "lowered", "scf"])
    parser.add_argument("--prefix", help="Prefix to prepend to all procedure names.")

    args = parser.parse_args()
    srcs = [Path(src) for src in args.source]

    if len(srcs) == 0:
        parser.error("Must provide at least one source file.")

    # construct opts
    opts = CompilerOptions()
    opts.target = args.target
    opts.prefix = args.prefix
    opts.verbose = args.verbose

    # single source file
    if len(srcs) == 1 and not args.output:
        dest = srcs[0].with_suffix(".mlir")
        os.makedirs(dest.parent, exist_ok=True)
        compile_path(srcs[0], dest, opts)
        return

    # check if output has no suffix - assumes a directory, we make it if it doesn't exist
    if args.output and not Path(args.output).suffix:
        os.makedirs(args.output, exist_ok=True)

    # check non-directory output
    if not args.output or not os.path.isdir(args.output) and args.output != "-":
        parser.error("Must provide a directory output for multiple source files.")

    # multiple source files
    dests = [Path(args.output) / src.with_suffix(".mlir").name for src in srcs]
    for src, dest in zip(srcs, dests):
        compile_path(src, dest if args.output != "-" else None, opts)


if __name__ == "__main__":
    main()
