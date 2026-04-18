"""Inference entrypoint via structured package path."""

from dot.infer import build_arg_parser, infer


def main() -> None:
    infer(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()

