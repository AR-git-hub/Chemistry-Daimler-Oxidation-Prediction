"""Entrypoint for inference."""

from dot.infer import build_arg_parser, infer


if __name__ == "__main__":
    infer(build_arg_parser().parse_args())

