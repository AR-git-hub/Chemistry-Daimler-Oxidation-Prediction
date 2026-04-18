"""Entrypoint for model training."""

from dot.train import build_arg_parser, train


if __name__ == "__main__":
    train(build_arg_parser().parse_args())

