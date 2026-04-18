"""Training entrypoint via structured package path."""

from dot.train import build_arg_parser, train


def main() -> None:
    train(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()

