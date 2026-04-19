"""Entrypoint for submission file checks."""

from dot.validate import build_arg_parser, validate_predictions


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    validate_predictions(args.predictions_path, args.test_path)

