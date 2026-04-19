"""Entrypoint for factor interpretation analysis."""

from interpret.factor_analysis import build_arg_parser, run_factor_analysis


if __name__ == "__main__":
    output = run_factor_analysis(build_arg_parser().parse_args())
    print(f"Saved factor analysis to {output}")

