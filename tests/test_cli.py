from pathlib import Path

from endodaveval.cli import parser


def test_cli_parser_accepts_full_evaluation_command():
    args = parser().parse_args(
        ["--config", "configs/scared.local.json", "--stage", "all"]
    )

    assert args.config == Path("configs/scared.local.json")
    assert args.stage == "all"
    assert args.limit_sequences is None
