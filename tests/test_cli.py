import pytest

from src.main import main


def test_cli_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_cli_invalid_mode_exits_two():
    with pytest.raises(SystemExit) as exc:
        main(["--mode", "not-a-mode"])
    assert exc.value.code == 2


def test_cli_image_without_source_fails_before_model():
    code = main(["--mode", "image", "--no-display"])
    assert code == 1
