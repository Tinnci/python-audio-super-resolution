from audio_super_resolution.cli import main


def test_env_info(capsys) -> None:
    assert main(["--env-info"]) == 0
    output = capsys.readouterr().out
    assert "audio-super-resolution:" in output
    assert "python:" in output
