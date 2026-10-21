"""The .env auto-loader: keys load, shell wins, gates never load from file."""
import os

from webaggs.config import _load_env_file


def test_env_file_loads_keys_but_never_gates(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n"
        "OPENAI_API_KEY=file-key\n"
        "SERPER_API_KEY=\"quoted-key\"\n"
        "RUN_LIVE=1\n"                      # must NOT load: spend gate
        "RUN_ONLINE=1\n"                    # must NOT load: network gate
        "EMPTY=\n"
        "garbage line without equals\n")
    for k in ("OPENAI_API_KEY", "SERPER_API_KEY", "RUN_LIVE", "RUN_ONLINE",
              "EMPTY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SERPER_API_KEY", "shell-key")   # shell precedence
    _load_env_file(env)
    assert os.environ["OPENAI_API_KEY"] == "file-key"
    assert os.environ["SERPER_API_KEY"] == "shell-key"  # setdefault: shell won
    assert "RUN_LIVE" not in os.environ
    assert "RUN_ONLINE" not in os.environ
    assert "EMPTY" not in os.environ


def test_missing_env_file_is_a_noop(tmp_path):
    _load_env_file(tmp_path / "nope.env")               # simply returns
