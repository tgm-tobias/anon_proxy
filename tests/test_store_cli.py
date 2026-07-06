import json

import pytest

from anon_proxy.store_cli import filter_entries, main, purge_tokens


def _store_data():
    return {
        "reverse": {
            "<PERSON_1>": "Alice Smith",
            "<PERSON_2>": "la",
            "<EMAIL_1>": "alice@x.com",
        },
        "counters": {"PERSON": 2, "EMAIL": 1},
    }


class TestFilterEntries:
    def test_by_label(self):
        rows = filter_entries(_store_data(), "PERSON", None, None)
        assert [r[0] for r in rows] == ["<PERSON_1>", "<PERSON_2>"]

    def test_by_max_len(self):
        rows = filter_entries(_store_data(), None, None, 3)
        assert [r[0] for r in rows] == ["<PERSON_2>"]


class TestPurge:
    def test_purge_removes_mapping_keeps_counter(self):
        data, removed, missing = purge_tokens(_store_data(), ["<PERSON_2>"])
        assert removed == ["<PERSON_2>"]
        assert missing == []
        assert "<PERSON_2>" not in data["reverse"]
        assert data["counters"]["PERSON"] == 2

    def test_purge_unknown_token_reports_missing(self):
        data, removed, missing = purge_tokens(_store_data(), ["<PERSON_99>"])
        assert removed == []
        assert missing == ["<PERSON_99>"]
        assert len(data["reverse"]) == 3


def _write_store(path):
    path.write_text(json.dumps(_store_data()))


def test_cli_prune_dry_run_changes_nothing(tmp_path, capsys):
    path = tmp_path / "store.json"
    _write_store(path)

    rc = main(
        [
            "--store",
            str(path),
            "prune",
            "--label",
            "PERSON",
            "--max-len",
            "3",
            "--dry-run",
        ]
    )

    assert rc == 0
    assert "would remove <PERSON_2>" in capsys.readouterr().out
    assert json.loads(path.read_text()) == _store_data()
    assert not (tmp_path / "store.json.bak").exists()


def test_cli_prune_requires_filter_or_all(tmp_path, capsys):
    path = tmp_path / "store.json"
    _write_store(path)

    with pytest.raises(SystemExit) as exc:
        main(["--store", str(path), "prune", "--dry-run"])

    assert exc.value.code == 2
    assert "prune requires at least one filter or --all" in capsys.readouterr().err
    assert json.loads(path.read_text()) == _store_data()


def test_cli_prune_all_requires_explicit_all(tmp_path, capsys):
    path = tmp_path / "store.json"
    _write_store(path)

    assert main(["--store", str(path), "prune", "--all", "--dry-run"]) == 0

    out = capsys.readouterr().out
    assert "would remove <PERSON_1>" in out
    assert "would remove <PERSON_2>" in out
    assert "would remove <EMAIL_1>" in out


def test_cli_prune_writes_backup_then_modifies(tmp_path):
    path = tmp_path / "store.json"
    _write_store(path)

    rc = main(
        [
            "--store",
            str(path),
            "prune",
            "--label",
            "PERSON",
            "--max-len",
            "3",
        ]
    )

    assert rc == 0
    assert json.loads((tmp_path / "store.json.bak").read_text()) == _store_data()
    new = json.loads(path.read_text())
    assert "<PERSON_2>" not in new["reverse"]
    assert new["counters"]["PERSON"] == 2


def test_cli_purge_writes_backup_then_modifies(tmp_path, capsys):
    path = tmp_path / "store.json"
    _write_store(path)

    rc = main(["--store", str(path), "purge", "<EMAIL_1>"])

    assert rc == 0
    assert "removed <EMAIL_1>" in capsys.readouterr().out
    assert json.loads((tmp_path / "store.json.bak").read_text()) == _store_data()
    new = json.loads(path.read_text())
    assert "<EMAIL_1>" not in new["reverse"]
    assert new["counters"]["EMAIL"] == 1


def test_cli_purge_unknown_token_exits_nonzero(tmp_path, capsys):
    path = tmp_path / "store.json"
    _write_store(path)

    assert main(["--store", str(path), "purge", "<PERSON_99>"]) == 1

    captured = capsys.readouterr()
    assert "not found: <PERSON_99>" in captured.err
    assert captured.out == ""
    assert json.loads(path.read_text()) == _store_data()
    assert not (tmp_path / "store.json.bak").exists()


def test_cli_list_and_show(tmp_path, capsys):
    path = tmp_path / "store.json"
    _write_store(path)

    assert main(["--store", str(path), "list", "--label", "PERSON"]) == 0
    list_out = capsys.readouterr().out
    assert "<PERSON_1>" in list_out
    assert "Alice Smith" in list_out
    assert "<EMAIL_1>" not in list_out

    assert main(["--store", str(path), "show", "<EMAIL_1>"]) == 0
    assert "alice@x.com" in capsys.readouterr().out


def test_cli_accepts_store_after_command(tmp_path, capsys):
    path = tmp_path / "store.json"
    _write_store(path)

    assert main(["show", "<PERSON_1>", "--store", str(path)]) == 0

    assert "Alice Smith" in capsys.readouterr().out
