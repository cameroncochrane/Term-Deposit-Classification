from pathlib import Path
import sqlite3

from monitoring.normalize_mlflow_paths import PATH_COLUMNS, main, normalize, rebase_path

MAC_ROOT = "/Users/someone/Data Projects/Term Deposit Classification"
RUNNER_ROOT = "/home/runner/work/Term-Deposit-Classification/Term-Deposit-Classification"


def _make_db(path: Path, values: dict[tuple[str, str], list[str]]) -> None:
    conn = sqlite3.connect(path)
    for table, column in PATH_COLUMNS:
        conn.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, {column} TEXT)")
        for value in values.get((table, column), []):
            conn.execute(f"INSERT INTO {table} ({column}) VALUES (?)", (value,))
    conn.commit()
    conn.close()


def _read(path: Path, table: str, column: str) -> list[str]:
    conn = sqlite3.connect(path)
    rows = [row[0] for row in conn.execute(f"SELECT {column} FROM {table} ORDER BY id")]
    conn.close()
    return rows


def test_rebase_path_moves_mac_and_runner_paths_onto_new_root(tmp_path):
    for old_root in (MAC_ROOT, RUNNER_ROOT):
        assert (
            rebase_path(f"{old_root}/mlruns/1/abc/artifacts", tmp_path)
            == f"{tmp_path.as_posix()}/mlruns/1/abc/artifacts"
        )


def test_rebase_path_keeps_file_scheme(tmp_path):
    result = rebase_path(f"file://{MAC_ROOT}/mlruns/1", tmp_path)

    assert result == f"file://{tmp_path.as_posix()}/mlruns/1"


def test_rebase_path_leaves_remote_and_unrelated_values_alone(tmp_path):
    for value in (
        "s3://bucket/mlruns/1/abc",
        "models:/m-0dd524e0b877489ba42e776b760fde9d",
        "/some/other/place/model.pkl",
    ):
        assert rebase_path(value, tmp_path) == value


def test_normalize_rewrites_every_path_column_and_is_idempotent(tmp_path):
    db = tmp_path / "mlflow.db"
    _make_db(
        db,
        {
            ("experiments", "artifact_location"): [f"{MAC_ROOT}/mlruns/1"],
            ("runs", "artifact_uri"): [f"{MAC_ROOT}/mlruns/1/run-a/artifacts"],
            ("logged_models", "artifact_location"): [
                f"{RUNNER_ROOT}/mlruns/1/models/m-1/artifacts"
            ],
            ("model_versions", "storage_location"): [f"{MAC_ROOT}/mlruns/1/models/m-2/artifacts"],
        },
    )

    changed = normalize(db, tmp_path)
    changed_again = normalize(db, tmp_path)

    assert changed == len(PATH_COLUMNS)
    assert changed_again == 0
    for table, column in PATH_COLUMNS:
        (value,) = _read(db, table, column)
        assert value.startswith(f"{tmp_path.as_posix()}/mlruns/1")


def test_normalize_skips_missing_tables_and_null_values(tmp_path):
    db = tmp_path / "mlflow.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE experiments (id INTEGER PRIMARY KEY, artifact_location TEXT)")
    conn.execute("INSERT INTO experiments (artifact_location) VALUES (NULL)")
    conn.commit()
    conn.close()

    assert normalize(db, tmp_path) == 0


def test_main_is_a_no_op_when_the_db_does_not_exist(tmp_path, capsys):
    exit_code = main([str(tmp_path / "missing.db")])

    assert exit_code == 0
    assert "nothing to normalize" in capsys.readouterr().out


def test_main_honours_root_flag(tmp_path):
    db = tmp_path / "mlflow.db"
    _make_db(db, {("experiments", "artifact_location"): [f"{MAC_ROOT}/mlruns/1"]})
    root = tmp_path / "workspace"
    root.mkdir()

    assert main([str(db), "--root", str(root)]) == 0

    assert _read(db, "experiments", "artifact_location") == [
        f"{root.resolve().as_posix()}/mlruns/1"
    ]
