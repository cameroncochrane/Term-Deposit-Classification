"""
Rewrite machine-specific artifact paths stored inside an MLflow sqlite backend store.

MLflow records each experiment's / run's / logged model's artifact location as an absolute
path at creation time. The drift-monitor workflow shares one `mlflow.db` between a developer
laptop and the CI runner (via S3), so a DB created on a Mac contains `/Users/<name>/.../mlruns/...`
and one last written by CI contains `/home/runner/work/.../mlruns/...`. Training on the other
machine then tries to create directories under a path that doesn't exist there
(`PermissionError: '/Users'`).

This re-roots every local `.../mlruns/...` path onto the current project root. It is idempotent,
works regardless of which machine wrote the original path, and leaves remote locations
(`s3://...`, `models:/...`) alone.

Usage (from the project root, after pulling mlflow.db from S3):

    python app/monitoring/normalize_mlflow_paths.py [path/to/mlflow.db] [--root DIR]
"""

from contextlib import closing
from pathlib import Path
import sqlite3
import sys

# (table, column) pairs MLflow populates with local filesystem locations.
PATH_COLUMNS = [
    ("experiments", "artifact_location"),
    ("runs", "artifact_uri"),
    ("logged_models", "artifact_location"),
    ("model_versions", "storage_location"),
]

ARTIFACT_DIR_MARKER = "/mlruns/"


def rebase_path(value: str, root: Path) -> str:
    """
    Re-root `value` onto `<root>/mlruns/...`. Returns `value` unchanged if it isn't a local
    mlruns path (a remote URI, a `models:/` URI, or anything without an `/mlruns/` segment).
    """
    scheme = ""
    path = value
    if value.startswith("file://"):
        scheme, path = "file://", value[len("file://") :]
    elif "://" in value or value.startswith("models:/"):
        return value

    marker_at = path.find(ARTIFACT_DIR_MARKER)
    if marker_at == -1:
        return value

    remainder = path[marker_at + len(ARTIFACT_DIR_MARKER) :]
    return f"{scheme}{root.as_posix()}{ARTIFACT_DIR_MARKER}{remainder}"


def normalize(db_path: Path, root: Path) -> int:
    """Rewrite stored paths in `db_path` in place. Returns the number of distinct values changed."""
    changed = 0
    with closing(sqlite3.connect(db_path)) as conn, conn:
        existing_tables = {
            name for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        for table, column in PATH_COLUMNS:
            if table not in existing_tables:
                continue
            columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            if column not in columns:
                continue

            rows = conn.execute(
                f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL"
            ).fetchall()
            for (old_value,) in rows:
                new_value = rebase_path(old_value, root)
                if new_value != old_value:
                    conn.execute(
                        f"UPDATE {table} SET {column} = ? WHERE {column} = ?",
                        (new_value, old_value),
                    )
                    changed += 1
    return changed


def main(argv: list[str]) -> int:
    args = list(argv)
    root = Path.cwd()
    if "--root" in args:
        flag_at = args.index("--root")
        root = Path(args[flag_at + 1]).resolve()
        del args[flag_at : flag_at + 2]
    db_path = Path(args[0]) if args else Path("mlflow.db")

    if not db_path.exists():
        print(f"{db_path} not found - nothing to normalize.")
        return 0

    changed = normalize(db_path, root.resolve())
    print(
        f"Re-rooted {changed} stored artifact path(s) in {db_path} onto {root.resolve()}/mlruns/"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
