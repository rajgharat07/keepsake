"""Seed sample memories. Prefer: python -m keepsake seed"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from keepsake import db, memory  # noqa: E402


def main() -> None:
    db.init_schema()
    folder = ROOT / "data" / "sample_memories"
    for f in sorted(folder.glob("*.md")):
        ids = memory.ingest_file(f)
        print(f"{f.name}: {ids}")

    # Historical rows so `onthisday` has something to show
    today = datetime.now(timezone.utc)
    for years_ago, title, content in [
        (
            12,
            "First monsoon after the move",
            "Dadi hung mango leaves on the door and said a new house needs old blessings.",
        ),
        (
            40,
            "Letter from the village post",
            "Dadaji's letter arrived late. Dadi read it twice before she sat down.",
        ),
    ]:
        created = today.replace(year=today.year - years_ago)
        memory.ingest_text(
            content,
            kind="event",
            title=title,
            people=["Dadi"],
            tags=["onthisday", "archive"],
            created_at=created,
        )
        print(f"onthisday seed: {title} ({created.date()})")
    print("done. total=", db.count_memories())


if __name__ == "__main__":
    main()
