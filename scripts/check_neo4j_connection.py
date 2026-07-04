from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.graph.neo4j_connection import check_neo4j_connection  # noqa: E402


def main() -> int:
    status = check_neo4j_connection(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    print(
        json.dumps(
            {
                "uri": status.uri,
                "user": status.user,
                "password_configured": status.password_configured,
                "database": status.database,
                "available": status.available,
                "error": status.error,
                "reason": status.reason,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if status.available else 2


if __name__ == "__main__":
    raise SystemExit(main())
