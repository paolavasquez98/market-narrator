"""Shared fixtures for tests that need a real Postgres+pgvector connection.

Only test files that explicitly request `skip_if_db_unreachable` pay the
cost of checking DB connectivity; pure unit tests elsewhere are
unaffected. Locally (no `docker compose up -d` running), these tests
skip; in CI, a real `pgvector/pgvector:pg16` service container is always
up (see .github/workflows/ci.yml), so they run for real there.
"""

from __future__ import annotations

import psycopg
import pytest

from finrag.config.settings import get_settings


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def skip_if_db_unreachable(settings):
    try:
        conn = psycopg.connect(settings.database_url, connect_timeout=2)
        conn.close()
    except psycopg.OperationalError:
        pytest.skip(
            f"Postgres not reachable at {settings.database_url} -- "
            "run `docker compose up -d` to enable this test"
        )
