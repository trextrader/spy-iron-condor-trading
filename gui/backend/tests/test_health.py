"""
Health Check Tests
Phase 6.1 - Core Infrastructure
"""

import pytest
from fastapi.testclient import TestClient


def test_health_endpoint():
    """Test that health endpoint returns healthy status."""
    # Import here to avoid import issues during collection
    from gui.backend.main import app

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_root_endpoint():
    """Test that root endpoint returns API info."""
    from gui.backend.main import app

    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert data["status"] == "running"


def test_readiness_endpoint():
    """Test that readiness endpoint returns check results."""
    from gui.backend.main import app

    client = TestClient(app)
    response = client.get("/api/readiness/ready")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "checks" in data
