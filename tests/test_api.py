import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from agent.api.main import app, init_db
from agent.data_pipeline.models import Base, Satellite, TLEHistory

@pytest.fixture
def test_client():
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    init_db('sqlite:///:memory:')
    return TestClient(app)

def test_list_satellites(test_client):
    response = test_client.get("/satellites?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert 'count' in data
    assert 'data' in data

def test_tle_history_not_found(test_client):
    response = test_client.get("/tle/99999/history")
    assert response.status_code == 200
    data = response.json()
    assert 'error' in data

def test_maneuvers(test_client):
    response = test_client.get("/maneuvers?days=30")
    assert response.status_code == 200
    data = response.json()
    assert 'count' in data
    assert 'data' in data

def test_status(test_client):
    response = test_client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert 'status' in data
