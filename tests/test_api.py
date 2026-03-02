import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_register_image():
    response = client.post("/register", json={"image": "path/to/image.jpg", "name": "Test User"})
    assert response.status_code == 200
    assert "Registered:" in response.text

def test_recognize_image():
    response = client.post("/recognize", json={"image": "path/to/unknown_image.jpg"})
    assert response.status_code == 200
    assert "No matches" in response.text or "score" in response.text

def test_register_image_invalid():
    response = client.post("/register", json={"image": "", "name": "Test User"})
    assert response.status_code == 422

def test_recognize_image_invalid():
    response = client.post("/recognize", json={"image": ""})
    assert response.status_code == 422