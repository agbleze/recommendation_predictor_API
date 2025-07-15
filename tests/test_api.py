from reviewoler.api.api import app
import pytest

@pytest.fixture(scope="module")
def client():
    with app.test_client() as client:
        yield client
        
def test_root_endpoint(client):
    """Test the root endpoint of the API"""
    response = client.get('/')
    assert response.status_code == 200
    assert isinstance(response.get_json(), dict)
    assert "message" in response.get_json()
    
def test_request_prediction(client):
    """Test review prediction"""
    result = client.post('/predict', json={'review': 'its a complete scam'}).get_json()
    assert isinstance(result, dict)
    assert "probability" in result
    assert "category" in result
    assert isinstance(result["probability"], float)

