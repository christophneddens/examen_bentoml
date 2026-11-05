import requests
import jwt
from datetime import datetime, timedelta

# Address of the running service
BASE_URL = "http://localhost:3000"

# Same secret and algorithm as in src/service.py
JWT_SECRET_KEY = "your_jwt_secret_key_here"
JWT_ALGORITHM = "HS256"

GOOD_LOGIN = {"credentials": {"username": "user123", "password": "password123"}}
BAD_LOGIN = {"credentials": {"username": "user123", "password": "wrong"}}

GOOD_INPUT = {
    "input_data": {
        "gre_score": 320,
        "toefl_score": 110,
        "university_rating": 4,
        "sop": 4.0,
        "lor": 4.5,
        "cgpa": 9.1,
        "research": 1,
    }
}


def get_token():
    response = requests.post(f"{BASE_URL}/login", json=GOOD_LOGIN)
    return response.json()["token"]


# ---------- 1. JWT authentication ----------

def test_missing_token():
    response = requests.post(f"{BASE_URL}/predict", json=GOOD_INPUT)
    assert response.status_code == 401


def test_invalid_token():
    headers = {"Authorization": "Bearer this-is-not-a-valid-token"}
    response = requests.post(f"{BASE_URL}/predict", json=GOOD_INPUT, headers=headers)
    assert response.status_code == 401


def test_expired_token():
    # A token that expired one hour ago
    payload = {"sub": "user123", "exp": datetime.utcnow() - timedelta(hours=1)}
    expired_token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    headers = {"Authorization": f"Bearer {expired_token}"}
    response = requests.post(f"{BASE_URL}/predict", json=GOOD_INPUT, headers=headers)
    assert response.status_code == 401


def test_valid_token():
    headers = {"Authorization": f"Bearer {get_token()}"}
    response = requests.post(f"{BASE_URL}/predict", json=GOOD_INPUT, headers=headers)
    assert response.status_code == 200


# ---------- 2. Login API ----------

def test_login_success():
    response = requests.post(f"{BASE_URL}/login", json=GOOD_LOGIN)
    assert response.status_code == 200
    assert "token" in response.json()


def test_login_wrong_credentials():
    response = requests.post(f"{BASE_URL}/login", json=BAD_LOGIN)
    assert response.status_code == 401


# ---------- 3. Prediction API ----------

def test_predict_valid_input():
    headers = {"Authorization": f"Bearer {get_token()}"}
    response = requests.post(f"{BASE_URL}/predict", json=GOOD_INPUT, headers=headers)
    assert response.status_code == 200
    prediction = response.json()["prediction"]
    assert isinstance(prediction, list)
    assert len(prediction) == 1
    assert isinstance(prediction[0], float)


def test_predict_invalid_input():
    headers = {"Authorization": f"Bearer {get_token()}"}
    bad_input = {"input_data": {"gre_score": "abc"}}   # wrong type, fields missing
    response = requests.post(f"{BASE_URL}/predict", json=bad_input, headers=headers)
    assert response.status_code == 400
