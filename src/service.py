import numpy as np
import bentoml
from pydantic import BaseModel
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import jwt
from datetime import datetime, timedelta

# Secret key for the tokens (do not hardcode secrets in a real project)
JWT_SECRET_KEY = "your_jwt_secret_key_here"
JWT_ALGORITHM = "HS256"

# Users allowed to log in
USERS = {
    "user123": "password123",
    "user456": "password456",
}


# What /login expects
class Credentials(BaseModel):
    username: str
    password: str


# What /predict expects: the 7 features of one student
class InputModel(BaseModel):
    gre_score: int
    toefl_score: int
    university_rating: int
    sop: float
    lor: float
    cgpa: float
    research: int


# Create a token that is valid for 1 hour
def create_jwt_token(user_id: str) -> str:
    expiration = datetime.utcnow() + timedelta(hours=1)
    payload = {"sub": user_id, "exp": expiration}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


# The "doorman": checks the token before a request reaches /predict
class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path.endswith("/predict"):
            token = request.headers.get("Authorization")
            if not token:
                return JSONResponse(status_code=401, content={"detail": "Missing authentication token"})
            try:
                token = token.split()[1]  # "Bearer <token>"
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            except jwt.ExpiredSignatureError:
                return JSONResponse(status_code=401, content={"detail": "Token has expired"})
            except jwt.InvalidTokenError:
                return JSONResponse(status_code=401, content={"detail": "Invalid token"})
            request.state.user = payload.get("sub")
        return await call_next(request)


@bentoml.service
class AdmissionPredictionService:
    def __init__(self):
        # Load the model once, when the service starts
        self.model = bentoml.sklearn.load_model("admission_lr:latest")

    @bentoml.api(route="/login")
    def login(self, credentials: Credentials):
        if USERS.get(credentials.username) == credentials.password:
            token = create_jwt_token(credentials.username)
            return {"token": token}
        return JSONResponse(status_code=401, content={"detail": "Invalid credentials"})

    @bentoml.api(route="/predict")
    def predict(self, input_data: InputModel):
        # Same order as the columns used for training
        x = np.array([
            input_data.gre_score, input_data.toefl_score, input_data.university_rating,
            input_data.sop, input_data.lor, input_data.cgpa, input_data.research,
        ], dtype=float).reshape(1, -1)
        pred = self.model.predict(x)
        return {"prediction": pred.tolist()}


# Put the doorman in front of the whole service
AdmissionPredictionService.add_asgi_middleware(JWTAuthMiddleware)
