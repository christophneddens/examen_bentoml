#BentoMlexam - admission prediction service

Linear regression model predicts a student's chance of admission to a university.
The model is stored in the BentoML Model Store as `admission_lr` and served by a
BentoML 1.4.35 service with two endpoints:

- `POST /login` - returns a JWT token (valid 1 hour). Test user: `user123` / `password123`
- `POST /predict` - returns the prediction; needs the header `Authorization: Bearer <token>`

Model performance on the test set: R2 = 0.82, RMSE = 0.061, MAE = 0.043

##Run the service and the tests (in this order)

1. Load the Docker image
```bash
docker load -i admission_prediction_service.tar
```

2. Start the service in the background on port 3000
```bash
docker run --rm -d -p 3000:3000 --name admission_api admission_prediction_service:latest
```

3. Wait until the service is ready
```bash
sleep 10
```

4. Create a virtual environment and install the requirements
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

5. Run the tests (8 tests, all must pass)
```bash
pytest -v tests/
```

6. Stop the service
```bash
docker stop admission_api
```

##example requests

```bash
curl -X POST http://localhost:3000/login -H "Content-Type: application/json" -d '{"credentials": {"username": "user123", "password": "password123"}}'

curl -X POST http://localhost:3000/predict -H "Content-Type: application/json" -H "Authorization: Bearer <token>" -d '{"input_data": {"gre_score": 320, "toefl_score": 110, "university_rating": 4, "sop": 4.0, "lor": 4.5, "cgpa": 9.1, "research": 1}}'
```

##how the image was built

```bash
python src/prepare_data.py
python src/train_model.py
bentoml build
bentoml containerize admission_prediction_service:latest -t admission_prediction_service:latest
docker save admission_prediction_service:latest -o admission_prediction_service.tar
```
