poetry run mlflow models build-docker -m server/outputs -n mlops_model_image
cd server/
poetry run docker compose up --build
cd ..