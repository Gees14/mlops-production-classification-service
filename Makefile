.PHONY: install train evaluate run-api test mlflow-ui docker-build docker-up docker-down clean

install:
	pip install -r requirements.txt

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

run-api:
	python scripts/run_api.py

test:
	pytest tests/ -v

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

docker-build:
	docker build -t classification-service .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache
	rm -rf artifacts/model/model.joblib
	rm -rf artifacts/preprocessing/preprocessor.joblib
	rm -rf artifacts/preprocessing/training_stats.json
	rm -rf logs/predictions.jsonl
	rm -rf reports/metrics.json reports/data_validation.json reports/model_card.md
	rm -rf reports/figures/
	@echo "Clean complete."
