.PHONY: test smoke eval diagnose run-api run-ui compose-up compose-full compose-down

smoke:
	python -m compileall -q .
	python -m tests.smoke_test

test:
	python -m pytest -q

eval:
	python evaluation/eval_demo.py
	python evaluation/eval_stress.py

diagnose:
	python scripts/diagnose_stack.py --ingest-demo

run-api:
	uvicorn app.api:app --reload --port 8000

run-ui:
	streamlit run app/ui.py

compose-up:
	docker compose up -d --build

compose-full:
	docker compose -f docker-compose.yml -f docker-compose.optional.yml --profile full build --build-arg INSTALL_FULL=true
	docker compose -f docker-compose.yml -f docker-compose.optional.yml --profile full up -d

compose-down:
	docker compose down
