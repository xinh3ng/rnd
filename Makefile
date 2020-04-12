install:
	pip install -r requirements.txt

lint:
	black AI BI data_eng eng -l 120 --target-version=py37

test:
	black AI BI data_eng eng -l 120 --target-version=py37 --check
	# pytest -s -v --cov=dashboard_gold_lake tests --cov-fail-under=60 --disable-pytest-warnings
