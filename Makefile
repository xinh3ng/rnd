install:
	pip install -r requirements.txt

lint:
	black ai bi data_eng eng -l 120 --target-version=py37

test:
	black ai bi data_eng eng -l 120 --target-version=py37 --check
	# pytest -s -v --cov= tests --cov-fail-under=60 --disable-pytest-warnings
