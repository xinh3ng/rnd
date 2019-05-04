HOST=127.0.0.1
TEST_PATH=./

clean-pyc:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
	find . -name '*~' -exec rm --force  {} +

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info

isort:
	sh -c "isort --skip-glob=.tox --recursive . "

install:
	virtualenv venv 
	. venv/bin/activate && pip install -r requirements.txt

lint:
	black ML data_eng eng -l 120 --target-version=py36

test:
	# black churnover tests -l 120 --target-version=py36 --check
	# pytest -s -v --cov=churnover tests --cov-fail-under=18 --disable-pytest-warnings
