init:
	pip install -r requirements.txt

build: clean
	python3 setup.py build

wheel: clean
	python3 setup.py bdist_wheel

install:
	python setup.py install

clean:
	rm -rf *egg-info
	rm -rf build dist
	rm -rf __pycache__
	rm -rf .pytest_cache

test: clean
	pytest -sv --show-capture all
