init:
	pip install -r requirements.txt

build:
	python setup.py build

install:
	python setup.py install

test:
	pytest -sv

clean:
	rm -rf build dist
	rm -rf .pytest_cache