init:
	pip install -r requirements.txt

build: clean
	python setup.py build

wheel: clean
	python setup.py bdist_wheel

install:
	python setup.py install

clean:
	rm -rf *egg-info
	rm -rf build dist
	rm -rf __pycache__
	rm -rf .pytest_cache

test: clean
	pytest -sv --show-capture all --disable-pytest-warnings

bsr-data: clean
	wget --tries=10 http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
	tar -xzvf BSR_bsds500.tgz -C images/