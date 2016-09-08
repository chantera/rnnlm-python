PY = python
# PY = python3

all:
	./scripts/download.sh
	swig -c++ -python ./rnnlm-python/rnnlm.i
	$(PY) setup.py build_ext --inplace

install:
	$(PY) setup.py install

test:
	$(PY) ./scripts/main.py -train ./rnnlm/train -valid ./rnnlm/valid -rnnlm test_model -hidden 15 -rand-seed 1 -max-iter 10 -debug 2 -class 100 -bptt 4 -bptt-block 10 -direct-order 3 -direct 2 -binary

.PHONY: clean
clean:
	find . -name "*.pyc" -delete
	find . -name "*.so" -delete
	find . -name "__pycache__" -delete
	find . -name "test_model" -type f -delete
	find . -name "test_model.output.txt" -type f -delete
	find . -name "rnnlm.py" -type l -delete
	rm -rf rnnlm
	rm -rf build
