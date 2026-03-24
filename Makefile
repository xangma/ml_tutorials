.PHONY: install execute-notebooks clean-notebooks

NOTEBOOKS = iris.ipynb mnist.ipynb

install:
	python -m pip install -r requirements.txt

execute-notebooks:
	python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 iris.ipynb
	python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 mnist.ipynb

clean-notebooks:
	python -m jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace $(NOTEBOOKS)
