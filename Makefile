.PHONY: install
install:
	poetry self update
	poetry install --no-root

.PHONY: notebooks
notebooks:
	poetry run jupyter lab --ip='*' --no-browser --NotebookApp.token='' --NotebookApp.password=''