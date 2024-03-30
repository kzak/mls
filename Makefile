.PHONY: install
install:
	poetry self update
	poetry install --no-root

.PHONY: fmt
fmt:
	poetry run isort .
	poetry run black .

.PHONY: notebooks
notebooks:
	poetry run jupyter lab --ip='*' --no-browser --NotebookApp.token='' --NotebookApp.password=''