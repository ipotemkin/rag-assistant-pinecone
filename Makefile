PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python

.PHONY: venv install refresh doctor clean-wheelhouse clean-venv reinstall \
	cli-index cli-search

CLI := $(VENV_PYTHON) -m lang_chain
INDEX ?= demo-index

venv:
	$(PYTHON) -m venv $(VENV)

install:
	pipi

offline-install:
	$(VENV)/bin/python -m pip install --no-index \
		--find-links $$HOME/.local/pip/wheelhouse \
		-r requirements.txt

upgrade-pip:
	./$(VENV)/bin/python -m pip install --upgrade pip --timeout 60

refresh:
	pipi-refresh

doctor:
	pipi-doctor

clean-wheelhouse:
	pipi-clean

clean-venv:
	rm -rf $(VENV)

reinstall: clean-venv venv install

cli-index:
	$(CLI) index --name $(INDEX) --file etc/sample.txt

cli-search:
	$(CLI) search --name $(INDEX) --query "векторная база данных"
