VENV := .venv
PYTHON := python3

.PHONY: setup
setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

.PHONY: run
run:
	$(VENV)/bin/python main.py

.PHONY: clean
clean:
	rm -rf $(VENV)

