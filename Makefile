KERNEL := hpc_for_datascience_demos
KERNEL_DIR = $(HOME)/.local/share/jupyter/kernels/$(KERNEL)

NOTEBOOKS := notebooks/hyperparameters_search_basic.ipynb
HTML_FILES := $(NOTEBOOKS:.ipynb=.html)

all: $(NOTEBOOKS) $(HTML_FILES)

notebooks/%.html: notebooks/%.ipynb
	. venv/bin/activate; jupyter nbconvert --to html "$<"

notebooks/%.ipynb: src/%.py requirements-pinned.txt
	mkdir -p notebooks
	. venv/bin/activate; jupytext --to notebook --execute --set-kernel $(KERNEL) "$<"
	mv "src/$(@F)" "$@"

requirements-pinned.txt: venv/.canary
	venv/bin/pip freeze > "$@"

venv/.canary: requirements.txt requirements-dev.txt
	python3 -m venv venv
	venv/bin/pip install -U pip
	venv/bin/pip install -r requirements.txt
	venv/bin/pip install -r requirements-dev.txt
	venv/bin/python3 -m ipykernel install --user --name $(KERNEL)
	sed -i 's|"argv": \[|"argv": \[\n  "$(PWD)/src/kernel_wrapper.bash",|' $(KERNEL_DIR)/kernel.json
	touch "$@"

venv: venv/.canary

.PHONY: venv
