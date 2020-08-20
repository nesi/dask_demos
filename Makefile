KERNEL := hpc_for_datascience_demos
KERNEL_DIR = $(HOME)/.local/share/jupyter/kernels/$(KERNEL)

NOTEBOOKS := notebooks/hyperparameters_search_basic.ipynb
HTML_FILES := $(NOTEBOOKS:.ipynb=.html)

all: $(NOTEBOOKS) $(HTML_FILES)

help:
	@echo 'Available targets:'
	@echo ''
	@echo '	help	Display this help message'
	@echo '	venv	Create a Python virtual environment and register it to jupyter kernels'
	@echo '	all	Run all scripts and convert them into notebooks and .html files'
	@echo ''

# convert a notebook into .html document
notebooks/%.html: notebooks/%.ipynb
	. venv/bin/activate; jupyter nbconvert --to html "$<"

# convert a script into a notebook, run it and store it in the notebooks folder
notebooks/%.ipynb: src/%.py requirements-pinned.txt
	mkdir -p notebooks
	. venv/bin/activate; jupytext --to notebook --execute --set-kernel $(KERNEL) "$<"
	mv "src/$(@F)" "$@"

# freeze the dependencies installed in the virtual environment for reproducibility
requirements-pinned.txt: venv/.canary
	venv/bin/pip freeze > "$@"

# create a virtual environment and register it as a jupyter kernel
venv/.canary: requirements.txt requirements-dev.txt
	python3 -m venv venv
	venv/bin/pip install -U pip
	venv/bin/pip install -r requirements.txt
	venv/bin/pip install -r requirements-dev.txt
	venv/bin/python3 -m ipykernel install --user --name $(KERNEL)
	# use a wrapper script to load required modules before starting the kernel
	sed -i 's|"argv": \[|"argv": \[\n  "$(PWD)/src/kernel_wrapper.bash",|' \
	    $(KERNEL_DIR)/kernel.json
	touch "$@"

# alias for virtual environment creation
venv: venv/.canary

.PHONY: venv help
