PYTHON_VERSION ?= 3.8
CONDA_VENV_PATH ?= $(PWD)/venv
KERNEL_NAME ?= $(shell basename $(CURDIR))

NOTEBOOKS := notebooks/hyperparameters_search_basic.ipynb notebooks/hyperparameters_search_nzrse.ipynb
HTML_FILES := $(NOTEBOOKS:.ipynb=.html)

CONDA_BASE := $(shell conda info --base)
CONDA_VENV := . $(CONDA_BASE)/etc/profile.d/conda.sh && conda activate $(CONDA_VENV_PATH) &&
KERNEL_DIR := $(HOME)/.local/share/jupyter/kernels/$(KERNEL_NAME)

all: help

help:
	@echo 'Available targets:'
	@echo ''
	@echo '        help        Display this help message'
	@echo '        venv        Create a Conda virtual environment and register it as a Jupyter kernel'
	@echo '        venv_nesi   Create a Conda virtual environment for Jupyter on NeSI'
	@echo '        clean_venv  Remove the Conda virtual environment'
	@echo '        notebooks   Convert some scripts into notebooks and .html files'
	@echo '        format      Format Python scripts'
	@echo ''
	@echo 'Configurable variables (current value):'
	@echo ''
	@echo '        PYTHON_VERSION   Python interpreter version ($(PYTHON_VERSION))'
	@echo '        CONDA_VENV_PATH  Path of the Conda virtual environment ($(CONDA_VENV_PATH))'
	@echo '        KERNAL_NAME      Jupyter kernel name ($(KERNEL_NAME))'
	@echo ''

notebooks: $(NOTEBOOKS) $(HTML_FILES)

# convert a notebook into .html document
notebooks/%.html: notebooks/%.ipynb
	$(CONDA_VENV) jupyter nbconvert --to html "$<"

# convert a script into a notebook, run it and store it in the notebooks folder
notebooks/%.ipynb: src/%.py requirements.txt
	$(CONDA_VENV) jupytext --to notebook --execute --set-kernel $(KERNEL_NAME) "$<"
	mkdir -p notebooks
	mv "src/$(@F)" "$@"

# freeze the dependencies installed in the virtual environment for reproducibility
requirements.txt: venv/.canary
	$(CONDA_VENV) pip freeze > "$@"

# create a virtual environment and register it as a jupyter kernel
venv/.canary: setup.cfg setup.py
	conda create -y -p $(CONDA_VENV_PATH) python=$(PYTHON_VERSION)
	$(CONDA_VENV) pip install -e .[dev]
	$(CONDA_VENV) python -m ipykernel install --user --name $(KERNEL_NAME)
	touch "$@"

# alias for virtual environment creation
venv: venv/.canary

# remove the conda virtual environment
clean_venv:
	conda env remove -p $(CONDA_VENV_PATH)

# use a wrapper script to load required modules before starting the kernel on NeSI
venv_nesi: venv/.canary nesi/template_wrapper.bash nesi/template_kernel.json
	cp nesi/template_wrapper.bash $(KERNEL_DIR)/wrapper.bash
	sed -i 's|##CONDA_VENV_PATH##|$(CONDA_VENV_PATH)|' $(KERNEL_DIR)/wrapper.bash
	cp nesi/template_kernel.json $(KERNEL_DIR)/kernel.json
	sed -i 's|##KERNEL_DIR##|$(KERNEL_DIR)|; s|##KERNEL_NAME##|$(KERNEL_NAME)|' $(KERNEL_DIR)/kernel.json

# reformat all python files using black
format:
	$(CONDA_VENV) black src/*.py

.PHONY: help venv clean_venv venv_nesi notebooks format
