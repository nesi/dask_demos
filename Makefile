KERNEL := hpc_for_datascience_demos

NOTEBOOKS := notebooks/hyperparameters_search_basic.ipynb
HTML_FILES := $(NOTEBOOKS:.ipynb=.html)

all: $(NOTEBOOKS) $(HTML_FILES)

notebooks/%.html: notebooks/%.ipynb
	. venv/bin/activate; jupyter nbconvert --to html "$<"

notebooks/%.ipynb: src/%.py requirements-pinned.txt
	mkdir -p notebooks
	. venv/bin/activate; jupytext --to notebook --execute --set-kernel $(KERNEL) "$<"
	mv "src/$(@F)" "$@"

requirements-pinned.txt: venv/bin/activate
	venv/bin/pip freeze > "$@"

venv/bin/activate: requirements.txt requirements-dev.txt
	python3 -m venv venv
	venv/bin/pip install -r requirements.txt
	venv/bin/bin/pip install -r requirements-dev.txt
	python3 -m ipykernel install --user --name $(KERNEL)
	touch "$@"

venv: venv/bin/activate

.PHONY: venv
