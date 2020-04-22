CURRENT_PATH := $(shell pwd)
MODULE := $(shell basename "$(CURRENT_PATH)")
VERSION := $(shell python -c "import sys; import $(MODULE); sys.stdout.write($(MODULE).__version__)")
SOURCES := $(shell find $(MODULE) -name '*.py')
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXSOURCEDIR = ./docs
SPHINXBUILDDIR  = docs/_build

test: FORCE
	py.test --doctest-modules --ignore=make --ignore=docs
	# nosetests --with-doctest --logging-level=ERROR

coverage:
	# coverage run $(MODULE) test
	py.test --doctest-modules --ignore=make --ignore=docs --cov=$(MODULE) || true
	# nosetests --with-doctest --with-coverage --cover-erase --cover-package=$(MODULE)
	# coverage report
	coverage html
	python -m webbrowser htmlcov/index.html

clean:
	coverage erase
	rm -rf htmlcov/
	rm -rf docs/_build/*
	rm -rf docs/source

build:
	@while [ -z "$$CONTINUE" ]; do \
			read -r -p "Have you changed the __version__ attribute in tf_neiss/__about__.py??? If yes, type Y or y to continue. Type anything else to exit. [y/N]: " CONTINUE; \
	done ; \
	[ $$CONTINUE = "y" ] || [ $$CONTINUE = "Y" ] || (echo "Exiting."; exit 1;)
	@echo "... publishing ..."
	@echo "New Version: $(VERSION)"

	git tag -a v$(VERSION) -m "Version $(VERSION) tag"  # tag version
	git push origin v$(VERSION)  # explicitly push tag to the shared server
	rm -f dist/*  # could become unnecessary, if you find a way to specify 'twine upload dist/transcoding-$(VERSION).tag.gz' in last step
	python setup.py sdist  # building package

publish_test: build
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*  # publish to test.pypi.org

publish: build
	twine upload dist/*  # publish package

untag:
	# remove last tag. mostly, because publishing failed
	git tag -d v$(VERSION)
	git push origin :refs/tags/v$(VERSION)


docs: Makefile $(SOURCES)
	# link apidoc to source
	sphinx-apidoc -o docs/source/ $(MODULE)
	# build html documentation with sphinx
	# @$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(0)
	sphinx-build -M html docs docs/_build
	# open the html slides
	python -m webbrowser docs/_build/html/index.html

FORCE: ;
