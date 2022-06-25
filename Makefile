# Makefile to automate recurring tasks in package development
#
# Usage: A "Make tool" such as GNU Make can run so called "targets" defined in this file.
# It has to be invoked on the command line in the directory that the Makefile is located.
# - General: make <target>
# - Example: make help
#
# References
# - GNU Make
#   - https://www.gnu.org/software/make
# - Style checking
#   - isort: https://github.com/timothycrosley/isort
#   - Flake8: http://flake8.pycqa.org (uses pycodestyle, pyflakes and mccabe)
#   - Pylint: https://www.pylint.org
# - URL checking
#   - urlchecker: https://github.com/urlstechie/urlchecker-python
# - Testing
#   - pytest: https://docs.pytest.org
#   - pytest-cov: https://github.com/pytest-dev/pytest-cov
#   - pytest-timeout: https://bitbucket.org/pytest-dev/pytest-timeout
#   - tox: https://tox.readthedocs.io


PKG_NAME := alogos
PKG_ABBREVIATION := al

PKG_DIR := $(PKG_NAME)
EXAMPLES_DIR := examples
TEST_DIR := tests


.DEFAULT: help
.PHONY: help
help:
	@echo "--------------------------------------------------------------------------------------------"
	@echo
	@echo "How to use this Makefile with a \"Make tool\" like GNU Make"
	@echo
	@echo "  Help"
	@echo "    make help          show this help message"
	@echo
	@echo "  Installation"
	@echo "    make install       install the package with pip from this folder"
	@echo "    make install-edit  install the package with pip in editable mode for development"
	@echo "    make install-dev   install development tools and the package in editable mode"
	@echo "    make uninstall     uninstall the package"
	@echo
	@echo "  Usage"
	@echo "    make run           launch a Python shell and import the installed package"
	@echo "    make run-ipython   launch an IPython shell and import the installed package"
	@echo
	@echo "  Development"
	@echo '    make clean         delete everything that can be generated
	@echo "    make style-isort   stylecheck with isort (check import order)"
	@echo "    make style-flake8  stylecheck with flake8 (check basic code conventions)"
	@echo "    make style-pylint  stylecheck with pylint (detailed code analysis)"
	@echo "    make test-unit     run unit and integration tests with pytest"
	@echo "    make test-system   run system tests in virtual environments with tox"
	@echo "    make todo          check for TODO comments in source code with pylint"
	@echo "    make urls          check if all URLs in code lead to a server response"
	@echo
	@echo "  Deployment"
	@echo "    make package       Create two distribution packages for upload to TestPyPI or PyPI"
	@echo "                         1. A source distribution (sdist)"
	@echo "                         2. A built distribution in form of a wheel (bdist_wheel)"
	@echo "    make upload-test   Upload distribution packages (source & built) with twine to TestPyPI"
	@echo "    make upload-pypi   Upload distribution packages (source & built) with twine to PyPI"
	@echo
	@echo "--------------------------------------------------------------------------------------------"


# Installation

.PHONY: install
install:
	pip install .

.PHONY: install-edit
install-edit:
	pip install -e .

.PHONY: install-dev
install-dev:
	# Linting
	pip install isort flake8 pylint
	# Testing
	pip install pytest pytest-cov pytest-timeout tox
	# Upload to PyPI
	pip install twine
	# Package in editable form
	pip install -e .

.PHONY: uninstall
uninstall:
	pip uninstall -y $(PKG_DIR)


# Usage

.PHONY: run
run:
	python -i -c "import $(PKG_NAME) as $(PKG_ABBREVIATION)"

.PHONY: run-ipython
run-ipython:
	ipython -i -c "import $(PKG_NAME) as $(PKG_ABBREVIATION)"


# Development

.PHONY: clean
clean:
	# Directories
	-@rm -rf tests/out* tests/htmlcov tests/.coverage*
	-@rm -rf .tox
	-@rm -rf build dist
	-@find . -type d \
	    \( \
	    -name "__pycache__" \
	    -or -name "*.egg-info" \
	    -or -name ".cache" \
	    -or -name ".pytest_cache" \
	    -or -name "dask-worker-space" \
	    -or -name ".ipynb_checkpoints" \
	    \) \
	    -exec rm -rf {} \; 2> /dev/null || true
	# Files
	-@find . \( -name '*.pyc' -or -name '*.pyd' -or -name '*.pyo' -or -name '*~' \) \
	    -exec rm --force {} + 2> /dev/null || true
	@echo "Deleted all unnecessary files and directories."

.PHONY: style-isort
style-isort:
	isort --lines-after-imports 2 --recursive --diff $(PKG_DIR) $(TEST_DIR)

.PHONY: style-flake8
style-flake8:
	flake8 $(PKG_DIR) $(TEST_DIR)

.PHONY: style-pylint
style-pylint:
	pylint $(PKG_DIR) $(TEST_DIR) --disable=W0511

.PHONY: test-unit
test-unit:
	@cd tests; \
	rm -rf dask-worker-space output htmlcov .coverage*; \
	pytest --timeout=240 --cov=$(PKG_NAME) --cov-report html

.PHONY: test-system
test-system:
	tox --recreate --result-json tox_results.json
	@echo
	@echo "Results of system test with tox can be found in tox_results.json"

.PHONY: todo
todo:
	pylint $(PKG_DIR) $(TEST_DIR) --disable=all --enable=W0511

.PHONY: urls
urls:
	-urlchecker check $(PKG_DIR) --file-types .ipynb,.md,.py,.rst,.txt --white-listed-urls http://127.0.0.1
	-urlchecker check $(TEST_DIR) --file-types .ipynb,.md,.py,.rst,.txt --white-listed-urls http://127.0.0.1


# Deployment

.PHONY: package
package:
	-@rm -rf build dist
	python3 setup.py sdist bdist_wheel
	@echo
	@echo 'Two distribution packages (source and built) can now be found in directory "dist".'
	-@rm -rf build

.PHONY: upload-testpypi
upload-testpypi:
	@echo 'Uploading the distribution packages (source and built) from directory "dist" with twine to TestPyPI.'
	@echo
	@echo "Suggested manual checks afterwards:"
	@echo "- Is the upload available at https://test.pypi.org/project/$(PKG_NAME)"
	@echo "- Can it be installed with: pip install --index-url https://test.pypi.org/simple --extra-index-url https://pypi.python.org/pypi $(PKG_NAME)"
	@echo
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

.PHONY: upload-pypi
upload-pypi:
	@echo 'Uploading the distribution packages (source and built) from directory "dist" with twine to PyPI.'
	@echo
	@echo "Suggested manual checks afterwards:"
	@echo "- Is the upload available at https://pypi.org/project/$(PKG_NAME)"
	@echo "- Can it be installed with: pip install $(PKG_NAME)"
	@echo
	twine upload dist/*
