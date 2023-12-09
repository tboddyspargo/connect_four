
PIP_LOCKFILES = requirements.txt all-requirements.txt
# PIP_COMPILE_EXTRA_INPUTS = ../other-package/requirements.in

# DOT_IN_BASE_DEP_PREREQS is the default '.in' file to use. All targe lockfiles are expected to depend on it.
DOT_IN_BASE_DEP_PREREQS ?= requirements.in
DOT_IN_OTHER_DEP_PREREQS =  %-requirements.in $(DOT_IN_BASE_DEP_PREREQS) $(PIP_COMPILE_EXTRA_INPUTS)
# PIP_BASE_LOCKFILE is the default lockfile to compile and will be used as a constraints file for others.
PIP_BASE_LOCKFILE ?= requirements.txt

PIP_LOCKFILES ?= $(PIP_BASE_LOCKFILE)

PYPROJECT = $(wildcard pyproject.toml)
PYPROJECT_BASE_DEP_PREREQS = pyproject.toml $(wildcard setup.*)
PYPROJECT_OTHER_DEP_PREREQS = $(PYPROJECT_BASE_DEP_PREREQS) $(PIP_COMPILE_EXTRA_INPUTS)

# PIP_COMPILE_EXTRA_INPUTS is a list of additional (non-standard) input files to list when compiling ANY lockfile.
# e.g. An input file from a separate package that's accessible via relative path: ../other-package/pyproject.toml
PIP_COMPILE_EXTRA_INPUTS ?=

PIP_COMPILE_ARGS += -q --strip-extras


$(PIP_BASE_LOCKFILE): $(if $(PYPROJECT),$(PYPROJECT_BASE_DEP_PREREQS),$(DOT_IN_BASE_DEP_PREREQS)) $(PIP_COMPILE_EXTRA_INPUTS)
	pip-compile $^ -o $@ $(PIP_COMPILE_ARGS)

%-requirements.txt: $(if $(PYPROJECT),$(PYPROJECT_OTHER_DEP_PREREQS),$(DOT_IN_OTHER_DEP_PREREQS)) $(PIP_BASE_LOCKFILE)
	pip-compile $(wordlist 1, $(shell echo $$(( $(words $^) - 1 ))), $^) -o $@ $(if $(findstring pyproject,$<),$(if $(findstring all,$*),--all-extras,--extra $*)) --constraint $(lastword $^) $(PIP_COMPILE_ARGS)


.PHONY: update
update: | $(PIP_LOCKFILES)
