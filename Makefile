
.PHONY: git-hooks
git-hooks:
	@echo "Setting up git hooks..."
	git config core.hooksPath .githooks
	@echo "Git hooks set up successfully."
