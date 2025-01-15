# sets up a local Python development environment
setup:
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "Installing uv via curl..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	uv sync

check:
	uv run ruff check

format:
	uv run ruff check --fix
	uv run ruff format

test:
	uv run pytest eval_test.py

run:
	uv run streamlit run eval.py --server.headless True
