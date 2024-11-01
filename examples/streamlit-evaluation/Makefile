# sets up a local Python development environment
setup:
	@command -v uv >/dev/null 2>&1 || { echo >&2 "uv is not installed. Installing via Homebrew..."; brew install uv; }
	uv sync

check:
	ruff check

format:
	ruff check --fix
	ruff format

test:
	pytest eval_test.py

run:
	uv run streamlit run eval.py --server.headless True
