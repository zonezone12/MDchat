"""
Entry point for MDChat.

Works in three ways:
    python -m src.mdchat          # from repo root
    mdchat                        # after pip install -e '.[chat]'
    F5 in VS Code / Cursor        # via .vscode/launch.json
"""

import argparse
import os
import sys

# Ensure the repo root is on sys.path so `from src.xxx import ...` works
# both when run as `python -m src.mdchat` and as an installed console script.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


def _load_dotenv() -> None:
    """Load .env from the repo root if python-dotenv is available."""
    try:
        from dotenv import load_dotenv

        env_path = os.path.join(_repo_root, ".env")
        if os.path.isfile(env_path):
            load_dotenv(env_path, override=False)
    except ImportError:
        pass


def main() -> None:
    _load_dotenv()

    parser = argparse.ArgumentParser(
        prog="mdchat",
        description="MDChat — LLM-powered Molecular Dynamics trajectory analysis",
    )
    parser.add_argument(
        "--provider",
        default=os.environ.get("MDCHAT_PROVIDER", "anthropic"),
        choices=("anthropic", "gemini"),
        help="LLM backend: anthropic (Claude) or gemini (set MDCHAT_PROVIDER in .env)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the selected provider (overrides env vars)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MDCHAT_MODEL"),
        help="Model id (Claude or Gemini; or set MDCHAT_MODEL in .env)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("MDCHAT_OUTPUT_DIR"),
        help="Directory for output artifacts (or set MDCHAT_OUTPUT_DIR in .env)",
    )
    args = parser.parse_args()

    provider = (args.provider or "anthropic").lower()
    if args.api_key:
        api_key = args.api_key
    elif provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")

    from src.mdchat.cli import run_cli

    run_cli(
        provider=provider,
        api_key=api_key,
        model=args.model,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
