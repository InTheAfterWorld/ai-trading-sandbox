"""
Main Entry Point for AI TRADING SANDBOX.

Usage:

    python run.py             # Start the Web Dashboard (default)
    python run.py --cli       # Run terminal interactive mode
    python run.py --port 8080 # Custom port for Web Dashboard
    python run.py --debug     # Enable Flask debug mode (auto-reload)
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="AI TRADING SANDBOX — Multi-Agent Stock Market Sandbox",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--cli", "-c",
        action="store_true",
        help="Run in terminal CLI mode instead of starting the Web Dashboard",
    )

    parser.add_argument(
        "--port", "-p",
        type=int,
        default=5000,
        help="Port number for the Web Dashboard (default: 5000)",
    )

    parser.add_argument(
        "--provider",
        help="CLI: override the API provider from the saved configuration",
    )

    parser.add_argument(
        "--model",
        help="CLI: override the model from the saved configuration",
    )

    parser.add_argument(
        "--api-key",
        dest="api_key",
        help="CLI: override the API key from the saved configuration",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run the Web Dashboard with Flask debug mode (auto-reload, verbose errors)",
    )

    parser.add_argument(
        "--steps",
        type=int,
        help="CLI: number of simulation steps (default: saved config)",
    )

    args = parser.parse_args()

    if args.cli:
        from ai_trading_society.__main__ import main as cli_main

        cli_main(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            steps=args.steps,
        )

    else:
        from ai_trading_society.web import app

        port = int(os.environ.get("PORT", args.port))

        print("\n" + "=" * 55)
        print("  AI TRADING SANDBOX -- Web Dashboard")
        print(f"  Running on port {port}")
        print("=" * 55 + "\n")

        app.run(
            host="0.0.0.0",
            debug=args.debug,
            port=port,
        )


if __name__ == "__main__":
    main()