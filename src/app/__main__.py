"""Allow `python -m app` as an alias of `python -m app.main`."""

from app.main import main

if __name__ == "__main__":
    raise SystemExit(main())
