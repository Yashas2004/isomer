from .cli import main

# The guard matters on Windows: worker processes re-import this module.
if __name__ == "__main__":
    raise SystemExit(main())
