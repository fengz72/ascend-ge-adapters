#!/usr/bin/env python3
"""Convenience entry point for verify_outputs.py perf (the same strict checks)."""
import sys
import verify_outputs

if __name__ == "__main__":
    sys.argv.insert(1, "perf")
    sys.exit(verify_outputs.main())
