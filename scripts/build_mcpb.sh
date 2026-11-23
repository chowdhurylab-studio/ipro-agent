#!/usr/bin/env bash
# Thin wrapper; the builder is Python so it needs no staging directory.
exec python3 "$(dirname "$0")/build_mcpb.py" "$@"
