#!/bin/sh
# Wrap llama-server so its log output is filtered to a single updating line per request.
exec /app/llama-server "$@" 2>&1 | awk -W interactive -f /scripts/log-filter.awk
