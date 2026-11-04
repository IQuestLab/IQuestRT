#!/usr/bin/env bash
# Usage: bash tools/claude_code.sh [http://HOST:8000] [claude arguments...]
set -euo pipefail
URL=http://localhost:8000
if [[ ${1:-} == http://* || ${1:-} == https://* ]]; then
    URL=${1%/}
    shift
fi
if [[ $URL == */v1 ]]; then
    echo 'Pass the server root URL, without /v1.' >&2
    exit 2
fi
command -v claude >/dev/null || { echo 'claude is not installed or is not on PATH.' >&2; exit 1; }
# Feed headers through stdin so credentials never appear in curl's process arguments.
header=
if [[ -n ${ANTHROPIC_AUTH_TOKEN:-} ]]; then
    header="Authorization: Bearer $ANTHROPIC_AUTH_TOKEN"
elif [[ -n ${ANTHROPIC_API_KEY:-} ]]; then
    header="x-api-key: $ANTHROPIC_API_KEY"
else
    export ANTHROPIC_AUTH_TOKEN=dummy
fi
if [[ $header == *$'\n'* || $header == *$'\r'* ]]; then
    echo 'Authentication values must not contain line breaks.' >&2
    exit 2
fi
header=${header//\\/\\\\}
header=${header//\"/\\\"}
probe() {
    { if [[ -n $header ]]; then printf 'header = "%s"\n' "$header"; fi; } |
        curl --config - -fsS --max-time 5 "$URL/$1"
}
probe health >/dev/null
MODEL=$(probe v1/models |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
# Optional: a gateway may expose only /v1.
CONTEXT=$(probe metrics 2>/dev/null |
    awk '$1 == "iquestrt:max_model_len" { print $2; exit }') || true
export ANTHROPIC_BASE_URL="$URL"
unset CLAUDE_CODE_OAUTH_TOKEN
unset CLAUDE_CODE_USE_BEDROCK CLAUDE_CODE_USE_VERTEX CLAUDE_CODE_USE_FOUNDRY
export ANTHROPIC_MODEL="$MODEL" ANTHROPIC_SMALL_FAST_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL" ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL"
if [[ -n $CONTEXT ]]; then
    export CLAUDE_CODE_MAX_CONTEXT_TOKENS="$CONTEXT"
fi
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export ENABLE_TOOL_SEARCH=false
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-8192}
# --disallowedTools takes several values, so it must not sit right before "$@".
exec claude --disallowedTools WebSearch --model "$MODEL" "$@"
