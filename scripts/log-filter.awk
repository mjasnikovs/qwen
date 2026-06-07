# Collapses llama-server's per-request "slot/srv" chatter into a single
# carriage-return-updating status line. Startup logs and errors pass through.
#
# llama.cpp prefixes most lines with "<ts> <LEVEL> " (e.g. "0.42.936.607 I ").
# We strip that prefix into `rest` before matching component/function names.
# The timing SUMMARY block (prompt eval time / eval time / total time) is
# still printed raw without a prefix, so those rules match $0 directly.

BEGIN {
    n_prompt = 0
    pp_tokens = "0"; pp_rate = "0.0"
    g_tokens  = "0"; g_rate  = "0.0"

    # ANSI styling
    R   = "\033[0m"
    BLD = "\033[1m"
    DIM = "\033[2m"
    CYA = "\033[36m"
    GRN = "\033[32m"
    YEL = "\033[33m"
    RED = "\033[31m"
    GRY = "\033[90m"
    CLR = "\r\033[2K"   # CR + erase entire line
}

# Strip the "<ts> <LEVEL> " prefix into `rest` (fallback: the whole line).
{
    if (match($0, /^[0-9][0-9.]*[ \t]+[IWEDV][ \t]+/))
        rest = substr($0, RLENGTH + 1)
    else
        rest = $0
}

# Cancellation
rest ~ /cancel task/ {
    printf "%s%scanceled%s\n", CLR, RED, R
    fflush()
    next
}

# New task — prime the status line (children/draft tasks excluded)
rest ~ /launch_slot_.*processing task, is_child = 0/ {
    n_prompt = 0
    pp_tokens = "0"; pp_rate = "0.0"
    g_tokens  = "0"; g_rate  = "0.0"
    printf "%s%sprompt%s %sloading...%s", CLR, CYA, R, GRY, R
    fflush()
    next
}

# Prompt processing progress — update in place. Total prompt size is derived
# from n_tokens/progress (no separate "new prompt" line in this format).
rest ~ /print_timing:.*prompt processing, n_tokens = / {
    n_past = 0; prog = 0
    if (match(rest, /n_tokens =[ ]*[0-9]+/)) {
        s = substr(rest, RSTART, RLENGTH); gsub(/[^0-9]/, "", s); n_past = s + 0
    }
    if (match(rest, /progress = [0-9.]+/)) {
        s = substr(rest, RSTART, RLENGTH); gsub(/[^0-9.]/, "", s); prog = s + 0
    }
    if (prog > 0) n_prompt = int(n_past / prog + 0.5)
    pct = int(prog * 100 + 0.5)
    if (pct >= 100) {
        printf "%s%sprompt%s %s%d%s tok done %s|%s %sgenerating...%s", \
            CLR, CYA, R, BLD, n_past, R, GRY, R, YEL, R
    } else {
        printf "%s%sprompt%s %s%d%s/%d tok %s(%d%%)%s", \
            CLR, CYA, R, BLD, n_past, R, n_prompt, GRY, pct, R
    }
    fflush()
    next
}

# Generation progress — update in place
rest ~ /print_timing:.*n_decoded = / {
    nd = 0; tg = "0.0"
    if (match(rest, /n_decoded =[ ]*[0-9]+/)) {
        s = substr(rest, RSTART, RLENGTH); gsub(/[^0-9]/, "", s); nd = s + 0
    }
    if (match(rest, /tg =[ ]*[0-9.]+/)) {
        s = substr(rest, RSTART, RLENGTH); gsub(/[^0-9.]/, "", s); tg = s
    }
    printf "%s%sgen%s %s%d%s tok %s@%s %s%s%s t/s", \
        CLR, YEL, R, BLD, nd, R, GRY, R, BLD, tg, R
    fflush()
    next
}

# Prompt eval timing (raw, no prefix) — buffer values for the summary
$0 ~ /prompt eval time =/ {
    if (match($0, /\/[ ]+[0-9]+[ ]+tokens/)) {
        s = substr($0, RSTART, RLENGTH); gsub(/[^0-9]/, "", s); pp_tokens = s
    }
    if (match($0, /[0-9.]+[ ]+tokens per second/)) {
        s = substr($0, RSTART, RLENGTH); gsub(/[^0-9.]/, "", s); pp_rate = s
    }
    next
}

# Generation eval timing (raw, no prefix) — emit final summary
$0 ~ /^[ \t]*eval time =/ {
    if (match($0, /\/[ ]+[0-9]+[ ]+tokens/)) {
        s = substr($0, RSTART, RLENGTH); gsub(/[^0-9]/, "", s); g_tokens = s
    }
    if (match($0, /[0-9.]+[ ]+tokens per second/)) {
        s = substr($0, RSTART, RLENGTH); gsub(/[^0-9.]/, "", s); g_rate = s
    }
    printf "%s%s✓ done%s  %spp%s %s%s%s tok %s@%s %s%s%s t/s  %sgen%s %s%s%s tok %s@%s %s%s%s t/s\n", \
        CLR, GRN, R, \
        CYA, R, BLD, pp_tokens, R, GRY, R, BLD, pp_rate, R, \
        CYA, R, BLD, g_tokens, R, GRY, R, BLD, g_rate, R
    fflush()
    next
}

# Suppress remaining per-request slot/srv chatter and speculative-decode stats,
# but keep startup variants (`... load_model:`, `srv ... init:`).
rest ~ /^slot /       && rest !~ /load_model:/                      { next }
rest ~ /^srv /        && rest !~ /load_model:/ && rest !~ /init:/   { next }
rest ~ /^statistics / { next }
rest ~ /draft acceptance rate/                                     { next }
$0 ~ /^[ \t]*total time =/                                         { next }

# Pass through everything else (model load, llama_*, common_*, errors)
{ print; fflush() }
