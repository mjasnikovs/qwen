# Collapses llama-server's per-request "slot/srv" chatter into a single
# carriage-return-updating status line. Startup logs and errors pass through.

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

# Cancellation
/cancel task/ {
    printf "%s%scanceled%s\n", CLR, RED, R
    fflush()
    next
}

# New prompt — capture total prompt size, prime the status line
/new prompt.*task\.n_tokens/ {
    if (match($0, /task\.n_tokens = [0-9]+/)) {
        s = substr($0, RSTART, RLENGTH); gsub(/[^0-9]/, "", s)
        n_prompt = s + 0
    }
    pp_tokens = "0"; pp_rate = "0.0"
    g_tokens  = "0"; g_rate  = "0.0"
    printf "%s%sprompt%s %s0%s/%d tok %s(0%%)%s", CLR, CYA, R, BLD, R, n_prompt, GRY, R
    fflush()
    next
}

# Prompt processing progress — update in place
/prompt processing progress.*progress = / {
    n_past = 0; prog = 0
    if (match($0, /n_tokens = [0-9]+/)) {
        s = substr($0, RSTART, RLENGTH); gsub(/[^0-9]/, "", s)
        n_past = s + 0
    }
    if (match($0, /progress = [0-9.]+/)) {
        s = substr($0, RSTART, RLENGTH); gsub(/[^0-9.]/, "", s)
        prog = s + 0
    }
    pct = int(prog * 100 + 0.5)
    if (pct >= 100) {
        printf "%s%sprompt%s %s%d%s tok done %s|%s %sgenerating...%s", \
            CLR, CYA, R, BLD, n_prompt, R, GRY, R, YEL, R
    } else {
        printf "%s%sprompt%s %s%d%s/%d tok %s(%d%%)%s", \
            CLR, CYA, R, BLD, n_past, R, n_prompt, GRY, pct, R
    }
    fflush()
    next
}

# Prompt eval timing — buffer values for the summary
/prompt eval time =/ {
    if (match($0, /\/[ ]+[0-9]+[ ]+tokens/)) {
        s = substr($0, RSTART, RLENGTH); gsub(/[^0-9]/, "", s); pp_tokens = s
    }
    if (match($0, /[0-9.]+[ ]+tokens per second/)) {
        s = substr($0, RSTART, RLENGTH); gsub(/[^0-9.]/, "", s); pp_rate = s
    }
    next
}

# Generation eval timing — emit final summary
/^[ \t]*eval time =/ {
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

# Suppress per-request slot/srv chatter, but keep the two startup variants
# (`slot ... load_model:` and `srv ... load_model:`/`srv ... init:`).
/^slot / && !/^slot +load_model:/                       { next }
/^srv /  && !/^srv +load_model:/ && !/^srv +init:/      { next }
/^[ \t]*total time =/                                   { next }

# Pass through everything else (model load, llama_*, common_*, errors)
{ print; fflush() }
