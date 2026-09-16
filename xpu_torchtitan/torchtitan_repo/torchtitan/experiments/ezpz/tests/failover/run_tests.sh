#!/bin/bash
# Test the failover_lib.sh exit-code detection logic against synthetic
# log fixtures. Each fixture mirrors a real-world failure mode we've
# observed in production (with byte-identical ANSI escapes).
#
# We mirror the rc-determination block from failover_lib.sh into
# `evaluate_rc()` below — keep them in sync by hand. When you change
# failover_lib.sh's rc logic, update evaluate_rc + add a new fixture.
#
# Run from repo root:
#   bash torchtitan/experiments/ezpz/tests/failover/run_tests.sh

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
FAILOVER_LIB="$REPO_ROOT/torchtitan/experiments/ezpz/scripts/failover_lib.sh"
FIXTURES_DIR="$(dirname "$0")/fixtures"

if [[ ! -f "$FAILOVER_LIB" ]]; then
    echo "ERROR: failover_lib.sh not found at $FAILOVER_LIB"
    exit 1
fi

# Mirror of failover_lib.sh's inner_rc + crash_lines block.
# Keep in sync with the source. When the source changes, change this too.
evaluate_rc() {
    local logf="$1"
    local rc="$2"
    local inner_rc
    inner_rc=$(sed -r 's/\x1b\[[0-9;]*m//g' "$logf" 2>/dev/null | grep -oE "Execution finished with [0-9]+" | tail -1 | grep -oE "[0-9]+$")
    if [[ -n "$inner_rc" && "$inner_rc" != "0" ]]; then
        if (( rc == 0 )); then
            rc=$inner_rc
        fi
    elif (( rc == 0 )); then
        local crash_lines
        crash_lines=$(grep -cE "RuntimeError: \[.*gloo.*\] Connection closed by peer|RuntimeError: \[.*gloo.*\] Timed out waiting|OutOfMemoryError|UR_RESULT_ERROR_OUT_OF_RESOURCES|died from signal|EOFError: No data left in file" "$logf" 2>/dev/null)
        if (( crash_lines >= 1 )); then
            rc=1
        fi
    fi
    echo "$rc"
}

# Mirror of failover_lib.sh's walltime guard
should_retry_after_143() {
    local logf="$1"
    local bad_crash_lines
    bad_crash_lines=$(grep -cE "RuntimeError: \[.*gloo.*\] Connection closed by peer|RuntimeError: \[.*gloo.*\] Timed out waiting|OutOfMemoryError|UR_RESULT_ERROR_OUT_OF_RESOURCES|died from signal|EOFError: No data left in file" "$logf" 2>/dev/null)
    if (( bad_crash_lines == 0 )); then
        echo "no_retry"
    else
        echo "retry"
    fi
}

# fixture | shell_rc | expected_final_rc | expected_decision
declare -a TESTS=(
    "clean_success.log|0|0|succeeded"
    "gloo_cascade_mass.log|0|143|retry"
    "gloo_cascade_few.log|0|143|retry"
    "silent_hang.log|124|124|watchdog"
    "walltime_clean.log|143|143|no_retry"
    "walltime_with_crash.log|143|143|retry"
    "mpiexec_help_dump.log|0|1|retry"
    "set_determinism_oom.log|0|143|retry"
    "blendcorpus_eoferror.log|0|143|retry"
)

PASS=0
FAIL=0
FAILED_TESTS=()

for tc in "${TESTS[@]}"; do
    IFS='|' read -r fixture shell_rc expected_rc expected_decision <<< "$tc"
    fpath="$FIXTURES_DIR/$fixture"
    if [[ ! -f "$fpath" ]]; then
        echo "SKIP: $fixture — fixture missing"
        continue
    fi

    actual_rc=$(evaluate_rc "$fpath" "$shell_rc")

    case "$actual_rc" in
        0)   actual_decision="succeeded" ;;
        124) actual_decision="watchdog" ;;
        143) actual_decision=$(should_retry_after_143 "$fpath") ;;
        *)   actual_decision="retry" ;;
    esac

    if [[ "$actual_rc" == "$expected_rc" && "$actual_decision" == "$expected_decision" ]]; then
        echo "PASS: $fixture (shell_rc=$shell_rc → rc=$actual_rc, $actual_decision)"
        PASS=$((PASS + 1))
    else
        echo "FAIL: $fixture"
        echo "  expected: rc=$expected_rc, decision=$expected_decision"
        echo "  got:      rc=$actual_rc, decision=$actual_decision"
        FAIL=$((FAIL + 1))
        FAILED_TESTS+=("$fixture")
    fi
done

echo
echo "=== summary ==="
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
if (( FAIL > 0 )); then
    echo "  failed: ${FAILED_TESTS[*]}"
    exit 1
fi
echo "  all tests passed"
