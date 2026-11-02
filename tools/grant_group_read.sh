#!/usr/bin/env bash
# Grant read + traverse on the project store to one or more Unix groups, including
# default ACLs so files written later inherit the grant. Read-only for the groups:
# nothing is moved, copied or deleted, and existing ACL entries are kept.
#
# Runs on the xfer partition (data-management node, no GPU). Must be submitted by the
# owner of the files (sadamov until the handover; afterwards whoever owns them).
#
#   sbatch --account=${AIENS_ACCOUNT:-ab016} tools/grant_group_read.sh            # s83 + ab016
#   GROUPS_RW="s83" sbatch --account=... tools/grant_group_read.sh <dir> [<dir>...]
#
# A group also needs traverse (x) on every parent directory. For the default store
# that means /capstor/store/cscs/mch/s83 (root:s83, mode 770): members of s83 can
# reach the store, members of other groups only if CSCS adds them there.
#
#SBATCH --job-name=aiens_acl
#SBATCH --partition=xfer
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=aiens_acl_%j.log

set -euo pipefail

GROUPS_RW="${GROUPS_RW:-s83 ab016}"
TARGETS=("${@:-${AIENS_STORE:-/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles}}")

spec=""
for g in $GROUPS_RW; do spec+="g:${g}:rX,"; done
spec="${spec%,}"

for t in "${TARGETS[@]}"; do
    echo "$(date -Is) access ACL ${spec} on ${t}"
    # -R + capital X: directories get r-x, files get r-- (x only where already executable).
    # Errors on files owned by someone else are reported but do not stop the run.
    setfacl -R -m "$spec" "$t" || echo "setfacl reported errors (see above)"
    echo "$(date -Is) default ACL on directories under ${t}"
    find "$t" -type d -exec setfacl -m "$(sed 's/g:/d:g:/g' <<< "$spec")" {} + || true
    getfacl -p "$t" | grep -E "^(default:)?group:" || true
done
echo "$(date -Is) done"
