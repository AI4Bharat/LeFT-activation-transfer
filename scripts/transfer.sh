#!/usr/bin/env bash
# Step 2-3 -- FREEZE and TRANSFER, shown on their own: take a source
# checkpoint, write the coefficient artifact, and verify it round-trips into a
# target network that shares no weight with the source.
#
#   scripts/transfer.sh <source-checkpoint> [out-artifact]
#
# This is the whole of the transfer step: five numbers out of one network and
# into another. The verification at the end is the point -- it asserts that the
# target network's activation is exactly the source's curve, that it is frozen,
# and that no source weight came along.
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="${1:?usage: scripts/transfer.sh <source-checkpoint> [out-artifact]}"
ART="${2:-artifacts/fact/transferred.json}"

python -m left.transfer.extract_cli --checkpoint "$CKPT" --out "$ART" --print

python - "$CKPT" "$ART" <<'PY'
import sys
from left.activations import FAct
from left.models.vit import build_vit
from left.transfer import assert_frozen, verify_curve_matches, verify_weight_independence
from left.utils.seed import set_seed

ckpt, art = sys.argv[1], sys.argv[2]
act = FAct.from_artifact(art)                 # frozen by default
set_seed(0)
model = build_vit("cifar10", act)   # act is an instance: shared by reference
assert_frozen(model)
verify_curve_matches(act, art)
verify_weight_independence(model, ckpt)
print(f"OK  transferred curve installed, frozen, and weight-independent of {ckpt}")
print(f"    {act}")
PY
