#!/bin/bash

# Generates code.
#
# Instructions:
#   1. Clone https://github.com/apple/swift to any directory.
#   2. Let `SWIFT_BASE` be the directory where you cloned it.
#   3. Run `./generate $SWIFT_BASE`.

SWIFT_BASE="$1"
if [ -z "$SWIFT_BASE" ]
then
  echo "SWIFT_BASE not specified. See instructions at the top of generate.sh."
  exit 1
fi

GYB="$SWIFT_BASE"/utils/gyb

GENERATED_FILES=(
  "Sources/SwiftFusion/Core/Vector.swift"
  "Tests/SwiftFusionTests/Core/VectorTests.swift"
)

for GENERATED_FILE in "${GENERATED_FILES[@]}"
do
  cat > "$GENERATED_FILE" << EOF
// WARNING: This is a generated file. Do not edit it. Instead, edit the corresponding ".gyb" file.
// See "generate.sh" in the root of this repository for instructions how to regenerate files.

EOF

  "$GYB" "$GENERATED_FILE.gyb" >> "$GENERATED_FILE"
done
