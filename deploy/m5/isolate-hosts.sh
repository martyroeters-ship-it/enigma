#!/bin/sh
# Operator steps for M5 host isolation. This script does not create users,
# does not copy credentials, and does not enable trading.
# Exit 0 only when the three unix users exist and, if ENIGMA_VALIDATION_DIR is
# set, the research user cannot read it. Otherwise it fails closed.
set -eu

missing=0
for user in enigma-research enigma-validation enigma-execution; do
  if ! id "$user" >/dev/null 2>&1; then
    echo "missing unix user: $user"
    missing=1
  fi
done

if [ "$missing" -ne 0 ]; then
  echo "Create three accounts and separate homes. Example:"
  echo "  sudo dscl . -create /Users/enigma-validation   # macOS"
  echo "  sudo useradd --system --create-home enigma-validation   # Linux"
  echo "Then: chown enigma-validation \"\$ENIGMA_VALIDATION_DIR\" && chmod 0700 \"\$ENIGMA_VALIDATION_DIR\""
  echo "Run the validation process as that user with ENIGMA_HOST_ROLE=validation."
  echo "Run research with ENIGMA_HOST_ROLE=research and without ENIGMA_VALIDATION_CREDENTIAL."
  echo "Run execution with ENIGMA_HOST_ROLE=execution. Do not set ENIGMA_SANDBOX_ALLOW_WEAK=1."
  exit 2
fi

if [ -n "${ENIGMA_VALIDATION_DIR:-}" ]; then
  if ! sudo -u enigma-research test -r "$ENIGMA_VALIDATION_DIR"; then
    echo "research user cannot read $ENIGMA_VALIDATION_DIR"
  else
    echo "research user can read the validation directory; refusing"
    exit 2
  fi
fi

echo "unix users exist; confirm each process has its own ENIGMA_HOST_ROLE"
exit 0
