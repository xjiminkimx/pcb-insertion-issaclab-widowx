#!/usr/bin/env bash
# One-time setup: allow this user to power off after train_insert.sh without a password.
# You will be prompted for your sudo password ONCE when running this script.
set -euo pipefail

USER_NAME="$(id -un)"
SUDOERS_FILE="/etc/sudoers.d/widowx-train-shutdown"

RULE="${USER_NAME} ALL=(ALL) NOPASSWD: /usr/bin/systemctl poweroff, /usr/bin/systemctl halt, /sbin/shutdown, /usr/sbin/shutdown"

echo "This will add the following rule to ${SUDOERS_FILE}:"
echo "  ${RULE}"
echo ""
read -r -p "Continue? [y/N] " confirm
if [[ "${confirm}" != "y" && "${confirm}" != "Y" ]]; then
  echo "Cancelled."
  exit 0
fi

TMP="$(mktemp)"
printf '%s\n' "${RULE}" > "${TMP}"
sudo install -o root -g root -m 0440 "${TMP}" "${SUDOERS_FILE}"
rm -f "${TMP}"

sudo visudo -cf "${SUDOERS_FILE}"
echo "[OK] Passwordless shutdown enabled for train_insert.sh"
echo "     Test (does NOT power off): sudo -n systemctl poweroff --help || true"
