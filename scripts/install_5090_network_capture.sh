#!/usr/bin/env bash
# Install observation only. No experiment, SSH, routing, firewall, or Tailscale settings change.
set -euo pipefail
mode="${1:-user}"
source_path="${2:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/paper/scripts/collect_5090_network_incident.py}"
target_user="${3:-leekwanhyeong}"
[[ -f "$source_path" ]] || { echo "Collector source missing: $source_path" >&2; exit 2; }
[[ "$mode" == user || "$mode" == root ]] || { echo "Usage: $0 user|root [collector.py] [user]" >&2; exit 2; }
command -v systemctl >/dev/null
command -v python3 >/dev/null
if [[ "$mode" == user ]]; then
    [[ "$(id -u)" != 0 ]] || { echo "Run user mode as the intended non-root account." >&2; exit 2; }
    code_dir="$HOME/.local/lib/paper-5090-network"
    log_dir="$HOME/.local/state/paper-5090-network"
    unit_dir="$HOME/.config/systemd/user"
    install -d -m 0750 "$code_dir" "$log_dir" "$unit_dir"
    install -m 0750 "$source_path" "$code_dir/collector.py"
    cat > "$unit_dir/paper-5090-network.service" <<EOF
[Unit]
Description=5090 read-only network incident capture (user scope)

[Service]
Type=simple
ExecStart=/usr/bin/python3 $code_dir/collector.py --output $log_dir --owner-uid $(id -u) --interval 10 --deadline 60
Restart=on-failure
RestartSec=15
RestartPreventExitStatus=73
Nice=15
UMask=0027
NoNewPrivileges=yes
StandardOutput=null
StandardError=journal

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload
    systemctl --user enable --now paper-5090-network.service
    systemctl --user is-active --quiet paper-5090-network.service
    printf 'Installed user collector. Logs: %s\n' "$log_dir"
    echo 'User service starts with the user manager; unattended boot persistence requires existing linger or the root option.'
    echo 'System/kernel journal access is not granted by this installer.'
else
    [[ "$(id -u)" == 0 ]] || { echo "Root mode requires sudo; no privilege grant is installed." >&2; exit 2; }
    target_uid="$(id -u "$target_user")"
    target_group="$(id -gn "$target_user")"
    # Acquire the exact global collector lock before root installation. If user mode
    # is active, stop that unit explicitly first; never silently replace its process.
    /usr/bin/python3 - "$target_uid" <<'PY'
import fcntl, os, stat, sys
uid = int(sys.argv[1])
path = f"/tmp/paper-5090-network-{uid}.lock"
try:
    fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
except FileNotFoundError:
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
s = os.fstat(fd)
if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1 or s.st_uid not in (0, uid):
    raise SystemExit("Unexpected collector lock owner/type")
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit("Collector already active. Stop its user unit before root installation; existing capture is preserved.")
os.close(fd)
PY
    code_dir="/usr/local/lib/paper-5090-network"
    log_dir="/var/log/paper-5090-network"
    install -d -o root -g root -m 0755 "$code_dir"
    install -o root -g root -m 0755 "$source_path" "$code_dir/collector.py"
    install -d -o root -g "$target_group" -m 0750 "$log_dir"
    cat > /etc/systemd/system/paper-5090-network.service <<EOF
[Unit]
Description=5090 read-only network and privileged journal incident capture
After=local-fs.target

[Service]
Type=simple
User=root
Group=$target_group
ExecStart=/usr/bin/python3 $code_dir/collector.py --output $log_dir --owner-uid $target_uid --interval 10 --deadline 60
Restart=on-failure
RestartSec=15
RestartPreventExitStatus=73
Nice=15
UMask=0027
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=read-only
PrivateDevices=yes
ReadWritePaths=$log_dir /tmp
StandardOutput=null
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    chmod 0644 /etc/systemd/system/paper-5090-network.service
    systemctl daemon-reload
    systemctl enable --now paper-5090-network.service
    systemctl is-active --quiet paper-5090-network.service
    printf 'Installed boot-persistent root collector. Readable by group %s: %s\n' "$target_group" "$log_dir"
fi
