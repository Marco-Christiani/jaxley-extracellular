#!/usr/bin/env bash
set -euo pipefail

# Bootstrap Prometheus + Grafana on a Debian-based Ray head VM so Ray's Metrics
# tab works reliably through SSH tunnels.

if [ "${EUID}" -ne 0 ]; then
  # sudo's secure_path drops relative paths, so resolve to absolute first.
  # Invoke through bash so we don't depend on the script being chmod +x.
  exec sudo --preserve-env=PATH bash "$(readlink -f "${BASH_SOURCE[0]}")" "$@"
fi

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y curl gpg ca-certificates jq prometheus

install -d -m 0755 /etc/apt/keyrings
if [ ! -f /etc/apt/keyrings/grafana.gpg ]; then
  curl -fsSL https://apt.grafana.com/gpg.key | gpg --dearmor -o /etc/apt/keyrings/grafana.gpg
fi

cat > /etc/apt/sources.list.d/grafana.list <<'EOF'
deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main
EOF

apt-get update
apt-get install -y grafana

cat > /etc/prometheus/prometheus.yml <<'EOF'
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: ray
    static_configs:
      - targets: ['localhost:8080']
EOF

install -d -m 0755 /var/lib/grafana/ray-dashboards
install -d -m 0755 /etc/grafana/provisioning/datasources
install -d -m 0755 /etc/grafana/provisioning/dashboards
install -d -m 0755 /etc/systemd/system/grafana-server.service.d

cat > /etc/grafana/provisioning/datasources/ray-prometheus.yaml <<'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://localhost:9090
    isDefault: true
    editable: false
EOF

cat > /etc/grafana/provisioning/dashboards/ray-dashboards.yaml <<'EOF'
apiVersion: 1

providers:
  - name: Ray
    orgId: 1
    folder: Ray
    type: file
    disableDeletion: false
    editable: false
    updateIntervalSeconds: 30
    options:
      path: /var/lib/grafana/ray-dashboards
EOF

cat > /etc/systemd/system/grafana-server.service.d/ray-observability.conf <<'EOF'
[Service]
Environment=GF_SECURITY_ALLOW_EMBEDDING=true
Environment=GF_SECURITY_COOKIE_SECURE=false
Environment=GF_SECURITY_COOKIE_SAMESITE=none
Environment=GF_AUTH_ANONYMOUS_ENABLED=true
Environment=GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer
EOF

cat > /usr/local/bin/ray-refresh-grafana-dashboards <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

src="/tmp/ray/session_latest/metrics/grafana/dashboards"
dst="/var/lib/grafana/ray-dashboards"

if [ ! -d "$src" ]; then
  echo "Ray dashboard source not found at $src" >&2
  exit 1
fi

find "$dst" -maxdepth 1 -type f -name '*.json' -delete
cp -f "$src"/*.json "$dst"/
EOF
chmod +x /usr/local/bin/ray-refresh-grafana-dashboards

systemctl daemon-reload
systemctl enable --now prometheus
systemctl enable --now grafana-server

if [ -d /tmp/ray/session_latest/metrics/grafana/dashboards ]; then
  /usr/local/bin/ray-refresh-grafana-dashboards || true
  systemctl restart grafana-server
fi

echo "Prometheus: http://localhost:9090"
echo "Grafana:    http://localhost:3000"
echo "Next: restart Ray with infra/scripts/ray-head-start.sh if it was already running before bootstrap."
