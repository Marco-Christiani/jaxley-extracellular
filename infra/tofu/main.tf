provider "google" {
  project = var.project_id
  zone    = var.zone
}

provider "google-beta" {
  project = var.project_id
  zone    = var.zone
}

locals {
  startup_script = <<-EOT
    #!/usr/bin/env bash
    set -euo pipefail

    # Best-effort TPU perf tuning.
    if [ -w /sys/kernel/mm/transparent_hugepage/enabled ]; then
      echo always > /sys/kernel/mm/transparent_hugepage/enabled || true
    fi

    if ! command -v uv >/dev/null 2>&1; then
      curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
    fi
  EOT
}

# ---------- Cloud SQL for experiment tracking (opt-in) ----------

resource "google_sql_database_instance" "tracking" {
  count = var.enable_tracking_db ? 1 : 0

  project          = var.project_id
  name             = "${var.name}-tracking"
  region           = replace(var.zone, "/-[a-z]$/", "")
  database_version = "POSTGRES_16"

  settings {
    tier              = var.tracking_db_tier
    availability_type = "ZONAL"
    edition           = "ENTERPRISE"

    ip_configuration {
      ipv4_enabled = true
      authorized_networks {
        name  = "all"
        value = "0.0.0.0/0"
      }
    }

    backup_configuration {
      enabled = false
    }
  }

  deletion_protection = false
}

resource "google_sql_database" "tracking" {
  count    = var.enable_tracking_db ? 1 : 0
  project  = var.project_id
  instance = google_sql_database_instance.tracking[0].name
  name     = var.tracking_db_name
}

resource "google_sql_user" "tracking" {
  count    = var.enable_tracking_db ? 1 : 0
  project  = var.project_id
  instance = google_sql_database_instance.tracking[0].name
  name     = var.tracking_db_user
  password = var.tracking_db_password
}

# ---------- Tracking server service account ----------

resource "google_service_account" "tracking" {
  count        = var.enable_tracking_server ? 1 : 0
  project      = var.project_id
  account_id   = "${var.name}-tracking"
  display_name = "Tracking server (${var.name})"
}

# ---------- Secret Manager (DB password for tracking server) ----------

resource "google_secret_manager_secret" "tracking_db_password" {
  count     = var.enable_tracking_server ? 1 : 0
  project   = var.project_id
  secret_id = "${var.name}-tracking-db-password"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "tracking_db_password" {
  count       = var.enable_tracking_server ? 1 : 0
  secret      = google_secret_manager_secret.tracking_db_password[0].id
  secret_data = var.tracking_db_password
}

resource "google_secret_manager_secret_iam_member" "tracking_server" {
  count     = var.enable_tracking_server ? 1 : 0
  project   = var.project_id
  secret_id = google_secret_manager_secret.tracking_db_password[0].secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.tracking[0].email}"
}

# ---------- GCS artifact bucket (opt-in) ----------

resource "google_storage_bucket" "artifacts" {
  count    = var.enable_artifact_bucket ? 1 : 0
  project  = var.project_id
  name     = "${var.project_id}-${var.name}-artifacts"
  location = replace(var.zone, "/-[a-z]$/", "")

  uniform_bucket_level_access = true
  force_destroy               = true

  lifecycle_rule {
    condition { age = var.artifact_retention_days }
    action { type = "Delete" }
  }
}

resource "google_storage_bucket_iam_member" "tracking_server" {
  count  = var.enable_tracking_server && var.enable_artifact_bucket ? 1 : 0
  bucket = google_storage_bucket.artifacts[0].name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.tracking[0].email}"
}

# ---------- Tracking server (opt-in) ----------

resource "google_compute_address" "tracking" {
  count   = var.enable_tracking_server ? 1 : 0
  project = var.project_id
  region  = replace(var.zone, "/-[a-z]$/", "")
  name    = "${var.name}-tracking"
}

resource "google_compute_firewall" "tracking" {
  count   = var.enable_tracking_server ? 1 : 0
  project = var.project_id
  network = var.network != null ? var.network : "default"
  name    = "${var.name}-tracking"

  allow {
    protocol = "tcp"
    ports    = [tostring(var.tracking_server_port)]
  }

  # Internal VPC access (other GCE/TPU instances)
  source_ranges = concat(
    ["10.0.0.0/8"],
    var.tracking_server_allowed_cidrs,
  )

  target_tags = ["tracking-server"]
}

resource "google_compute_instance" "tracking" {
  count        = var.enable_tracking_server ? 1 : 0
  project      = var.project_id
  zone         = var.zone
  name         = "${var.name}-tracking"
  machine_type = var.tracking_server_machine_type

  tags = ["tracking-server"]

  # Tracking server requires Cloud SQL and GCS bucket.
  lifecycle {
    precondition {
      condition     = var.enable_tracking_db
      error_message = "enable_tracking_db must be true when enable_tracking_server is true."
    }
    precondition {
      condition     = var.enable_artifact_bucket
      error_message = "enable_artifact_bucket must be true when enable_tracking_server is true."
    }
  }

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-12"
      size  = 10
    }
  }

  network_interface {
    network    = var.network != null ? var.network : "default"
    subnetwork = var.subnetwork

    access_config {
      nat_ip = google_compute_address.tracking[0].address
    }
  }

  metadata_startup_script = <<-EOT
#!/usr/bin/env bash
set -euo pipefail

UV_BIN=/usr/local/bin
TOOL_BIN=/root/.local/bin

# Install uv if not present
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$UV_BIN sh
fi

# Install tracking server package
$UV_BIN/uv tool install '${var.tracking_server_package}'

# Write a launcher that fetches the DB password from Secret Manager at runtime.
# The password never appears in instance metadata or systemd unit files.
# Tofu resolves interpolations at plan time.
# Bare $VAR passes through to the shell script unchanged.
cat > /usr/local/bin/tracking-server-start <<'LAUNCHER'
#!/usr/bin/env bash
set -euo pipefail
DB_PASSWORD=$(gcloud secrets versions access latest \
  --secret="${var.name}-tracking-db-password" \
  --project="${var.project_id}")
exec /root/.local/bin/${var.tracking_server_command} server \
  --backend-store-uri "postgresql://${var.tracking_db_user}:$DB_PASSWORD@${google_sql_database_instance.tracking[0].public_ip_address}/${var.tracking_db_name}" \
  --default-artifact-root "gs://${google_storage_bucket.artifacts[0].name}/mlflow" \
  --host 0.0.0.0 --port ${var.tracking_server_port}
LAUNCHER
chmod +x /usr/local/bin/tracking-server-start

# Systemd unit -- no secrets in the unit file
cat > /etc/systemd/system/tracking-server.service <<'UNIT'
[Unit]
Description=Experiment Tracking Server
After=network.target

[Service]
Type=simple
Environment=PATH=/root/.local/bin:/usr/local/bin:/usr/sbin:/usr/bin
ExecStart=/usr/local/bin/tracking-server-start
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now tracking-server
EOT

  labels = var.labels

  service_account {
    email  = google_service_account.tracking[0].email
    scopes = ["cloud-platform"]
  }

  depends_on = [
    google_sql_database.tracking,
    google_storage_bucket.artifacts,
    google_secret_manager_secret_version.tracking_db_password,
  ]
}

# ---------- TPU ----------

resource "google_tpu_v2_vm" "this" {
  provider = google-beta

  project          = var.project_id
  zone             = var.zone
  name             = var.name
  description      = var.description
  runtime_version  = var.runtime_version
  accelerator_type = var.accelerator_type

  scheduling_config {
    spot        = var.spot
    preemptible = var.preemptible
  }

  dynamic "network_config" {
    for_each = var.network == null && var.subnetwork == null ? [] : [1]
    content {
      network             = var.network
      subnetwork          = var.subnetwork
      enable_external_ips = var.enable_external_ips
    }
  }

  labels = var.labels
  metadata = merge(
    {
      startup-script = local.startup_script
    },
    var.metadata
  )
}
