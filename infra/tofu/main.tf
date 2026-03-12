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
