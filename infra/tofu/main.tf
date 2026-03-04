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
