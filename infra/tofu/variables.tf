variable "project_id" {
  description = "GCP project ID."
  type        = string
}

variable "enable_tpu" {
  description = "Whether to create the TPU VM. Set false to provision only the tracking stack."
  type        = bool
  default     = true
}

variable "zone" {
  description = "Default zone for regional resources (SQL region, bucket location, tracking server). Also the TPU zone unless tpu_zone is set."
  type        = string
}

variable "tpu_zone" {
  description = "Override zone for the TPU VM only. Lets the TPU live in a different zone (e.g. for capacity) than the SQL/bucket/tracking server, which stay in zone's region. Null means use var.zone."
  type        = string
  default     = null
}

variable "name" {
  description = "TPU VM name."
  type        = string
  default     = "jx-tpu-dev"
}

variable "runtime_version" {
  description = "TPU runtime version (see gcloud compute tpus tpu-vm versions list)."
  type        = string
}

variable "accelerator_type" {
  description = "TPU accelerator type (see gcloud compute tpus accelerator-types list)."
  type        = string
}

variable "description" {
  description = "Optional TPU description."
  type        = string
  default     = "Managed by OpenTofu"
}

variable "spot" {
  description = "Whether TPU VM is spot."
  type        = bool
  default     = false
}

variable "preemptible" {
  description = "Whether TPU VM is preemptible."
  type        = bool
  default     = false
}

variable "labels" {
  description = "Labels for TPU VM."
  type        = map(string)
  default     = {}
}

variable "metadata" {
  description = "Metadata map for TPU VM startup/shutdown scripts etc."
  type        = map(string)
  default     = {}
}

variable "network" {
  description = "Optional VPC network self-link or name. Null uses default behavior."
  type        = string
  default     = null
}

variable "subnetwork" {
  description = "Optional subnetwork self-link or name. Null uses default behavior."
  type        = string
  default     = null
}

variable "enable_external_ips" {
  description = "Whether TPU workers get external IPs when using network_config."
  type        = bool
  default     = true
}

# ---------- Cloud SQL (experiment tracking) ----------

variable "enable_tracking_db" {
  description = "Whether to create a Cloud SQL Postgres instance for experiment tracking."
  type        = bool
  default     = true
}

variable "tracking_db_tier" {
  description = "Cloud SQL machine tier."
  type        = string
  default     = "db-f1-micro"
}

variable "tracking_db_name" {
  description = "Database name."
  type        = string
  default     = "tracking"
}

variable "tracking_db_user" {
  description = "Database user name."
  type        = string
  default     = "tracker"
}

variable "tracking_db_password" {
  description = "Password for the tracking database user."
  type        = string
  sensitive   = true
  default     = ""
}

# ---------- Tracking server ----------

variable "enable_tracking_server" {
  description = "Whether to create a GCE instance for the tracking server."
  type        = bool
  default     = true
}

variable "tracking_server_machine_type" {
  description = "GCE machine type for the tracking server."
  type        = string
  default     = "e2-micro"
}

variable "tracking_server_port" {
  description = "Port the tracking server listens on."
  type        = number
  default     = 5000
}

variable "tracking_server_package" {
  description = "Python package to install via uv tool install, e.g. 'mlflow[db]>=2.12'."
  type        = string
  default     = "mlflow[extras,db]>=2.12"
}

variable "tracking_server_constraints" {
  description = <<-EOT
    Extra requirements pinned alongside tracking_server_package via
    `uv tool install ... --with '<constraints>'`. Default pins
    google-auth<2.21 to avoid the universe-domain HTTPS probe of
    metadata.google.internal that fails on this stack: the metadata
    endpoint's TLS cert is signed by a Google-internal CA which isn't
    in any standard trust bundle (system or certifi), so token refresh
    SSL-errors and mlflow returns 500 to artifact PUTs. Pinning to a
    pre-2.21 google-auth restores the HTTP-only metadata path.
  EOT
  type        = string
  default     = "google-auth<2.21"
}

variable "tracking_server_command" {
  description = "Binary name to invoke, e.g. 'mlflow'."
  type        = string
  default     = "mlflow"
}

variable "tracking_server_args" {
  description = <<-EOT
    Arguments passed to tracking_server_command. The launcher exports three
    environment variables before running the command, which you can reference here:
      $DB_URI        full Postgres connection URI (password injected at runtime)
      $ARTIFACT_ROOT gs://bucket/mlflow
      $PORT          value of tracking_server_port
  EOT
  type        = string
  default     = "server --backend-store-uri $DB_URI --artifacts-destination $ARTIFACT_ROOT --serve-artifacts --host 0.0.0.0 --port $PORT --workers 1"
}

variable "iap_users" {
  description = "Google accounts allowed to IAP-tunnel to the tracking server, e.g. [\"user:you@gmail.com\"]."
  type        = list(string)
  default     = []
}

# ---------- GCS artifact bucket ----------

variable "enable_artifact_bucket" {
  description = "Whether to create a GCS bucket for experiment artifacts."
  type        = bool
  default     = true
}

variable "artifact_retention_days" {
  description = "Days before artifacts are auto-deleted (lifecycle policy)."
  type        = number
  default     = 90
}
