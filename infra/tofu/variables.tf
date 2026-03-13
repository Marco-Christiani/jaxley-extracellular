variable "project_id" {
  description = "GCP project ID."
  type        = string
}

variable "zone" {
  description = "TPU zone, e.g. us-central2-b."
  type        = string
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
  description = "Python package to install for the tracking server (e.g. mlflow, aim)."
  type        = string
  default     = "mlflow>=2.12"
}

variable "tracking_server_command" {
  description = "Binary name installed by tracking_server_package (e.g. mlflow, aim)."
  type        = string
  default     = "mlflow"
}

variable "tracking_server_allowed_cidrs" {
  description = "External CIDRs allowed to reach the tracking server (in addition to VPC internal)."
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
