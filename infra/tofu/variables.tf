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
