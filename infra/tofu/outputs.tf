output "tpu_name" {
  value       = var.enable_tpu ? google_tpu_v2_vm.this[0].name : ""
  description = "TPU VM name"
}

output "zone" {
  value       = var.zone
  description = "Zone"
}

output "project_id" {
  value       = var.project_id
  description = "GCP project id"
}

output "tpu_id" {
  value       = var.enable_tpu ? google_tpu_v2_vm.this[0].id : ""
  description = "Full TPU resource id"
}

output "tpu_network_endpoints" {
  value       = var.enable_tpu ? google_tpu_v2_vm.this[0].network_endpoints : []
  description = "TPU worker endpoints"
}

# ---------- Cloud SQL ----------

output "tracking_db_internal_ip" {
  value       = var.enable_tracking_db ? google_sql_database_instance.tracking[0].public_ip_address : ""
  description = "Cloud SQL IP (public but VPC-accessible)"
}

output "tracking_db_uri" {
  value = var.enable_tracking_db ? (
    "postgresql://${var.tracking_db_user}:${var.tracking_db_password}@${google_sql_database_instance.tracking[0].public_ip_address}/${var.tracking_db_name}"
  ) : ""
  description = "Postgres connection URI for experiment tracking"
  sensitive   = true
}

# ---------- Tracking server ----------

output "tracking_server_name" {
  value       = var.enable_tracking_server ? google_compute_instance.tracking[0].name : ""
  description = "GCE instance name of the tracking server"
}

output "tracking_server_internal_ip" {
  value       = var.enable_tracking_server ? google_compute_instance.tracking[0].network_interface[0].network_ip : ""
  description = "Internal IP of the tracking server (VPC-accessible)"
}

output "tracking_server_external_ip" {
  value       = var.enable_tracking_server ? google_compute_address.tracking[0].address : ""
  description = "Static external IP of the tracking server"
}

output "tracking_server_internal_uri" {
  value       = var.enable_tracking_server ? "http://${google_compute_instance.tracking[0].network_interface[0].network_ip}:${var.tracking_server_port}" : ""
  description = "Tracking server URI for same-VPC clients (TPU VMs)"
}

output "tracking_server_external_uri" {
  value       = var.enable_tracking_server ? "http://${google_compute_address.tracking[0].address}:${var.tracking_server_port}" : ""
  description = "Tracking server URI for external clients (IAP tunnel endpoint)"
}

# ---------- GCS artifact bucket ----------

output "artifact_bucket_name" {
  value       = var.enable_artifact_bucket ? google_storage_bucket.artifacts[0].name : ""
  description = "GCS bucket name for experiment artifacts"
}

output "artifact_bucket_uri" {
  value       = var.enable_artifact_bucket ? "gs://${google_storage_bucket.artifacts[0].name}" : ""
  description = "GCS bucket URI for experiment artifacts (gs://...)"
}
