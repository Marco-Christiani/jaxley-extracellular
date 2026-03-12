output "name" {
  value       = google_tpu_v2_vm.this.name
  description = "TPU VM name"
}

output "zone" {
  value       = google_tpu_v2_vm.this.zone
  description = "TPU VM zone"
}

output "project_id" {
  value       = var.project_id
  description = "GCP project id"
}

output "id" {
  value       = google_tpu_v2_vm.this.id
  description = "Full TPU resource id"
}

output "network_endpoints" {
  value       = google_tpu_v2_vm.this.network_endpoints
  description = "TPU worker endpoints"
}

# ---------- Cloud SQL ----------

output "tracking_db_ip" {
  value       = var.enable_tracking_db ? google_sql_database_instance.tracking[0].public_ip_address : ""
  description = "Cloud SQL public IP"
}

output "tracking_db_uri" {
  value = var.enable_tracking_db ? (
    "postgresql://${var.tracking_db_user}:${var.tracking_db_password}@${google_sql_database_instance.tracking[0].public_ip_address}/${var.tracking_db_name}"
  ) : ""
  description = "Postgres connection URI for experiment tracking"
  sensitive   = true
}

# ---------- Tracking server ----------

output "tracking_server_internal_ip" {
  value       = var.enable_tracking_server ? google_compute_instance.tracking[0].network_interface[0].network_ip : ""
  description = "Internal IP of the tracking server (for same-VPC access)"
}

output "tracking_server_external_ip" {
  value       = var.enable_tracking_server ? google_compute_address.tracking[0].address : ""
  description = "Static external IP of the tracking server"
}

output "tracking_server_url" {
  value       = var.enable_tracking_server ? "http://${google_compute_instance.tracking[0].network_interface[0].network_ip}:${var.tracking_server_port}" : ""
  description = "Tracking server URL for same-VPC experiments (TPU VMs)"
}

output "tracking_server_external_url" {
  value       = var.enable_tracking_server ? "http://${google_compute_address.tracking[0].address}:${var.tracking_server_port}" : ""
  description = "Tracking server URL for external experiments (local machine)"
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
