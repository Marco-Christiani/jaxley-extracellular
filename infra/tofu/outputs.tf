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

# ---------- GCS artifact bucket ----------

output "artifact_bucket_name" {
  value       = var.enable_artifact_bucket ? google_storage_bucket.artifacts[0].name : ""
  description = "GCS bucket name for experiment artifacts"
}

output "artifact_bucket_uri" {
  value       = var.enable_artifact_bucket ? "gs://${google_storage_bucket.artifacts[0].name}" : ""
  description = "GCS bucket URI for experiment artifacts (gs://...)"
}
