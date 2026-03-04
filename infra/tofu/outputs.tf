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
