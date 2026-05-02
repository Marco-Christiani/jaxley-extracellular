# Infrastructure architecture

Text summary of what `main.tf` provisions and how the pieces talk to each
other. Source of truth for the `architecture.d2` diagram in this directory.

## Components

| Resource                                  | Tofu var              | Purpose                                             |
|-------------------------------------------|-----------------------|-----------------------------------------------------|
| `google_tpu_v2_vm.this`                   | `enable_tpu`          | Compute node. Bootstraps `uv` on startup.           |
| `google_sql_database_instance.tracking`   | `enable_tracking_db`  | Postgres 16 backend store for tracking metadata.    |
| `google_storage_bucket.artifacts`         | `enable_artifact_bucket` | Artifact storage (retention-bounded).            |
| `google_compute_instance.tracking`        | `enable_tracking_server` | GCE VM running the tracking server (MLflow).    |
| `google_service_account.tracking`         | (implicit)            | SA bound to the tracking server VM.                 |
| `google_secret_manager_secret.tracking_db_password` | (implicit)  | DB password; never in metadata or unit files.       |
| `google_compute_address.tracking`         | (implicit)            | Static external IP for the tracking server.         |
| `google_compute_firewall.tracking`        | (implicit)            | Allows VPC + IAP to reach the tracking server port. |
| `google_iap_tunnel_instance_iam_member`   | `iap_users`           | Grants IAP-tunnel access to named users.            |

## Dependencies

- Tracking server has two `lifecycle.precondition` blocks: `enable_tracking_db == true` and `enable_artifact_bucket == true`. Provisioning fails fast if either is off.
- Tracking server SA has two IAM bindings:
  - `roles/secretmanager.secretAccessor` on the DB-password secret.
  - `roles/storage.objectAdmin` on the artifact bucket.
- The server boots via `metadata_startup_script` which writes a launcher (`/usr/local/bin/tracking-server-start`) that fetches the DB password from Secret Manager, constructs the DB URI in memory, and execs the server process under a systemd unit.

## Network layout

- Default VPC unless overridden by `var.network` / `var.subnetwork`.
- Firewall rule opens the tracking server port (default 5000) only to:
  - `10.0.0.0/8` (internal VPC, used by TPU workers)
  - `35.235.240.0/20` (Google IAP tunnel range)
- TPU workers reach the server via VPC-internal connectivity.
- Developers reach the server via an IAP tunnel (`gcloud compute start-iap-tunnel` forwards `localhost:5000` to the server port).

## Runtime flow

1. User grants IAP access, provisions the stack with `tofu apply`.
2. User starts the tracking server and TPU VM with `task remote:tracking:up` and `task remote:tpu:up`.
3. User opens an IAP tunnel (`task remote:tracking:tunnel:bg`) to reach the server UI at `localhost:5000`.
4. TPU VM runs experiments. Code logs to the tracking server via the internal VPC URI (`tracking_server_internal_uri` output).
5. Tracking server writes metadata to Cloud SQL and streams artifacts to GCS via its `--serve-artifacts` mode, so workers only need the tracking URI (no direct GCS credentials on the TPU).

## Variable summary (opt-in switches)

- `enable_tpu` (default `true`): TPU VM.
- `enable_tracking_db` (default `true`): Cloud SQL.
- `enable_artifact_bucket` (default `true`): GCS bucket.
- `enable_tracking_server` (default `true`): GCE tracking server.

Setting any of these to `false` elides the corresponding resource block.
