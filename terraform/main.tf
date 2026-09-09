terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 6.0, < 8.0"
    }
  }
}

provider "google" {
  project = var.project_id
  zone    = var.zone
}

resource "google_project_service" "compute" {
  project            = var.project_id
  service            = "compute.googleapis.com"
  disable_on_destroy = false
}

data "google_compute_image" "deep_learning" {
  family  = "common-cu129-ubuntu-2204-nvidia-580"
  project = "deeplearning-platform-release"

  depends_on = [google_project_service.compute]
}

resource "google_compute_instance" "training" {
  name                      = "ablation-study-l4-spot"
  machine_type              = "g2-standard-4"
  zone                      = var.zone
  allow_stopping_for_update = true

  boot_disk {
    initialize_params {
      image = data.google_compute_image.deep_learning.self_link
      size  = 50
      type  = "pd-balanced"
    }
  }

  scheduling {
    automatic_restart           = false
    instance_termination_action = "STOP"
    on_host_maintenance         = "TERMINATE"
    preemptible                 = true
    provisioning_model          = "SPOT"
  }

  network_interface {
    network = "default"

    access_config {}
  }

  metadata = {
    install-nvidia-driver = "True"
  }

  labels = {
    workload = "distilbert-ablation-spot"
  }

  lifecycle {
    create_before_destroy = true
  }

  depends_on = [google_project_service.compute]
}

output "external_ip" {
  description = "Public IP address of the training VM."
  value       = google_compute_instance.training.network_interface[0].access_config[0].nat_ip
}

output "ssh_command" {
  description = "Connect to the training VM."
  value       = "gcloud compute ssh ${google_compute_instance.training.name} --zone=${var.zone} --project=${var.project_id}"
}

output "upload_command" {
  description = "Run from the project directory to upload its files."
  value       = "gcloud compute scp --recurse . ${google_compute_instance.training.name}:~/ablation-study --zone=${var.zone} --project=${var.project_id}"
}

output "download_command" {
  description = "Download completed results to the current local directory."
  value       = "gcloud compute scp --recurse ${google_compute_instance.training.name}:~/ablation-study/results ./gcp-results --zone=${var.zone} --project=${var.project_id}"
}
