variable "project_id" {
  description = "Google Cloud project ID with billing and GPU quota enabled."
  type        = string

  validation {
    condition     = length(trimspace(var.project_id)) > 0
    error_message = "project_id cannot be empty."
  }
}

variable "zone" {
  description = "GCP zone with capacity and quota for a Spot NVIDIA L4 GPU."
  type        = string
  default     = "europe-west3-a"
}
