# GCP Terraform runner

This configuration creates exactly one Compute Engine VM for the study.

- Spot `g2-standard-4` (4 vCPUs and 16 GB system memory)
- one NVIDIA L4 GPU with 24 GB VRAM
- a 50 GB balanced persistent disk
- an Ubuntu 22.04 Deep Learning VM image with CUDA support
- NVIDIA driver installation through VM metadata

See [PROVISIONING_LOG.md](PROVISIONING_LOG.md) for the latest creation attempts
and quota findings for this project.

The project needs billing and quota for at least one Spot L4 GPU. Terraform can enable the Compute Engine API,
but it cannot approve quota requests or guarantee that a zone has free GPUs.

Spot capacity is cheaper but can be preempted. The VM stops rather than deleting
itself after preemption, so completed files remain on its persistent boot disk.

## Create the VM

Authenticate Terraform on your computer:

```powershell
gcloud auth application-default login
```

Copy the example variables file and replace the project ID:

```powershell
cd "D:\abilation study\terraform"
Copy-Item terraform.tfvars.example terraform.tfvars
notepad terraform.tfvars
```

Review and create the VM:

```powershell
terraform init
terraform plan
terraform apply
```

Terraform prints commands named `upload_command`, `ssh_command`, and
`download_command`. View one again with, for example:

```powershell
terraform output -raw ssh_command
```

## Upload and run the study

Run the printed upload command from the project directory, not from the
`terraform` directory:

```powershell
cd "D:\abilation study"
gcloud compute scp --recurse . ablation-study-l4-spot:~/ablation-study --zone=europe-west3-a --project=YOUR_PROJECT_ID
```

Connect using the printed SSH command. On the VM, verify and install:

```bash
nvidia-smi -L
cd ~/ablation-study
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

The last command should print `True 1`. Use `tmux` so an SSH disconnect does
not stop the study:

```bash
sudo apt-get update && sudo apt-get install -y tmux
tmux new -s ablation
bash run.sh 2>&1 | tee training.log
```

Alternatively, the supplied detached launcher writes the process ID and suite
log under `results/`:

```bash
bash start_gcp.sh
tail -f results/suite.log
```

Detach with `Ctrl+B`, then `D`. Reconnect later with `tmux attach -t ablation`.
The training runs and final comparison are also stored in `results/mlflow.db`
and `results/mlartifacts` for
MLflow. After downloading `results`, view them locally from the project folder:

```powershell
mlflow ui --backend-store-uri sqlite:///gcp-results/mlflow.db --host 127.0.0.1 --port 5000
```

## Download and destroy

After training, run the printed download command on your computer. Confirm that
`gcp-results/report/` and `gcp-results/runs/` were copied before destroying the
VM:

```powershell
cd "D:\abilation study\terraform"
terraform destroy
```

Destroying the VM also deletes its boot disk, so downloaded results are the
recoverable copy. This project has no automatic cost deadline or shutdown.

References: [Terraform for Compute Engine](https://docs.cloud.google.com/compute/docs/terraform),
[GPU VM requirements](https://docs.cloud.google.com/compute/docs/gpus/create-vm-with-gpus),
and [GPU locations](https://docs.cloud.google.com/compute/docs/regions-zones/gpu-regions-zones).
