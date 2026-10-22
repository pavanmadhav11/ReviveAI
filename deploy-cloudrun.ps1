$ErrorActionPreference = 'Stop'

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw 'Install the Google Cloud CLI and sign in with `gcloud auth login`, then rerun this script.'
}

# Cloud Run request-based billing scales to zero and has a monthly free tier.
# A billing account is still required; usage beyond free allowances can incur charges.
gcloud run deploy reviveai `
    --source . `
    --region asia-south1 `
    --memory 2Gi `
    --cpu 1 `
    --concurrency 1 `
    --max 1 `
    --min 0 `
    --port 5000 `
    --timeout 300 `
    --allow-unauthenticated `
    --set-env-vars 'HOST=0.0.0.0,REVIVEAI_DISABLE_AI_SCRATCH=0,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1'
