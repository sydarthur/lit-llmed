# GitHub Actions Setup for Cloud Run Deployment

## Required GitHub Secrets

Add these secrets to your GitHub repository under Settings > Secrets and variables > Actions:

### 1. GCP_PROJECT_ID
Your Google Cloud Project ID
```
your-gcp-project-id
```

### 2. GCP_SA_KEY
Service Account JSON key with the following permissions:
- Cloud Run Admin
- Storage Admin  
- Artifact Registry Administrator
- Artifact Registry Create on Push Admin

## Creating the Service Account

1. Go to Google Cloud Console > IAM & Admin > Service Accounts
2. Create a new service account: `github-actions-deploy`
3. Grant these roles:
   - Cloud Run Admin
   - Storage Admin
   - Artifact Registry Administrator
   - Artifact Registry Create on Push Admin
4. Create and download JSON key
5. Copy the entire JSON content to GitHub secret `GCP_SA_KEY`

## Enable Required APIs

Run these commands in Google Cloud Shell:

```bash
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

## Deployment Triggers

The workflow automatically deploys when:
- Code is pushed to `main` or `dev-gcp-connect` branches
- Changes are made to `src/gcp/**` files
- Manual trigger via GitHub Actions tab

## Repository Settings

1. Go to Settings > Actions > General
2. Set "Workflow permissions" to "Read and write permissions"
3. Allow actions to create pull requests: ✓

## Testing the Setup

1. Add the secrets to your repository
2. Push changes to the `dev-gcp-connect` branch
3. Check the Actions tab for deployment status
4. Service will be available at: `https://SERVICE_NAME-HASH-REGION.a.run.app`

## Environment Variables

The deployment sets these environment variables automatically:
- `ENV=production`
- `GITHUB_SHA` (current commit hash)
- `PORT=8080`

## Security Notes

- Service runs as non-root user
- Uses gunicorn in production mode
- Container Registry images are private by default
- Cloud Run service allows unauthenticated access (can be changed)