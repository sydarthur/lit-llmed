#!/bin/bash

# Cloud Run deployment script for lit-llmed-gcp
# Usage: ./deploy.sh PROJECT_ID REGION

PROJECT_ID=$1
REGION=${2:-us-central1}
SERVICE_NAME=lit-llmed-gcp

if [ -z "$PROJECT_ID" ]; then
    echo "Usage: $0 PROJECT_ID [REGION]"
    echo "Example: $0 my-gcp-project us-central1"
    exit 1
fi

echo "Deploying $SERVICE_NAME to $PROJECT_ID in $REGION..."

# Build and push the Docker image
echo "Building Docker image..."
cd src/gcp
docker build -t gcr.io/$PROJECT_ID/$SERVICE_NAME:latest .

echo "Pushing Docker image to GCR..."
docker push gcr.io/$PROJECT_ID/$SERVICE_NAME:latest

echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$SERVICE_NAME:latest \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 512Mi \
    --cpu 1 \
    --max-instances 10 \
    --min-instances 0 \
    --port 8080 \
    --set-env-vars ENV=production

echo "Deployment complete!"
echo "Service URL:"
gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)'