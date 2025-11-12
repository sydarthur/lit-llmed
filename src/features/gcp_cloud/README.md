# GCP Cloud Run Service

Basic Flask application for running on Google Cloud Run.

## Local Testing

### Option 1: Direct Python Testing
1. Install dependencies:
```bash
pip install -r ../../requirements.txt
```

2. Run the automated test:
```bash
cd src/gcp
python test_local.py
```

### Option 2: Manual Testing
1. Start the server:
```bash
python app.py
```

2. Test endpoints manually:
```bash
# Health check
curl http://localhost:8080/

# Status check
curl http://localhost:8080/api/status

# Process endpoint
curl -X POST http://localhost:8080/api/process \
  -H "Content-Type: application/json" \
  -d '{"message": "test", "data": [1,2,3]}'
```

### Option 3: Docker Testing
1. Build the image:
```bash
docker build -t lit-llmed-gcp .
```

2. Run the container:
```bash
docker run -p 8080:8080 lit-llmed-gcp
```

3. Test in another terminal:
```bash
curl http://localhost:8080/
```

## Endpoints

- `GET /` - Health check
- `GET /api/status` - Service status
- `POST /api/process` - Basic processing endpoint

## Cloud Run Deployment

1. Set up your GCP project and enable Cloud Run API
2. Configure your project ID in the deployment script
3. Run the deployment:

```bash
cd ../../config
./deploy.sh YOUR_PROJECT_ID us-central1
```

## Docker Build

```bash
docker build -t lit-llmed-gcp .
docker run -p 8080:8080 lit-llmed-gcp
```