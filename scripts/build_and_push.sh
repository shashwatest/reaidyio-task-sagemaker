#!/bin/bash

# Build and push Docker container to ECR
# Usage: ./build_and_push.sh <region> <account-id>

set -e

REGION=${1:-us-east-1}
ACCOUNT_ID=${2}

if [ -z "$ACCOUNT_ID" ]; then
    echo "Getting AWS account ID..."
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
fi

IMAGE_NAME="sagemaker-pytorch-training"
ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE_NAME}"

echo "Building Docker image..."
docker build -t ${IMAGE_NAME}:latest .

echo "Logging in to ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

echo "Creating ECR repository (if not exists)..."
aws ecr describe-repositories --repository-names ${IMAGE_NAME} --region ${REGION} || \
    aws ecr create-repository --repository-name ${IMAGE_NAME} --region ${REGION}

echo "Tagging image..."
docker tag ${IMAGE_NAME}:latest ${ECR_REPO}:latest

echo "Pushing image to ECR..."
docker push ${ECR_REPO}:latest

echo "✓ Image pushed successfully!"
echo "Image URI: ${ECR_REPO}:latest"
