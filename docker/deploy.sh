#!/bin/bash

# Check if a version number was provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <version-number>"
  exit 1
fi

VERSION_NUMBER=$1

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi

REGION=us-west-2

REPOSITORY_NAME="vllm"
CONTAINER_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPOSITORY_NAME}"

# Log in to ECR
aws ecr get-login-password --region "${REGION}" | docker login --username AWS --password-stdin "${ACCOUNT_ID}".dkr.ecr."${REGION}".amazonaws.com

# Check if repository exists and create it if not.
aws ecr describe-repositories --repository-names "${REPOSITORY_NAME}" --region "${REGION}" > /dev/null 2>&1
if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${REPOSITORY_NAME}" --region "${REGION}" > /dev/null
fi

# Build the Docker image with specified build arguments.
if DOCKER_BUILDKIT=1 docker build . --target vllm-openai --tag "${REPOSITORY_NAME}" --build-arg max_jobs=8 --build-arg nvcc_threads=2
then 
    # Tagging both specific version and latest tag.
    docker tag "${REPOSITORY_NAME}" "$CONTAINER_URI:${VERSION_NUMBER}"
    docker tag "${REPOSITORY_NAME}" "$CONTAINER_URI:latest"

    # Pushing both tags to ECR.
    docker push "$CONTAINER_URI:${VERSION_NUMBER}"
    docker push "$CONTAINER_URI:latest"
else 
	echo "Docker build failed."
	exit 2	
fi  