@echo off
REM Build and push Docker container to ECR (Windows)
REM Usage: build_and_push.bat <region> <account-id>

setlocal enabledelayedexpansion

set REGION=%1
if "%REGION%"=="" set REGION=us-east-1

set ACCOUNT_ID=%2
if "%ACCOUNT_ID%"=="" (
    echo Getting AWS account ID...
    for /f "tokens=*" %%i in ('aws sts get-caller-identity --query Account --output text') do set ACCOUNT_ID=%%i
)

set IMAGE_NAME=sagemaker-pytorch-training
set ECR_REPO=%ACCOUNT_ID%.dkr.ecr.%REGION%.amazonaws.com/%IMAGE_NAME%

echo Building Docker image...
docker build -t %IMAGE_NAME%:latest .

echo Logging in to ECR...
aws ecr get-login-password --region %REGION% | docker login --username AWS --password-stdin %ACCOUNT_ID%.dkr.ecr.%REGION%.amazonaws.com

echo Creating ECR repository (if not exists)...
aws ecr describe-repositories --repository-names %IMAGE_NAME% --region %REGION% 2>nul || aws ecr create-repository --repository-name %IMAGE_NAME% --region %REGION%

echo Tagging image...
docker tag %IMAGE_NAME%:latest %ECR_REPO%:latest

echo Pushing image to ECR...
docker push %ECR_REPO%:latest

echo.
echo Image pushed successfully!
echo Image URI: %ECR_REPO%:latest

endlocal
