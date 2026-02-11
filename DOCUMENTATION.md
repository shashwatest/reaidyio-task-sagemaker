# Complete Documentation

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Setup Instructions](#setup-instructions)
4. [Running the Pipeline](#running-the-pipeline)
5. [Troubleshooting](#troubleshooting)
6. [Architecture](#architecture)
7. [Cost Information](#cost-information)
8. [Command Reference](#command-reference)

---

## Overview

This is an end-to-end distributed training pipeline for image classification using Amazon SageMaker and PyTorch. It includes:

- Data preprocessing and augmentation
- Distributed training with PyTorch DDP
- Hyperparameter optimization
- Model deployment to SageMaker endpoints
- Real-time inference API
- Cost optimization with spot instances

**Tech Stack:** AWS SageMaker, PyTorch, Docker, Python 3.8+

---

## Prerequisites

### Required
- AWS Account with billing enabled
- AWS CLI installed and configured
- Docker Desktop installed and running
- Python 3.8 or higher
- 10GB free disk space

### AWS Resources Needed
- IAM user with access keys
- SageMaker execution role
- S3 bucket
- ECR repository
- SageMaker training quotas

---

## Setup Instructions

### 1. AWS Account Configuration

**Create IAM User:**
1. Go to AWS IAM Console
2. Create user with programmatic access
3. Attach policies: `AmazonSageMakerFullAccess`, `AmazonS3FullAccess`, `CloudWatchLogsFullAccess`, `AmazonEC2ContainerRegistryFullAccess`
4. Save access keys

**Configure AWS CLI:**
```bash
aws configure
# Enter: Access Key ID, Secret Access Key, Region (e.g., us-east-1), Output format (json)
```

**Create SageMaker Execution Role:**
1. Go to IAM Console → Roles → Create role
2. Select AWS service → SageMaker
3. Attach policies: `AmazonSageMakerFullAccess`, `AmazonS3FullAccess`, `CloudWatchLogsFullAccess`
4. Name: `SageMakerExecutionRole`
5. Copy the Role ARN

**Create S3 Bucket:**
```bash
aws s3 mb s3://your-unique-bucket-name --region us-east-1
```

### 2. Local Environment Setup

**Install Dependencies:**
```bash
pip install -r requirements.txt
```

**Set Environment Variables:**

Copy the example environment file and fill in your AWS details:
```bash
cp .env.example .env
```

Edit `.env` with your actual values:
```
SAGEMAKER_ROLE=arn:aws:iam::<your-account-id>:role/SageMakerExecutionRole
S3_BUCKET=your-bucket-name
AWS_REGION=us-east-1
```

Or set them in your shell:
```bash
# Windows PowerShell
$env:SAGEMAKER_ROLE="arn:aws:iam::<account-id>:role/SageMakerExecutionRole"
$env:S3_BUCKET="your-bucket-name"
$env:AWS_REGION="us-east-1"

# Linux/Mac
export SAGEMAKER_ROLE="arn:aws:iam::<account-id>:role/SageMakerExecutionRole"
export S3_BUCKET="your-bucket-name"
export AWS_REGION="us-east-1"
```

### 3. Build Docker Container

**Get AWS Account ID:**
```bash
aws sts get-caller-identity --query Account --output text
```

**Build and Push:**
```bash
# Build image
docker build -t sagemaker-pytorch-training .

# Create ECR repository
aws ecr create-repository --repository-name sagemaker-pytorch-training --region $AWS_REGION

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin <account-id>.dkr.ecr.$AWS_REGION.amazonaws.com

# Tag and push
docker tag sagemaker-pytorch-training:latest <account-id>.dkr.ecr.$AWS_REGION.amazonaws.com/sagemaker-pytorch-training:latest
docker push <account-id>.dkr.ecr.$AWS_REGION.amazonaws.com/sagemaker-pytorch-training:latest
```

Or use the automated script:
```bash
# Windows
scripts\build_and_push.bat %AWS_REGION%

# Linux/Mac
./scripts/build_and_push.sh $AWS_REGION
```

### 4. Request SageMaker Quotas

**Important:** New AWS accounts have 0 quota for SageMaker training instances.

1. Go to AWS Service Quotas Console: https://console.aws.amazon.com/servicequotas/
2. Search for "SageMaker"
3. Request these quotas:
   - `ml.m5.xlarge for spot training job usage` → Request 2-6 instances
   - `ml.m5.2xlarge for spot training job usage` → Request 2-4 instances
   - `ml.g4dn.xlarge for spot training job usage` → Request 2 instances (optional, for GPU)
4. Wait for approval (usually instant if within default limits, otherwise 24-48 hours)

---

## Running the Pipeline

### Step 1: Preprocess Data

```bash
python src/preprocessing.py --s3-bucket $S3_BUCKET
```

This will:
- Download CIFAR-10 dataset
- Apply data augmentation
- Split into train/validation/test
- Upload to S3

**Expected output:**
```
Training samples: 45000
Validation samples: 5000
Test samples: 10000
Preprocessing complete!
```

### Step 2: Train Model

```bash
python scripts/setup_sagemaker.py --mode train --s3-bucket $S3_BUCKET
```

**Options:**
- `--no-spot`: Disable spot instances
- `--dataset cifar10`: Specify dataset name

**Expected time:** 1-2 hours (CPU), 30-45 minutes (GPU)

### Step 3: Monitor Training

```bash
# List all training jobs
python scripts/monitor_training.py --list

# Monitor specific job
python scripts/monitor_training.py --job-name <job-name>
```

Or view in AWS Console: https://console.aws.amazon.com/sagemaker/ → Training jobs

### Step 4: Hyperparameter Tuning (Optional)

```bash
python scripts/setup_sagemaker.py --mode tune --s3-bucket $S3_BUCKET
```

This runs 20 training jobs with different hyperparameters using Bayesian optimization.

**Analyze results:**
```bash
jupyter notebook notebooks/hyperparameter_tuning.ipynb
```

### Step 5: Deploy Model

```bash
python scripts/deploy_endpoint.py --endpoint-name my-classifier
```

**Options:**
- `--instance-type ml.m5.xlarge`: Specify instance type
- `--instance-count 1`: Number of instances
- `--enable-autoscaling`: Enable auto-scaling
- `--model-data s3://path/to/model.tar.gz`: Deploy specific model

**Expected time:** 5-10 minutes

### Step 6: Test Inference

```bash
# Test with random image
python scripts/test_inference.py --endpoint-name my-classifier

# Test with your image
python scripts/test_inference.py --endpoint-name my-classifier --image-path path/to/image.jpg

# Test with S3 image
python scripts/test_inference.py --endpoint-name my-classifier --s3-bucket $S3_BUCKET --s3-key images/test.jpg
```

### Step 7: Cleanup

**Important:** Delete endpoints when not in use to avoid charges!

```bash
python scripts/cleanup.py --endpoint-name my-classifier --s3-bucket $S3_BUCKET
```

Or manually:
```bash
aws sagemaker delete-endpoint --endpoint-name my-classifier
```

---

## Troubleshooting

### Common Issues

**1. "ModuleNotFoundError: No module named 'sagemaker.pytorch'"**

Solution: Downgrade to SageMaker SDK 2.x
```bash
pip install "sagemaker<3.0" --force-reinstall
```

**2. "ResourceLimitExceeded: account-level service limit"**

Solution: Request quota increase in AWS Service Quotas Console (see Setup Instructions)

**3. "Could not determine SageMaker role"**

Solution: Set SAGEMAKER_ROLE environment variable or create .env file

**4. Docker build fails**

Solution: Ensure Docker Desktop is running

**5. Training job fails**

Solution: Check CloudWatch logs in AWS Console for error details

**6. "Access Denied" errors**

Solution: Verify IAM permissions and SageMaker role has correct policies

### Checking Status

**Training Jobs:**
```bash
aws sagemaker list-training-jobs --sort-by CreationTime --sort-order Descending
aws sagemaker describe-training-job --training-job-name <job-name>
```

**Endpoints:**
```bash
aws sagemaker list-endpoints
aws sagemaker describe-endpoint --endpoint-name <endpoint-name>
```

**CloudWatch Logs:**
- Training: `/aws/sagemaker/TrainingJobs`
- Endpoints: `/aws/sagemaker/Endpoints/<endpoint-name>`

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                         AWS Cloud                            │
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │    S3    │◄──►│ SageMaker│───►│CloudWatch│             │
│  │          │    │          │    │          │             │
│  │ Datasets │    │ Training │    │ Metrics  │             │
│  │ Models   │    │ Endpoints│    │ Logs     │             │
│  └──────────┘    └──────────┘    └──────────┘             │
│       ▲               ▲                                     │
│       │               │                                     │
│  ┌────┴────┐     ┌────┴────┐                              │
│  │   ECR   │     │   IAM   │                              │
│  │ Docker  │     │  Roles  │                              │
│  └─────────┘     └─────────┘                              │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Preprocessing:** Download CIFAR-10 → Augment → Upload to S3
2. **Training:** S3 → SageMaker Training → Model Artifacts → S3
3. **Deployment:** S3 Model → SageMaker Endpoint
4. **Inference:** Client → Endpoint → Prediction

### Model Architecture

- **Base:** ResNet-18 (modified for CIFAR-10)
- **Input:** 32x32 RGB images
- **Output:** 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Modifications:** 
  - First conv layer: 3x3 instead of 7x7
  - No max pooling after first conv
  - Final FC layer: 10 classes

---

## Cost Information

### Training Costs

**CPU Instances (ml.m5.xlarge):**
- On-demand: ~$0.27/hour
- Spot: ~$0.08/hour (70% savings)
- Typical training: 1-2 hours = $0.16-0.32 with spot

**GPU Instances (ml.g4dn.xlarge):**
- On-demand: ~$0.75/hour
- Spot: ~$0.23/hour (70% savings)
- Typical training: 30-45 minutes = $0.17-0.26 with spot

### Inference Costs

**Endpoint (ml.m5.xlarge):**
- Cost: ~$0.19/hour = ~$140/month
- **Important:** Delete when not in use!

### Storage Costs

- S3: ~$0.023/GB/month
- ECR: ~$0.10/GB/month
- Typical usage: ~$0.50/month

### Total Estimated Costs

- **One-time setup:** ~$0.10 (Docker image storage)
- **Single training run:** ~$0.30 (with spot instances)
- **Hyperparameter tuning:** ~$6 (20 jobs with spot)
- **Endpoint (24/7):** ~$140/month
- **Testing/Learning:** ~$5-10 total (if you cleanup promptly)

### Cost Optimization Tips

1. **Use spot instances** (enabled by default) - 70% savings
2. **Delete endpoints** when not in use
3. **Use auto-scaling** for production endpoints
4. **Set up billing alerts** in AWS Console
5. **Use S3 lifecycle policies** to archive old data

---

## Command Reference

### Data Preprocessing
```bash
python src/preprocessing.py --s3-bucket <bucket> --dataset cifar10
```

### Training
```bash
# Standard training
python scripts/setup_sagemaker.py --mode train --s3-bucket <bucket>

# Without spot instances
python scripts/setup_sagemaker.py --mode train --s3-bucket <bucket> --no-spot

# Hyperparameter tuning
python scripts/setup_sagemaker.py --mode tune --s3-bucket <bucket>
```

### Monitoring
```bash
# List jobs
python scripts/monitor_training.py --list

# Monitor specific job
python scripts/monitor_training.py --job-name <job-name> --refresh 30
```

### Deployment
```bash
# Deploy endpoint
python scripts/deploy_endpoint.py --endpoint-name <name>

# With auto-scaling
python scripts/deploy_endpoint.py --endpoint-name <name> --enable-autoscaling --min-capacity 1 --max-capacity 3

# Deploy specific model
python scripts/deploy_endpoint.py --endpoint-name <name> --model-data s3://bucket/path/model.tar.gz
```

### Inference
```bash
# Test endpoint
python scripts/test_inference.py --endpoint-name <name> --num-tests 5

# Test with image
python scripts/test_inference.py --endpoint-name <name> --image-path image.jpg

# Sample inference
python examples/sample_inference.py --endpoint-name <name> --image-path image.jpg
```

### Cleanup
```bash
# Cleanup all resources
python scripts/cleanup.py --endpoint-name <name> --s3-bucket <bucket>

# Dry run (see what would be deleted)
python scripts/cleanup.py --dry-run --s3-bucket <bucket>

# List resources only
python scripts/cleanup.py --list-only
```

### AWS CLI Commands
```bash
# List training jobs
aws sagemaker list-training-jobs --sort-by CreationTime --sort-order Descending

# Describe training job
aws sagemaker describe-training-job --training-job-name <job-name>

# List endpoints
aws sagemaker list-endpoints

# Delete endpoint
aws sagemaker delete-endpoint --endpoint-name <name>

# View CloudWatch logs
aws logs tail /aws/sagemaker/TrainingJobs --follow
```

---

## Project Structure

```
.
├── src/
│   ├── preprocessing.py       # Data preprocessing and augmentation
│   ├── train.py              # Distributed PyTorch training
│   └── inference.py          # Model inference handler
├── scripts/
│   ├── setup_sagemaker.py    # Launch training/tuning jobs
│   ├── deploy_endpoint.py    # Deploy model to endpoint
│   ├── test_inference.py     # Test deployed endpoint
│   ├── monitor_training.py   # Monitor training progress
│   ├── cleanup.py            # Cleanup AWS resources
│   ├── train_cpu.py          # CPU-only training script
│   ├── build_and_push.bat    # Docker build (Windows)
│   └── build_and_push.sh     # Docker build (Linux/Mac)
├── examples/
│   ├── sample_inference.py   # Inference examples
│   └── batch_inference.py    # Batch transform
├── notebooks/
│   └── hyperparameter_tuning.ipynb  # HPO analysis
├── Dockerfile                 # Training container
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables
├── .gitignore               # Git ignore rules
├── README.md                # Project overview
└── DOCUMENTATION.md         # This file
```

---

## Additional Resources

### AWS Documentation
- SageMaker: https://docs.aws.amazon.com/sagemaker/
- Service Quotas: https://docs.aws.amazon.com/servicequotas/
- ECR: https://docs.aws.amazon.com/ecr/

### PyTorch Documentation
- PyTorch: https://pytorch.org/docs/
- torchvision: https://pytorch.org/vision/

### Support
- AWS Support: https://console.aws.amazon.com/support/
- SageMaker Forums: https://forums.aws.amazon.com/forum.jspa?forumID=285

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- AWS SageMaker team for excellent documentation
- PyTorch team for the framework
- CIFAR-10 dataset creators
