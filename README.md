# SageMaker Distributed Image Classification Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red.svg)](https://pytorch.org/)
[![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker-orange.svg)](https://aws.amazon.com/sagemaker/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

End-to-end distributed training pipeline for image classification using Amazon SageMaker and PyTorch.

## Features

- ✅ Data preprocessing and augmentation (CIFAR-10)
- ✅ Distributed training with PyTorch DDP
- ✅ Hyperparameter optimization with Bayesian search
- ✅ Model deployment to SageMaker endpoints
- ✅ Real-time inference API
- ✅ Cost optimization with spot instances (70% savings)
- ✅ Auto-scaling endpoints
- ✅ CloudWatch monitoring and logging

## Quick Start

### Prerequisites

- AWS Account with billing enabled
- AWS CLI configured
- Docker Desktop running
- Python 3.8+

### Setup (5 minutes)

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Configure environment:**

Copy `.env.example` to `.env` and fill in your AWS details:
```bash
cp .env.example .env
# Edit .env with your AWS account details
```

3. **Build Docker container:**
```bash
# Windows
scripts\build_and_push.bat %AWS_REGION%

# Linux/Mac
./scripts/build_and_push.sh $AWS_REGION
```

4. **Request SageMaker quotas:**
   - Go to AWS Service Quotas Console
   - Request `ml.m5.xlarge for spot training job usage` = 2-6 instances
   - Wait for approval (usually instant)

### Run Pipeline

```bash
# 1. Preprocess data
python src/preprocessing.py --s3-bucket $S3_BUCKET

# 2. Train model
python scripts/setup_sagemaker.py --mode train --s3-bucket $S3_BUCKET

# 3. Monitor training
python scripts/monitor_training.py --list

# 4. Deploy endpoint
python scripts/deploy_endpoint.py --endpoint-name my-classifier

# 5. Test inference
python scripts/test_inference.py --endpoint-name my-classifier

# 6. Cleanup (important!)
python scripts/cleanup.py --endpoint-name my-classifier
```

## Project Structure

```
├── src/                    # Core training code
├── scripts/                # Utility scripts
├── examples/               # Usage examples
├── notebooks/              # Analysis notebooks
├── Dockerfile             # Training container
├── requirements.txt       # Dependencies
├── README.md             # This file
└── DOCUMENTATION.md      # Complete documentation
```

## Documentation

📚 **[Complete Documentation](DOCUMENTATION.md)** - Detailed setup, troubleshooting, architecture, and command reference

## Key Commands

| Task | Command |
|------|---------|
| Preprocess data | `python src/preprocessing.py --s3-bucket <bucket>` |
| Train model | `python scripts/setup_sagemaker.py --mode train --s3-bucket <bucket>` |
| Hyperparameter tuning | `python scripts/setup_sagemaker.py --mode tune --s3-bucket <bucket>` |
| Monitor training | `python scripts/monitor_training.py --list` |
| Deploy endpoint | `python scripts/deploy_endpoint.py --endpoint-name <name>` |
| Test inference | `python scripts/test_inference.py --endpoint-name <name>` |
| Cleanup | `python scripts/cleanup.py --endpoint-name <name>` |

## Cost Estimate

- **Training:** ~$0.30 per job (with spot instances)
- **Hyperparameter tuning:** ~$6 (20 jobs)
- **Endpoint:** ~$0.19/hour (delete when not in use!)
- **Storage:** ~$0.50/month

**Total for testing:** ~$5-10 (if you cleanup promptly)

## Architecture

```
Local Machine          AWS Cloud
     │                      │
     ├─ Docker ────────► ECR
     ├─ Data ──────────► S3
     │                      │
     └─ Scripts ────────► SageMaker
                            ├─ Training Jobs
                            ├─ Endpoints
                            └─ CloudWatch
```

## Troubleshooting

**"ResourceLimitExceeded" error:**
- Request quota increase in AWS Service Quotas Console
- See [DOCUMENTATION.md](DOCUMENTATION.md#troubleshooting) for details

**"ModuleNotFoundError: sagemaker.pytorch":**
```bash
pip install "sagemaker<3.0" --force-reinstall
```

**Training fails:**
- Check CloudWatch logs in AWS Console
- See [DOCUMENTATION.md](DOCUMENTATION.md#troubleshooting)

## Tech Stack

- **ML Framework:** PyTorch 2.0.1
- **Cloud Platform:** AWS SageMaker
- **Container:** Docker
- **Dataset:** CIFAR-10
- **Model:** ResNet-18 (modified)

## Performance

- **Training time:** 1-2 hours (CPU), 30-45 minutes (GPU)
- **Inference latency:** 30-50ms
- **Model accuracy:** ~90% on CIFAR-10

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For detailed instructions, troubleshooting, and architecture information, see [DOCUMENTATION.md](DOCUMENTATION.md).

---

**Ready to start?** Follow the [Quick Start](#quick-start) guide above!
