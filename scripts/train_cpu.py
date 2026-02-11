"""
Quick training script using CPU instances (ml.m5.xlarge) which don't require quota increases.
This is slower than GPU but works immediately on new AWS accounts.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.setup_sagemaker import *

# Override the create_training_job function to use CPU instances
def create_training_job_cpu(s3_bucket, dataset_name='cifar10'):
    """Create training job with CPU instances (no quota needed)."""
    
    session = sagemaker.Session()
    region = session.boto_region_name
    role = get_execution_role()
    
    print(f"Using role: {role}")
    print(f"Region: {region}")
    print("⚠️  Using CPU instances (ml.m5.xlarge) - training will be slower but no quota needed")
    
    # Training data location
    train_input = f's3://{s3_bucket}/datasets/{dataset_name}/train.pt'
    val_input = f's3://{s3_bucket}/datasets/{dataset_name}/validation.pt'
    
    # Get custom training image
    image_uri = get_training_image_uri(region)
    print(f"Using training image: {image_uri}")
    
    # Define hyperparameters (reduced epochs for CPU)
    hyperparameters = {
        'epochs': 10,  # Reduced for CPU
        'batch-size': 64,  # Smaller batch for CPU
        'learning-rate': 0.1,
        'momentum': 0.9,
        'weight-decay': 5e-4,
        'num-classes': 10
    }
    
    # Metric definitions for CloudWatch
    metric_definitions = [
        {'Name': 'train:loss', 'Regex': 'train_loss=([0-9\\.]+)'},
        {'Name': 'train:accuracy', 'Regex': 'train_acc=([0-9\\.]+)'},
        {'Name': 'validation:loss', 'Regex': 'val_loss=([0-9\\.]+)'},
        {'Name': 'validation:accuracy', 'Regex': 'val_acc=([0-9\\.]+)'},
    ]
    
    # Create estimator with CPU instance
    estimator = PyTorch(
        entry_point='train.py',
        source_dir='./src',
        image_uri=image_uri,
        role=role,
        instance_count=1,  # Single instance for CPU
        instance_type='ml.m5.xlarge',  # CPU instance
        hyperparameters=hyperparameters,
        output_path=f's3://{s3_bucket}/models/{dataset_name}',
        sagemaker_session=session,
        max_run=3600 * 2,  # 2 hours max
        base_job_name='pytorch-cpu-classifier',
        metric_definitions=metric_definitions
    )
    
    # Launch training job
    print("\nLaunching CPU training job...")
    print("This will take longer than GPU training (expect 1-2 hours)")
    estimator.fit({
        'train': train_input,
        'validation': val_input
    }, wait=False)
    
    print(f"\nTraining job started: {estimator.latest_training_job.name}")
    print(f"Monitor at: https://console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{estimator.latest_training_job.name}")
    
    return estimator


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train with CPU instances')
    parser.add_argument('--s3-bucket', type=str, help='S3 bucket name')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
    
    args = parser.parse_args()
    
    # Get S3 bucket from env if not provided
    s3_bucket = args.s3_bucket or os.environ.get('S3_BUCKET')
    if not s3_bucket:
        raise ValueError("S3 bucket must be specified via --s3-bucket or S3_BUCKET env variable")
    
    estimator = create_training_job_cpu(
        s3_bucket=s3_bucket,
        dataset_name=args.dataset
    )
    
    print("\nTo check job status:")
    print(f"  python scripts/monitor_training.py --job-name {estimator.latest_training_job.name}")
