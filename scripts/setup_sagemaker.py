import argparse
import boto3
import sagemaker
try:
    from sagemaker.estimator import Estimator
except ImportError:
    # For SageMaker SDK 3.x
    from sagemaker import Estimator
try:
    from sagemaker.pytorch import PyTorch
except ImportError:
    # For SageMaker SDK 3.x compatibility
    PyTorch = None
try:
    from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter
except ImportError:
    try:
        from sagemaker.parameter import IntegerParameter, ContinuousParameter
        from sagemaker.tuner import HyperparameterTuner
    except ImportError:
        # SDK 3.x
        from sagemaker import HyperparameterTuner, IntegerParameter, ContinuousParameter
import time
import os

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip


def get_execution_role():
    """Get SageMaker execution role."""
    role = os.environ.get('SAGEMAKER_ROLE')
    if not role:
        try:
            role = sagemaker.get_execution_role()
        except:
            raise ValueError(
                "Could not determine SageMaker role. "
                "Please set SAGEMAKER_ROLE environment variable or run from SageMaker notebook."
            )
    return role


def get_training_image_uri(region):
    """Get ECR image URI for training container."""
    account_id = boto3.client('sts').get_caller_identity()['Account']
    ecr_image = f"{account_id}.dkr.ecr.{region}.amazonaws.com/sagemaker-pytorch-training:latest"
    return ecr_image


def create_training_job(s3_bucket, dataset_name='cifar10', use_spot=True):
    """Create and launch a SageMaker training job."""
    
    session = sagemaker.Session()
    region = session.boto_region_name
    role = get_execution_role()
    
    print(f"Using role: {role}")
    print(f"Region: {region}")
    
    # Training data location
    train_input = f's3://{s3_bucket}/datasets/{dataset_name}/train.pt'
    val_input = f's3://{s3_bucket}/datasets/{dataset_name}/validation.pt'
    
    # Get custom training image
    image_uri = get_training_image_uri(region)
    print(f"Using training image: {image_uri}")
    
    # Define hyperparameters
    hyperparameters = {
        'epochs': 50,
        'batch-size': 128,
        'learning-rate': 0.1,
        'momentum': 0.9,
        'weight-decay': 5e-4,
        'num-classes': 10
    }
    
    # Configure distributed training
    distribution = {
        'pytorchddp': {
            'enabled': True
        }
    }
    
    # Spot instance configuration
    if use_spot:
        max_run = 3600 * 3  # 3 hours
        max_wait = 3600 * 6  # 6 hours
        checkpoint_s3_uri = f's3://{s3_bucket}/checkpoints/{dataset_name}'
    else:
        max_run = 3600 * 3  # 3 hours
        max_wait = None
        checkpoint_s3_uri = None
    
    # Metric definitions for CloudWatch
    metric_definitions = [
        {'Name': 'train:loss', 'Regex': 'train_loss=([0-9\\.]+)'},
        {'Name': 'train:accuracy', 'Regex': 'train_acc=([0-9\\.]+)'},
        {'Name': 'validation:loss', 'Regex': 'val_loss=([0-9\\.]+)'},
        {'Name': 'validation:accuracy', 'Regex': 'val_acc=([0-9\\.]+)'},
    ]
    
    # Create estimator using generic Estimator class (compatible with SDK 3.x)
    if PyTorch is not None:
        # Use PyTorch estimator if available (SDK 2.x)
        estimator = PyTorch(
            entry_point='train.py',
            source_dir='./src',
            image_uri=image_uri,
            role=role,
            instance_count=2,
            instance_type='ml.m5.2xlarge',  # Using CPU instance with spot
            hyperparameters=hyperparameters,
            distribution=distribution,
            output_path=f's3://{s3_bucket}/models/{dataset_name}',
            sagemaker_session=session,
            use_spot_instances=use_spot,
            max_run=max_run,
            max_wait=max_wait,
            checkpoint_s3_uri=checkpoint_s3_uri,
            base_job_name='pytorch-image-classifier',
            metric_definitions=metric_definitions
        )
    else:
        # Use generic Estimator for SDK 3.x
        estimator = Estimator(
            image_uri=image_uri,
            role=role,
            instance_count=2,
            instance_type='ml.g4dn.xlarge',
            hyperparameters=hyperparameters,
            output_path=f's3://{s3_bucket}/models/{dataset_name}',
            sagemaker_session=session,
            use_spot_instances=use_spot,
            max_run=max_run,
            max_wait=max_wait,
            checkpoint_s3_uri=checkpoint_s3_uri,
            base_job_name='pytorch-image-classifier',
            metric_definitions=metric_definitions,
            # Distribution config for SDK 3.x
            distribution=distribution
        )
    
    # Launch training job
    print("\nLaunching training job...")
    estimator.fit({
        'train': train_input,
        'validation': val_input
    }, wait=False)
    
    print(f"\nTraining job started: {estimator.latest_training_job.name}")
    print(f"Monitor at: https://console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{estimator.latest_training_job.name}")
    
    return estimator


def create_hyperparameter_tuning_job(s3_bucket, dataset_name='cifar10'):
    """Create and launch hyperparameter tuning job."""
    
    session = sagemaker.Session()
    region = session.boto_region_name
    role = get_execution_role()
    
    print(f"Using role: {role}")
    print(f"Region: {region}")
    
    # Training data location
    train_input = f's3://{s3_bucket}/datasets/{dataset_name}/train.pt'
    val_input = f's3://{s3_bucket}/datasets/{dataset_name}/validation.pt'
    
    # Get custom training image
    image_uri = get_training_image_uri(region)
    
    # Base hyperparameters
    hyperparameters = {
        'epochs': 30,
        'num-classes': 10
    }
    
    # Metric definitions
    metric_definitions = [
        {'Name': 'validation:accuracy', 'Regex': 'val_acc=([0-9\\.]+)'},
    ]
    
    # Create estimator (compatible with both SDK 2.x and 3.x)
    if PyTorch is not None:
        # Use PyTorch estimator if available (SDK 2.x)
        estimator = PyTorch(
            entry_point='train.py',
            source_dir='./src',
            image_uri=image_uri,
            role=role,
            instance_count=1,
            instance_type='ml.g4dn.xlarge',
            hyperparameters=hyperparameters,
            output_path=f's3://{s3_bucket}/tuning/{dataset_name}',
            sagemaker_session=session,
            use_spot_instances=True,
            max_run=3600 * 2,
            max_wait=3600 * 4,
            base_job_name='pytorch-hpo',
            metric_definitions=metric_definitions
        )
    else:
        # Use generic Estimator for SDK 3.x
        estimator = Estimator(
            image_uri=image_uri,
            role=role,
            instance_count=1,
            instance_type='ml.g4dn.xlarge',
            hyperparameters=hyperparameters,
            output_path=f's3://{s3_bucket}/tuning/{dataset_name}',
            sagemaker_session=session,
            use_spot_instances=True,
            max_run=3600 * 2,
            max_wait=3600 * 4,
            base_job_name='pytorch-hpo',
            metric_definitions=metric_definitions
        )
    
    # Define hyperparameter ranges
    hyperparameter_ranges = {
        'learning-rate': ContinuousParameter(0.001, 0.2),
        'batch-size': IntegerParameter(64, 256),
        'momentum': ContinuousParameter(0.8, 0.99),
        'weight-decay': ContinuousParameter(1e-5, 1e-3)
    }
    
    # Create tuner
    tuner = HyperparameterTuner(
        estimator=estimator,
        objective_metric_name='validation:accuracy',
        objective_type='Maximize',
        hyperparameter_ranges=hyperparameter_ranges,
        metric_definitions=[
            {'Name': 'validation:accuracy', 'Regex': 'val_acc=([0-9\\.]+)'},
        ],
        max_jobs=20,
        max_parallel_jobs=2,
        base_tuning_job_name='pytorch-hpo'
    )
    
    # Launch tuning job
    print("\nLaunching hyperparameter tuning job...")
    tuner.fit({
        'train': train_input,
        'validation': val_input
    }, wait=False)
    
    print(f"\nTuning job started: {tuner.latest_tuning_job.name}")
    print(f"Monitor at: https://console.aws.amazon.com/sagemaker/home?region={region}#/hyper-tuning-jobs/{tuner.latest_tuning_job.name}")
    
    return tuner


def main():
    parser = argparse.ArgumentParser(description='Setup SageMaker training')
    parser.add_argument('--mode', type=str, choices=['train', 'tune'], required=True,
                       help='Mode: train (single job) or tune (hyperparameter tuning)')
    parser.add_argument('--s3-bucket', type=str, help='S3 bucket name')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('--no-spot', action='store_true', help='Disable spot instances')
    
    args = parser.parse_args()
    
    # Get S3 bucket from env if not provided
    s3_bucket = args.s3_bucket or os.environ.get('S3_BUCKET')
    if not s3_bucket:
        raise ValueError("S3 bucket must be specified via --s3-bucket or S3_BUCKET env variable")
    
    if args.mode == 'train':
        estimator = create_training_job(
            s3_bucket=s3_bucket,
            dataset_name=args.dataset,
            use_spot=not args.no_spot
        )
        print("\nTo check job status:")
        print(f"  aws sagemaker describe-training-job --training-job-name {estimator.latest_training_job.name}")
    
    elif args.mode == 'tune':
        tuner = create_hyperparameter_tuning_job(
            s3_bucket=s3_bucket,
            dataset_name=args.dataset
        )
        print("\nTo check tuning status:")
        print(f"  aws sagemaker describe-hyper-parameter-tuning-job --hyper-parameter-tuning-job-name {tuner.latest_tuning_job.name}")


if __name__ == '__main__':
    main()
