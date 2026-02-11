import argparse
import boto3
import sagemaker
try:
    from sagemaker.pytorch import PyTorchModel
except ImportError:
    # For SageMaker SDK 3.x compatibility
    from sagemaker.model import Model as PyTorchModel
from sagemaker.predictor import Predictor
try:
    from sagemaker.serializers import IdentitySerializer
    from sagemaker.deserializers import JSONDeserializer
except ImportError:
    # For SageMaker SDK 3.x
    IdentitySerializer = None
    JSONDeserializer = None
import os
import time


def get_execution_role():
    """Get SageMaker execution role."""
    role = os.environ.get('SAGEMAKER_ROLE')
    if not role:
        try:
            role = sagemaker.get_execution_role()
        except:
            raise ValueError(
                "Could not determine SageMaker role. "
                "Please set SAGEMAKER_ROLE environment variable."
            )
    return role


def get_latest_training_job(job_prefix='pytorch-image-classifier'):
    """Get the latest completed training job."""
    sagemaker_client = boto3.client('sagemaker')
    
    response = sagemaker_client.list_training_jobs(
        SortBy='CreationTime',
        SortOrder='Descending',
        StatusEquals='Completed',
        NameContains=job_prefix,
        MaxResults=1
    )
    
    if not response['TrainingJobSummaries']:
        raise ValueError(f"No completed training jobs found with prefix: {job_prefix}")
    
    return response['TrainingJobSummaries'][0]['TrainingJobName']


def deploy_model(model_data_url=None, endpoint_name='image-classifier-endpoint', 
                 instance_type='ml.m5.xlarge', instance_count=1):
    """Deploy trained model to SageMaker endpoint."""
    
    session = sagemaker.Session()
    region = session.boto_region_name
    role = get_execution_role()
    
    print(f"Using role: {role}")
    print(f"Region: {region}")
    
    # If model_data_url not provided, get from latest training job
    if not model_data_url:
        training_job_name = get_latest_training_job()
        print(f"Using model from training job: {training_job_name}")
        
        sagemaker_client = boto3.client('sagemaker')
        job_details = sagemaker_client.describe_training_job(TrainingJobName=training_job_name)
        model_data_url = job_details['ModelArtifacts']['S3ModelArtifacts']
    
    print(f"Model data: {model_data_url}")
    
    # Get inference image URI
    account_id = boto3.client('sts').get_caller_identity()['Account']
    inference_image = f"{account_id}.dkr.ecr.{region}.amazonaws.com/sagemaker-pytorch-training:latest"
    
    # Create PyTorch model
    pytorch_model = PyTorchModel(
        model_data=model_data_url,
        role=role,
        image_uri=inference_image,
        entry_point='inference.py',
        source_dir='./src',
        framework_version='2.0.1',
        py_version='py310',
        sagemaker_session=session
    )
    
    print(f"\nDeploying model to endpoint: {endpoint_name}")
    print("This may take 5-10 minutes...")
    
    # Deploy model
    predictor = pytorch_model.deploy(
        endpoint_name=endpoint_name,
        instance_type=instance_type,
        initial_instance_count=instance_count,
        serializer=IdentitySerializer(content_type='application/x-image'),
        deserializer=JSONDeserializer(),
        wait=True
    )
    
    print(f"\n✓ Model deployed successfully!")
    print(f"Endpoint name: {endpoint_name}")
    print(f"Endpoint URL: https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{endpoint_name}/invocations")
    
    # Save endpoint info
    endpoint_info = {
        'endpoint_name': endpoint_name,
        'region': region,
        'instance_type': instance_type,
        'model_data': model_data_url
    }
    
    import json
    with open('endpoint_info.json', 'w') as f:
        json.dump(endpoint_info, f, indent=2)
    
    print("\nEndpoint info saved to endpoint_info.json")
    
    return predictor


def configure_autoscaling(endpoint_name, min_capacity=1, max_capacity=3, target_value=70.0):
    """Configure auto-scaling for the endpoint."""
    
    client = boto3.client('application-autoscaling')
    
    # Register scalable target
    resource_id = f'endpoint/{endpoint_name}/variant/AllTraffic'
    
    try:
        client.register_scalable_target(
            ServiceNamespace='sagemaker',
            ResourceId=resource_id,
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            MinCapacity=min_capacity,
            MaxCapacity=max_capacity
        )
        
        # Define scaling policy
        client.put_scaling_policy(
            PolicyName=f'{endpoint_name}-scaling-policy',
            ServiceNamespace='sagemaker',
            ResourceId=resource_id,
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            PolicyType='TargetTrackingScaling',
            TargetTrackingScalingPolicyConfiguration={
                'TargetValue': target_value,
                'PredefinedMetricSpecification': {
                    'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
                },
                'ScaleInCooldown': 300,
                'ScaleOutCooldown': 60
            }
        )
        
        print(f"\n✓ Auto-scaling configured:")
        print(f"  Min instances: {min_capacity}")
        print(f"  Max instances: {max_capacity}")
        print(f"  Target invocations per instance: {target_value}")
        
    except Exception as e:
        print(f"Warning: Could not configure auto-scaling: {e}")


def main():
    parser = argparse.ArgumentParser(description='Deploy model to SageMaker endpoint')
    parser.add_argument('--model-data', type=str, help='S3 path to model.tar.gz')
    parser.add_argument('--endpoint-name', type=str, default='image-classifier-endpoint',
                       help='Name for the endpoint')
    parser.add_argument('--instance-type', type=str, default='ml.m5.xlarge',
                       help='Instance type for endpoint')
    parser.add_argument('--instance-count', type=int, default=1,
                       help='Number of instances')
    parser.add_argument('--enable-autoscaling', action='store_true',
                       help='Enable auto-scaling')
    parser.add_argument('--min-capacity', type=int, default=1,
                       help='Minimum instances for auto-scaling')
    parser.add_argument('--max-capacity', type=int, default=3,
                       help='Maximum instances for auto-scaling')
    
    args = parser.parse_args()
    
    # Deploy model
    predictor = deploy_model(
        model_data_url=args.model_data,
        endpoint_name=args.endpoint_name,
        instance_type=args.instance_type,
        instance_count=args.instance_count
    )
    
    # Configure auto-scaling if requested
    if args.enable_autoscaling:
        configure_autoscaling(
            endpoint_name=args.endpoint_name,
            min_capacity=args.min_capacity,
            max_capacity=args.max_capacity
        )
    
    print("\nNext steps:")
    print(f"  python scripts/test_inference.py --endpoint-name {args.endpoint_name}")


if __name__ == '__main__':
    main()
