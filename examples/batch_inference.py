"""
Batch inference example using SageMaker Batch Transform.
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
import os
import time


def create_batch_transform_job(
    model_name,
    input_s3_path,
    output_s3_path,
    instance_type='ml.m5.xlarge',
    instance_count=1,
    max_payload=6,
    batch_strategy='MultiRecord'
):
    """
    Create a batch transform job for bulk inference.
    
    Args:
        model_name: Name of the SageMaker model
        input_s3_path: S3 path to input data
        output_s3_path: S3 path for output results
        instance_type: Instance type for batch transform
        instance_count: Number of instances
        max_payload: Maximum payload size in MB
        batch_strategy: Batching strategy
    
    Returns:
        str: Transform job name
    """
    sagemaker_client = boto3.client('sagemaker')
    
    # Generate job name
    job_name = f'batch-transform-{int(time.time())}'
    
    # Create transform job
    response = sagemaker_client.create_transform_job(
        TransformJobName=job_name,
        ModelName=model_name,
        MaxConcurrentTransforms=instance_count,
        MaxPayloadInMB=max_payload,
        BatchStrategy=batch_strategy,
        TransformInput={
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': input_s3_path
                }
            },
            'ContentType': 'application/x-image',
            'SplitType': 'None'
        },
        TransformOutput={
            'S3OutputPath': output_s3_path,
            'Accept': 'application/json',
            'AssembleWith': 'Line'
        },
        TransformResources={
            'InstanceType': instance_type,
            'InstanceCount': instance_count
        }
    )
    
    print(f"Batch transform job created: {job_name}")
    print(f"Input: {input_s3_path}")
    print(f"Output: {output_s3_path}")
    
    return job_name


def monitor_batch_job(job_name):
    """Monitor batch transform job progress."""
    sagemaker_client = boto3.client('sagemaker')
    
    print(f"\nMonitoring job: {job_name}")
    
    while True:
        response = sagemaker_client.describe_transform_job(
            TransformJobName=job_name
        )
        
        status = response['TransformJobStatus']
        print(f"Status: {status}")
        
        if status in ['Completed', 'Failed', 'Stopped']:
            break
        
        time.sleep(30)
    
    if status == 'Completed':
        print(f"\n✓ Batch transform completed!")
        print(f"Output location: {response['TransformOutput']['S3OutputPath']}")
    else:
        print(f"\n✗ Batch transform {status.lower()}")
        if 'FailureReason' in response:
            print(f"Reason: {response['FailureReason']}")


def prepare_batch_input(image_paths, output_s3_path):
    """
    Upload images to S3 for batch processing.
    
    Args:
        image_paths: List of local image paths
        output_s3_path: S3 path to upload images
    """
    s3_client = boto3.client('s3')
    
    # Parse S3 path
    if output_s3_path.startswith('s3://'):
        output_s3_path = output_s3_path[5:]
    
    bucket, prefix = output_s3_path.split('/', 1)
    
    print(f"Uploading {len(image_paths)} images to S3...")
    
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        s3_key = f"{prefix}/{filename}"
        
        s3_client.upload_file(image_path, bucket, s3_key)
        print(f"  Uploaded: {filename}")
    
    print(f"\n✓ All images uploaded to s3://{bucket}/{prefix}/")


# Example usage
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch inference example')
    parser.add_argument('--model-name', type=str, required=True,
                       help='SageMaker model name')
    parser.add_argument('--input-s3-path', type=str, required=True,
                       help='S3 path to input images')
    parser.add_argument('--output-s3-path', type=str, required=True,
                       help='S3 path for output results')
    parser.add_argument('--instance-type', type=str, default='ml.m5.xlarge',
                       help='Instance type')
    parser.add_argument('--instance-count', type=int, default=1,
                       help='Number of instances')
    
    args = parser.parse_args()
    
    # Create batch transform job
    job_name = create_batch_transform_job(
        model_name=args.model_name,
        input_s3_path=args.input_s3_path,
        output_s3_path=args.output_s3_path,
        instance_type=args.instance_type,
        instance_count=args.instance_count
    )
    
    # Monitor progress
    monitor_batch_job(job_name)
