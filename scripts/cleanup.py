import argparse
import boto3
import time


def delete_endpoint(endpoint_name):
    """Delete SageMaker endpoint."""
    sagemaker_client = boto3.client('sagemaker')
    
    try:
        print(f"Deleting endpoint: {endpoint_name}")
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        print(f"✓ Endpoint {endpoint_name} deleted")
        return True
    except sagemaker_client.exceptions.ClientError as e:
        if 'Could not find endpoint' in str(e):
            print(f"Endpoint {endpoint_name} not found")
        else:
            print(f"Error deleting endpoint: {e}")
        return False


def delete_endpoint_config(config_name):
    """Delete SageMaker endpoint configuration."""
    sagemaker_client = boto3.client('sagemaker')
    
    try:
        print(f"Deleting endpoint config: {config_name}")
        sagemaker_client.delete_endpoint_config(EndpointConfigName=config_name)
        print(f"✓ Endpoint config {config_name} deleted")
        return True
    except sagemaker_client.exceptions.ClientError as e:
        if 'Could not find endpoint configuration' in str(e):
            print(f"Endpoint config {config_name} not found")
        else:
            print(f"Error deleting endpoint config: {e}")
        return False


def delete_model(model_name):
    """Delete SageMaker model."""
    sagemaker_client = boto3.client('sagemaker')
    
    try:
        print(f"Deleting model: {model_name}")
        sagemaker_client.delete_model(ModelName=model_name)
        print(f"✓ Model {model_name} deleted")
        return True
    except sagemaker_client.exceptions.ClientError as e:
        if 'Could not find model' in str(e):
            print(f"Model {model_name} not found")
        else:
            print(f"Error deleting model: {e}")
        return False


def list_endpoints():
    """List all SageMaker endpoints."""
    sagemaker_client = boto3.client('sagemaker')
    
    response = sagemaker_client.list_endpoints(
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=100
    )
    
    return response['Endpoints']


def list_models():
    """List all SageMaker models."""
    sagemaker_client = boto3.client('sagemaker')
    
    response = sagemaker_client.list_models(
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=100
    )
    
    return response['Models']


def cleanup_s3(bucket_name, prefixes):
    """Delete objects from S3 bucket."""
    s3_client = boto3.client('s3')
    
    for prefix in prefixes:
        print(f"\nDeleting objects with prefix: {prefix}")
        
        try:
            # List objects
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
            
            delete_count = 0
            for page in pages:
                if 'Contents' not in page:
                    continue
                
                objects = [{'Key': obj['Key']} for obj in page['Contents']]
                
                if objects:
                    s3_client.delete_objects(
                        Bucket=bucket_name,
                        Delete={'Objects': objects}
                    )
                    delete_count += len(objects)
            
            print(f"✓ Deleted {delete_count} objects from s3://{bucket_name}/{prefix}")
            
        except Exception as e:
            print(f"Error deleting from S3: {e}")


def cleanup_all(endpoint_name=None, s3_bucket=None, dry_run=False):
    """Cleanup all SageMaker resources."""
    
    print("="*60)
    print("SageMaker Resource Cleanup")
    print("="*60)
    
    if dry_run:
        print("\n[DRY RUN MODE - No resources will be deleted]\n")
    
    # Cleanup endpoints
    print("\n1. Cleaning up endpoints...")
    endpoints = list_endpoints()
    
    if endpoint_name:
        endpoints = [ep for ep in endpoints if ep['EndpointName'] == endpoint_name]
    
    if not endpoints:
        print("No endpoints found")
    else:
        for ep in endpoints:
            name = ep['EndpointName']
            status = ep['EndpointStatus']
            print(f"\nEndpoint: {name} (Status: {status})")
            
            if not dry_run:
                # Get endpoint config name before deleting
                sagemaker_client = boto3.client('sagemaker')
                try:
                    ep_details = sagemaker_client.describe_endpoint(EndpointName=name)
                    config_name = ep_details['EndpointConfigName']
                    
                    # Delete endpoint
                    delete_endpoint(name)
                    
                    # Wait a bit for endpoint to be deleted
                    time.sleep(2)
                    
                    # Delete endpoint config
                    delete_endpoint_config(config_name)
                    
                except Exception as e:
                    print(f"Error: {e}")
    
    # Cleanup models
    print("\n2. Cleaning up models...")
    models = list_models()
    
    if not models:
        print("No models found")
    else:
        for model in models:
            name = model['ModelName']
            print(f"\nModel: {name}")
            
            if not dry_run:
                delete_model(name)
    
    # Cleanup S3
    if s3_bucket:
        print(f"\n3. Cleaning up S3 bucket: {s3_bucket}")
        
        prefixes = [
            'datasets/',
            'models/',
            'tuning/',
            'checkpoints/',
            'code/'
        ]
        
        if not dry_run:
            cleanup_s3(s3_bucket, prefixes)
        else:
            print(f"Would delete objects with prefixes: {prefixes}")
    
    print("\n" + "="*60)
    if dry_run:
        print("Dry run complete. Run without --dry-run to actually delete resources.")
    else:
        print("Cleanup complete!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Cleanup SageMaker resources')
    parser.add_argument('--endpoint-name', type=str, 
                       help='Specific endpoint to delete (if not specified, deletes all)')
    parser.add_argument('--s3-bucket', type=str, 
                       help='S3 bucket to clean up')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be deleted without actually deleting')
    parser.add_argument('--list-only', action='store_true',
                       help='Only list resources without deleting')
    
    args = parser.parse_args()
    
    if args.list_only:
        print("\nActive Endpoints:")
        print("-" * 60)
        endpoints = list_endpoints()
        for ep in endpoints:
            print(f"  {ep['EndpointName']} - {ep['EndpointStatus']}")
        
        print("\nModels:")
        print("-" * 60)
        models = list_models()
        for model in models:
            print(f"  {model['ModelName']}")
        
        print("\nTo delete resources, run:")
        print("  python scripts/cleanup.py --endpoint-name <name> --s3-bucket <bucket>")
        print("  python scripts/cleanup.py --dry-run  # See what would be deleted")
    
    else:
        cleanup_all(
            endpoint_name=args.endpoint_name,
            s3_bucket=args.s3_bucket,
            dry_run=args.dry_run
        )


if __name__ == '__main__':
    main()
