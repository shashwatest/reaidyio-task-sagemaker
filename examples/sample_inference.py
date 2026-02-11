"""
Sample inference script demonstrating how to use the deployed SageMaker endpoint.
"""

import boto3
import json
import base64
from PIL import Image
import io


def predict_from_file(endpoint_name, image_path, region='us-east-1'):
    """
    Make prediction from local image file.
    
    Args:
        endpoint_name: Name of SageMaker endpoint
        image_path: Path to image file
        region: AWS region
    
    Returns:
        dict: Prediction results
    """
    runtime = boto3.client('sagemaker-runtime', region_name=region)
    
    # Load and prepare image
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    # Invoke endpoint
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/x-image',
        Body=image_bytes
    )
    
    # Parse response
    result = json.loads(response['Body'].read().decode())
    return result


def predict_from_url(endpoint_name, image_url, region='us-east-1'):
    """
    Make prediction from image URL.
    
    Args:
        endpoint_name: Name of SageMaker endpoint
        image_url: URL to image
        region: AWS region
    
    Returns:
        dict: Prediction results
    """
    import requests
    
    runtime = boto3.client('sagemaker-runtime', region_name=region)
    
    # Download image
    response = requests.get(image_url)
    image_bytes = response.content
    
    # Invoke endpoint
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/x-image',
        Body=image_bytes
    )
    
    # Parse response
    result = json.loads(response['Body'].read().decode())
    return result


def predict_from_s3(endpoint_name, s3_bucket, s3_key, region='us-east-1'):
    """
    Make prediction from S3 image.
    
    Args:
        endpoint_name: Name of SageMaker endpoint
        s3_bucket: S3 bucket name
        s3_key: S3 object key
        region: AWS region
    
    Returns:
        dict: Prediction results
    """
    s3 = boto3.client('s3', region_name=region)
    runtime = boto3.client('sagemaker-runtime', region_name=region)
    
    # Download from S3
    response = s3.get_object(Bucket=s3_bucket, Key=s3_key)
    image_bytes = response['Body'].read()
    
    # Invoke endpoint
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/x-image',
        Body=image_bytes
    )
    
    # Parse response
    result = json.loads(response['Body'].read().decode())
    return result


def predict_batch(endpoint_name, image_paths, region='us-east-1'):
    """
    Make predictions on multiple images.
    
    Args:
        endpoint_name: Name of SageMaker endpoint
        image_paths: List of image file paths
        region: AWS region
    
    Returns:
        list: List of prediction results
    """
    results = []
    
    for image_path in image_paths:
        try:
            result = predict_from_file(endpoint_name, image_path, region)
            results.append({
                'image': image_path,
                'prediction': result
            })
        except Exception as e:
            results.append({
                'image': image_path,
                'error': str(e)
            })
    
    return results


def predict_with_json(endpoint_name, image_path, region='us-east-1'):
    """
    Make prediction using JSON format with base64 encoded image.
    
    Args:
        endpoint_name: Name of SageMaker endpoint
        image_path: Path to image file
        region: AWS region
    
    Returns:
        dict: Prediction results
    """
    runtime = boto3.client('sagemaker-runtime', region_name=region)
    
    # Load and encode image
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # Create JSON payload
    payload = {
        'image': image_base64
    }
    
    # Invoke endpoint
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(payload)
    )
    
    # Parse response
    result = json.loads(response['Body'].read().decode())
    return result


# Example usage
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Sample inference script')
    parser.add_argument('--endpoint-name', type=str, required=True,
                       help='SageMaker endpoint name')
    parser.add_argument('--image-path', type=str, help='Path to image file')
    parser.add_argument('--image-url', type=str, help='URL to image')
    parser.add_argument('--s3-bucket', type=str, help='S3 bucket name')
    parser.add_argument('--s3-key', type=str, help='S3 object key')
    parser.add_argument('--region', type=str, default='us-east-1',
                       help='AWS region')
    
    args = parser.parse_args()
    
    # Make prediction based on input type
    if args.image_path:
        print(f"Predicting from file: {args.image_path}")
        result = predict_from_file(args.endpoint_name, args.image_path, args.region)
    
    elif args.image_url:
        print(f"Predicting from URL: {args.image_url}")
        result = predict_from_url(args.endpoint_name, args.image_url, args.region)
    
    elif args.s3_bucket and args.s3_key:
        print(f"Predicting from S3: s3://{args.s3_bucket}/{args.s3_key}")
        result = predict_from_s3(args.endpoint_name, args.s3_bucket, 
                                args.s3_key, args.region)
    
    else:
        print("Please specify --image-path, --image-url, or --s3-bucket and --s3-key")
        exit(1)
    
    # Print results
    print("\nPrediction Results:")
    print(f"  Class: {result['class_name']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"\nTop 3 Predictions:")
    
    probs = result['probabilities']
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    import numpy as np
    top_3_indices = np.argsort(probs)[-3:][::-1]
    
    for idx in top_3_indices:
        print(f"  {class_names[idx]}: {probs[idx]:.4f}")
