import argparse
import boto3
import json
import numpy as np
from PIL import Image
import io
import time
import os


# CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']


def create_test_image(image_path=None):
    """Create or load a test image."""
    if image_path and os.path.exists(image_path):
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    else:
        # Create a random test image
        print("Creating random test image (32x32)...")
        random_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        return Image.fromarray(random_array)


def invoke_endpoint(endpoint_name, image, region='us-east-1'):
    """Invoke SageMaker endpoint with an image."""
    
    runtime_client = boto3.client('sagemaker-runtime', region_name=region)
    
    # Convert image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Invoke endpoint
    start_time = time.time()
    
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/x-image',
        Body=img_byte_arr
    )
    
    inference_time = (time.time() - start_time) * 1000  # Convert to ms
    
    # Parse response
    result = json.loads(response['Body'].read().decode())
    
    return result, inference_time


def test_endpoint(endpoint_name, image_path=None, num_tests=5):
    """Test the deployed endpoint."""
    
    # Get region from environment or use default
    region = os.environ.get('AWS_REGION', 'us-east-1')
    
    print(f"Testing endpoint: {endpoint_name}")
    print(f"Region: {region}")
    print(f"Number of test requests: {num_tests}\n")
    
    # Create test image
    test_image = create_test_image(image_path)
    print(f"Test image size: {test_image.size}")
    print(f"Test image mode: {test_image.mode}\n")
    
    # Run multiple tests
    inference_times = []
    
    for i in range(num_tests):
        try:
            result, inference_time = invoke_endpoint(endpoint_name, test_image, region)
            inference_times.append(inference_time)
            
            print(f"Test {i+1}/{num_tests}:")
            print(f"  Predicted class: {result['predicted_class']}")
            print(f"  Class name: {result['class_name']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  Inference time: {inference_time:.2f} ms")
            
            if i == 0:
                print(f"  Top 3 predictions:")
                probs = result['probabilities']
                top_3_indices = np.argsort(probs)[-3:][::-1]
                for idx in top_3_indices:
                    print(f"    {CLASS_NAMES[idx]}: {probs[idx]:.4f}")
            print()
            
        except Exception as e:
            print(f"Error in test {i+1}: {e}\n")
    
    # Print statistics
    if inference_times:
        print("Performance Statistics:")
        print(f"  Average inference time: {np.mean(inference_times):.2f} ms")
        print(f"  Min inference time: {np.min(inference_times):.2f} ms")
        print(f"  Max inference time: {np.max(inference_times):.2f} ms")
        print(f"  Std deviation: {np.std(inference_times):.2f} ms")


def load_test_from_s3(s3_bucket, s3_key):
    """Load test image from S3."""
    s3_client = boto3.client('s3')
    
    response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
    image_bytes = response['Body'].read()
    
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image


def main():
    parser = argparse.ArgumentParser(description='Test SageMaker endpoint')
    parser.add_argument('--endpoint-name', type=str, required=True,
                       help='Name of the SageMaker endpoint')
    parser.add_argument('--image-path', type=str, help='Path to test image')
    parser.add_argument('--s3-bucket', type=str, help='S3 bucket for test image')
    parser.add_argument('--s3-key', type=str, help='S3 key for test image')
    parser.add_argument('--num-tests', type=int, default=5,
                       help='Number of test requests')
    parser.add_argument('--region', type=str, help='AWS region')
    
    args = parser.parse_args()
    
    # Set region if provided
    if args.region:
        os.environ['AWS_REGION'] = args.region
    
    # Load test image
    if args.s3_bucket and args.s3_key:
        print(f"Loading test image from S3: s3://{args.s3_bucket}/{args.s3_key}")
        test_image = load_test_from_s3(args.s3_bucket, args.s3_key)
        test_image.save('test_image.png')
        image_path = 'test_image.png'
    else:
        image_path = args.image_path
    
    # Test endpoint
    test_endpoint(
        endpoint_name=args.endpoint_name,
        image_path=image_path,
        num_tests=args.num_tests
    )
    
    print("\nTo delete the endpoint when done:")
    print(f"  aws sagemaker delete-endpoint --endpoint-name {args.endpoint_name}")


if __name__ == '__main__':
    main()
