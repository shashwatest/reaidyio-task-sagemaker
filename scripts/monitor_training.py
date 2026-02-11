import argparse
import boto3
import time
from datetime import datetime


def get_training_job_status(job_name):
    """Get current status of training job."""
    sagemaker_client = boto3.client('sagemaker')
    
    try:
        response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
        return response
    except Exception as e:
        print(f"Error getting job status: {e}")
        return None


def get_cloudwatch_logs(log_group, log_stream, limit=50):
    """Fetch recent CloudWatch logs."""
    logs_client = boto3.client('logs')
    
    try:
        response = logs_client.get_log_events(
            logGroupName=log_group,
            logStreamName=log_stream,
            limit=limit,
            startFromHead=False
        )
        return response['events']
    except Exception as e:
        print(f"Error fetching logs: {e}")
        return []


def monitor_training_job(job_name, refresh_interval=30):
    """Monitor training job progress."""
    
    print(f"Monitoring training job: {job_name}")
    print(f"Refresh interval: {refresh_interval} seconds")
    print("Press Ctrl+C to stop monitoring\n")
    
    sagemaker_client = boto3.client('sagemaker')
    previous_status = None
    
    try:
        while True:
            response = get_training_job_status(job_name)
            
            if not response:
                print("Could not retrieve job status. Retrying...")
                time.sleep(refresh_interval)
                continue
            
            status = response['TrainingJobStatus']
            
            # Print status update if changed
            if status != previous_status:
                print(f"\n{'='*60}")
                print(f"Status Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*60}")
                print(f"Job Status: {status}")
                
                if 'SecondaryStatus' in response:
                    print(f"Secondary Status: {response['SecondaryStatus']}")
                
                if 'TrainingStartTime' in response:
                    start_time = response['TrainingStartTime']
                    elapsed = datetime.now(start_time.tzinfo) - start_time
                    print(f"Elapsed Time: {elapsed}")
                
                if 'BillableTimeInSeconds' in response:
                    billable = response['BillableTimeInSeconds']
                    print(f"Billable Time: {billable} seconds ({billable/3600:.2f} hours)")
                
                if 'ResourceConfig' in response:
                    config = response['ResourceConfig']
                    print(f"Instance Type: {config['InstanceType']}")
                    print(f"Instance Count: {config['InstanceCount']}")
                
                previous_status = status
            
            # Check if job completed
            if status in ['Completed', 'Failed', 'Stopped']:
                print(f"\n{'='*60}")
                print(f"Training job {status.lower()}!")
                print(f"{'='*60}")
                
                if status == 'Completed':
                    if 'ModelArtifacts' in response:
                        print(f"Model artifacts: {response['ModelArtifacts']['S3ModelArtifacts']}")
                    
                    if 'FinalMetricDataList' in response:
                        print("\nFinal Metrics:")
                        for metric in response['FinalMetricDataList']:
                            print(f"  {metric['MetricName']}: {metric['Value']:.4f}")
                
                elif status == 'Failed':
                    if 'FailureReason' in response:
                        print(f"Failure Reason: {response['FailureReason']}")
                
                break
            
            # Wait before next check
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        print(f"Job is still running. Current status: {status}")
        print(f"\nTo check status later:")
        print(f"  aws sagemaker describe-training-job --training-job-name {job_name}")


def list_recent_jobs(max_results=10):
    """List recent training jobs."""
    sagemaker_client = boto3.client('sagemaker')
    
    response = sagemaker_client.list_training_jobs(
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=max_results
    )
    
    print(f"\nRecent Training Jobs (last {max_results}):")
    print(f"{'='*100}")
    print(f"{'Job Name':<50} {'Status':<15} {'Creation Time':<25}")
    print(f"{'='*100}")
    
    for job in response['TrainingJobSummaries']:
        name = job['TrainingJobName']
        status = job['TrainingJobStatus']
        created = job['CreationTime'].strftime('%Y-%m-%d %H:%M:%S')
        print(f"{name:<50} {status:<15} {created:<25}")
    
    return response['TrainingJobSummaries']


def main():
    parser = argparse.ArgumentParser(description='Monitor SageMaker training jobs')
    parser.add_argument('--job-name', type=str, help='Training job name to monitor')
    parser.add_argument('--list', action='store_true', help='List recent training jobs')
    parser.add_argument('--refresh', type=int, default=30, 
                       help='Refresh interval in seconds (default: 30)')
    
    args = parser.parse_args()
    
    if args.list:
        jobs = list_recent_jobs()
        
        if not args.job_name and jobs:
            print("\nTo monitor a job, run:")
            print(f"  python scripts/monitor_training.py --job-name {jobs[0]['TrainingJobName']}")
    
    elif args.job_name:
        monitor_training_job(args.job_name, args.refresh)
    
    else:
        print("Please specify --job-name or --list")
        print("\nExamples:")
        print("  python scripts/monitor_training.py --list")
        print("  python scripts/monitor_training.py --job-name pytorch-image-classifier-2024-01-01-12-00-00-000")


if __name__ == '__main__':
    main()
