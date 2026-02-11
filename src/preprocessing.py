import argparse
import os
import boto3
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import pickle


class DataPreprocessor:
    """Handles data preprocessing and augmentation for image classification."""
    
    def __init__(self, dataset_name='cifar10', s3_bucket=None):
        self.dataset_name = dataset_name
        self.s3_bucket = s3_bucket
        self.s3_client = boto3.client('s3')
        
    def get_transforms(self, train=True):
        """Define data augmentation and normalization transforms."""
        if train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    
    def download_dataset(self, data_dir='./data'):
        """Download and prepare dataset."""
        os.makedirs(data_dir, exist_ok=True)
        
        print(f"Downloading {self.dataset_name} dataset...")
        
        if self.dataset_name == 'cifar10':
            train_dataset = torchvision.datasets.CIFAR10(
                root=data_dir,
                train=True,
                download=True,
                transform=self.get_transforms(train=True)
            )
            
            test_dataset = torchvision.datasets.CIFAR10(
                root=data_dir,
                train=False,
                download=True,
                transform=self.get_transforms(train=False)
            )
            
            # Split train into train and validation
            train_size = int(0.9 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(
                train_dataset, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            return train_dataset, val_dataset, test_dataset
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
    
    def save_to_s3(self, dataset, split_name, data_dir='./data'):
        """Save processed dataset to S3."""
        if not self.s3_bucket:
            raise ValueError("S3 bucket not specified")
        
        print(f"Saving {split_name} split to S3...")
        
        # Save dataset locally first
        local_path = os.path.join(data_dir, f'{split_name}.pt')
        torch.save(dataset, local_path)
        
        # Upload to S3
        s3_key = f'datasets/{self.dataset_name}/{split_name}.pt'
        self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
        
        print(f"Uploaded to s3://{self.s3_bucket}/{s3_key}")
        
        return f's3://{self.s3_bucket}/{s3_key}'
    
    def process_and_upload(self):
        """Complete preprocessing pipeline."""
        train_dataset, val_dataset, test_dataset = self.download_dataset()
        
        print(f"\nDataset Statistics:")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        # Upload to S3
        train_s3_path = self.save_to_s3(train_dataset, 'train')
        val_s3_path = self.save_to_s3(val_dataset, 'validation')
        test_s3_path = self.save_to_s3(test_dataset, 'test')
        
        # Save metadata
        metadata = {
            'dataset': self.dataset_name,
            'train_path': train_s3_path,
            'val_path': val_s3_path,
            'test_path': test_s3_path,
            'num_classes': 10,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset)
        }
        
        metadata_path = './data/metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        s3_metadata_key = f'datasets/{self.dataset_name}/metadata.pkl'
        self.s3_client.upload_file(metadata_path, self.s3_bucket, s3_metadata_key)
        
        print(f"\nPreprocessing complete!")
        print(f"Metadata saved to s3://{self.s3_bucket}/{s3_metadata_key}")
        
        return metadata


def main():
    parser = argparse.ArgumentParser(description='Preprocess image dataset for SageMaker')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('--s3-bucket', type=str, required=True, help='S3 bucket name')
    
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor(dataset_name=args.dataset, s3_bucket=args.s3_bucket)
    metadata = preprocessor.process_and_upload()
    
    print("\nNext steps:")
    print("1. Build and push Docker container")
    print("2. Run: python scripts/setup_sagemaker.py --mode train")


if __name__ == '__main__':
    main()
