import argparse
import json
import logging
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.models as models


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class ResNetClassifier(nn.Module):
    """ResNet-based image classifier."""
    
    def __init__(self, num_classes=10, pretrained=False):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        # Modify first conv layer for CIFAR-10 (32x32 images)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        # Modify final layer for num_classes
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)


def setup_distributed():
    """Initialize distributed training environment."""
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        
        logger.info(f"Distributed training initialized: rank={rank}, world_size={world_size}")
        return True, rank, world_size, local_rank
    else:
        logger.info("Single GPU training")
        return False, 0, 1, 0


def load_data(data_dir, batch_size, is_distributed, world_size, rank):
    """Load training and validation datasets."""
    train_path = os.path.join(data_dir, 'train.pt')
    val_path = os.path.join(data_dir, 'validation.pt')
    
    logger.info(f"Loading data from {data_dir}")
    
    train_dataset = torch.load(train_path)
    val_dataset = torch.load(val_path)
    
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_sampler


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, rank):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 50 == 0 and rank == 0:
            logger.info(f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | '
                       f'Loss: {loss.item():.3f} | Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    val_loss = val_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def save_model(model, model_dir, is_distributed):
    """Save model to SageMaker model directory."""
    if is_distributed:
        model_to_save = model.module
    else:
        model_to_save = model
    
    model_path = os.path.join(model_dir, 'model.pth')
    torch.save(model_to_save.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")


def main():
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--num-classes', type=int, default=10)
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', './data'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))
    
    args = parser.parse_args()
    
    # Setup distributed training
    is_distributed, rank, world_size, local_rank = setup_distributed()
    
    # Set device
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, train_sampler = load_data(
        args.train, args.batch_size, is_distributed, world_size, rank
    )
    
    # Create model
    model = ResNetClassifier(num_classes=args.num_classes)
    model = model.to(device)
    
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        if is_distributed and train_sampler:
            train_sampler.set_epoch(epoch)
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, rank
        )
        
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        if rank == 0:
            logger.info(f'Epoch {epoch}/{args.epochs}:')
            logger.info(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            logger.info(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            
            # Log metrics for CloudWatch
            print(f'#quality_metric: epoch={epoch}, train_loss={train_loss:.4f}, '
                  f'train_acc={train_acc:.2f}, val_loss={val_loss:.4f}, val_acc={val_acc:.2f}')
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_model(model, args.model_dir, is_distributed)
                logger.info(f'New best validation accuracy: {best_val_acc:.2f}%')
    
    if rank == 0:
        logger.info(f'Training complete! Best validation accuracy: {best_val_acc:.2f}%')
        
        # Save final metrics
        metrics = {
            'best_val_acc': best_val_acc,
            'final_train_loss': train_loss,
            'final_val_loss': val_loss
        }
        
        metrics_path = os.path.join(args.output_data_dir, 'metrics.json')
        os.makedirs(args.output_data_dir, exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
    
    if is_distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
