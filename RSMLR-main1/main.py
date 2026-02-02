import torch  # Add torch import
import torch.optim.lr_scheduler as lr_scheduler  # Add lr_scheduler import
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts  # Add scheduler import

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import RichProgressBar
import pytorch_lightning as pl
from dataset import MultiSceneDataModule
from model import MyCLSModel
import subprocess
import os

import torchvision
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.callbacks import LearningRateMonitor  # Add learning rate monitor import


#pl.utilities.distributed.log.setLevel("ERROR")
model_name = "svm"  # Select model [VGG16, resnet50, ViT-b, ResNeXt50, ResNext101]
print("-----------------------")
print(model_name)
print("-----------------------")
pretrained_status = True  # Whether to use pretrained weights [True, False]
criterion = "BCE"  # [BCE, ASL]

if model_name == "vgg19":
    result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True,
                            text=True)
    output = result.stdout
    for line in output.splitlines():
        if '=' in line:
            var, value = line.split('=', 1)
            os.environ[var] = value

MultiSceneClean = MultiSceneDataModule(data_dir=r'/root/autodl-tmp/RSMLR_images')
MultiSceneClean.setup(stage='fit')

# Get some random training images
train_dataloader = MultiSceneClean.train_dataloader()
images, labels = next(iter(train_dataloader))


# Since it's multi-label classification, we need to modify how labels are displayed
def print_labels(labels):
    # Assuming we have a class list, which should be defined in your data module or somewhere
    classes = MultiSceneClean.classes
    batch_labels = labels.numpy()
    for i in range(batch_labels.shape[0]):
        # For each sample, find the positive labels
        label_indexes = np.where(batch_labels[i] == 1)[0]
        label_names = [classes[idx] for idx in label_indexes]
        print('Image', i + 1, ':', ' '.join(label_names))


# Print labels
print_labels(labels)
num_cls = MultiSceneClean.num_classes

# Use model with scheduler
model = MyCLSModel(model_name, pretrained_status, criterion, learning_rate=0.02, num_classes=num_cls)

checkpoint_dir = 'ckpt/'
from pytorch_lightning.callbacks import Callback


class CustomLoggerCallback(pl.Callback):
    def __init__(self):
        self.last_printed_epoch = -1

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        # Ensure only printing once per epoch
        if epoch == self.last_printed_epoch:
            return

        self.last_printed_epoch = epoch

        # Skip first epoch as it may lack some metrics
        if epoch == 0:
            return

        # Get learning rate
        scheduler = pl_module.lr_schedulers()
        if scheduler and hasattr(scheduler, 'get_last_lr'):  # Use scheduler object directly
            lr = scheduler.get_last_lr()[0]
        else:
            lr = pl_module.learning_rate

        # Clear screen and print formatted output
        print("\033c", end="")  # ANSI clear screen
        print("\n" + "-" * 50)
        print(
            f"Epoch: {epoch + 1}/{trainer.max_epochs}\n"
            f"Train Loss: {metrics.get('train_loss_epoch', 0):.3f} | Train Accuracy: {metrics.get('train_accuracy_epoch', 0):.3f}\n"
            f"Val   Loss: {metrics.get('val_loss', 0):.3f} | Val   Accuracy: {metrics.get('val_accuracy', 0):.3f}\n"
            f"Val   mAP : {metrics.get('val_mAP', 0) :.3f}%\n"
            f"Learning Rate: {lr:.6f}"
        )

        print("\nClass-wise AP:")
        # Collect AP for all classes
        aps = []
        for i in range(num_cls):
            ap_val = metrics.get(f'val_AP_class_{i}', 0)
            aps.append(ap_val)

        # Print 4 classes per line
        for i in range(0, num_cls, 4):
            # Build content for this line
            line = []
            for j in range(i, min(i + 4, num_cls)):
                line.append(f"Class {j}: {aps[j]:.3f}")
            print(" | ".join(line))

        # Check and display test metrics (if exist)
        test_keys = [k for k in metrics.keys() if k.startswith("test_")]
        if test_keys:
            print("\nTest Metrics:")
            test_metrics = []
            for key in ["OP", "OR", "OF1", "CP", "CR", "CF1", "EP", "ER", "EF1"]:
                metric_key = f"test_{key}"
                if metric_key in metrics:
                    value = metrics[metric_key]
                    test_metrics.append(f"{key}: {value:.3f}")

            # Display 3 metrics per line
            for i in range(0, len(test_metrics), 3):
                print(" | ".join(test_metrics[i:i + 3]))

        print("-" * 50)

    def on_train_epoch_end(self, trainer, pl_module):
        # Print same information at training end to ensure complete display
        self.on_validation_epoch_end(trainer, pl_module)

# Set up model checkpoint callback to save top 3 models with highest mAP
checkpoint_callback = ModelCheckpoint(
    monitor='val_mAP',
    dirpath=checkpoint_dir,
    filename=f'{model_name}-{pretrained_status}-{criterion}-{{epoch:02d}}-{{val_mAP:.4f}}',
    save_top_k=3,
    mode='max',
)

# Set up early stopping callback, stop training if validation loss doesn't improve for 10 consecutive epochs
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.00,
    patience=10,
    verbose=False,
    mode='min'
)

# Learning rate monitor callback (displayed in TensorBoard)
lr_monitor = LearningRateMonitor(logging_interval='epoch')

trainer = pl.Trainer(
    max_epochs=100,
    accelerator="auto",
    devices=[0],
    logger=TensorBoardLogger(save_dir=f"logs/train/{model_name}"),
    callbacks=[
        CustomLoggerCallback(),
        checkpoint_callback,
        lr_monitor,  # Add learning rate monitor
        #RichProgressBar(),  # Progress bar
    ],
)

train_dataloader = MultiSceneClean.train_dataloader()
val_dataloader = MultiSceneClean.val_dataloader()

# Train model
trainer.fit(model, train_dataloader, val_dataloader)

print("-----------------------")
print(model_name)
print("-----------------------")