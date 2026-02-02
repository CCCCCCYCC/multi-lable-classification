from typing import Optional
import os
import torch
from torch import nn
from torchvision import models
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score, MetricCollection
from torchmetrics.classification import MultilabelAccuracy
from transformers import ViTConfig, ViTModel, ViTForImageClassification
from sklearn.metrics import average_precision_score
import numpy as np
from optimized import AsymmetricLossOptimized
from accuracy import AveragePrecisionMeter
from cnn import *
from relationnet import *
from cnn import *
from saff import *
from facnn import *
from kfb import *
import timm
from relationnet import *
from backbone import *
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
class MyCLSModel(pl.LightningModule):
    def __init__(self, model_name, pretrained_status, criterion="BCE", learning_rate=0.05,
                 num_classes=10):  # num_classes should be passed as an argument
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.pretrained = False
        self.criterion = criterion
        nb_classes = num_classes
        pretrained = False
        self.is_deep_learning = model_name in [
            "VGG16", "resnet50", "ViT-b", "ResNeXt50", "ResNext101",
            "vggnet16", "vggnet19", "inceptionv3", "resnet101",
            "resnet152", "squeezenet", "densenet121", "densenet169",
            "shufflenetv2", "mobilenetv2", "resnext50", "resnext101",
            "mnasnet", "lr-vggnet16", "lr-resnet50", "saff", "facnn", "kfb",'ViT-L','ViT-b',"Swin-T"
        ]
        if self.is_deep_learning:
            if model_name == "ViT-L":
                self.model = timm.create_model(
                    'vit_large_patch16_224',  # ViT-Large 预设模型
                    pretrained=False,  # 不使用预训练权重
                    num_classes=18  # 自定义分类头
                )
            elif model_name == "ViT-b":
                self.model = timm.create_model(
                    'vit_base_patch16_224',  # ViT-Base 预设模型
                    pretrained=False,  # 不使用预训练权重
                    num_classes=18  # 自定义分类头
                )
            elif model_name == "Swin-T":

                self.model = timm.create_model(
                    'swin_tiny_patch4_window7_224',
                    pretrained=self.pretrained,
                    num_classes=18  # 修改输出维度为 18
                )
            elif model_name == "Swin-B":
                self.model = timm.create_model('swin_base_patch4_window7_224', pretrained=self.pretrained,num_classes=18)
            elif model_name == "Swin-L":
                self.model = timm.create_model('swin_large_patch4_window7_224', pretrained=self.pretrained)
            elif model_name == 'vggnet16':
                self.model = VGGNetBaseline(models.vgg16(pretrained=pretrained), nb_classes)
            elif model_name == 'vggnet19':
                self.model = VGGNetBaseline(models.vgg19(pretrained=pretrained), nb_classes)
            elif model_name == 'inceptionv3':
                self.model = InceptionBaseline(models.inception_v3(pretrained=pretrained), nb_classes)
            elif model_name == 'resnet50':
                self.model = ResNetBaseline(models.resnet50(pretrained=pretrained), nb_classes)
            elif model_name == 'resnet101':
                self.model = ResNetBaseline(models.resnet101(pretrained=pretrained), nb_classes)
            elif model_name == 'resnet152':
                self.model = ResNetBaseline(models.resnet152(pretrained=pretrained), nb_classes)
            elif model_name == 'squeezenet':
                self.model = SqueezeNetBaseline(models.squeezenet1_0(pretrained=pretrained), nb_classes)
            elif model_name == 'densenet121':
                self.model = DenseNetBaseline(models.densenet121(pretrained=pretrained), nb_classes)
            elif model_name == 'densenet169':
                self.model = DenseNetBaseline(models.densenet169(pretrained=pretrained), nb_classes)
            elif model_name == 'shufflenetv2':
                self.model = ShuffleNetBaseline(models.shufflenet_v2_x1_0(pretrained=pretrained), nb_classes)
            elif model_name == 'mobilenetv2':
                self.model = MobileNetBaseline(models.mobilenet_v2(pretrained=pretrained), nb_classes)
            elif model_name == 'resnext50':
                self.model = ResNeXtBaseline(models.resnext50_32x4d(pretrained=pretrained), nb_classes)
            elif model_name == 'resnext101':
                self.model = ResNeXtBaseline(models.resnext101_32x8d(pretrained=pretrained), nb_classes)
            elif model_name == 'mnasnet':
                self.model = MNASNetBaseline(models.mnasnet1_0(pretrained=pretrained), nb_classes)
            elif model_name == 'lr-vggnet16':
                self.model = RelationNet('vggnet16', nb_classes, num_moda=args.nb_moda, num_units=args.nb_units)
            elif model_name == 'lr-resnet50':
                self.model = RelationNet('resnet50', nb_classes, num_moda=args.nb_moda, num_units=args.nb_units)

            elif model_name == 'saff':
                self.model = SAFF(models.vgg16(pretrained=pretrained), nb_classes, 256)
            elif model_name == 'facnn':
                self.model = FACNN(models.vgg16(pretrained=pretrained), nb_classes)
            elif model_name == 'kfb':
                self.model = KFB(models.vgg16(pretrained=pretrained), nb_classes)
            elif self.model_name == 'ResNeXt50':  # ResNeXt
                self.model = models.resnext50_32x4d(pretrained=self.pretrained)
                num_ftrs = self.model.fc.in_features
                self.model.fc = nn.Linear(num_ftrs, num_classes)
            elif self.model_name == 'ResNeXt101':  # ResNext101
                self.model = models.resnext101_32x8d(pretrained=self.pretrained)  # 使用预训练权重或不使用
                num_ftrs = self.model.fc.in_features
                self.model.fc = nn.Linear(num_ftrs, num_classes)

        else:
            if model_name == 'svm':
                self.model = MultiOutputClassifier(
                    SVC(random_state=0, tol=1e-5, max_iter=100000, verbose=1),
                    n_jobs=-1  # 使用所有 CPU 核心
                )
            elif model_name == 'xgboost':
                self.model = MultiOutputClassifier(
                    XGBClassifier(booster='gbtree', n_jobs=100, n_estimators=200, verbosity=1, use_label_encoder=False,
                                  gpu_id=0))
            elif model_name == 'rf':
                self.model = MultiOutputClassifier(RandomForestClassifier(random_state=0, n_estimators=200, verbose=1))



        # self.criterion = nn.CrossEntropyLoss()
        if self.criterion == "BCE":
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.criterion == "ASL":
            self.criterion = AsymmetricLossOptimized()

        # self.train_accuracy = Accuracy(task="multilabel", num_labels=num_classes)
        self.train_accuracy = MultilabelAccuracy(num_labels=num_classes)
        self.val_accuracy = MultilabelAccuracy(num_labels=num_classes)
        self.test_accuracy = MultilabelAccuracy(num_labels=num_classes)

        self.average_precision_meter = AveragePrecisionMeter()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        #x, y = batch
        #if self.model_name == 'ViT-b':
        #    outputs = self(x)
        #    logits = outputs.logits
        #else:
        #    logits = self(x)
        #loss = self.criterion(logits, y)
        #acc = self.train_accuracy(logits, y)
        if self.is_deep_learning:
            x, y = batch
            logits = self(x)
            loss = self.criterion(logits, y)
            acc = self.train_accuracy(logits, y)
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_accuracy", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss
        else:
            # 传统模型需单独训练（不在PyTorch Lightning中处理）
            return None

        # 添加样本到评估类中
        # self.average_precision_meter.add(logits, y)  # 不需要在训练步骤中更新 mAP

        # self.log("train_loss", loss)
        # self.log("train_accuracy", acc, on_step=False, on_epoch=True)
        #self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log("train_accuracy", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        #return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.model_name == 'ViT-b11':
            outputs = self(x)
            logits = outputs.logits
        else:
            logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.val_accuracy(logits, y)

        # 添加样本到评估类中
        self.average_precision_meter.add(logits, y)

        # self.log("val_loss", loss)
        # self.log("val_accuracy", acc)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuracy", acc, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        if self.model_name == 'ViT-b':
            outputs = self(x)
            logits = outputs.logits
        else:
            logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.test_accuracy(logits, y)

        # 添加样本到评估类中
        self.average_precision_meter.add(logits, y)

        # self.log("test_accuracy", acc)
        # self.log("test_loss", loss)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        # 计算并记录性能指标
        ap_values = self.average_precision_meter.value()
        self.log('val_mAP', ap_values.mean() * 100, prog_bar=True, logger=True)

        for i, ap_value in enumerate(ap_values):
            self.log(f'val_AP_class_{i}', ap_value, prog_bar=True, logger=True)
        # self.log('val_per-class AP:', ap_values*100, prog_bar=True, logger=True)

        op, or_, of1, cp, cr, cf1, ep, er, ef1 = self.average_precision_meter.overall()
        self.log('test_OP', op * 100, prog_bar=True, logger=True)
        self.log('test_OR', or_ * 100, prog_bar=True, logger=True)
        self.log('test_OF1', of1 * 100, prog_bar=True, logger=True)
        self.log('test_CP', cp * 100, prog_bar=True, logger=True)
        self.log('test_CR', cr * 100, prog_bar=True, logger=True)
        self.log('test_CF1', cf1 * 100, prog_bar=True, logger=True)
        self.log('test_EP', ep * 100, prog_bar=True, logger=True)
        self.log('test_ER', er * 100, prog_bar=True, logger=True)
        self.log('test_EF1', ef1 * 100, prog_bar=True, logger=True)

        self.average_precision_meter.reset()

    def on_test_epoch_end(self):
        # 计算并记录性能指标
        ap_values = self.average_precision_meter.value()
        self.log('test_mAP', ap_values.mean() * 100, prog_bar=True, logger=True)

        for i, ap_value in enumerate(ap_values):
            self.log(f'test_AP_class_{i}', ap_value, prog_bar=True, logger=True)
        # self.log('test_per-class AP:', ap_values*100, prog_bar=True, logger=True)

        op, or_, of1, cp, cr, cf1, ep, er, ef1 = self.average_precision_meter.overall()
        self.log('test_OP', op * 100, prog_bar=True, logger=True)
        self.log('test_OR', or_ * 100, prog_bar=True, logger=True)
        self.log('test_OF1', of1 * 100, prog_bar=True, logger=True)
        self.log('test_CP', cp * 100, prog_bar=True, logger=True)
        self.log('test_CR', cr * 100, prog_bar=True, logger=True)
        self.log('test_CF1', cf1 * 100, prog_bar=True, logger=True)
        self.log('test_EP', ep * 100, prog_bar=True, logger=True)
        self.log('test_ER', er * 100, prog_bar=True, logger=True)
        self.log('test_EF1', ef1 * 100, prog_bar=True, logger=True)

        self.average_precision_meter.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,  # 每10个epoch更新一次学习率
            gamma=0.5  # 学习率乘以的因子（比如从0.05降到0.005）
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }