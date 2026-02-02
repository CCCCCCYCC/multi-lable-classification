import math
import numpy as np
import torch
from accuracy import AveragePrecisionMeter
from dataset import MultiSceneDataModule


class EnhancedMLTrainer:
    def __init__(self, model_type, data_dir, num_classes=19, batch_size=64, threshold=0.7):
        self.model_type = model_type
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.threshold = threshold  # Fixed threshold

        # Initialize data loaders
        MultiSceneClean = MultiSceneDataModule(data_dir=data_dir)
        MultiSceneClean.setup(stage='fit')
        self.train_dataloader = MultiSceneClean.train_dataloader()
        self.val_dataloader = MultiSceneClean.val_dataloader()

        # Initialize model
        self._init_model()

        # Initialize AP Meter
        self.ap_meter = AveragePrecisionMeter()

    def _init_model(self):
        """Initialize model"""
        if self.model_type == "xgboost":
            from xgboost import XGBClassifier
            base_model = XGBClassifier(
                booster="gbtree",
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                reg_alpha=1.0,
                reg_lambda=1.0,
                tree_method="gpu_hist" if torch.cuda.is_available() else "auto",
                use_label_encoder=False,
                eval_metric="logloss"
            )
            from sklearn.multioutput import MultiOutputClassifier
            self.model = MultiOutputClassifier(base_model)

        elif self.model_type == "rf":
            from sklearn.ensemble import RandomForestClassifier
            base_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                class_weight="balanced_subsample",
                n_jobs=-1
            )
            from sklearn.multioutput import MultiOutputClassifier
            self.model = MultiOutputClassifier(base_model)

        elif self.model_type == "svm":
            from sklearn.svm import LinearSVC
            from sklearn.multiclass import OneVsRestClassifier
            self.model = OneVsRestClassifier(
                LinearSVC(
                    dual=False,  # Recommended False when n_samples > n_features
                    max_iter=1000,  # Increase iterations to ensure convergence
                    random_state=42,
                    class_weight='balanced'  # Handle class imbalance
                ),
                n_jobs=2
            )

    def _load_data(self, stage='train'):
        """Load data"""
        dataloader = {
            'train': self.train_dataloader,
            'test': self.val_dataloader
        }[stage]

        features, labels = [], []
        for images, batch_labels in dataloader:
            batch_features = images.numpy().reshape(images.shape[0], -1)
            features.append(batch_features)
            labels.append(batch_labels.numpy())

        X = np.concatenate(features, axis=0)
        y = np.concatenate(labels, axis=0)

        return X, y

    def train(self):
        """Enhanced training process"""
        X_train, y_train = self._load_data('train')
        print(f"\n{'=' * 40}\nTraining {self.model_type.upper()} without Data Balancing\n{'=' * 40}")
        self.model.fit(X_train, y_train)

    def evaluate(self, X, y, stage='Validation'):
        """Evaluate using raw scores, avoid passing binary predictions"""
        try:
            # Try to get probabilities
            y_proba = self.model.predict_proba(X)

            # Correctly extract probabilities for XGBoost/RF (each label's result is an (n_samples, 2) array)
            y_scores = []
            for class_probs in y_proba:
                # Take probability of positive class (second column for each label)
                y_scores.append(class_probs[:, 1])

            # Combine into (n_samples, n_classes) array
            y_scores = np.stack(y_scores, axis=1)

        except AttributeError:
            # If model doesn't support predict_proba, use decision function
            y_scores = self.model.decision_function(X)

        # Convert to tensor and add to meter
        scores_tensor = torch.from_numpy(y_scores).float()
        targets_tensor = torch.from_numpy(y).long()

        self.ap_meter.reset()
        self.ap_meter.add(scores_tensor, targets_tensor)

        # Calculate metrics
        ap_values = self.ap_meter.value().numpy()
        mAP = float(np.nanmean(ap_values))

        # Overall metrics (OP/OR/OF1, CP/CR/CF1, EP/ER/EF1)
        OP, OR, OF1, CP, CR, CF1, EP, ER, EF1 = self.ap_meter.overall()

        # Print in target format
        print(f"{stage} Report - {self.model_type.upper()}")
        print("=" * 40)
        print(f"OP: {OP:.4f}  OR: {OR:.4f}  OF1: {OF1:.4f}")
        print(f"CP: {CP:.4f}  CR: {CR:.4f}  CF1: {CF1:.4f}")
        print(f"EP: {EP:.4f}  ER: {ER:.4f}  EF1: {EF1:.4f}")
        print(f"mAP: {mAP:.4f}")

        # Print per-class AP
        print("Class AP:")
        for idx, ap in enumerate(ap_values):
            print(f"{idx}: {ap:.4f}")

        return {
            "OP": round(OP, 4),
            "OR": round(OR, 4),
            "OF1": round(OF1, 4),
            "CP": round(CP, 4),
            "CR": round(CR, 4),
            "CF1": round(CF1, 4),
            "EP": round(EP, 4),
            "ER": round(ER, 4),
            "EF1": round(EF1, 4),
            "mAP": round(mAP, 4),
            "class_ap": {str(i): round(ap, 4) for i, ap in enumerate(ap_values)}
        }


if __name__ == "__main__":
    config = {
        "model_type": "xgboost",  # Can be "rf" or "xgboost" here
        "data_dir": "/root/RSMLR_images",
        "num_classes": 18,
        "batch_size": 512,
        "threshold": 0.5  # Fixed threshold
    }

    trainer = EnhancedMLTrainer(**config)
    trainer.train()

    # Final evaluation
    X_test, y_test = trainer._load_data('test')
    results = trainer.evaluate(X_test, y_test, stage='Final Test')