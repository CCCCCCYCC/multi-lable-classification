# predict.py
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import pandas as pd
import os
from model import MyCLSModel
from dataset import MultiSceneDataModule
import pytorch_lightning as pl


class Predictor:
    def __init__(self, ckpt_path, data_dir, model_name='ResNeXt101', threshold=0.5):
        """
        Initialize predictor
        :param ckpt_path: Path to model checkpoint
        :param data_dir: Data directory (for retrieving class information)
        :param model_name: Model name (must match training configuration)
        :param threshold: Classification threshold
        """
        # Load data module to get class information
        self.data_module = MultiSceneDataModule(data_dir=data_dir)
        self.data_module.setup(stage='fit')

        # Load model
        self.model = MyCLSModel.load_from_checkpoint(
            ckpt_path,
            model_name=model_name,
            pretrained_status=False,  # No need for pretrained weights during inference
            criterion="BCE",  # Must match training configuration
            num_classes=self.data_module.num_classes
        )
        self.model.eval()
        self.model.freeze()

        # Set up preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.classes = self.data_module.classes
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict_single(self, image_path):
        """
        Predict on a single image
        :param image_path: Path to input image
        :return: Dictionary containing prediction results
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            if self.model.model_name == 'ViT-b':
                outputs = self.model.model(tensor)
                logits = outputs.logits
            else:
                logits = self.model.model(tensor)

        # Post-processing
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        predictions = (probs > self.threshold).astype(int)

        return {
            'image_path': image_path,
            'probabilities': probs,
            'predictions': predictions,
            'labels': [self.classes[i] for i in np.where(predictions == 1)[0]]
        }

    def predict_batch(self, image_dir):
        """
        Batch prediction
        :param image_dir: Directory containing images for prediction
        :return: List of prediction results
        """
        results = []
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

        for filename in image_files:
            image_path = os.path.join(image_dir, filename)
            try:
                result = self.predict_single(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

        return results

    def save_predictions(self, results, output_file):
        """
        Save prediction results to CSV file
        :param results: List of prediction results
        :param output_file: Output file path
        """
        df = pd.DataFrame([{
            'image_path': res['image_path'],
            'labels': ', '.join(res['labels']),
            **{cls: prob for cls, prob in zip(self.classes, res['probabilities'])}
        } for res in results])

        df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")


if __name__ == '__main__':
    # Usage example
    predictor = Predictor(
        ckpt_path="/root/autodl-tmp/RSMLR-main/ckpt/resnext101-True-BCE-epoch=43-val_mAP=70.6157.ckpt",  # Replace with actual checkpoint path
        data_dir="/root/autodl-tmp/RSMLR_images",
        model_name="resnext101",
        threshold=0.5
    )

    # Single image prediction
    # single_result = predictor.predict_single("path/to/your/image.jpg")
    # print("Single prediction:")
    # print(f"Image: {single_result['image_path']}")
    # print(f"Labels: {', '.join(single_result['labels'])}")
    # print("Probabilities:", dict(zip(predictor.classes, single_result['probabilities'])))

    # Batch prediction
    batch_results = predictor.predict_batch("/root/autodl-tmp/RSMLR_images/Images")
    predictor.save_predictions(batch_results, "predictions.csv")