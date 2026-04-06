#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Best Model for Kanji Recognition
Based on experimental results: HOG + Data Augmentation

認識率:
  30サンプル/クラス (450サンプル、135テスト): 69.63%
  全サンプル (2415サンプル、725テスト、拡張なし): 97.38% ★最高精度

Optimal Settings:
- Feature extraction: HOG only
- Data augmentation: Enabled (5x)
- Split method: projection
- Pyramid levels: 5
- PNN sigma method: adaptive

Usage:
    from kanji_best_model import KanjiBestModel

    # Training
    model = KanjiBestModel()
    model.train(train_images, train_labels)

    # Prediction
    prediction = model.predict(test_image)
    accuracy, predictions = model.evaluate(test_images, test_labels)
"""

import numpy as np
import cv2
import os
import pickle
from hierarchical_recognizer_optimized import HierarchicalPatternRecognizer


class KanjiBestModel:
    """
    Optimized Kanji Recognition Model

    This model uses the best configuration found through experimentation:
    - HOG features (no Gabor to avoid overfitting and reduce computation time)
    - Data augmentation (5x) for improved generalization
    - Projection-based left-right splitting
    - 5-level hierarchical pyramid
    - Adaptive PNN sigma determination

    Performance:
    - Accuracy: 69.63% (15-class kanji recognition)
    - Training time: ~5 seconds (315 samples)
    - Prediction time: ~0.37 seconds (135 samples)
    """

    def __init__(self):
        """Initialize the best model with optimal parameters"""
        self.recognizer = HierarchicalPatternRecognizer(
            num_pyramid_levels=5,
            split_method='projection',
            pnn_sigma_method='adaptive',
            use_enhanced_features=True,
            feature_types=['hog'],  # HOG only - best performance
            use_data_augmentation=True,
            augmentation_factor=5
        )

        self.trained = False

    def train(self, images, labels):
        """
        Train the model

        Parameters:
        -----------
        images : np.ndarray
            Training images (N, H, W) where H=W=64
        labels : np.ndarray
            Training labels (N,)

        Returns:
        --------
        training_time : float
            Time taken for training in seconds
        """
        import time

        print("="*80)
        print("Best Kanji Recognition Model - Training")
        print("="*80)
        print(f"Configuration:")
        print(f"  - Features: HOG only")
        print(f"  - Data augmentation: 5x")
        print(f"  - Split method: projection")
        print(f"  - Pyramid levels: 5")
        print(f"  - Samples: {len(images)}")
        print(f"  - Classes: {len(np.unique(labels))}")
        print()

        start_time = time.time()
        self.recognizer.train(images, labels)
        training_time = time.time() - start_time

        self.trained = True

        print(f"\nTraining completed in {training_time:.2f} seconds")
        print("="*80)

        return training_time

    def predict(self, image):
        """
        Predict the class of a single image

        Parameters:
        -----------
        image : np.ndarray
            Input image (H, W) where H=W=64

        Returns:
        --------
        prediction : int
            Predicted class label
        """
        if not self.trained:
            raise RuntimeError("Model not trained yet. Call train() first.")

        return self.recognizer.predict(image)

    def predict_proba(self, image):
        """
        Get class probabilities for a single image

        Parameters:
        -----------
        image : np.ndarray
            Input image (H, W) where H=W=64

        Returns:
        --------
        probabilities : np.ndarray
            Class probabilities
        """
        if not self.trained:
            raise RuntimeError("Model not trained yet. Call train() first.")

        return self.recognizer.predict_proba(image)

    def evaluate(self, images, labels):
        """
        Evaluate the model on test data

        Parameters:
        -----------
        images : np.ndarray
            Test images (N, H, W)
        labels : np.ndarray
            Test labels (N,)

        Returns:
        --------
        accuracy : float
            Classification accuracy (%)
        predictions : np.ndarray
            Predicted labels
        """
        if not self.trained:
            raise RuntimeError("Model not trained yet. Call train() first.")

        return self.recognizer.evaluate(images, labels)

    def save(self, filepath):
        """
        Save the trained model to file

        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        if not self.trained:
            raise RuntimeError("Model not trained yet. Call train() first.")

        with open(filepath, 'wb') as f:
            pickle.dump(self.recognizer, f)

        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """
        Load a trained model from file

        Parameters:
        -----------
        filepath : str
            Path to the saved model
        """
        with open(filepath, 'rb') as f:
            self.recognizer = pickle.load(f)

        self.trained = True
        print(f"Model loaded from {filepath}")


def load_etl8b_data(base_path, target_classes, max_samples=30):
    """
    Load ETL8B dataset

    Parameters:
    -----------
    base_path : str
        Path to ETL8B-img-full directory
    target_classes : list
        List of class hex codes (e.g., ['3a2c', '3a2e', ...])
    max_samples : int
        Maximum samples per class

    Returns:
    --------
    images : np.ndarray
        Images array (N, 64, 64)
    labels : np.ndarray
        Labels array (N,)
    class_info : dict
        Class information dictionary
    """
    images = []
    labels = []
    class_info = {}

    print("Loading ETL8B data...")
    for class_idx, class_hex in enumerate(target_classes):
        class_dir = os.path.join(base_path, class_hex)
        if not os.path.exists(class_dir):
            print(f"  Warning: {class_dir} not found")
            continue

        image_files = [f for f in os.listdir(class_dir) if f.endswith('.png')]

        if len(image_files) > max_samples:
            np.random.seed(42)
            image_files = np.random.choice(image_files, max_samples, replace=False)

        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if img.shape != (64, 64):
                    img = cv2.resize(img, (64, 64))
                img = img.astype(np.float64) / 255.0
                images.append(img)
                labels.append(class_idx)

        decimal = int(class_hex, 16)
        char = chr(decimal) if decimal < 0x10000 else f'[{decimal}]'
        class_info[class_idx] = {'hex': class_hex, 'char': char}
        print(f"  Class {class_idx} ({class_hex}): {char} - {len([l for l in labels if l == class_idx])} samples")

    print(f"Loaded: {len(images)} samples, {len(target_classes)} classes\n")
    return np.array(images), np.array(labels), class_info


# Example usage and demonstration
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("="*80)
    print("Best Kanji Recognition Model - Demonstration")
    print("="*80)

    # Load data (same 15 classes as experiment)
    target_classes = [
        '3a2c', '3a2e', '3a4e', '3a6e', '3a51',
        '3a60', '3a64', '3a72', '3b4f', '3b6b',
        '4f40', '3c23', '3d26', '3d3b', '3d3e'
    ]

    base_path = "./ETL8B-img-full"
    images, labels, class_info = load_etl8b_data(base_path, target_classes, max_samples=30)

    # Train-test split
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )

    print(f"Training samples: {len(train_images)}")
    print(f"Test samples: {len(test_images)}\n")

    # Create and train model
    model = KanjiBestModel()
    training_time = model.train(train_images, train_labels)

    # Evaluate
    print("\n" + "="*80)
    print("Evaluation")
    print("="*80)

    import time
    start_time = time.time()
    accuracy, predictions = model.evaluate(test_images, test_labels)
    eval_time = time.time() - start_time

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Evaluation time: {eval_time:.2f} seconds")

    # Confusion matrix
    cm = confusion_matrix(test_labels, predictions)

    # Save confusion matrix plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True)
    plt.title(f'Best Model Confusion Matrix\nAccuracy: {accuracy:.2f}%',
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/best_model_confusion_matrix.png', dpi=150, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: results/best_model_confusion_matrix.png")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, predictions,
                                target_names=[class_info[i]['char'] for i in range(len(class_info))]))

    # Save model
    model.save('models/kanji_best_model.pkl')

    print("\n" + "="*80)
    print("Demonstration completed!")
    print("="*80)
    print("\nModel Performance Summary:")
    print(f"  - Accuracy: {accuracy:.2f}%")
    print(f"  - Training time: {training_time:.2f}s")
    print(f"  - Evaluation time: {eval_time:.2f}s")
    print(f"  - Configuration: HOG + Data Augmentation (5x)")
    print("\nModel saved to: models/kanji_best_model.pkl")
    print("Confusion matrix saved to: results/best_model_confusion_matrix.png")
