import os
import yaml
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch

from src.data.data_loader import EmergencyDataLoader
from src.features.feature_engineering import FeatureEngineer
from src.models import EmergencyClassifier
from src.models import EmergencyModelTrainer
from src.data import EmergencyDataset


def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('emergency_detection.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def main():
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Emergency Event Detection System")

    # Load configuration
    config = load_config()

    # Data Loading
    data_loader = EmergencyDataLoader(config['data_paths'])
    combined_data, encoding_info = data_loader.load_and_preprocess_data()

    # Clean data
    combined_data = combined_data.dropna()
    combined_data = combined_data[combined_data['text'].str.strip() != '']

    # Drop timestamp if it exists
    if 'timestamp' in combined_data.columns:
        combined_data = combined_data.drop(columns=['timestamp'])

    # Feature Engineering
    feature_engineer = FeatureEngineer()
    processed_data = feature_engineer.engineer_features(combined_data)

    # Split data
    train_data, test_data = train_test_split(
        processed_data,
        test_size=config['training']['test_size'],
        random_state=42,
        stratify=processed_data['emergency_level_encoded']
    )

    # Reset indices
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    # Create datasets
    train_dataset = EmergencyDataset(
        texts=train_data['text'],
        emergency_level_labels=train_data['emergency_level_encoded'],
        sentiment_labels=train_data['sentiment_encoded'],
        disaster_type_labels=train_data['disaster_type_encoded']
    )

    test_dataset = EmergencyDataset(
        texts=test_data['text'],
        emergency_level_labels=test_data['emergency_level_encoded'],
        sentiment_labels=test_data['sentiment_encoded'],
        disaster_type_labels=test_data['disaster_type_encoded']
    )

    # Model configuration
    model_config = {
        'num_emergency_levels': len(feature_engineer.emergency_level_encoder.classes_),
        'num_disaster_types': len(feature_engineer.disaster_type_encoder.classes_),
        'num_sentiments': len(feature_engineer.sentiment_encoder.classes_)
    }

    # Initialize model
    model = EmergencyClassifier(model_config)

    # Trainer configuration
    trainer_config = config['training'].copy()
    trainer_config.update({
        'label_mappings': feature_engineer.label_mappings
    })

    # Initialize trainer
    model_trainer = EmergencyModelTrainer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=trainer_config
    )

    # Training
    logger.info("Starting model training...")
    model_trainer.train(epochs=config['training']['num_epochs'])

    # Save model
    model_save_path = os.path.join(
        'models/saved_models',
        f'emergency_classifier_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
    )
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    main()