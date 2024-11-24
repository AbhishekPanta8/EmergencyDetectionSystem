import torch
from transformers import RobertaTokenizer
import yaml
from src.models.model import EmergencyClassifier
import logging
import json

class EmergencyPredictor:
    def __init__(self, model_path, config_path='config/config.yaml'):
        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Load config
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        
        # Define label mappings
        self.label_mappings = {
            'emergency_level': {
                0: 'HIGH_EMERGENCY',
                1: 'MODERATE_EMERGENCY',
                2: 'LOW_EMERGENCY',
                3: 'NON_EMERGENCY',
                4: 'POTENTIAL_EMERGENCY'
            },
            'disaster_type': {
                0: 'EARTHQUAKE',
                1: 'FIRE',
                2: 'FLOOD',
                3: 'HURRICANE',
                4: 'TORNADO',
                5: 'TSUNAMI',
                6: 'OTHER',
                7: 'STORM',
                8: 'LANDSLIDE',
                9: 'VOLCANIC_ERUPTION'
            },
            'sentiment': {
                0: 'PANIC',
                1: 'FEAR',
                2: 'URGENT',
                3: 'NEUTRAL'
            }
        }
        
        # Load model
        self.model = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

    def load_model(self, model_path):
        try:
            # Define model configuration
            model_config = {
                'num_emergency_levels': len(self.label_mappings['emergency_level']),
                'num_disaster_types': len(self.label_mappings['disaster_type']),
                'num_sentiments': len(self.label_mappings['sentiment'])
            }
            
            # Initialize model
            model = EmergencyClassifier(model_config)
            
            # Load saved weights
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.logger.info(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def preprocess_text(self, text):
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }

    def get_label_name(self, category, index):
        """Get the human-readable label name from the numeric index"""
        return self.label_mappings[category].get(index, f"UNKNOWN_{index}")

    def predict(self, texts):
        """
        Make predictions for a list of texts
        
        Args:
            texts (list): List of text strings to predict
            
        Returns:
            list: List of dictionaries containing predictions for each text
        """
        results = []
        
        with torch.no_grad():
            for text in texts:
                # Preprocess text
                inputs = self.preprocess_text(text)
                
                # Get model predictions
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
                
                # Get predicted classes
                emergency_level = torch.argmax(outputs['emergency_level_logits'], dim=1).item()
                disaster_type = torch.argmax(outputs['disaster_type_logits'], dim=1).item()
                sentiment = torch.argmax(outputs['sentiment_logits'], dim=1).item()
                
                
                results.append({
                    'text': text,
                    'predictions': {
                        'emergency_level': self.get_label_name('emergency_level', emergency_level),
                        'disaster_type': self.get_label_name('disaster_type', disaster_type),
                        'sentiment': self.get_label_name('sentiment', sentiment),
                    }
                })
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = EmergencyPredictor(
        model_path='models/saved_models/emergency_classifier_20241124_010648.pth'
    )
    
    test_tweets = [
        "Major earthquake just hit downtown! Buildings shaking, people evacuating #emergency",
        "Flash flood warning in effect for the next 24 hours. Seek higher ground immediately!",
    ]
    
    # Make predictions
    predictions = predictor.predict(test_tweets)
    
    # Print results
    for pred in predictions:
        print("\nText:", pred['text'])
        print("Predictions:")
        print(f"Emergency Level: {pred['predictions']['emergency_level']} ")
        print(f"Disaster Type: {pred['predictions']['disaster_type']} ")
        print(f"Sentiment: {pred['predictions']['sentiment']} ")