import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer

class EmergencyClassifier(nn.Module):
    def __init__(self, config):
        super(EmergencyClassifier, self).__init__()
        
        # Initialize RoBERTa
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.hidden_size = 768  # RoBERTa base hidden size
        
        # Multi-task classification heads
        self.num_emergency_levels = config.get('num_emergency_levels', 5)
        self.num_disaster_types = config.get('num_disaster_types', 1)
        self.num_sentiments = config.get('num_sentiments', 4)
        
        # Classification heads
        self.emergency_level_head = nn.Linear(self.hidden_size, self.num_emergency_levels)
        self.disaster_type_head = nn.Linear(self.hidden_size, self.num_disaster_types)
        self.sentiment_head = nn.Linear(self.hidden_size, self.num_sentiments)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Loss functions
        self.emergency_level_loss = nn.CrossEntropyLoss()
        self.disaster_type_loss = nn.CrossEntropyLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask=None, emergency_level_labels=None, 
                disaster_type_labels=None, sentiment_labels=None):
        # Get RoBERTa embeddings
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        pooled_output = self.dropout(pooled_output)
        
        # Multi-task predictions
        emergency_level_logits = self.emergency_level_head(pooled_output)
        disaster_type_logits = self.disaster_type_head(pooled_output)
        sentiment_logits = self.sentiment_head(pooled_output)
        
        output = {
            'emergency_level_logits': emergency_level_logits,
            'disaster_type_logits': disaster_type_logits,
            'sentiment_logits': sentiment_logits
        }
        
        if emergency_level_labels is not None and sentiment_labels is not None:
            losses = {
                'emergency_level_loss': self.emergency_level_loss(emergency_level_logits, emergency_level_labels),
                'disaster_type_loss': self.disaster_type_loss(disaster_type_logits, disaster_type_labels),
                'sentiment_loss': self.sentiment_loss(sentiment_logits, sentiment_labels)
            }
            output['losses'] = losses
        
        return output
    
    def predict(self, input_ids, attention_mask):
        with torch.no_grad():
            output = self.forward(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
        
        predictions = {
            'emergency_level': torch.argmax(output['emergency_level_logits'], dim=1),
            'disaster_type': torch.argmax(output['disaster_type_logits'], dim=1),
            'sentiment': torch.argmax(output['sentiment_logits'], dim=1)
        }
        
        return predictions

    def save(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, config, path):
        model = cls(config)
        model.load_state_dict(torch.load(path))
        return model