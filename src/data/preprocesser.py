import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class EmergencyDataProcessor:
    def __init__(self, datasets, test_size=0.2, random_state=42):
        self.datasets = datasets
        self.test_size = test_size
        self.random_state = random_state
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        
        # Label Encoders
        self.emergency_level_encoder = LabelEncoder()
        self.disaster_type_encoder = LabelEncoder()
        self.sentiment_encoder = LabelEncoder()

    def combine_datasets(self):
        combined_data = []
        
        # Process individual datasets
        for dataset_path in self.datasets:
            df = pd.read_csv(dataset_path, sep='\t' if dataset_path.endswith('.tsv') else ',')
            
            # Handling different dataset structures
            if 'text' not in df.columns:
                df['text'] = df.get('tweet text', df.get('tweet_id', ''))
            
            # Add preprocessing logic for different datasets
            if 'class_label' in df.columns:
                # For event-specific datasets like earthquake, wildfires
                df['emergency_level'] = 'HIGH'
                df['sentiment'] = 'PANIC'
                df['disaster_type'] = dataset_path.split('_')[0].upper()
            
            combined_data.append(df)
        
        return pd.concat(combined_data, ignore_index=True)

    def prepare_data(self, max_length=128):
        # Combine and preprocess datasets
        df = self.combine_datasets()
        
        # Encode labels and ensure proper columns exist in the DataFrame
        df['emergency_level_encoded'] = self.emergency_level_encoder.fit_transform(df['emergency_level'])
        df['disaster_type_encoded'] = self.disaster_type_encoder.fit_transform(df['disaster_type'])
        df['sentiment_encoded'] = self.sentiment_encoder.fit_transform(df['sentiment'])
        
        # Select feature columns and target columns
        feature_columns = ['text']  # Ensure 'text' exists and is used as the input
        target_columns = ['emergency_level_encoded', 'disaster_type_encoded', 'sentiment_encoded']
        
        # Define features and targets
        X = df[['text']].reset_index(drop=True)  # Ensure X is a DataFrame
        y = df[['emergency_level_encoded', 'disaster_type_encoded', 'sentiment_encoded']].reset_index(drop=True)

        # Debugging: Print shapes and indices
        print("X shape:", X.shape, "y shape:", y.shape)
        print("X indices:", X.index, "y indices:", y.index)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Create PyTorch datasets
        train_dataset = EmergencyDataset(
        texts=X_train['text'].values,
        emergency_level_labels=y_train['emergency_level_encoded'].values,
        sentiment_labels=y_train['sentiment_encoded'].values,
        disaster_type_labels=y_train['disaster_type_encoded'].values,
        max_length=max_length
    )
        
        test_dataset = EmergencyDataset(
        texts=X_test['text'].values,
        emergency_level_labels=y_test['emergency_level_encoded'].values,
        sentiment_labels=y_test['sentiment_encoded'].values,
        disaster_type_labels=y_test['disaster_type_encoded'].values,
        max_length=max_length
    )
        
        return train_dataset, test_dataset


class EmergencyDataset(Dataset):
    def __init__(self, texts, emergency_level_labels, sentiment_labels, disaster_type_labels=None, max_length=512):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        
        # Convert to lists if they're pandas Series/DataFrame
        self.texts = texts.values if isinstance(texts, pd.Series) else texts
        self.emergency_level_labels = emergency_level_labels.values if isinstance(emergency_level_labels, pd.Series) else emergency_level_labels
        self.sentiment_labels = sentiment_labels.values if isinstance(sentiment_labels, pd.Series) else sentiment_labels
        
        # Handle disaster type labels
        if disaster_type_labels is None:
            self.disaster_type_labels = [0] * len(self.texts)
        else:
            self.disaster_type_labels = disaster_type_labels.values if isinstance(disaster_type_labels, pd.Series) else disaster_type_labels
        
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Remove the batch dimension added by the tokenizer
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'emergency_level_labels': torch.tensor(self.emergency_level_labels[idx], dtype=torch.long),
            'disaster_type_labels': torch.tensor(self.disaster_type_labels[idx], dtype=torch.long),
            'sentiment_labels': torch.tensor(self.sentiment_labels[idx], dtype=torch.long)
        }
        
        return item
