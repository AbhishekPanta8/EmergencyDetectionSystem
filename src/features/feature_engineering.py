import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from transformers import RobertaTokenizer


class FeatureEngineer:
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.emergency_level_encoder = LabelEncoder()
        self.sentiment_encoder = LabelEncoder()
        self.disaster_type_encoder = LabelEncoder()

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def extract_text_features(self, texts):
        """
        Extract TF-IDF features from text.
        
        Args:
            texts (list): List of text strings.
        
        Returns:
            numpy.ndarray: TF-IDF feature matrix.
        """
        return self.tfidf_vectorizer.fit_transform(texts).toarray()
    
    def encode_labels(self, emergency_levels, sentiments):
        """
        Encode categorical labels for emergency levels and sentiments.
        
        Args:
            emergency_levels (list): Emergency level categories.
            sentiments (list): Sentiment categories.
        
        Returns:
            dict: Encoded labels and mappings.
        """
        emergency_level_encoded = self.emergency_level_encoder.fit_transform(emergency_levels)
        sentiment_encoded = self.sentiment_encoder.fit_transform(sentiments)
        
        return {
            'emergency_level': emergency_level_encoded,
            'sentiment': sentiment_encoded,
            'emergency_level_mapping': dict(zip(
                self.emergency_level_encoder.classes_, 
                self.emergency_level_encoder.transform(self.emergency_level_encoder.classes_)
            )),
            'sentiment_mapping': dict(zip(
                self.sentiment_encoder.classes_, 
                self.sentiment_encoder.transform(self.sentiment_encoder.classes_)
            ))
        }
    
    def engineer_features(self, dataframe):
        """
         feature engineering pipeline that returns a DataFrame
        """
        required_columns = ['text', 'emergency_level', 'sentiment', 'disaster_type']
        if not all(col in dataframe.columns for col in required_columns):
            raise ValueError(f"Dataframe must contain the following columns: {required_columns}")

        # Create a copy of the dataframe
        processed_df = dataframe.copy()

        # Encode labels
        processed_df['emergency_level_encoded'] = self.emergency_level_encoder.fit_transform(
            processed_df['emergency_level'])
        processed_df['sentiment_encoded'] = self.sentiment_encoder.fit_transform(processed_df['sentiment'])
        processed_df['disaster_type_encoded'] = self.disaster_type_encoder.fit_transform(processed_df['disaster_type'])

        # Store mappings
        self.label_mappings = {
            'emergency_level': dict(zip(
                self.emergency_level_encoder.classes_,
                self.emergency_level_encoder.transform(self.emergency_level_encoder.classes_)
            )),
            'sentiment': dict(zip(
                self.sentiment_encoder.classes_,
                self.sentiment_encoder.transform(self.sentiment_encoder.classes_)
            )),
            'disaster_type': dict(zip(
                self.disaster_type_encoder.classes_,
                self.disaster_type_encoder.transform(self.disaster_type_encoder.classes_)
            ))
        }

        return processed_df