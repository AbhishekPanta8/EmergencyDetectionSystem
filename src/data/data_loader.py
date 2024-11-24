import pandas as pd
import logging
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class EmergencyDataLoader:
    def __init__(self, data_paths: Dict[str, str]):
        self.data_paths = data_paths
        self.logger = logging.getLogger(__name__)

    def _read_csv_with_encoding(self, file_path: str, sep: str = '\t') -> pd.DataFrame:
        """Try reading CSV with different encodings."""
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']

        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, sep=sep, encoding=encoding)
                return df.fillna('')
            except Exception as e:
                self.logger.error(f"Error reading {file_path} with {encoding}: {str(e)}")
        raise ValueError(f"Could not read {file_path} with any encoding")

    def load_and_preprocess_data(self) -> Tuple[pd.DataFrame, Dict]:
        """Load and preprocess only the emergency dataset and 911 calls."""
        try:
            # Load datasets
            datasets = {
                'emergency_dataset': self._read_csv_with_encoding(self.data_paths['emergency_dataset'], sep=','),
                '911_calls': self._read_csv_with_encoding(self.data_paths['911_calls'], sep=',')
            }

            # Preprocess each dataset
            processed_datasets = [
                self._preprocess_emergency_dataset(datasets['emergency_dataset']),
                self._preprocess_911_calls(datasets['911_calls'])
            ]

            # Combine datasets
            combined_data = pd.concat(processed_datasets, ignore_index=True)

            # Encode categorical features
            label_encoder = LabelEncoder()
            combined_data['emergency_level_encoded'] = label_encoder.fit_transform(combined_data['emergency_level'])
            combined_data['disaster_type_encoded'] = label_encoder.fit_transform(combined_data['disaster_type'])
            combined_data['sentiment_encoded'] = label_encoder.fit_transform(combined_data['sentiment'])

            # Metadata about encoding
            encoding_info = {
                'emergency_level_mapping': dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))),
                'disaster_type_mapping': dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))),
                'sentiment_mapping': dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
            }

            return combined_data, encoding_info

        except Exception as e:
            self.logger.error(f"Data loading error: {str(e)}")
            raise

    def _preprocess_emergency_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocessing for large emergency dataset."""
        df['emergency_level'] = df['emergency_level'].fillna('NON_EMERGENCY')
        df['disaster_type'] = df['disaster_type'].fillna('NONE')
        df['sentiment'] = df['sentiment'].fillna('NEUTRAL')
        return df

    def _preprocess_911_calls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocessing for 911 calls dataset."""
        df['emergency_level'] = df['emergency_level'].fillna('NON_EMERGENCY')
        df['disaster_type'] = df['disaster_type'].fillna('OTHER')
        df['sentiment'] = df['sentiment'].fillna('NEUTRAL')
        return df

    def split_data(self, data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Dict:
        """Split data into train, validation, and test sets with stratification."""
        # Ensure balanced split across emergency levels
        train_data, test_data = train_test_split(
            data,
            test_size=test_size,
            stratify=data['emergency_level'],
            random_state=random_state
        )

        train_data, val_data = train_test_split(
            train_data,
            test_size=0.25,
            stratify=train_data['emergency_level'],
            random_state=random_state
        )

        return {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }
