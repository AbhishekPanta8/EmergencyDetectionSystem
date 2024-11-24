import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)
import logging
from tqdm import tqdm
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Union

class EmergencyModelTrainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Create save directories
        self.save_dir = "models/saved_models"
        self.plots_dir = "models/plots"
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.get('batch_size', 16),
            shuffle=True,
            num_workers=0
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=config.get('batch_size', 16),
            shuffle=False,
            num_workers=0
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=float(config.get('learning_rate', 2e-5)),
            weight_decay=float(config.get('weight_decay', 0.01))
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Early stopping parameters
        self.early_stopping_patience = config.get('early_stopping_patience', 5)
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        
        # Store label mappings
        self.label_mappings = config.get('label_mappings', {})

    def calculate_metrics(self, y_true, y_pred, y_prob=None, task_name=None):
        """Calculate comprehensive classification metrics"""
        try:
            metrics = {}
            
            # Basic metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            
            # Precision, recall, F1
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true,
                y_pred,
                average='weighted'
            )
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1'] = f1
            
            # Detailed classification report
            metrics['classification_report'] = classification_report(
                y_true,
                y_pred,
                output_dict=True
            )
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = conf_matrix
            
            # Plot confusion matrix if task name is provided
            if task_name and self.label_mappings.get(task_name):
                labels = [str(v) for v in self.label_mappings[task_name].values()]
                self.plot_confusion_matrix(
                    conf_matrix,
                    labels,
                    save_path=os.path.join(self.plots_dir, f'{task_name}_confusion_matrix.png')
                )
            
            # ROC AUC if probabilities are provided
            if y_prob is not None:
                try:
                    metrics['roc_auc'] = roc_auc_score(
                        y_true,
                        y_prob,
                        multi_class='ovr',
                        average='weighted'
                    )
                except Exception as e:
                    self.logger.warning(f"Could not calculate ROC AUC: {str(e)}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return None

    def plot_confusion_matrix(self, conf_matrix, labels, save_path=None):
        """Plot confusion matrix using seaborn"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def save_model(self, path, is_best=False):
        """Save model checkpoint"""
        try:
            state = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_loss': self.best_val_loss,
            }
            torch.save(state, path)
            self.logger.info(f"Successfully saved model to {path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Clear gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                emergency_level_labels=batch['emergency_level_labels'],
                disaster_type_labels=batch['disaster_type_labels'],
                sentiment_labels=batch['sentiment_labels']
            )
            
            # Calculate total loss
            loss = sum(outputs['losses'].values())
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Update progress bar
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss/(progress_bar.n + 1)})
        
        # Save model after each epoch
        self.save_model(os.path.join(self.save_dir, f'model_epoch_{epoch+1}.pth'))
        
        return total_loss / num_batches

    def validate(self):
        self.model.eval()
        total_val_loss = 0
        predictions = {
            'emergency_level': [],
            'disaster_type': [],
            'sentiment': []
        }
        true_labels = {
            'emergency_level': [],
            'disaster_type': [],
            'sentiment': []
        }
        probabilities = {
            'emergency_level': [],
            'disaster_type': [],
            'sentiment': []
        }
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Validating'):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    emergency_level_labels=batch['emergency_level_labels'],
                    disaster_type_labels=batch['disaster_type_labels'],
                    sentiment_labels=batch['sentiment_labels']
                )
                
                # Accumulate loss
                total_val_loss += sum(outputs['losses'].values()).item()
                
                # Get predictions and probabilities
                for task in predictions.keys():
                    logits = outputs[f'{task}_logits']
                    probs = torch.softmax(logits, dim=1)
                    preds = torch.argmax(logits, dim=1)
                    
                    predictions[task].extend(preds.cpu().numpy())
                    probabilities[task].extend(probs.cpu().numpy())
                    true_labels[task].extend(batch[f'{task}_labels'].cpu().numpy())
        
        # Calculate and log metrics for each task
        metrics = {}
        for task in predictions.keys():
            self.logger.info(f"\n{task.upper()} Metrics:")
            task_metrics = self.calculate_metrics(
                true_labels[task],
                predictions[task],
                probabilities[task],
                task_name=task
            )
            
            if task_metrics:
                metrics[task] = task_metrics
                self.logger.info(f"Accuracy: {task_metrics['accuracy']:.4f}")
                self.logger.info(f"Precision: {task_metrics['precision']:.4f}")
                self.logger.info(f"Recall: {task_metrics['recall']:.4f}")
                self.logger.info(f"F1 Score: {task_metrics['f1']:.4f}")
                if 'roc_auc' in task_metrics:
                    self.logger.info(f"ROC AUC: {task_metrics['roc_auc']:.4f}")
                
                # Save detailed classification report
                report_df = pd.DataFrame(task_metrics['classification_report']).transpose()
                report_df.to_csv(os.path.join(self.plots_dir, f'{task}_classification_report.csv'))
        
        return total_val_loss / len(self.test_loader)

    def train(self, epochs=1):
        self.logger.info(f"Training on device: {self.device}")
        
        for epoch in range(epochs):
            self.logger.info(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training phase
            train_loss = self.train_epoch(epoch)
            self.logger.info(f"Training Loss: {train_loss:.4f}")
            
            # Save model before validation
            self.save_model(os.path.join(self.save_dir, 'latest_model.pth'))
            
            # Validation phase
            val_loss = self.validate()
            self.logger.info(f"Validation Loss: {val_loss:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                # Save best model
                self.save_model(os.path.join(self.save_dir, 'best_model.pth'), is_best=True)
                self.logger.info("Saved new best model")
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.early_stopping_patience:
                    self.logger.info("Early stopping triggered")
                    break
    
    def predict(self, text_input):
        """
        Make predictions on new text input
        """
        self.model.eval()
        # Implementation for prediction on new data
        # This would need the tokenizer and proper preprocessing
        pass