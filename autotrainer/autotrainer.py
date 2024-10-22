# autotrainer.py

import numpy as np
from autotrainer.model_catalog import ModelCatalog
from transformers import AutoTokenizer
from autotrainer.utils import detect_task_type, get_input_size
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class AutoTrainer:
    def __init__(self, X_train, y_train, X_val=None, y_val=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model_catalog = ModelCatalog()
        self.task_type = detect_task_type(X_train, y_train)
        self.selected_model = None
        self.hyperparameters = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_fn = None
        self.optimizer = None
        self.tokenizer = None  # For text models

    def select_model(self):
        models = self.model_catalog.get_models_for_task(self.task_type)
        if not models:
            raise ValueError(f"No models available for task type: {self.task_type}")

        # Intelligent model selection based on data properties
        if self.task_type == 'tabular_regression':
            input_size = self.X_train.shape[1]
            hidden_sizes = self.hyperparameters.get('hidden_sizes', [64, 32])
            self.selected_model = models['mlp_regressor'](input_size, hidden_sizes).to(self.device)
        elif self.task_type == 'tabular_classification':
            input_size = self.X_train.shape[1]
            num_classes = len(np.unique(self.y_train))
            hidden_sizes = self.hyperparameters.get('hidden_sizes', [64, 32])
            self.selected_model = models['mlp_classifier'](input_size, hidden_sizes, num_classes=num_classes).to(self.device)
        elif self.task_type == 'text_classification':
            # Select transformer models for text classification
            model_name = 'distilbert'  # Choose based on data size or other criteria
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.selected_model = models[model_name](model_name).to(self.device)
        elif self.task_type == 'image_classification':
            # Choose deeper models for larger datasets
            data_size = len(self.X_train)
            if data_size > 10000:
                model_name = 'resnet50'
            else:
                model_name = 'resnet18'
            self.selected_model = models[model_name](pretrained=True).to(self.device)
            # Modify the final layer to match the number of classes
            num_classes = len(np.unique(self.y_train))
            if hasattr(self.selected_model, 'fc'):
                in_features = self.selected_model.fc.in_features
                self.selected_model.fc = nn.Linear(in_features, num_classes).to(self.device)
        elif self.task_type in ['tabular_regression', 'tabular_classification']:
            input_size = self.X_train.shape[1]
            if self.task_type == 'tabular_regression':
                self.selected_model = models['mlp_regressor'](input_size).to(self.device)
            else:
                num_classes = len(np.unique(self.y_train))
                self.selected_model = models['mlp_classifier'](input_size, num_classes=num_classes).to(self.device)
        elif self.task_type == 'sequence_modeling':
            input_size = get_input_size(self.X_train)
            self.selected_model = models['lstm'](input_size).to(self.device)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        print(f"Selected model for {self.task_type}: {self.selected_model}")

    def adjust_hyperparameters(self):
        # Adjust hyperparameters based on data size and task type
        data_size = len(self.X_train)
        self.hyperparameters.setdefault('learning_rate', 1e-3)
        self.hyperparameters.setdefault('hidden_sizes', [64, 32])  # Default hidden layer sizes
        if data_size < 1000:
            self.hyperparameters.setdefault('batch_size', 16)
            self.hyperparameters.setdefault('epochs', 10)
        elif data_size < 10000:
            self.hyperparameters.setdefault('batch_size', 32)
            self.hyperparameters.setdefault('epochs', 20)
        else:
            self.hyperparameters.setdefault('batch_size', 64)
            self.hyperparameters.setdefault('epochs', 30)

        print(f"Adjusted hyperparameters: {self.hyperparameters}")

    def _prepare_data(self):
        # Data preparation based on task type
        if self.task_type == 'text_classification':
            # Tokenize text data
            inputs = self.tokenizer(self.X_train.tolist(), return_tensors='pt', padding=True, truncation=True)
            labels = torch.tensor(self.y_train)
            dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
            self.train_loader = DataLoader(dataset, batch_size=self.hyperparameters['batch_size'], shuffle=True)
        else:
            # Convert data to tensors
            X_tensor = torch.tensor(self.X_train).float()
            y_tensor = torch.tensor(self.y_train)
            dataset = TensorDataset(X_tensor, y_tensor)
            self.train_loader = DataLoader(dataset, batch_size=self.hyperparameters['batch_size'], shuffle=True)

    def _configure_training(self):
        # Loss function and optimizer configuration
        if self.task_type.endswith('classification'):
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.task_type.endswith('regression'):
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()  # Default to classification

        self.optimizer = optim.Adam(self.selected_model.parameters(), lr=self.hyperparameters['learning_rate'])

    def train(self):
        self._prepare_data()
        self._configure_training()
        epochs = self.hyperparameters['epochs']

        print("Starting training...")
        self.selected_model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                if self.task_type == 'text_classification':
                    input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                    outputs = self.selected_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                else:
                    inputs, labels = [b.to(self.device) for b in batch]
                    outputs = self.selected_model(inputs)
                    loss = self.loss_fn(outputs.squeeze(), labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(self.train_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    def run(self):
        self.select_model()
        self.adjust_hyperparameters()
        self.train()
