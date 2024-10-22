# model_catalog.py

import torch.nn as nn
import torchvision.models as vision_models
import transformers
import torch

class ModelCatalog:
    def __init__(self):
        self.models = {
            'image_classification': self._get_image_classification_models(),
            'text_classification': self._get_text_classification_models(),
            'sequence_modeling': self._get_sequence_models(),
            'tabular_regression': self._get_tabular_regression_models(),
            'tabular_classification': self._get_tabular_classification_models(),
            # Add more task types as needed
        }

    def _get_image_classification_models(self):
        return {
            'resnet18': vision_models.resnet18,
            'resnet50': vision_models.resnet50,
            'vgg16': vision_models.vgg16,
            'efficientnet_b0': vision_models.efficientnet_b0,
            'mobilenet_v2': vision_models.mobilenet_v2,
            # Add more CNN architectures here
        }

    def _get_text_classification_models(self):
        return {
            'bert': transformers.BertForSequenceClassification.from_pretrained,
            'roberta': transformers.RobertaForSequenceClassification.from_pretrained,
            'distilbert': transformers.DistilBertForSequenceClassification.from_pretrained,
            # Add more transformer models here
        }

    def _get_sequence_models(self):
        return {
            'lstm': self._build_lstm_model,
            'gru': self._build_gru_model,
        }

    def _get_tabular_regression_models(self):
        return {
            'mlp_regressor': self._build_mlp_regressor,
        }

    def _get_tabular_classification_models(self):
        return {
            'mlp_classifier': self._build_mlp_classifier,
        }

    # Custom model builders
    def _build_lstm_model(self, input_size, hidden_size=128, num_layers=2, num_classes=1):
        return nn.Sequential(
            nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True),
            nn.Linear(hidden_size, num_classes)
        )

    def _build_gru_model(self, input_size, hidden_size=128, num_layers=2, num_classes=1):
        return nn.Sequential(
            nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True),
            nn.Linear(hidden_size, num_classes)
        )

    def _build_mlp_regressor(self, input_size, hidden_sizes):
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        return nn.Sequential(*layers)

    def _build_mlp_classifier(self, input_size, hidden_sizes, num_classes):
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, num_classes))
        return nn.Sequential(*layers)

    # Update the tabular regression models to allow for adjustable parameters
    def _get_tabular_regression_models(self):
        return {
            'mlp_regressor': self._build_mlp_regressor,
        }

    def _get_tabular_classification_models(self):
        return {
            'mlp_classifier': self._build_mlp_classifier,
        }

    def get_models_for_task(self, task_type):
        return self.models.get(task_type, {})
