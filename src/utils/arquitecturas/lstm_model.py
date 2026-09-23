"""
Esta clase fue creada para envolver un modelo LSTM de PyTorch y hacerlo compatible con el ecosistema de scikit-learn.
Esto permite integrar el modelo fácilmente en flujos de trabajo que utilizan herramientas como `GridSearchCV`, `Pipeline`
y funciones de evaluación que esperan métodos estándar como `fit()`, `predict()`, `get_params()` y `set_params()`.
Además, facilita la experimentación y reproducción de 4 modelos al permitir ajustar hiperparámetros y reconstruir
el modelo de forma automática, manteniendo la flexibilidad y el control que ofrece PyTorch.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union
import matplotlib.pyplot as plt


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        attention: bool = True,
    ):
        """
        Modelo LSTM con opción de atención

        Args:
            input_size: Número de características de entrada
            hidden_size: Tamaño de la capa oculta LSTM
            num_layers: Número de capas LSTM apiladas
            dropout: Tasa de dropout entre capas LSTM (solo si num_layers > 1)
            bidirectional: Si es True, usa LSTM bidireccional
            attention: Si es True, usa mecanismo de atención
        """
        super(LSTMModel, self).__init__()

        self.use_attention = attention
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Calculamos el factor multiplicador para capas bidireccionales
        self.directions = 2 if bidirectional else 1

        # Definimos la capa LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Capa de atención (si está activada)
        if attention:
            self.attn_fc = nn.Linear(hidden_size * self.directions, 1)

        # Capa final para la predicción
        self.fc = nn.Linear(hidden_size * self.directions, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor de entrada de forma [batch_size, seq_len, input_size]

        Returns:
            Tensor de salida de forma [batch_size, 1]
        """
        # Pasar por LSTM: output es [batch, seq_len, hidden_size * directions]
        lstm_out, (hn, _) = self.lstm(x)

        if self.use_attention:
            # Calcular pesos de atención
            attn_weights = torch.softmax(
                self.attn_fc(lstm_out).squeeze(-1), dim=1
            )  # [batch, seq_len]

            # Aplicar atención al output de LSTM
            context = torch.sum(
                lstm_out * attn_weights.unsqueeze(-1), dim=1
            )  # [batch, hidden_size * directions]
            output = self.fc(context)
        else:
            # Sin atención, usar el último estado oculto
            if self.bidirectional:
                # Concatenar las direcciones del último estado oculto
                last_hidden = hn.view(
                    self.num_layers, self.directions, x.size(0), self.hidden_size
                )
                last_hidden = (
                    last_hidden[-1].transpose(0, 1).contiguous().view(x.size(0), -1)
                )
                output = self.fc(last_hidden)
            else:
                # Usar directamente el último estado oculto
                output = self.fc(hn[-1])

        return output

    def get_attention_weights(self, x: torch.Tensor) -> Optional[np.ndarray]:
        """
        Obtiene los pesos de atención para visualización

        Args:
            x: Tensor de entrada

        Returns:
            Pesos de atención o None si no se usa atención
        """
        if not self.use_attention:
            return None

        self.eval()
        with torch.no_grad():
            lstm_out, _ = self.lstm(x)
            attn_weights = torch.softmax(self.attn_fc(lstm_out).squeeze(-1), dim=1)

        return attn_weights.cpu().numpy()


class SklearnLikeLSTM(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        input_shape: Optional[Tuple] = None,
        hidden_size: int = 50,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        attention: bool = True,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 0.001,
        weight_decay: float = 0,
        early_stopping_patience: int = None,
        device: Optional[str] = None,
    ):
        """
        Wrapper scikit-learn para modelo LSTM de PyTorch

        Args:
            input_shape: Forma de los datos de entrada (seq_len, features)
            hidden_size: Tamaño de la capa oculta LSTM
            num_layers: Número de capas LSTM
            dropout: Tasa de dropout entre capas LSTM
            bidirectional: Si es True, usa LSTM bidireccional
            attention: Si es True, usa mecanismo de atención
            epochs: Número de épocas de entrenamiento
            batch_size: Tamaño del lote para entrenamiento
            lr: Tasa de aprendizaje
            weight_decay: Regularización L2
            early_stopping_patience: Si no es None, para el entrenamiento si la pérdida
                                     de validación no mejora durante este número de épocas
            device: Dispositivo para entrenar ('cuda' o 'cpu')
        """
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.attention = attention
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.history = {"train_loss": [], "val_loss": []}

    def _build_model(self) -> LSTMModel:
        """Construye el modelo LSTM con los parámetros actuales"""
        input_size = self.input_shape[-1]
        model = LSTMModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            attention=self.attention,
        )
        return model.to(self.device)

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        validation_data: Optional[
            Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]
        ] = None,
        verbose: bool = True,
    ) -> "SklearnLikeLSTM":
        """
        Entrena el modelo

        Args:
            X: Datos de entrada de forma [batch, seq_len, features]
            y: Valores objetivo
            validation_data: Tupla opcional (X_val, y_val) para validación
            verbose: Si es True, muestra progreso del entrenamiento

        Returns:
            self: Instancia entrenada
        """
        # Convertir datos a tensores PyTorch si es necesario
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        # Preparar datos de validación si se proporcionan
        if validation_data is not None:
            X_val, y_val = validation_data
            if isinstance(X_val, np.ndarray):
                X_val = torch.tensor(X_val, dtype=torch.float32)
            if isinstance(y_val, np.ndarray):
                y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
            X_val, y_val = X_val.to(self.device), y_val.to(self.device)

        # Determinar la forma de entrada si no se especificó
        self.input_shape = X.shape[1:] if self.input_shape is None else self.input_shape

        # Construir el modelo
        self.model = self._build_model()

        # Definir función de pérdida y optimizador
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Crear dataloader para entrenamiento
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Variables para early stopping
        best_val_loss = float("inf")
        patience_counter = 0

        # Entrenamiento
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batches = 0

            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batches += 1

            # Calcular pérdida promedio de la época
            avg_train_loss = epoch_loss / batches
            self.history["train_loss"].append(avg_train_loss)

            # Validación si se proporcionan datos
            if validation_data is not None:
                with torch.no_grad():
                    self.model.eval()
                    val_output = self.model(X_val)
                    val_loss = criterion(val_output, y_val).item()
                    self.history["val_loss"].append(val_loss)
                    self.model.train()

                    # Early stopping
                    if self.early_stopping_patience is not None:
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1

                        if patience_counter >= self.early_stopping_patience:
                            if verbose:
                                print(
                                    f"Early stopping en época {epoch + 1}/{self.epochs}"
                                )
                            break

                if verbose:
                    print(
                        f"Época {epoch + 1}/{self.epochs}, Pérdida: {avg_train_loss:.6f}, "
                        f"Pérdida val: {val_loss:.6f}"
                    )
            elif verbose:
                print(f"Época {epoch + 1}/{self.epochs}, Pérdida: {avg_train_loss:.6f}")

        return self

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Realiza predicciones con el modelo entrenado

        Args:
            X: Datos de entrada

        Returns:
            Array con predicciones
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado. Llama a fit() primero.")

        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(X.to(self.device))

        return preds.cpu().numpy().squeeze()

    def get_attention_weights(
        self, X: Union[np.ndarray, torch.Tensor]
    ) -> Optional[np.ndarray]:
        """
        Obtiene los pesos de atención para una entrada dada

        Args:
            X: Datos de entrada

        Returns:
            Array con pesos de atención o None si no se usa atención
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado. Llama a fit() primero.")

        if not self.attention:
            return None

        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)

        return self.model.get_attention_weights(X.to(self.device))

    def plot_training_history(self) -> None:
        """Visualiza el historial de entrenamiento"""
        if not self.history["train_loss"]:
            raise ValueError("No hay historial de entrenamiento disponible.")

        plt.figure(figsize=(10, 5))
        plt.plot(self.history["train_loss"], label="Pérdida de entrenamiento")

        if self.history["val_loss"]:
            plt.plot(self.history["val_loss"], label="Pérdida de validación")

        plt.xlabel("Época")
        plt.ylabel("Pérdida")
        plt.legend()
        plt.grid(True)
        plt.title("Historial de entrenamiento")
        plt.show()

    def save_model(self, path: str) -> None:
        """
        Guarda el modelo en disco

        Args:
            path: Ruta del archivo para guardar
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado. Llama a fit() primero.")

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_params": self.get_params(),
                "history": self.history,
            },
            path,
        )

    @classmethod
    def load_model(cls, path: str, device: Optional[str] = None) -> "SklearnLikeLSTM":
        """
        Carga un modelo desde disco

        Args:
            path: Ruta del archivo a cargar
            device: Dispositivo para cargar el modelo

        Returns:
            Instancia cargada de SklearnLikeLSTM
        """
        checkpoint = torch.load(path, map_location=device or "cpu")

        # Crear instancia con los parámetros guardados
        instance = cls(**checkpoint["model_params"])

        # Construir modelo y cargar pesos
        instance.model = instance._build_model()
        instance.model.load_state_dict(checkpoint["model_state_dict"])

        # Cargar historial
        instance.history = checkpoint["history"]

        return instance

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Obtiene parámetros (para compatibilidad con scikit-learn)"""
        return {
            "input_shape": self.input_shape,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
            "attention": self.attention,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "early_stopping_patience": self.early_stopping_patience,
            "device": self.device,
        }

    def set_params(self, **params) -> "SklearnLikeLSTM":
        """Establece parámetros (para compatibilidad con scikit-learn)"""
        for param, value in params.items():
            setattr(self, param, value)
        return self
