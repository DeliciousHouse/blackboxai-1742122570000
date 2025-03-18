import json
import logging
import os
import pickle
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial import Delaunay
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .db import execute_query, execute_write_query

# Define model directory path
MODEL_DIR = Path("/data/models")

logger = logging.getLogger(__name__)

class AIProcessor:
    """AI processor for enhancing blueprint generation with machine learning."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the AI processor."""
        self.config = self._load_config(config_path)

        # Ensure model directory exists
        os.makedirs(MODEL_DIR, exist_ok=True)

        # Initialize models
        self.rssi_distance_model = None
        self.room_clustering_model = None
        self.wall_prediction_model = None
        self.blueprint_refinement_model = None

        # Load models if they exist
        self._load_models()

    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration from file or use defaults."""
        if config_path:
            with open(config_path, 'r') as f:
                return json.load(f)
        return {
            'ai_settings': {
                'rssi_distance': {
                    'model_type': 'random_forest',
                    'features': ['rssi', 'tx_power', 'frequency'],
                    'hyperparams': {
                        'n_estimators': 100,
                        'max_depth': 10
                    }
                },
                'room_clustering': {
                    'algorithm': 'dbscan',
                    'eps': 2.0,
                    'min_samples': 3,
                    'features': ['x', 'y', 'z'],
                    'temporal_weight': 0.2
                },
                'wall_prediction': {
                    'model_type': 'cnn',
                    'input_shape': [64, 64, 1],
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 50
                },
                'blueprint_refinement': {
                    'reward_weights': {
                        'room_size': 0.3,
                        'wall_alignment': 0.4,
                        'flow_efficiency': 0.3
                    },
                    'learning_rate': 0.01,
                    'discount_factor': 0.9
                }
            }
        }

    def setup_database_tables(self) -> bool:
        """Set up database tables for AI training data."""
        try:
            # Create table for RSSI-to-distance training data
            execute_write_query("""
            CREATE TABLE IF NOT EXISTS ai_rssi_distance_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                device_id VARCHAR(255) NOT NULL,
                sensor_id VARCHAR(255) NOT NULL,
                rssi INT NOT NULL,
                distance FLOAT NOT NULL,
                tx_power INT,
                frequency FLOAT,
                environment_type VARCHAR(50),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # Create table for room clustering training data
            execute_write_query("""
            CREATE TABLE IF NOT EXISTS ai_room_clustering_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                device_id VARCHAR(255) NOT NULL,
                x FLOAT NOT NULL,
                y FLOAT NOT NULL,
                z FLOAT NOT NULL,
                room_label VARCHAR(255),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # Create table for wall prediction training data
            execute_write_query("""
            CREATE TABLE IF NOT EXISTS ai_wall_prediction_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                blueprint_id VARCHAR(255) NOT NULL,
                positions_data TEXT NOT NULL,
                walls_data TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # Create table for blueprint refinement feedback
            execute_write_query("""
            CREATE TABLE IF NOT EXISTS ai_blueprint_feedback (
                id INT AUTO_INCREMENT PRIMARY KEY,
                blueprint_id VARCHAR(255) NOT NULL,
                original_blueprint TEXT NOT NULL,
                modified_blueprint TEXT NOT NULL,
                feedback_score FLOAT,
                feedback_comments TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # Create table for trained models
            execute_write_query("""
            CREATE TABLE IF NOT EXISTS ai_models (
                id INT AUTO_INCREMENT PRIMARY KEY,
                model_name VARCHAR(255) NOT NULL,
                model_type VARCHAR(255) NOT NULL,
                model_path VARCHAR(255) NOT NULL,
                metrics TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY (model_name)
            )
            """)

            logger.info("AI database tables created successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to set up AI database tables: {str(e)}")
            return False

    def _load_models(self) -> None:
        """Load trained models from disk."""
        try:
            # Load RSSI-to-distance model
            rssi_model_path = MODEL_DIR / 'rssi_distance_model.pkl'
            if rssi_model_path.exists():
                self.rssi_distance_model = joblib.load(rssi_model_path)
                logger.info("Loaded RSSI-to-distance model")

            # Load room clustering model
            room_model_path = MODEL_DIR / 'room_clustering_model.pkl'
            if room_model_path.exists():
                self.room_clustering_model = joblib.load(room_model_path)
                logger.info("Loaded room clustering model")

            # Load wall prediction model
            wall_model_path = MODEL_DIR / 'wall_prediction_model.pt'
            if wall_model_path.exists() and torch.cuda.is_available():
                self.wall_prediction_model = torch.load(wall_model_path)
                self.wall_prediction_model.eval()
                logger.info("Loaded wall prediction model")
            elif wall_model_path.exists():
                self.wall_prediction_model = torch.load(wall_model_path, map_location=torch.device('cpu'))
                self.wall_prediction_model.eval()
                logger.info("Loaded wall prediction model (CPU)")

            # Load blueprint refinement model
            bp_model_path = MODEL_DIR / 'blueprint_refinement_model.zip'
            if bp_model_path.exists():
                self.blueprint_refinement_model = PPO.load(bp_model_path)
                logger.info("Loaded blueprint refinement model")

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")

    def get_models_status(self) -> Dict[str, Dict[str, Any]]:
        """Get the status of all AI models."""
        status = {
            "rssi_distance_model": {
                "loaded": self.rssi_distance_model is not None,
                "type": type(self.rssi_distance_model).__name__ if self.rssi_distance_model else None,
                "training_samples": self._get_training_sample_count("ai_rssi_distance_data")
            },
            "room_clustering_model": {
                "loaded": self.room_clustering_model is not None,
                "type": type(self.room_clustering_model).__name__ if self.room_clustering_model else None,
                "training_samples": self._get_training_sample_count("ai_room_clustering_data")
            },
            "wall_prediction_model": {
                "loaded": self.wall_prediction_model is not None,
                "type": "PyTorch Neural Network" if self.wall_prediction_model else None,
                "training_samples": self._get_training_sample_count("ai_wall_prediction_data")
            },
            "blueprint_refinement_model": {
                "loaded": self.blueprint_refinement_model is not None,
                "type": "Reinforcement Learning (PPO)" if self.blueprint_refinement_model else None,
                "feedback_samples": self._get_training_sample_count("ai_blueprint_feedback")
            }
        }
        return status

    def _get_training_sample_count(self, table_name: str) -> int:
        """Get the count of training samples for a model."""
        try:
            query = f"SELECT COUNT(*) FROM {table_name}"
            result = execute_query(query)
            return result[0][0] if result else 0
        except Exception as e:
            logger.error(f"Error getting sample count for {table_name}: {str(e)}")
            return 0

    # RSSI-to-Distance ML Model methods

    def save_rssi_distance_sample(self, device_id: str, sensor_id: str, rssi: int,
                                 distance: float, tx_power: Optional[int] = None,
                                 frequency: Optional[float] = None,
                                 environment_type: Optional[str] = None) -> bool:
        """Save a training sample for the RSSI-to-distance model."""
        try:
            query = """
            INSERT INTO ai_rssi_distance_data
            (device_id, sensor_id, rssi, distance, tx_power, frequency, environment_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            execute_write_query(query, (
                device_id, sensor_id, rssi, distance, tx_power,
                frequency, environment_type
            ))
            logger.debug(f"Saved RSSI-distance sample: {rssi} -> {distance}m")
            return True
        except Exception as e:
            logger.error(f"Failed to save RSSI-distance sample: {str(e)}")
            return False

    def train_models(self):
        """Train AI models with collected data."""
        try:
            # Get training data from database
            from .db import get_sqlite_connection
            conn = get_sqlite_connection()
            cursor = conn.cursor()

            # Get RSSI to distance training data
            cursor.execute("""
                SELECT rssi, distance, environment_type
                FROM rssi_distance_samples
                ORDER BY timestamp DESC
                LIMIT 10000
            """)
            rssi_distance_data = cursor.fetchall()

            if not rssi_distance_data:
                logger.info("Not enough training data for distance model")
                return False

            logger.info(f"Training models with {len(rssi_distance_data)} RSSI samples")

            # This is a placeholder - actual model training would happen here
            # In a real implementation, you'd train ML models with the data

            return True

        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            return False

    def train_rssi_distance_model(self, model_type: str = 'random_forest',
                                 test_size: float = 0.2,
                                 features: List[str] = None,
                                 hyperparams: Dict = None) -> Dict:
        """Train the RSSI-to-distance regression model."""
        if features is None:
            features = ['rssi']
        if hyperparams is None:
            hyperparams = {}

        try:
            # Get training data
            query = """
            SELECT rssi, distance, tx_power, frequency, environment_type
            FROM ai_rssi_distance_data
            """
            results = execute_query(query)

            if not results or len(results) < 10:
                return {
                    "success": False,
                    "error": "Not enough training data (minimum 10 samples required)"
                }

            # Prepare data
            data = []
            for row in results:
                sample = {'rssi': row[0], 'distance': row[1]}
                if 'tx_power' in features and row[2] is not None:
                    sample['tx_power'] = row[2]
                if 'frequency' in features and row[3] is not None:
                    sample['frequency'] = row[3]
                if 'environment_type' in features and row[4] is not None:
                    sample['environment_type'] = 1 if row[4] == 'indoor' else 0
                data.append(sample)

            df = pd.DataFrame(data)

            # Handle missing values
            for feature in features:
                if feature in df.columns and df[feature].isnull().any():
                    if feature == 'environment_type':
                        df[feature].fillna(0, inplace=True)
                    else:
                        df[feature].fillna(df[feature].mean(), inplace=True)

            # Prepare features and target
            X = df[[f for f in features if f in df.columns]]
            y = df['distance']

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model based on type
            if model_type == 'random_forest':
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(
                    n_estimators=hyperparams.get('n_estimators', 100),
                    max_depth=hyperparams.get('max_depth', 10),
                    random_state=42
                )
            elif model_type == 'xgboost':
                import xgboost as xgb
                model = xgb.XGBRegressor(
                    n_estimators=hyperparams.get('n_estimators', 100),
                    max_depth=hyperparams.get('max_depth', 6),
                    learning_rate=hyperparams.get('learning_rate', 0.1),
                    random_state=42
                )
            elif model_type == 'neural_network':
                from sklearn.neural_network import MLPRegressor
                model = MLPRegressor(
                    hidden_layer_sizes=hyperparams.get('hidden_layer_sizes', (100, 50)),
                    max_iter=hyperparams.get('max_iter', 500),
                    random_state=42
                )
            else:
                return {
                    "success": False,
                    "error": f"Unsupported model type: {model_type}"
                }

            # Train the model
            model.fit(X_train_scaled, y_train)

            # Evaluate the model
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Save the model
            model_data = {
                'model': model,
                'scaler': scaler,
                'features': features,
                'metrics': {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2
                },
                'model_type': model_type,
                'hyperparams': hyperparams
            }

            model_path = MODEL_DIR / 'rssi_distance_model.pkl'
            joblib.dump(model_data, model_path)

            # Update the model in memory
            self.rssi_distance_model = model_data

            # Save model info to database
            metrics_json = json.dumps({
                'mse': float(mse),
                'mae': float(mae),
                'r2': float(r2),
                'samples': len(df),
                'features': features,
                'hyperparams': hyperparams
            })

            execute_write_query("""
            INSERT INTO ai_models (model_name, model_type, model_path, metrics)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                model_type = VALUES(model_type),
                model_path = VALUES(model_path),
                metrics = VALUES(metrics),
                created_at = CURRENT_TIMESTAMP
            """, ('rssi_distance', model_type, model_path, metrics_json))

            logger.info(f"Trained RSSI-distance model: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

            return {
                "success": True,
                "metrics": {
                    "mse": float(mse),
                    "mae": float(mae),
                    "r2": float(r2)
                },
                "samples": len(df),
                "features": features,
                "model_type": model_type
            }

        except Exception as e:
            logger.error(f"Failed to train RSSI-distance model: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def estimate_distance(self, rssi: int, environment_type: Optional[str] = None) -> float:
        """Estimate distance from RSSI using the trained model."""
        try:
            if self.rssi_distance_model is None:
                # Fall back to physics-based model
                reference_power = -59  # Default reference power at 1m
                path_loss_exponent = 2.0  # Default path loss exponent
                return 10 ** ((reference_power - rssi) / (10 * path_loss_exponent))

            # Prepare input features
            features = self.rssi_distance_model['features']
            data = {'rssi': rssi}

            if 'environment_type' in features and environment_type is not None:
                data['environment_type'] = 1 if environment_type == 'indoor' else 0

            # Fill missing features with defaults
            for feature in features:
                if feature not in data:
                    if feature == 'tx_power':
                        data[feature] = -59  # Default TX power
                    elif feature == 'frequency':
                        data[feature] = 2400  # Default frequency (2.4 GHz)
                    elif feature == 'environment_type':
                        data[feature] = 1  # Default to indoor

            # Create DataFrame with the right features
            df = pd.DataFrame([data])
            X = df[features]

            # Scale features
            X_scaled = self.rssi_distance_model['scaler'].transform(X)

            # Predict distance
            distance = float(self.rssi_distance_model['model'].predict(X_scaled)[0])

            # Ensure distance is positive
            return max(0.1, distance)

        except Exception as e:
            logger.error(f"Error estimating distance: {str(e)}")
            # Fall back to physics-based model
            reference_power = -59
            path_loss_exponent = 2.0
            return 10 ** ((reference_power - rssi) / (10 * path_loss_exponent))

    # Room Clustering methods

    def configure_room_clustering(self, algorithm: str = 'dbscan',
                                 eps: float = 2.0, min_samples: int = 3,
                                 features: List[str] = None,
                                 temporal_weight: float = 0.2) -> Dict:
        """Configure the room clustering model."""
        if features is None:
            features = ['x', 'y', 'z']

        try:
            # Create clustering model based on algorithm
            if algorithm == 'dbscan':
                model = DBSCAN(eps=eps, min_samples=min_samples)
            elif algorithm == 'kmeans':
                model = KMeans(n_clusters=min_samples, random_state=42)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported clustering algorithm: {algorithm}"
                }

            # Save model configuration
            model_data = {
                'model': model,
                'algorithm': algorithm,
                'params': {
                    'eps': eps,
                    'min_samples': min_samples
                },
                'features': features,
                'temporal_weight': temporal_weight
            }

            model_path = MODEL_DIR / 'room_clustering_model.pkl'
            joblib.dump(model_data, model_path)

            # Update the model in memory
            self.room_clustering_model = model_data

            # Save model info to database
            metrics_json = json.dumps({
                'algorithm': algorithm,
                'eps': eps,
                'min_samples': min_samples,
                'features': features,
                'temporal_weight': temporal_weight
            })

            execute_write_query("""
            INSERT INTO ai_models (model_name, model_type, model_path, metrics)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                model_type = VALUES(model_type),
                model_path = VALUES(model_path),
                metrics = VALUES(metrics),
                created_at = CURRENT_TIMESTAMP
            """, ('room_clustering', algorithm, model_path, metrics_json))

            logger.info(f"Configured room clustering model: {algorithm}")

            return {
                "success": True,
                "algorithm": algorithm,
                "params": {
                    "eps": eps,
                    "min_samples": min_samples
                },
                "features": features,
                "temporal_weight": temporal_weight
            }

        except Exception as e:
            logger.error(f"Failed to configure room clustering model: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def detect_rooms_ml(self, positions: Dict[str, Dict[str, float]]) -> List[Dict]:
        """Detect rooms using ML-based clustering."""
        if not positions or not self.room_clustering_model:
            return []

        try:
            # Extract position coordinates
            device_ids = list(positions.keys())
            coords = np.array([[p['x'], p['y'], p['z']] for p in positions.values()])

            # Apply clustering
            model = self.room_clustering_model['model']
            labels = model.fit_predict(coords)

            # Group positions by cluster
            rooms = []
            for label in set(labels):
                if label == -1:  # Skip noise points
                    continue

                # Get positions in this cluster
                room_positions = coords[labels == label]
                room_device_ids = [device_ids[i] for i, l in enumerate(labels) if l == label]

                # Calculate room properties
                min_coords = np.min(room_positions, axis=0)
                max_coords = np.max(room_positions, axis=0)
                center = np.mean(room_positions, axis=0)

                # Create room object
                room = {
                    'id': f"room_{label}",
                    'name': f"Room {label + 1}",
                    'center': {
                        'x': float(center[0]),
                        'y': float(center[1]),
                        'z': float(center[2])
                    },
                    'dimensions': {
                        'width': float(max_coords[0] - min_coords[0]),
                        'length': float(max_coords[1] - min_coords[1]),
                        'height': float(max_coords[2] - min_coords[2])
                    },
                    'bounds': {
                        'min': {
                            'x': float(min_coords[0]),
                            'y': float(min_coords[1]),
                            'z': float(min_coords[2])
                        },
                        'max': {
                            'x': float(max_coords[0]),
                            'y': float(max_coords[1]),
                            'z': float(max_coords[2])
                        }
                    },
                    'devices': room_device_ids
                }

                rooms.append(room)

            return rooms

        except Exception as e:
            logger.error(f"Error in ML room detection: {str(e)}")
            return []

    # Wall Prediction Neural Network methods

    class WallPredictionDataset(Dataset):
        """Dataset for wall prediction model."""

        def __init__(self, positions_data, walls_data, transform=None):
            self.positions = positions_data
            self.walls = walls_data
            self.transform = transform

        def __len__(self):
            return len(self.positions)

        def __getitem__(self, idx):
            positions = self.positions[idx]
            walls = self.walls[idx]

            # Convert to tensors
            positions_tensor = torch.tensor(positions, dtype=torch.float32)
            walls_tensor = torch.tensor(walls, dtype=torch.float32)

            if self.transform:
                positions_tensor = self.transform(positions_tensor)

            return positions_tensor, walls_tensor

    class WallPredictionCNN(nn.Module):
        """CNN model for wall prediction."""

        def __init__(self, input_channels=1, output_channels=1):
            super().__init__()

            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU()
            )

            # Decoder
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, output_channels, kernel_size=3, padding=1),
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    def train_wall_prediction_model(self, model_type: str = 'cnn',
                                   training_data: List = None,
                                   epochs: int = 50,
                                   batch_size: int = 32,
                                   learning_rate: float = 0.001) -> Dict:
        """Train the wall prediction neural network."""
        try:
            # Get training data from database if not provided
            if not training_data:
                query = """
                SELECT positions_data, walls_data FROM ai_wall_prediction_data
                """
                results = execute_query(query)

                if not results or len(results) < 5:
                    return {
                        "success": False,
                        "error": "Not enough training data (minimum 5 samples required)"
                    }

                positions_data = [json.loads(row[0]) for row in results]
                walls_data = [json.loads(row[1]) for row in results]
            else:
                positions_data = [item['positions'] for item in training_data]
                walls_data = [item['walls'] for item in training_data]

            # Create dataset
            dataset = self.WallPredictionDataset(positions_data, walls_data)

            # Split into train/validation
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            # Initialize model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = self.WallPredictionCNN().to(device)

            # Define loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Training loop
            best_val_loss = float('inf')
            metrics = {'train_loss': [], 'val_loss': []}

            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0.0

                for positions, walls in train_loader:
                    positions, walls = positions.to(device), walls.to(device)

                    optimizer.zero_grad()
                    outputs = model(positions)
                    loss = criterion(outputs, walls)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * positions.size(0)

                train_loss /= len(train_loader.dataset)
                metrics['train_loss'].append(train_loss)

                # Validation
                model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for positions, walls in val_loader:
                        positions, walls = positions.to(device), walls.to(device)
                        outputs = model(positions)
                        loss = criterion(outputs, walls)
                        val_loss += loss.item() * positions.size(0)

                val_loss /= len(val_loader.dataset)
                metrics['val_loss'].append(val_loss)

                logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model_path = MODEL_DIR / 'wall_prediction_model.pt'
                    torch.save(model, model_path)

            # Load the best model
            model_path = MODEL_DIR / 'wall_prediction_model.pt'
            self.wall_prediction_model = torch.load(model_path)

            # Save model info to database
            metrics_json = json.dumps({
                'model_type': model_type,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'final_train_loss': metrics['train_loss'][-1],
                'final_val_loss': metrics['val_loss'][-1],
                'best_val_loss': best_val_loss,
                'samples': len(dataset)
            })

            execute_write_query("""
            INSERT INTO ai_models (model_name, model_type, model_path, metrics)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                model_type = VALUES(model_type),
                model_path = VALUES(model_path),
                metrics = VALUES(metrics),
                created_at = CURRENT_TIMESTAMP
            """, ('wall_prediction', model_type, model_path, metrics_json))

            logger.info(f"Trained wall prediction model: best_val_loss={best_val_loss:.4f}")

            return {
                "success": True,
                "metrics": {
                    "train_loss": metrics['train_loss'][-1],
                    "val_loss": metrics['val_loss'][-1],
                    "best_val_loss": best_val_loss
                },
                "samples": len(dataset),
                "model_type": model_type
            }

        except Exception as e:
            logger.error(f"Failed to train wall prediction model: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def predict_walls(self, positions: Dict[str, Dict[str, float]], rooms: List[Dict]) -> List[Dict]:
        """Predict walls using the trained neural network."""
        if not positions or not rooms or not self.wall_prediction_model:
            return []

        try:
            # Convert positions to grid representation
            grid_size = 64
            grid = np.zeros((grid_size, grid_size), dtype=np.float32)

            # Find bounds of the space
            all_coords = np.array([[p['x'], p['y']] for p in positions.values()])
            min_x, min_y = np.min(all_coords, axis=0)
            max_x, max_y = np.max(all_coords, axis=0)

            # Add padding
            padding = 0.1 * max(max_x - min_x, max_y - min_y)
            min_x -= padding
            min_y -= padding
            max_x += padding
            max_y += padding

            # Scale function to convert real coordinates to grid indices
            def scale_to_grid(x, y):
                grid_x = int((x - min_x) / (max_x - min_x) * (grid_size - 1))
                grid_y = int((y - min_y) / (max_y - min_y) * (grid_size - 1))
                return max(0, min(grid_size - 1, grid_x)), max(0, min(grid_size - 1, grid_y))

            # Place positions on grid
            for pos in positions.values():
                grid_x, grid_y = scale_to_grid(pos['x'], pos['y'])
                grid[grid_y, grid_x] = 1.0

            # Add room boundaries to grid
            for room in rooms:
                bounds = room['bounds']
                min_room_x, min_room_y = scale_to_grid(bounds['min']['x'], bounds['min']['y'])
                max_room_x, max_room_y = scale_to_grid(bounds['max']['x'], bounds['max']['y'])

                # Draw room boundaries on grid
                for x in range(min_room_x, max_room_x + 1):
                    grid[min_room_y, x] = 0.5
                    grid[max_room_y, x] = 0.5

                for y in range(min_room_y, max_room_y + 1):
                    grid[y, min_room_x] = 0.5
                    grid[y, max_room_x] = 0.5

            # Prepare input for model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            input_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            # Get prediction
            self.wall_prediction_model.eval()
            with torch.no_grad():
                output = self.wall_prediction_model(input_tensor)
                wall_grid = output.squeeze().cpu().numpy()

            # Convert wall grid to wall segments
            walls = []
            threshold = 0.5

            # Function to convert grid indices back to real coordinates
            def grid_to_real(grid_x, grid_y):
                x = min_x + grid_x / (grid_size - 1) * (max_x - min_x)
                y = min_y + grid_y / (grid_size - 1) * (max_y - min_y)
                return x, y

            # Detect horizontal walls
            for y in range(grid_size):
                wall_start = None
                for x in range(grid_size):
                    if wall_grid[y, x] > threshold:
                        if wall_start is None:
                            wall_start = x
                    elif wall_start is not None:
                        if x - wall_start > 2:  # Minimum wall length
                            start_x, start_y = grid_to_real(wall_start, y)
                            end_x, end_y = grid_to_real(x - 1, y)

                            # Calculate wall properties
                            length = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
                            angle = np.arctan2(end_y - start_y, end_x - start_x)

                            walls.append({
                                'start': {'x': float(start_x), 'y': float(start_y)},
                                'end': {'x': float(end_x), 'y': float(end_y)},
                                'thickness': 0.1,
                                'height': 2.4,
                                'angle': float(angle),
                                'confidence': float(np.mean(wall_grid[y, wall_start:x]))
                            })
                        wall_start = None

            # Detect vertical walls
            for x in range(grid_size):
                wall_start = None
                for y in range(grid_size):
                    if wall_grid[y, x] > threshold:
                        if wall_start is None:
                            wall_start = y
                    elif wall_start is not None:
                        if y - wall_start > 2:  # Minimum wall length
                            start_x, start_y = grid_to_real(x, wall_start)
                            end_x, end_y = grid_to_real(x, y - 1)

                            # Calculate wall properties
                            length = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
                            angle = np.arctan2(end_y - start_y, end_x - start_x)

                            walls.append({
                                'start': {'x': float(start_x), 'y': float(start_y)},
                                'end': {'x': float(end_x), 'y': float(end_y)},
                                'thickness': 0.1,
                                'height': 2.4,
                                'angle': float(angle),
                                'confidence': float(np.mean(wall_grid[wall_start:y, x]))
                            })
                        wall_start = None

            return walls

        except Exception as e:
            logger.error(f"Error in wall prediction: {str(e)}")
            return []

    # Blueprint Refinement methods

    def train_blueprint_refinement_model(self, feedback_data: List = None,
                                        reward_weights: Dict = None,
                                        learning_rate: float = 0.01,
                                        discount_factor: float = 0.9) -> Dict:
        """Train the blueprint refinement model using reinforcement learning."""
        if reward_weights is None:
            reward_weights = {
                'room_size': 0.3,
                'wall_alignment': 0.4,
                'flow_efficiency': 0.3
            }

        try:
            # Get feedback data from database if not provided
            if not feedback_data:
                query = """
                SELECT original_blueprint, modified_blueprint, feedback_score
                FROM ai_blueprint_feedback
                WHERE feedback_score IS NOT NULL
                """
                results = execute_query(query)

                if not results or len(results) < 5:
                    return {
                        "success": False,
                        "error": "Not enough feedback data (minimum 5 samples required)"
                    }

                feedback_data = [
                    {
                        'original': json.loads(row[0]),
                        'modified': json.loads(row[1]),
                        'score': row[2]
                    }
                    for row in results
                ]

            # Create blueprint environment
            env = self._create_blueprint_environment(reward_weights)

            # Initialize model
            model = PPO('MlpPolicy', env, learning_rate=learning_rate, gamma=discount_factor, verbose=1)

            # Train model
            model.learn(total_timesteps=10000)

            # Save model
            model_path = MODEL_DIR / 'blueprint_refinement_model.zip'
            model.save(model_path)

            # Update model in memory
            self.blueprint_refinement_model = model

            # Save model info to database
            metrics_json = json.dumps({
                'reward_weights': reward_weights,
                'learning_rate': learning_rate,
                'discount_factor': discount_factor,
                'samples': len(feedback_data)
            })

            execute_write_query("""
            INSERT INTO ai_models (model_name, model_type, model_path, metrics)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                model_type = VALUES(model_type),
                model_path = VALUES(model_path),
                metrics = VALUES(metrics),
                created_at = CURRENT_TIMESTAMP
            """, ('blueprint_refinement', 'ppo', model_path, metrics_json))

            logger.info(f"Trained blueprint refinement model with {len(feedback_data)} feedback samples")

            return {
                "success": True,
                "samples": len(feedback_data),
                "reward_weights": reward_weights
            }

        except Exception as e:
            logger.error(f"Failed to train blueprint refinement model: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def _create_blueprint_environment(self, reward_weights):
        """Create a gym environment for blueprint refinement."""
        # This is a simplified environment for blueprint refinement
        class BlueprintEnv(gym.Env):
            def __init__(self, reward_weights):
                super().__init__()
                self.reward_weights = reward_weights

                # Define action and observation space
                self.action_space = gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(10,), dtype=np.float32
                )

                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
                )

                self.blueprint = None
                self.step_count = 0

            def reset(self):
                self.blueprint = self._generate_random_blueprint()
                self.step_count = 0
                return self._get_observation()

            def step(self, action):
                self.step_count += 1

                # Apply action to modify blueprint
                self._apply_action(action)

                # Calculate reward
                reward = self._calculate_reward()

                # Check if done
                done = self.step_count >= 20

                return self._get_observation(), reward, done, {}

            def _generate_random_blueprint(self):
                # Generate a simple random blueprint for training
                return {
                    'rooms': [
                        {
                            'center': {'x': np.random.uniform(0, 10), 'y': np.random.uniform(0, 10), 'z': 0},
                            'dimensions': {'width': np.random.uniform(2, 5), 'length': np.random.uniform(2, 5), 'height': 2.4}
                        }
                        for _ in range(np.random.randint(1, 5))
                    ],
                    'walls': [
                        {
                            'start': {'x': np.random.uniform(0, 10), 'y': np.random.uniform(0, 10)},
                            'end': {'x': np.random.uniform(0, 10), 'y': np.random.uniform(0, 10)},
                            'thickness': 0.1,
                            'height': 2.4
                        }
                        for _ in range(np.random.randint(5, 15))
                    ]
                }

            def _apply_action(self, action):
                # Apply action to modify blueprint
                if len(self.blueprint['rooms']) > 0:
                    room_idx = np.random.randint(0, len(self.blueprint['rooms']))
                    room = self.blueprint['rooms'][room_idx]

                    # Modify room dimensions
                    room['dimensions']['width'] = max(1.5, room['dimensions']['width'] + action[0])
                    room['dimensions']['length'] = max(1.5, room['dimensions']['length'] + action[1])

                    # Modify room position
                    room['center']['x'] += action[2]
                    room['center']['y'] += action[3]

                if len(self.blueprint['walls']) > 0:
                    wall_idx = np.random.randint(0, len(self.blueprint['walls']))
                    wall = self.blueprint['walls'][wall_idx]

                    # Modify wall position
                    wall['start']['x'] += action[4]
                    wall['start']['y'] += action[5]
                    wall['end']['x'] += action[6]
                    wall['end']['y'] += action[7]

            def _get_observation(self):
                # Convert blueprint to observation vector
                obs = np.zeros(20)

                # Add room features
                room_count = min(2, len(self.blueprint['rooms']))
                for i in range(room_count):
                    room = self.blueprint['rooms'][i]
                    idx = i * 5
                    obs[idx] = room['center']['x']
                    obs[idx+1] = room['center']['y']
                    obs[idx+2] = room['dimensions']['width']
                    obs[idx+3] = room['dimensions']['length']
                    obs[idx+4] = room['dimensions']['height']

                # Add wall features
                wall_count = min(2, len(self.blueprint['walls']))
                for i in range(wall_count):
                    wall = self.blueprint['walls'][i]
                    idx = 10 + i * 5
                    obs[idx] = wall['start']['x']
                    obs[idx+1] = wall['start']['y']
                    obs[idx+2] = wall['end']['x']
                    obs[idx+3] = wall['end']['y']
                    obs[idx+4] = wall['thickness']

                return obs

            def _calculate_reward(self):
                # Calculate reward based on blueprint quality
                reward = 0

                # Room size reward
                room_size_reward = 0
                for room in self.blueprint['rooms']:
                    area = room['dimensions']['width'] * room['dimensions']['length']
                    if 6 <= area <= 30:  # Reasonable room size
                        room_size_reward += 1
                    else:
                        room_size_reward -= 1

                # Wall alignment reward
                wall_alignment_reward = 0
                for wall in self.blueprint['walls']:
                    # Check if wall is horizontal or vertical
                    dx = wall['end']['x'] - wall['start']['x']
                    dy = wall['end']['y'] - wall['start']['y']

                    if abs(dx) > abs(dy):  # Horizontal wall
                        wall_alignment_reward += 1 if abs(dy) < 0.1 else -1
                    else:  # Vertical wall
                        wall_alignment_reward += 1 if abs(dx) < 0.1 else -1

                # Flow efficiency reward (simplified)
                flow_efficiency_reward = 0

                # Combine rewards using weights
                reward = (
                    self.reward_weights['room_size'] * room_size_reward +
                    self.reward_weights['wall_alignment'] * wall_alignment_reward +
                    self.reward_weights['flow_efficiency'] * flow_efficiency_reward
                )

                return reward

        # Create and return the environment
        env = BlueprintEnv(reward_weights)
        return DummyVecEnv([lambda: env])

    def refine_blueprint(self, blueprint: Dict) -> Dict:
        """Refine a blueprint using the trained reinforcement learning model."""
        if not self.blueprint_refinement_model:
            return blueprint

        try:
            # Create a copy of the blueprint to refine
            refined_blueprint = json.loads(json.dumps(blueprint))

            # Convert blueprint to observation
            obs = self._blueprint_to_observation(refined_blueprint)

            # Get action from model
            action, _ = self.blueprint_refinement_model.predict(obs)

            # Apply action to refine blueprint
            refined_blueprint = self._apply_refinement_action(refined_blueprint, action)

            # Save refinement for future training
            self._save_blueprint_refinement(blueprint, refined_blueprint)

            return refined_blueprint

        except Exception as e:
            logger.error(f"Error refining blueprint: {str(e)}")
            return blueprint

    def _blueprint_to_observation(self, blueprint: Dict) -> np.ndarray:
        """Convert blueprint to observation vector for RL model."""
        obs = np.zeros(20)

        # Add room features
        room_count = min(2, len(blueprint.get('rooms', [])))
        for i in range(room_count):
            room = blueprint['rooms'][i]
            idx = i * 5
            obs[idx] = room.get('center', {}).get('x', 0)
            obs[idx+1] = room.get('center', {}).get('y', 0)
            obs[idx+2] = room.get('dimensions', {}).get('width', 0)
            obs[idx+3] = room.get('dimensions', {}).get('length', 0)
            obs[idx+4] = room.get('dimensions', {}).get('height', 0)

        # Add wall features
        wall_count = min(2, len(blueprint.get('walls', [])))
        for i in range(wall_count):
            wall = blueprint['walls'][i]
            idx = 10 + i * 5
            obs[idx] = wall.get('start', {}).get('x', 0)
            obs[idx+1] = wall.get('start', {}).get('y', 0)
            obs[idx+2] = wall.get('end', {}).get('x', 0)
            obs[idx+3] = wall.get('end', {}).get('y', 0)
            obs[idx+4] = wall.get('thickness', 0)

        return obs

    def _apply_refinement_action(self, blueprint: Dict, action: np.ndarray) -> Dict:
        """Apply refinement action to blueprint."""
        # Modify rooms
        if 'rooms' in blueprint and blueprint['rooms']:
            room_count = min(2, len(blueprint['rooms']))
            for i in range(room_count):
                room = blueprint['rooms'][i]

                # Modify room dimensions (with constraints)
                if 'dimensions' in room:
                    room['dimensions']['width'] = max(1.5, min(15, room['dimensions'].get('width', 3) + action[i*2]))
                    room['dimensions']['length'] = max(1.5, min(15, room['dimensions'].get('length', 3) + action[i*2+1]))

                # Update room bounds if present
                if 'center' in room and 'bounds' in room:
                    center_x = room['center'].get('x', 0)
                    center_y = room['center'].get('y', 0)
                    width = room['dimensions'].get('width', 3)
                    length = room['dimensions'].get('length', 3)

                    room['bounds']['min'] = {
                        'x': center_x - width/2,
                        'y': center_y - length/2,
                        'z': room['bounds']['min'].get('z', 0)
                    }

                    room['bounds']['max'] = {
                        'x': center_x + width/2,
                        'y': center_y + length/2,
                        'z': room['bounds']['max'].get('z', 2.4)
                    }

        # Align walls
        if 'walls' in blueprint and blueprint['walls']:
            for wall in blueprint['walls']:
                # Check if wall is nearly horizontal or vertical
                dx = wall['end'].get('x', 0) - wall['start'].get('x', 0)
                dy = wall['end'].get('y', 0) - wall['start'].get('y', 0)

                if abs(dx) > abs(dy):  # Horizontal wall
                    # Make it perfectly horizontal
                    wall['end']['y'] = wall['start'].get('y', 0)
                else:  # Vertical wall
                    # Make it perfectly vertical
                    wall['end']['x'] = wall['start'].get('x', 0)

        return blueprint

    def _save_blueprint_refinement(self, original: Dict, refined: Dict) -> None:
        """Save blueprint refinement for future training."""
        try:
            # Generate a unique ID for this refinement
            blueprint_id = original.get('id', str(uuid.uuid4()))

            # Save to database
            execute_write_query("""
            INSERT INTO ai_blueprint_feedback
            (blueprint_id, original_blueprint, modified_blueprint)
            VALUES (%s, %s, %s)
            """, (
                blueprint_id,
                json.dumps(original),
                json.dumps(refined)
            ))

            logger.debug(f"Saved blueprint refinement for future training: {blueprint_id}")

        except Exception as e:
            logger.error(f"Failed to save blueprint refinement: {str(e)}")
