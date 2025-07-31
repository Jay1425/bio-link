#!/usr/bin/env python3
"""
BioLink ML Service - Medical AI Model Integration
Handles loading and inference for multiple medical diagnostic models
GPU Optimized for RTX 4060
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import cv2
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
import time

# Configure GPU settings for RTX 4060
def configure_gpu():
    """Configure GPU settings for optimal performance on RTX 4060"""
    try:
        # Check if GPU is available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth to prevent TensorFlow from allocating all GPU memory
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Set up mixed precision for better performance on RTX cards
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                
                print(f"✅ GPU Configuration successful: {len(gpus)} GPU(s) detected")
                for i, gpu in enumerate(gpus):
                    print(f"   GPU {i}: {gpu.name}")
                
                return True
            except RuntimeError as e:
                print(f"⚠️  GPU configuration error: {e}")
                return False
        else:
            print("⚠️  No GPU detected, using CPU")
            return False
    except Exception as e:
        print(f"⚠️  GPU setup failed: {e}")
        return False

# Configure GPU on import
GPU_AVAILABLE = configure_gpu()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLModelService:
    """Service class for managing and running medical AI models"""
    
    def __init__(self, model_dir: str = "ml_dl_model"):
        self.model_dir = model_dir
        self.models = {}
        self.gpu_available = GPU_AVAILABLE
        self.model_configs = {
            'brain_tumor': {
                'file': 'brain_tumor_model.h5',
                'input_size': (224, 224),
                'classes': ['No Tumor', 'Tumor Detected'],  # Simplified for binary output
                'description': 'Brain Tumor Detection Model',
                'type': 'binary',
                'medical_category': 'Brain Cancer Detection',
                'cancer_indicator_classes': ['Tumor Detected']  # Classes that indicate cancer/abnormality
            },
            'lung_cancer': {
                'file': 'lung_cancer_classifier.h5',
                'input_size': (224, 224),
                'classes': ['Normal', 'Benign', 'Malignant'],  # Simplified for actual output structure
                'description': 'Lung Cancer Detection Model',
                'type': 'multiclass',
                'medical_category': 'Lung Cancer Detection',
                'cancer_indicator_classes': ['Malignant']  # Classes that indicate cancer
            },
            'skin_cancer_melanoma': {
                'file': 'skin_cancer_melonoma_model.h5',
                'input_size': (224, 224),
                'classes': ['Benign', 'Melanoma'],
                'description': 'Melanoma Detection Model',
                'type': 'binary',
                'medical_category': 'Skin Cancer Detection',
                'cancer_indicator_classes': ['Melanoma']
            },
            'skin_cancer_dermato': {
                'file': 'skin_cancer_dermato_model.h5',
                'input_size': (224, 224),
                'classes': ['Actinic Keratosis', 'Basal Cell Carcinoma', 'Dermatofibroma', 'Melanoma', 'Nevus', 'Pigmented Benign Keratosis', 'Seborrheic Keratosis', 'Squamous Cell Carcinoma', 'Vascular Lesion'],
                'description': 'Comprehensive Skin Cancer Classification Model',
                'type': 'multiclass',
                'medical_category': 'Skin Cancer Detection',
                'cancer_indicator_classes': ['Basal Cell Carcinoma', 'Melanoma', 'Squamous Cell Carcinoma']
            },
            'skin_cancer_ak': {
                'file': 'skin_cancer_ak_model.h5',
                'input_size': (224, 224),
                'classes': ['Normal', 'Actinic Keratosis'],
                'description': 'Actinic Keratosis Detection Model',
                'type': 'binary',
                'medical_category': 'Pre-Cancer Detection',
                'cancer_indicator_classes': ['Actinic Keratosis']  # Pre-cancerous condition
            },
            'skin_cancer_nevus': {
                'file': 'skin_cancer_nevus_model.h5',
                'input_size': (224, 224),
                'classes': ['Normal', 'Suspicious Nevus'],
                'description': 'Nevus (Mole) Analysis Model',
                'type': 'binary',
                'medical_category': 'Skin Analysis',
                'cancer_indicator_classes': ['Suspicious Nevus']
            },
            'seborrheic_keratosis': {
                'file': 'seborrheic_keratosis_classifier.h5',
                'input_size': (224, 224),
                'classes': ['Normal', 'Seborrheic Keratosis'],
                'description': 'Seborrheic Keratosis Detection Model',
                'type': 'binary',
                'medical_category': 'Skin Analysis',
                'cancer_indicator_classes': []  # Benign condition
            },
            'skin_disease': {
                'file': 'skin_disease_classifier.h5',
                'input_size': (224, 224),
                'classes': ['Acne', 'Eczema', 'Fungal', 'Normal', 'Psoriasis'],
                'description': 'General Skin Disease Classification Model',
                'type': 'multiclass',
                'medical_category': 'Skin Disease Detection',
                'cancer_indicator_classes': []  # No cancer indicators
            }
        }
        
        # Load models on initialization
        self.load_models()
    
    def load_models(self):
        """Load all available models with GPU optimization"""
        logger.info("Loading ML models...")
        logger.info(f"GPU Available: {self.gpu_available}")
        
        for model_name, config in self.model_configs.items():
            model_path = os.path.join(self.model_dir, config['file'])
            
            if os.path.exists(model_path):
                try:
                    start_time = time.time()
                    
                    # Handle DTypePolicy compatibility issues
                    custom_objects = {
                        'DTypePolicy': tf.keras.mixed_precision.Policy,
                        'Policy': tf.keras.mixed_precision.Policy
                    }
                    
                    # Load model with GPU context if available
                    if self.gpu_available:
                        with tf.device('/GPU:0'):
                            model = keras.models.load_model(
                                model_path, 
                                compile=False,
                                custom_objects=custom_objects
                            )
                    else:
                        model = keras.models.load_model(
                            model_path, 
                            compile=False,
                            custom_objects=custom_objects
                        )
                    
                    self.models[model_name] = model
                    load_time = time.time() - start_time
                    
                    device = "GPU" if self.gpu_available else "CPU"
                    logger.info(f"✅ Loaded {model_name} model on {device} ({load_time:.2f}s)")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load {model_name}: {e}")
            else:
                logger.warning(f"⚠️  Model file not found: {model_path}")
        
        logger.info(f"Loaded {len(self.models)} models successfully")
        if self.gpu_available:
            logger.info("🚀 Models optimized for RTX 4060 GPU acceleration")
    
    def preprocess_image(self, image_path: str, target_size: Tuple[int, int]) -> np.ndarray:
        """Preprocess image for model prediction"""
        try:
            # Load and resize image
            image = Image.open(image_path)
            image = image.convert('RGB')
            image = image.resize(target_size)
            
            # Convert to numpy array and normalize
            image_array = np.array(image, dtype=np.float32)
            image_array = image_array / 255.0
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            raise
    
    def predict(self, model_name: str, image_path: str) -> Dict:
        """Make prediction using specified model with GPU optimization"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not available")
        
        if model_name not in self.model_configs:
            raise ValueError(f"Configuration for model '{model_name}' not found")
        
        config = self.model_configs[model_name]
        model = self.models[model_name]
        
        try:
            start_time = time.time()
            
            # Preprocess image
            processed_image = self.preprocess_image(image_path, config['input_size'])
            preprocess_time = time.time() - start_time
            
            # Make prediction with GPU optimization
            prediction_start = time.time()
            if self.gpu_available:
                with tf.device('/GPU:0'):
                    predictions = model.predict(processed_image, verbose=0)
            else:
                predictions = model.predict(processed_image, verbose=0)
            
            prediction_time = time.time() - prediction_start
            total_time = time.time() - start_time
            
            # Handle different model types and get clear medical results
            if config['type'] == 'binary':
                # Binary classification (e.g., brain tumor model outputs single probability)
                confidence = float(predictions[0][0])
                predicted_class_idx = 1 if confidence > 0.5 else 0
                predicted_class = config['classes'][predicted_class_idx]
                
                # Create probabilities for both classes
                class_probabilities = {
                    config['classes'][0]: 1.0 - confidence,  # No cancer/normal
                    config['classes'][1]: confidence          # Cancer/abnormal
                }
            else:
                # Multi-class classification (e.g., lung cancer with 3 classes)
                probabilities = predictions[0]
                predicted_class_idx = np.argmax(probabilities)
                confidence = float(probabilities[predicted_class_idx])
                predicted_class = config['classes'][predicted_class_idx]
                
                # Create probabilities for all classes
                class_probabilities = {}
                for i, class_name in enumerate(config['classes']):
                    if i < len(probabilities):
                        class_probabilities[class_name] = float(probabilities[i])
            
            # Determine if this indicates cancer or abnormality
            cancer_indicators = config.get('cancer_indicator_classes', [])
            is_cancer_detected = predicted_class in cancer_indicators
            
            # Calculate cancer risk level
            if is_cancer_detected:
                if confidence > 0.8:
                    risk_level = "High"
                elif confidence > 0.6:
                    risk_level = "Moderate"
                else:
                    risk_level = "Low"
            else:
                risk_level = "Normal"
            
            # Create simplified medical interpretation
            if is_cancer_detected:
                medical_interpretation = f"⚠️ ABNORMALITY DETECTED: {predicted_class}"
                recommendation = "Please consult with a medical professional immediately for further evaluation."
            else:
                medical_interpretation = f"✅ NORMAL: {predicted_class}"
                recommendation = "No immediate concerns detected. Continue regular health monitoring."
            
            result = {
                'model_name': model_name,
                'model_description': config['description'],
                'predicted_class': predicted_class,
                'confidence': confidence,
                'confidence_percentage': round(confidence * 100, 2),
                'class_probabilities': class_probabilities,
                'all_classes': config['classes'],
                'prediction_timestamp': datetime.now().isoformat(),
                'image_path': image_path,
                
                # Medical Analysis Results
                'is_cancer_detected': is_cancer_detected,
                'risk_level': risk_level,
                'medical_interpretation': medical_interpretation,
                'recommendation': recommendation,
                'medical_category': config['medical_category'],
                'cancer_indicator_classes': cancer_indicators,
                
                # Performance metrics
                'processing_time_ms': round(total_time * 1000, 2),
                'preprocess_time_ms': round(preprocess_time * 1000, 2),
                'prediction_time_ms': round(prediction_time * 1000, 2),
                'device_used': 'GPU' if self.gpu_available else 'CPU'
            }
            
            device = "GPU" if self.gpu_available else "CPU"
            logger.info(f"Prediction completed for {model_name} on {device}: {predicted_class} ({confidence:.2%}) in {total_time*1000:.1f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction with {model_name}: {e}")
            raise
    
    def get_available_models(self) -> List[Dict]:
        """Get list of available models with their descriptions"""
        available_models = []
        
        for model_name, config in self.model_configs.items():
            if model_name in self.models:
                available_models.append({
                    'name': model_name,
                    'description': config['description'],
                    'classes': config['classes'],
                    'input_size': config['input_size'],
                    'loaded': True
                })
            else:
                available_models.append({
                    'name': model_name,
                    'description': config['description'],
                    'classes': config['classes'],
                    'input_size': config['input_size'],
                    'loaded': False
                })
        
        return available_models
    
    def get_model_recommendations(self, image_type: str = None) -> List[str]:
        """Get recommended models based on image type or medical area"""
        recommendations = {
            'brain': ['brain_tumor'],
            'lung': ['lung_cancer'],
            'skin': ['skin_cancer_dermato', 'skin_cancer_melanoma', 'skin_disease'],
            'dermatology': ['skin_cancer_dermato', 'skin_cancer_melanoma', 'seborrheic_keratosis', 'skin_disease'],
            'oncology': ['brain_tumor', 'lung_cancer', 'skin_cancer_melanoma', 'skin_cancer_dermato']
        }
        
        if image_type and image_type.lower() in recommendations:
            return recommendations[image_type.lower()]
        
        # Return all available models if no specific type
        return list(self.models.keys())
    
    def batch_predict(self, image_path: str, model_names: List[str] = None) -> Dict:
        """Run multiple models on the same image"""
        if model_names is None:
            model_names = list(self.models.keys())
        
        batch_start_time = time.time()
        
        results = {
            'image_path': image_path,
            'predictions': {},
            'batch_timestamp': datetime.now().isoformat(),
            'models_used': len(model_names),
            'models_processed': 0
        }
        
        for model_name in model_names:
            if model_name in self.models:
                try:
                    prediction = self.predict(model_name, image_path)
                    results['predictions'][model_name] = prediction
                    results['models_processed'] += 1
                except Exception as e:
                    logger.error(f"Batch prediction failed for {model_name}: {e}")
                    results['predictions'][model_name] = {
                        'error': str(e),
                        'model_name': model_name
                    }
        
        # Add total processing time
        total_time = time.time() - batch_start_time
        results['total_processing_time_ms'] = round(total_time * 1000, 2)
        
        return results

# Global ML service instance
ml_service = None

def initialize_ml_service(model_dir: str = "ml_dl_model") -> MLModelService:
    """Initialize the global ML service"""
    global ml_service
    ml_service = MLModelService(model_dir)
    return ml_service

def get_ml_service() -> MLModelService:
    """Get the global ML service instance"""
    global ml_service
    if ml_service is None:
        ml_service = initialize_ml_service()
    return ml_service

if __name__ == "__main__":
    # Test the ML service
    service = MLModelService()
    print("Available models:", service.get_available_models())
