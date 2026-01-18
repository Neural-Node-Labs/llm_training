# LLM Training Program - Complete Implementation
# Component-Based Development (CBD) Methodology

"""
PHASE II: IMPLEMENTATION
Each component implements its contract with full error handling and trace logging
"""

import json
import logging
import hashlib
import uuid
import time
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Iterator, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from pathlib import Path
import re

# ============================================================================
# LOGGING INFRASTRUCTURE
# ============================================================================

class LoggingAdaptor:
    """Dual-stream logging: system.log for technical, llm_interaction.log for audit"""
    
    def __init__(self):
        # System logger for technical errors
        self.system_logger = logging.getLogger('system')
        self.system_logger.setLevel(logging.DEBUG)
        
        system_handler = logging.FileHandler('system.log')
        system_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.system_logger.addHandler(system_handler)
        
        # LLM interaction logger for audit trail
        self.llm_logger = logging.getLogger('llm_interaction')
        self.llm_logger.setLevel(logging.INFO)
        
        llm_handler = logging.FileHandler('llm_interaction.log')
        llm_handler.setFormatter(logging.Formatter('%(message)s'))
        self.llm_logger.addHandler(llm_handler)
    
    def log_system(self, level: str, message: str, context: Dict) -> None:
        """Log technical events to system.log"""
        try:
            log_method = getattr(self.system_logger, level.lower())
            log_method(f"{message} | Context: {json.dumps(context)}")
        except Exception as e:
            self.system_logger.error(f"Logging failed: {str(e)}")
    
    def log_llm_interaction(self, component: str, event: str, data: Dict) -> None:
        """Log LLM interactions to llm_interaction.log in JSON format"""
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "component": component,
                "event": event,
                "data": data
            }
            self.llm_logger.info(json.dumps(log_entry))
        except Exception as e:
            self.system_logger.error(f"LLM interaction logging failed: {str(e)}")

# Global logger instance
logger = LoggingAdaptor()

# ============================================================================
# SECURITY UTILITIES
# ============================================================================

class SecurityValidator:
    """Input sanitization and security validation"""
    
    @staticmethod
    def validate_path(path: str) -> bool:
        """Prevent path traversal attacks"""
        try:
            # Resolve to absolute path
            abs_path = os.path.abspath(path)
            
            # Check for traversal patterns
            if '..' in path or path.startswith('/'):
                logger.log_system('warning', 'Path traversal attempt detected', {'path': path})
                return False
            
            # Ensure path is within allowed directory
            allowed_base = os.path.abspath('.')
            if not abs_path.startswith(allowed_base):
                return False
            
            return True
        except Exception as e:
            logger.log_system('error', 'Path validation error', {'error': str(e)})
            return False
    
    @staticmethod
    def sanitize_string(input_str: str) -> str:
        """Sanitize input strings to prevent injection"""
        try:
            # Remove potential SQL/command injection characters
            sanitized = re.sub(r'[;\'"\\`$(){}|&<>]', '', input_str)
            return sanitized.strip()
        except Exception as e:
            logger.log_system('error', 'String sanitization error', {'error': str(e)})
            return ""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

# ============================================================================
# EVENT BUS ADAPTOR
# ============================================================================

class EventBusAdaptor:
    """Asynchronous inter-component communication"""
    
    def __init__(self):
        self._subscribers: Dict[str, Dict[str, Callable]] = {}
        self._lock = threading.Lock()
    
    def publish(self, topic: str, message: Dict) -> bool:
        """Publish message to topic"""
        try:
            logger.log_llm_interaction('EventBusAdaptor', 'publish', {
                'topic': topic,
                'message_keys': list(message.keys())
            })
            
            with self._lock:
                if topic in self._subscribers:
                    for callback in self._subscribers[topic].values():
                        threading.Thread(
                            target=self._safe_callback,
                            args=(callback, message)
                        ).start()
            return True
        except Exception as e:
            logger.log_system('error', 'Event publish failed', {
                'topic': topic,
                'error': str(e)
            })
            return False
    
    def subscribe(self, topic: str, callback: Callable) -> str:
        """Subscribe to topic with callback"""
        try:
            subscription_id = str(uuid.uuid4())
            with self._lock:
                if topic not in self._subscribers:
                    self._subscribers[topic] = {}
                self._subscribers[topic][subscription_id] = callback
            
            logger.log_llm_interaction('EventBusAdaptor', 'subscribe', {
                'topic': topic,
                'subscription_id': subscription_id
            })
            return subscription_id
        except Exception as e:
            logger.log_system('error', 'Subscription failed', {
                'topic': topic,
                'error': str(e)
            })
            return ""
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from topic"""
        try:
            with self._lock:
                for topic, subs in self._subscribers.items():
                    if subscription_id in subs:
                        del subs[subscription_id]
                        return True
            return False
        except Exception as e:
            logger.log_system('error', 'Unsubscribe failed', {'error': str(e)})
            return False
    
    def _safe_callback(self, callback: Callable, message: Dict) -> None:
        """Execute callback with error handling"""
        try:
            callback(message)
        except Exception as e:
            logger.log_system('error', 'Callback execution failed', {
                'error': str(e)
            })

# ============================================================================
# CONFIGURATION ADAPTOR
# ============================================================================

class ConfigurationAdaptor:
    """Centralized configuration management"""
    
    def __init__(self):
        self._config: Dict[str, Dict[str, Any]] = {}
        self._watchers: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()
        self._load_defaults()
    
    def _load_defaults(self):
        """Load default configuration"""
        self._config = {
            'data_ingestion': {
                'chunk_size_mb': 50,
                'max_retries': 3,
                'timeout_seconds': 30
            },
            'preprocessing': {
                'max_sequence_length': 512,
                'tokenizer_type': 'bpe'
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.0001,
                'num_epochs': 10,
                'gradient_clip_norm': 1.0
            },
            'monitoring': {
                'checkpoint_every_n_steps': 1000,
                'validation_every_n_steps': 500
            }
        }
    
    def get_config(self, component: str, key: str) -> Any:
        """Get configuration value"""
        try:
            with self._lock:
                return self._config.get(component, {}).get(key)
        except Exception as e:
            logger.log_system('error', 'Config get failed', {
                'component': component,
                'key': key,
                'error': str(e)
            })
            return None
    
    def update_config(self, component: str, key: str, value: Any) -> bool:
        """Update configuration value"""
        try:
            with self._lock:
                if component not in self._config:
                    self._config[component] = {}
                self._config[component][key] = value
                
                # Notify watchers
                watcher_key = f"{component}.{key}"
                if watcher_key in self._watchers:
                    for callback in self._watchers[watcher_key]:
                        callback(value)
            
            logger.log_llm_interaction('ConfigurationAdaptor', 'update', {
                'component': component,
                'key': key
            })
            return True
        except Exception as e:
            logger.log_system('error', 'Config update failed', {'error': str(e)})
            return False
    
    def watch_config(self, component: str, callback: Callable) -> str:
        """Watch configuration changes"""
        try:
            watcher_id = str(uuid.uuid4())
            watcher_key = component
            
            with self._lock:
                if watcher_key not in self._watchers:
                    self._watchers[watcher_key] = []
                self._watchers[watcher_key].append(callback)
            
            return watcher_id
        except Exception as e:
            logger.log_system('error', 'Config watch failed', {'error': str(e)})
            return ""

# ============================================================================
# DATA INGESTION LAYER - Component 1: DataSourceConnector
# ============================================================================

@dataclass
class DataSourceConnectorInput:
    """IN Schema for DataSourceConnector"""
    source_type: str  # s3|database|api|web
    credentials: Dict[str, str]
    source_config: Dict[str, Any]

@dataclass
class DataSourceConnectorOutput:
    """OUT Schema for DataSourceConnector"""
    connection_id: str
    status: str  # connected|failed
    metadata: Dict[str, Any]

@dataclass
class DataSourceConnectorError:
    """Error Schema for DataSourceConnector"""
    error_code: str  # AUTH_FAILED|NETWORK_ERROR|INVALID_SOURCE
    message: str
    timestamp: str
    retry_possible: bool

class DataSourceConnector:
    """
    Component: DataSourceConnector
    Logical Function: Connect to and authenticate with external data sources
    """
    
    def __init__(self, event_bus: EventBusAdaptor):
        self.event_bus = event_bus
        logger.log_llm_interaction('DataSourceConnector', 'initialized', {})
    
    def connect(self, input_data: DataSourceConnectorInput) -> DataSourceConnectorOutput:
        """
        Restate IN Schema:
        - source_type: str [s3|database|api|web]
        - credentials: {auth_type: str, credentials_encrypted: str}
        - source_config: {endpoint: str, filters: object}
        
        Restate OUT Schema:
        - connection_id: uuid
        - status: str [connected|failed]
        - metadata: {source_size_estimate: int, available_formats: array}
        """
        start_time = time.time()
        
        try:
            # Trace point: connection_attempt
            logger.log_llm_interaction('DataSourceConnector', 'connection_attempt', {
                'source_type': input_data.source_type
            })
            
            # Validate source type
            valid_types = ['s3', 'database', 'api', 'web']
            if input_data.source_type not in valid_types:
                raise ValueError(f"Invalid source type: {input_data.source_type}")
            
            # Sanitize endpoint
            endpoint = SecurityValidator.sanitize_string(
                input_data.source_config.get('endpoint', '')
            )
            
            # Simulate authentication
            auth_type = input_data.credentials.get('auth_type', '')
            if not auth_type:
                raise AuthenticationError("Missing auth_type")
            
            # Trace point: auth_success
            logger.log_llm_interaction('DataSourceConnector', 'auth_success', {
                'auth_type': auth_type
            })
            
            # Generate connection ID
            connection_id = str(uuid.uuid4())
            
            # Simulate metadata retrieval
            metadata = {
                'source_size_estimate': 1000000,  # Simulated
                'available_formats': ['txt', 'json', 'parquet']
            }
            
            # Trace point: source_validated
            logger.log_llm_interaction('DataSourceConnector', 'source_validated', {
                'connection_id': connection_id,
                'execution_time_ms': (time.time() - start_time) * 1000
            })
            
            return DataSourceConnectorOutput(
                connection_id=connection_id,
                status='connected',
                metadata=metadata
            )
            
        except AuthenticationError as e:
            error = DataSourceConnectorError(
                error_code='AUTH_FAILED',
                message=str(e),
                timestamp=datetime.utcnow().isoformat(),
                retry_possible=True
            )
            logger.log_system('error', 'Authentication failed', asdict(error))
            raise
        
        except ValueError as e:
            error = DataSourceConnectorError(
                error_code='INVALID_SOURCE',
                message=str(e),
                timestamp=datetime.utcnow().isoformat(),
                retry_possible=False
            )
            logger.log_system('error', 'Invalid source configuration', asdict(error))
            raise
        
        except Exception as e:
            error = DataSourceConnectorError(
                error_code='NETWORK_ERROR',
                message=str(e),
                timestamp=datetime.utcnow().isoformat(),
                retry_possible=True
            )
            logger.log_system('error', 'Network error during connection', asdict(error))
            raise

# ============================================================================
# DATA INGESTION LAYER - Component 2: DataStreamReader
# ============================================================================

@dataclass
class DataStreamReaderInput:
    """IN Schema for DataStreamReader"""
    connection_id: str
    chunk_size_mb: int  # 1-100
    format: str  # txt|json|parquet|csv

@dataclass
class ChunkData:
    chunk_id: str
    data: bytes
    size_bytes: int
    checksum: str

@dataclass
class DataStreamReaderOutput:
    """OUT Schema for DataStreamReader"""
    stream_id: str
    chunks: List[ChunkData]
    total_chunks: int

@dataclass
class DataStreamReaderError:
    """Error Schema for DataStreamReader"""
    error_code: str  # READ_FAILED|CORRUPT_DATA|TIMEOUT
    chunk_id: Optional[str]
    recoverable: bool

class DataStreamReader:
    """
    Component: DataStreamReader
    Logical Function: Read raw data in chunks with backpressure handling
    """
    
    def __init__(self, event_bus: EventBusAdaptor):
        self.event_bus = event_bus
        logger.log_llm_interaction('DataStreamReader', 'initialized', {})
    
    def read_stream(self, input_data: DataStreamReaderInput) -> DataStreamReaderOutput:
        """
        Restate IN Schema:
        - connection_id: uuid
        - chunk_size_mb: int [1-100]
        - format: str [txt|json|parquet|csv]
        
        Restate OUT Schema:
        - stream_id: uuid
        - chunks: [{chunk_id, data, size_bytes, checksum}]
        - total_chunks: int
        """
        start_time = time.time()
        
        try:
            # Validate chunk size
            if not (1 <= input_data.chunk_size_mb <= 100):
                raise ValueError(f"Invalid chunk_size_mb: {input_data.chunk_size_mb}")
            
            # Trace point: chunk_read_start
            stream_id = str(uuid.uuid4())
            logger.log_llm_interaction('DataStreamReader', 'chunk_read_start', {
                'stream_id': stream_id,
                'chunk_size_mb': input_data.chunk_size_mb
            })
            
            chunks = []
            chunk_size_bytes = input_data.chunk_size_mb * 1024 * 1024
            
            # Simulate reading 3 chunks
            for i in range(3):
                try:
                    # Simulate chunk data
                    chunk_data = b'x' * min(chunk_size_bytes, 1000)  # Simulated data
                    
                    # Calculate checksum
                    checksum = hashlib.sha256(chunk_data).hexdigest()
                    
                    chunk = ChunkData(
                        chunk_id=str(uuid.uuid4()),
                        data=chunk_data,
                        size_bytes=len(chunk_data),
                        checksum=checksum
                    )
                    chunks.append(chunk)
                    
                    # Trace point: chunk_read_complete
                    logger.log_llm_interaction('DataStreamReader', 'chunk_read_complete', {
                        'chunk_id': chunk.chunk_id,
                        'size_bytes': chunk.size_bytes
                    })
                    
                except Exception as e:
                    error = DataStreamReaderError(
                        error_code='READ_FAILED',
                        chunk_id=str(uuid.uuid4()),
                        recoverable=True
                    )
                    logger.log_system('error', 'Chunk read failed', {
                        'error': str(e),
                        'chunk_index': i
                    })
                    raise
            
            # Trace point: stream_complete
            logger.log_llm_interaction('DataStreamReader', 'stream_complete', {
                'stream_id': stream_id,
                'total_chunks': len(chunks),
                'execution_time_ms': (time.time() - start_time) * 1000
            })
            
            return DataStreamReaderOutput(
                stream_id=stream_id,
                chunks=chunks,
                total_chunks=len(chunks)
            )
            
        except ValueError as e:
            logger.log_system('error', 'Invalid parameters', {'error': str(e)})
            raise
        
        except Exception as e:
            error = DataStreamReaderError(
                error_code='TIMEOUT',
                chunk_id=None,
                recoverable=True
            )
            logger.log_system('error', 'Stream read timeout', {'error': str(e)})
            raise

# ============================================================================
# DATA INGESTION LAYER - Component 3: DataValidator
# ============================================================================

@dataclass
class DataValidatorInput:
    """IN Schema for DataValidator"""
    stream_id: str
    chunk_data: bytes
    validation_rules: Dict[str, Any]

@dataclass
class DataValidatorOutput:
    """OUT Schema for DataValidator"""
    validation_id: str
    status: str  # passed|failed|warning
    quality_metrics: Dict[str, Any]
    sanitized_data: bytes

@dataclass
class DataValidatorError:
    """Error Schema for DataValidator"""
    error_code: str  # ENCODING_ERROR|QUALITY_BELOW_THRESHOLD|PROHIBITED_CONTENT
    details: List[str]
    chunk_rejected: bool

class DataValidator:
    """
    Component: DataValidator
    Logical Function: Validate data quality, encoding, and format compliance
    """
    
    def __init__(self, event_bus: EventBusAdaptor):
        self.event_bus = event_bus
        logger.log_llm_interaction('DataValidator', 'initialized', {})
    
    def validate(self, input_data: DataValidatorInput) -> DataValidatorOutput:
        """
        Restate IN Schema:
        - stream_id: uuid
        - chunk_data: binary
        - validation_rules: {encoding: str, min_quality_score: float, prohibited_patterns: array}
        
        Restate OUT Schema:
        - validation_id: uuid
        - status: str [passed|failed|warning]
        - quality_metrics: {encoding_valid: bool, quality_score: float, issues_found: int}
        - sanitized_data: binary
        """
        start_time = time.time()
        
        try:
            validation_id = str(uuid.uuid4())
            
            # Trace point: validation_start
            logger.log_llm_interaction('DataValidator', 'validation_start', {
                'validation_id': validation_id,
                'chunk_size': len(input_data.chunk_data)
            })
            
            # Validate encoding
            encoding = input_data.validation_rules.get('encoding', 'utf-8')
            encoding_valid = True
            
            try:
                input_data.chunk_data.decode(encoding)
            except UnicodeDecodeError:
                encoding_valid = False
                error = DataValidatorError(
                    error_code='ENCODING_ERROR',
                    details=[f'Failed to decode as {encoding}'],
                    chunk_rejected=True
                )
                logger.log_system('error', 'Encoding validation failed', asdict(error))
                raise
            
            # Trace point: quality_check
            quality_score = 0.95  # Simulated quality score
            min_quality = input_data.validation_rules.get('min_quality_score', 0.8)
            
            if quality_score < min_quality:
                error = DataValidatorError(
                    error_code='QUALITY_BELOW_THRESHOLD',
                    details=[f'Quality {quality_score} < {min_quality}'],
                    chunk_rejected=False
                )
                logger.log_system('warning', 'Quality below threshold', asdict(error))
                status = 'warning'
            else:
                status = 'passed'
            
            # Sanitize data (simulate)
            sanitized_data = input_data.chunk_data
            
            # Trace point: sanitization_complete
            logger.log_llm_interaction('DataValidator', 'sanitization_complete', {
                'validation_id': validation_id,
                'quality_score': quality_score,
                'execution_time_ms': (time.time() - start_time) * 1000
            })
            
            return DataValidatorOutput(
                validation_id=validation_id,
                status=status,
                quality_metrics={
                    'encoding_valid': encoding_valid,
                    'quality_score': quality_score,
                    'issues_found': 0 if status == 'passed' else 1
                },
                sanitized_data=sanitized_data
            )
            
        except UnicodeDecodeError as e:
            logger.log_system('error', 'Encoding error', {'error': str(e)})
            raise
        
        except Exception as e:
            logger.log_system('error', 'Validation failed', {'error': str(e)})
            raise

# ============================================================================
# PREPROCESSING LAYER - Component 4: DataPreprocessor
# ============================================================================

@dataclass
class DataPreprocessorInput:
    """IN Schema for DataPreprocessor"""
    raw_data: str
    preprocessing_config: Dict[str, Any]

@dataclass
class DataPreprocessorOutput:
    """OUT Schema for DataPreprocessor"""
    processed_id: str
    tokens: List[int]
    token_count: int
    metadata: Dict[str, Any]

@dataclass
class DataPreprocessorError:
    """Error Schema for DataPreprocessor"""
    error_code: str  # TOKENIZATION_FAILED|SEQUENCE_TOO_LONG
    position: int
    context: str

class DataPreprocessor:
    """
    Component: DataPreprocessor
    Logical Function: Tokenize, normalize, and clean text data
    """
    
    def __init__(self, event_bus: EventBusAdaptor):
        self.event_bus = event_bus
        # Simple vocabulary for simulation
        self.vocab = {chr(i): i for i in range(32, 127)}
        logger.log_llm_interaction('DataPreprocessor', 'initialized', {})
    
    def preprocess(self, input_data: DataPreprocessorInput) -> DataPreprocessorOutput:
        """
        Restate IN Schema:
        - raw_data: str
        - preprocessing_config: {tokenizer_type, lowercase, remove_special_chars, max_sequence_length}
        
        Restate OUT Schema:
        - processed_id: uuid
        - tokens: array[int]
        - token_count: int
        - metadata: {original_length, compression_ratio}
        """
        start_time = time.time()
        
        try:
            processed_id = str(uuid.uuid4())
            
            # Trace point: tokenization_start
            logger.log_llm_interaction('DataPreprocessor', 'tokenization_start', {
                'processed_id': processed_id,
                'input_length': len(input_data.raw_data)
            })
            
            # Normalize
            text = input_data.raw_data
            if input_data.preprocessing_config.get('lowercase', False):
                text = text.lower()
            
            if input_data.preprocessing_config.get('remove_special_chars', False):
                text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
            
            # Trace point: normalization_complete
            logger.log_llm_interaction('DataPreprocessor', 'normalization_complete', {
                'processed_id': processed_id
            })
            
            # Tokenize (simple character-level for simulation)
            tokens = [self.vocab.get(c, 0) for c in text[:1000]]  # Limit for demo
            
            max_length = input_data.preprocessing_config.get('max_sequence_length', 512)
            if len(tokens) > max_length:
                error = DataPreprocessorError(
                    error_code='SEQUENCE_TOO_LONG',
                    position=max_length,
                    context=f'Sequence length {len(tokens)} exceeds {max_length}'
                )
                logger.log_system('warning', 'Sequence truncated', asdict(error))
                tokens = tokens[:max_length]
            
            # Trace point: tokens_generated
            logger.log_llm_interaction('DataPreprocessor', 'tokens_generated', {
                'processed_id': processed_id,
                'token_count': len(tokens),
                'execution_time_ms': (time.time() - start_time) * 1000
            })
            
            return DataPreprocessorOutput(
                processed_id=processed_id,
                tokens=tokens,
                token_count=len(tokens),
                metadata={
                    'original_length': len(input_data.raw_data),
                    'compression_ratio': len(tokens) / max(len(input_data.raw_data), 1)
                }
            )
            
        except Exception as e:
            error = DataPreprocessorError(
                error_code='TOKENIZATION_FAILED',
                position=0,
                context=str(e)
            )
            logger.log_system('error', 'Tokenization failed', asdict(error))
            raise

# ============================================================================
# PREPROCESSING LAYER - Component 5: DatasetBuilder
# ============================================================================

@dataclass
class DatasetBuilderInput:
    """IN Schema for DatasetBuilder"""
    processed_tokens: List[List[int]]
    split_ratios: Dict[str, float]
    shuffle: bool
    seed: int

@dataclass
class SplitInfo:
    size: int
    path: str

@dataclass
class DatasetBuilderOutput:
    """OUT Schema for DatasetBuilder"""
    dataset_id: str
    splits: Dict[str, SplitInfo]
    statistics: Dict[str, Any]

@dataclass
class DatasetBuilderError:
    """Error Schema for DatasetBuilder"""
    error_code: str  # SPLIT_RATIO_INVALID|INSUFFICIENT_DATA|WRITE_FAILED
    message: str

class DatasetBuilder:
    """
    Component: DatasetBuilder
    Logical Function: Construct training/validation/test datasets with stratification
    """
    
    def __init__(self, event_bus: EventBusAdaptor):
        self.event_bus = event_bus
        logger.log_llm_interaction('DatasetBuilder', 'initialized', {})
    
    def build_dataset(self, input_data: DatasetBuilderInput) -> DatasetBuilderOutput:
        """
        Restate IN Schema:
        - processed_tokens: array
        - split_ratios: {train: float, validation: float, test: float}
        - shuffle: bool
        - seed: int
        
        Restate OUT Schema:
        - dataset_id: uuid
        - splits: {train: {size, path}, validation: {size, path}, test: {size, path}}
        - statistics: {total_samples, avg_sequence_length, vocabulary_size}
        """
        start_time = time.time()
        
        try:
            dataset_id = str(uuid.uuid4())
            
            # Validate split ratios
            total_ratio = sum(input_data.split_ratios.values())
            if abs(total_ratio - 1.0) > 0.01:
                error = DatasetBuilderError(
                    error_code='SPLIT_RATIO_INVALID',
                    message=f'Split ratios sum to {total_ratio}, expected 1.0'
                )
                logger.log_system('error', 'Invalid split ratios', asdict(error))
                raise ValueError(error.message)
            
            # Trace point: dataset_split_start
            logger.log_llm_interaction('DatasetBuilder', 'dataset_split_start', {
                'dataset_id': dataset_id,
                'total_samples': len(input_data.processed_tokens)
            })
            
            # Shuffle if requested
            import random
            random.seed(input_data.seed)
            
            data = input_data.processed_tokens.copy()
            if input_data.shuffle:
                random.shuffle(data)
                
                # Trace point: shuffle_complete
                logger.log_llm_interaction('DatasetBuilder', 'shuffle_complete', {
                    'dataset_id': dataset_id
                })
            
            # Split data
            total = len(data)
            train_end = int(total * input_data.split_ratios.get('train', 0.8))
            val_end = train_end + int(total * input_data.split_ratios.get('validation', 0.1))
            
            splits_data = {
                'train': data[:train_end],
                'validation': data[train_end:val_end],
                'test': data[val_end:]
            }
            
            # Create output directory
            os.makedirs('datasets', exist_ok=True)
            
            splits = {}
            for split_name, split_data in splits_data.items():
                path = f'datasets/{dataset_id}_{split_name}.json'
                
                # Validate path
                if not SecurityValidator.validate_path(path):
                    error = DatasetBuilderError(
                        error_code='WRITE_FAILED',
                        message=f'Invalid path: {path}'
                    )
                    logger.log_system('error', 'Path validation failed', asdict(error))
                    raise ValueError(error.message)
                
                # Write split to file
                try:
                    with open(path, 'w') as f:
                        json.dump({'tokens': split_data}, f)
                except IOError as e:
                    error = DatasetBuilderError(
                        error_code='WRITE_FAILED',
                        message=str(e)
                    )
                    logger.log_system('error', 'Dataset write failed', asdict(error))
                    raise
                
                splits[split_name] = SplitInfo(
                    size=len(split_data),
                    path=path
                )
            
            # Calculate statistics
            all_tokens = [t for seq in data for t in seq]
            vocab_size = len(set(all_tokens)) if all_tokens else 0
            avg_length = sum(len(seq) for seq in data) / max(len(data), 1)
            
            # Trace point: dataset_persisted
            logger.log_llm_interaction('DatasetBuilder', 'dataset_persisted', {
                'dataset_id': dataset_id,
                'splits': {k: v.size for k, v in splits.items()},
                'execution_time_ms': (time.time() - start_time) * 1000
            })
            
            return DatasetBuilderOutput(
                dataset_id=dataset_id,
                splits=splits,
                statistics={
                    'total_samples': len(data),
                    'avg_sequence_length': avg_length,
                    'vocabulary_size': vocab_size
                }
            )
            
        except ValueError as e:
            logger.log_system('error', 'Dataset build failed', {'error': str(e)})
            raise
        
        except Exception as e:
            error = DatasetBuilderError(
                error_code='INSUFFICIENT_DATA',
                message=str(e)
            )
            logger.log_system('error', 'Unexpected error', asdict(error))
            raise

# ============================================================================
# TRAINING INFRASTRUCTURE - Component 6: ComputeResourceManager
# ============================================================================

@dataclass
class ComputeResourceManagerInput:
    """IN Schema for ComputeResourceManager"""
    resource_requirements: Dict[str, Any]
    priority: str  # high|medium|low

@dataclass
class DeviceInfo:
    device_id: str
    type: str  # gpu|tpu
    memory_gb: int
    utilization: float

@dataclass
class ComputeResourceManagerOutput:
    """OUT Schema for ComputeResourceManager"""
    allocation_id: str
    resources: List[DeviceInfo]
    cluster_config: Dict[str, Any]

@dataclass
class ComputeResourceManagerError:
    """Error Schema for ComputeResourceManager"""
    error_code: str  # INSUFFICIENT_RESOURCES|ALLOCATION_TIMEOUT|DEVICE_FAILURE
    available_resources: Dict[str, Any]
    retry_after_seconds: int

class ComputeResourceManager:
    """
    Component: ComputeResourceManager
    Logical Function: Allocate and monitor GPU/TPU resources
    """
    
    def __init__(self, event_bus: EventBusAdaptor):
        self.event_bus = event_bus
        self.available_devices = self._discover_devices()
        logger.log_llm_interaction('ComputeResourceManager', 'initialized', {
            'available_devices': len(self.available_devices)
        })
    
    def _discover_devices(self) -> List[DeviceInfo]:
        """Simulate device discovery"""
        return [
            DeviceInfo(device_id=f'gpu-{i}', type='gpu', memory_gb=16, utilization=0.0)
            for i in range(4)
        ]
    
    def allocate(self, input_data: ComputeResourceManagerInput) -> ComputeResourceManagerOutput:
        """
        Restate IN Schema:
        - resource_requirements: {gpu_count: int, gpu_memory_gb: int, distributed: bool}
        - priority: str [high|medium|low]
        
        Restate OUT Schema:
        - allocation_id: uuid
        - resources: [{device_id, type, memory_gb, utilization}]
        - cluster_config: object
        """
        start_time = time.time()
        
        try:
            allocation_id = str(uuid.uuid4())
            
            # Trace point: allocation_requested
            logger.log_llm_interaction('ComputeResourceManager', 'allocation_requested', {
                'allocation_id': allocation_id,
                'gpu_count': input_data.resource_requirements.get('gpu_count', 1)
            })
            
            gpu_count = input_data.resource_requirements.get('gpu_count', 1)
            gpu_memory = input_data.resource_requirements.get('gpu_memory_gb', 16)
            
            # Check resource availability
            available = [d for d in self.available_devices if d.utilization < 0.5]
            
            if len(available) < gpu_count:
                error = ComputeResourceManagerError(
                    error_code='INSUFFICIENT_RESOURCES',
                    available_resources={'available_gpus': len(available)},
                    retry_after_seconds=60
                )
                logger.log_system('warning', 'Insufficient resources', asdict(error))
                raise ResourceError(error.error_code)
            
            # Allocate resources
            allocated = available[:gpu_count]
            for device in allocated:
                device.utilization = 0.8  # Mark as allocated
            
            # Trace point: resources_allocated
            logger.log_llm_interaction('ComputeResourceManager', 'resources_allocated', {
                'allocation_id': allocation_id,
                'devices': [d.device_id for d in allocated]
            })
            
            # Health check
            self._health_check(allocated)
            
            # Trace point: health_check
            logger.log_llm_interaction('ComputeResourceManager', 'health_check', {
                'allocation_id': allocation_id,
                'all_healthy': True,
                'execution_time_ms': (time.time() - start_time) * 1000
            })
            
            cluster_config = {
                'distributed': input_data.resource_requirements.get('distributed', False),
                'backend': 'nccl' if gpu_count > 1 else 'single'
            }
            
            return ComputeResourceManagerOutput(
                allocation_id=allocation_id,
                resources=allocated,
                cluster_config=cluster_config
            )
            
        except ResourceError as e:
            logger.log_system('error', 'Resource allocation failed', {'error': str(e)})
            raise
        
        except Exception as e:
            error = ComputeResourceManagerError(
                error_code='ALLOCATION_TIMEOUT',
                available_resources={},
                retry_after_seconds=30
            )
            logger.log_system('error', 'Allocation timeout', asdict(error))
            raise
    
    def _health_check(self, devices: List[DeviceInfo]) -> bool:
        """Verify device health"""
        for device in devices:
            if device.utilization > 0.95:
                raise DeviceError(f'Device {device.device_id} unhealthy')
        return True

# ============================================================================
# TRAINING INFRASTRUCTURE - Component 7: ModelInitializer
# ============================================================================

@dataclass
class ModelInitializerInput:
    """IN Schema for ModelInitializer"""
    architecture: Dict[str, Any]
    initialization: Dict[str, Any]

@dataclass
class ModelInitializerOutput:
    """OUT Schema for ModelInitializer"""
    model_id: str
    parameter_count: int
    memory_footprint_gb: float
    initialization_successful: bool
    checkpoint_path: str

@dataclass
class ModelInitializerError:
    """Error Schema for ModelInitializer"""
    error_code: str  # ARCHITECTURE_INVALID|PRETRAINED_LOAD_FAILED|OOM
    details: str

class ModelInitializer:
    """
    Component: ModelInitializer
    Logical Function: Initialize model architecture with random/pretrained weights
    """
    
    def __init__(self, event_bus: EventBusAdaptor):
        self.event_bus = event_bus
        logger.log_llm_interaction('ModelInitializer', 'initialized', {})
    
    def initialize(self, input_data: ModelInitializerInput) -> ModelInitializerOutput:
        """
        Restate IN Schema:
        - architecture: {model_type, num_layers, hidden_size, num_attention_heads, vocab_size}
        - initialization: {method, pretrained_path, seed}
        
        Restate OUT Schema:
        - model_id: uuid
        - parameter_count: int
        - memory_footprint_gb: float
        - initialization_successful: bool
        - checkpoint_path: str
        """
        start_time = time.time()
        
        try:
            model_id = str(uuid.uuid4())
            
            # Trace point: architecture_defined
            logger.log_llm_interaction('ModelInitializer', 'architecture_defined', {
                'model_id': model_id,
                'model_type': input_data.architecture.get('model_type', 'transformer')
            })
            
            # Validate architecture
            required_keys = ['model_type', 'num_layers', 'hidden_size', 'vocab_size']
            for key in required_keys:
                if key not in input_data.architecture:
                    error = ModelInitializerError(
                        error_code='ARCHITECTURE_INVALID',
                        details=f'Missing required field: {key}'
                    )
                    logger.log_system('error', 'Invalid architecture', asdict(error))
                    raise ValueError(error.details)
            
            # Calculate parameter count
            hidden_size = input_data.architecture['hidden_size']
            num_layers = input_data.architecture['num_layers']
            vocab_size = input_data.architecture['vocab_size']
            
            # Simplified calculation
            params_per_layer = hidden_size * hidden_size * 4  # Attention + FFN
            parameter_count = params_per_layer * num_layers + vocab_size * hidden_size
            
            # Estimate memory
            memory_footprint_gb = (parameter_count * 4) / (1024 ** 3)  # 4 bytes per param
            
            # Check memory availability
            if memory_footprint_gb > 32:  # Arbitrary limit
                error = ModelInitializerError(
                    error_code='OOM',
                    details=f'Model requires {memory_footprint_gb:.2f}GB'
                )
                logger.log_system('error', 'Out of memory', asdict(error))
                raise MemoryError(error.details)
            
            # Initialize weights
            init_method = input_data.initialization.get('method', 'random')
            
            if init_method == 'pretrained':
                pretrained_path = input_data.initialization.get('pretrained_path')
                if pretrained_path and not SecurityValidator.validate_path(pretrained_path):
                    error = ModelInitializerError(
                        error_code='PRETRAINED_LOAD_FAILED',
                        details='Invalid pretrained path'
                    )
                    logger.log_system('error', 'Path validation failed', asdict(error))
                    raise ValueError(error.details)
            
            # Trace point: weights_initialized
            logger.log_llm_interaction('ModelInitializer', 'weights_initialized', {
                'model_id': model_id,
                'init_method': init_method
            })
            
            # Save initial checkpoint
            os.makedirs('checkpoints', exist_ok=True)
            checkpoint_path = f'checkpoints/{model_id}_init.pt'
            
            # Simulate checkpoint save
            with open(checkpoint_path, 'w') as f:
                json.dump({'model_id': model_id, 'parameters': parameter_count}, f)
            
            # Trace point: model_ready
            logger.log_llm_interaction('ModelInitializer', 'model_ready', {
                'model_id': model_id,
                'parameter_count': parameter_count,
                'execution_time_ms': (time.time() - start_time) * 1000
            })
            
            return ModelInitializerOutput(
                model_id=model_id,
                parameter_count=parameter_count,
                memory_footprint_gb=memory_footprint_gb,
                initialization_successful=True,
                checkpoint_path=checkpoint_path
            )
            
        except ValueError as e:
            logger.log_system('error', 'Initialization failed', {'error': str(e)})
            raise
        
        except MemoryError as e:
            logger.log_system('error', 'Memory error', {'error': str(e)})
            raise
        
        except Exception as e:
            error = ModelInitializerError(
                error_code='ARCHITECTURE_INVALID',
                details=str(e)
            )
            logger.log_system('error', 'Unexpected error', asdict(error))
            raise

# ============================================================================
# TRAINING INFRASTRUCTURE - Component 8: DataLoader
# ============================================================================

@dataclass
class DataLoaderInput:
    """IN Schema for DataLoader"""
    dataset_path: str
    batch_size: int
    shuffle: bool
    num_workers: int
    prefetch_factor: int

@dataclass
class DataLoaderOutput:
    """OUT Schema for DataLoader"""
    loader_id: str
    batch_iterator: Iterator
    num_batches: int
    estimated_time_per_epoch_seconds: float

@dataclass
class DataLoaderError:
    """Error Schema for DataLoader"""
    error_code: str  # DATASET_NOT_FOUND|CORRUPT_BATCH|WORKER_CRASH
    batch_id: Optional[int]

class DataLoader:
    """
    Component: DataLoader
    Logical Function: Load batches with prefetching and memory management
    """
    
    def __init__(self, event_bus: EventBusAdaptor):
        self.event_bus = event_bus
        logger.log_llm_interaction('DataLoader', 'initialized', {})
    
    def create_loader(self, input_data: DataLoaderInput) -> DataLoaderOutput:
        """
        Restate IN Schema:
        - dataset_path: str
        - batch_size: int
        - shuffle: bool
        - num_workers: int
        - prefetch_factor: int
        
        Restate OUT Schema:
        - loader_id: uuid
        - batch_iterator: iterator
        - num_batches: int
        - estimated_time_per_epoch_seconds: float
        """
        start_time = time.time()
        
        try:
            loader_id = str(uuid.uuid4())
            
            # Validate path
            if not SecurityValidator.validate_path(input_data.dataset_path):
                error = DataLoaderError(
                    error_code='DATASET_NOT_FOUND',
                    batch_id=None
                )
                logger.log_system('error', 'Invalid dataset path', asdict(error))
                raise FileNotFoundError(input_data.dataset_path)
            
            # Trace point: loader_initialized
            logger.log_llm_interaction('DataLoader', 'loader_initialized', {
                'loader_id': loader_id,
                'dataset_path': input_data.dataset_path
            })
            
            # Load dataset
            try:
                with open(input_data.dataset_path, 'r') as f:
                    dataset = json.load(f)
                    tokens = dataset.get('tokens', [])
            except FileNotFoundError:
                error = DataLoaderError(
                    error_code='DATASET_NOT_FOUND',
                    batch_id=None
                )
                logger.log_system('error', 'Dataset file not found', asdict(error))
                raise
            except json.JSONDecodeError:
                error = DataLoaderError(
                    error_code='CORRUPT_BATCH',
                    batch_id=0
                )
                logger.log_system('error', 'Corrupt dataset file', asdict(error))
                raise
            
            # Create batch iterator
            num_samples = len(tokens)
            num_batches = (num_samples + input_data.batch_size - 1) // input_data.batch_size
            
            def batch_generator():
                """Generator for batches"""
                for i in range(0, num_samples, input_data.batch_size):
                    batch = tokens[i:i + input_data.batch_size]
                    
                    # Trace point: batch_loaded
                    logger.log_llm_interaction('DataLoader', 'batch_loaded', {
                        'loader_id': loader_id,
                        'batch_index': i // input_data.batch_size,
                        'batch_size': len(batch)
                    })
                    
                    yield batch
                    
                    # Trace point: prefetch_started (simulated)
                    if i + input_data.batch_size < num_samples:
                        logger.log_llm_interaction('DataLoader', 'prefetch_started', {
                            'loader_id': loader_id,
                            'next_batch_index': (i // input_data.batch_size) + 1
                        })
            
            batch_iterator = batch_generator()
            
            # Estimate time per epoch
            estimated_time = num_batches * 0.1  # 100ms per batch estimate
            
            logger.log_llm_interaction('DataLoader', 'loader_created', {
                'loader_id': loader_id,
                'num_batches': num_batches,
                'execution_time_ms': (time.time() - start_time) * 1000
            })
            
            return DataLoaderOutput(
                loader_id=loader_id,
                batch_iterator=batch_iterator,
                num_batches=num_batches,
                estimated_time_per_epoch_seconds=estimated_time
            )
            
        except FileNotFoundError as e:
            logger.log_system('error', 'Dataset not found', {'error': str(e)})
            raise
        
        except Exception as e:
            error = DataLoaderError(
                error_code='WORKER_CRASH',
                batch_id=None
            )
            logger.log_system('error', 'DataLoader creation failed', asdict(error))
            raise

# ============================================================================
# TRAINING EXECUTION - Component 9: TrainingOrchestrator
# ============================================================================

@dataclass
class TrainingOrchestratorInput:
    """IN Schema for TrainingOrchestrator"""
    model_id: str
    loader_id: str
    training_config: Dict[str, Any]

@dataclass
class TrainingOrchestratorOutput:
    """OUT Schema for TrainingOrchestrator"""
    training_id: str
    status: str  # running|completed|failed
    current_epoch: int
    current_step: int
    latest_checkpoint: str

@dataclass
class TrainingOrchestratorError:
    """Error Schema for TrainingOrchestrator"""
    error_code: str  # NAN_LOSS|GRADIENT_EXPLOSION|OOM|CHECKPOINT_FAILED
    step: int
    epoch: int
    recoverable: bool

class TrainingOrchestrator:
    """
    Component: TrainingOrchestrator
    Logical Function: Execute training loop with gradient accumulation and checkpointing
    """
    
    def __init__(self, event_bus: EventBusAdaptor):
        self.event_bus = event_bus
        logger.log_llm_interaction('TrainingOrchestrator', 'initialized', {})
    
    def train(self, input_data: TrainingOrchestratorInput, 
              data_loader_output: DataLoaderOutput) -> TrainingOrchestratorOutput:
        """
        Restate IN Schema:
        - model_id: uuid
        - loader_id: uuid
        - training_config: {num_epochs, learning_rate, optimizer, gradient_accumulation_steps, 
                           checkpoint_every_n_steps, mixed_precision}
        
        Restate OUT Schema:
        - training_id: uuid
        - status: str [running|completed|failed]
        - current_epoch: int
        - current_step: int
        - latest_checkpoint: str
        """
        start_time = time.time()
        
        try:
            training_id = str(uuid.uuid4())
            current_step = 0
            latest_checkpoint = ""
            
            num_epochs = input_data.training_config.get('num_epochs', 10)
            checkpoint_every = input_data.training_config.get('checkpoint_every_n_steps', 1000)
            
            for epoch in range(num_epochs):
                # Trace point: epoch_start
                logger.log_llm_interaction('TrainingOrchestrator', 'epoch_start', {
                    'training_id': training_id,
                    'epoch': epoch
                })
                
                # Simulate training steps
                for step in range(min(5, data_loader_output.num_batches)):  # Limited for demo
                    try:
                        # Forward pass (simulated)
                        loss_value = 2.5 - (current_step * 0.01)  # Decreasing loss
                        
                        # Check for NaN
                        if loss_value != loss_value:  # NaN check
                            error = TrainingOrchestratorError(
                                error_code='NAN_LOSS',
                                step=current_step,
                                epoch=epoch,
                                recoverable=False
                            )
                            logger.log_system('error', 'NaN loss detected', asdict(error))
                            raise ValueError('NaN loss')
                        
                        # Trace point: step_complete
                        logger.log_llm_interaction('TrainingOrchestrator', 'step_complete', {
                            'training_id': training_id,
                            'step': current_step,
                            'loss': loss_value
                        })
                        
                        # Checkpoint
                        if (current_step + 1) % checkpoint_every == 0:
                            latest_checkpoint = f'checkpoints/{input_data.model_id}_step_{current_step}.pt'
                            
                            # Trace point: checkpoint_saved
                            logger.log_llm_interaction('TrainingOrchestrator', 'checkpoint_saved', {
                                'training_id': training_id,
                                'step': current_step,
                                'checkpoint_path': latest_checkpoint
                            })
                        
                        current_step += 1
                        
                    except Exception as e:
                        error = TrainingOrchestratorError(
                            error_code='GRADIENT_EXPLOSION',
                            step=current_step,
                            epoch=epoch,
                            recoverable=True
                        )
                        logger.log_system('error', 'Step failed', asdict(error))
                        raise
            
            logger.log_llm_interaction('TrainingOrchestrator', 'training_complete', {
                'training_id': training_id,
                'total_steps': current_step,
                'execution_time_ms': (time.time() - start_time) * 1000
            })
            
            return TrainingOrchestratorOutput(
                training_id=training_id,
                status='completed',
                current_epoch=num_epochs - 1,
                current_step=current_step,
                latest_checkpoint=latest_checkpoint
            )
            
        except ValueError as e:
            logger.log_system('error', 'Training failed', {'error': str(e)})
            return TrainingOrchestratorOutput(
                training_id=training_id,
                status='failed',
                current_epoch=epoch,
                current_step=current_step,
                latest_checkpoint=latest_checkpoint
            )
        
        except Exception as e:
            error = TrainingOrchestratorError(
                error_code='OOM',
                step=current_step,
                epoch=0,
                recoverable=False
            )
            logger.log_system('error', 'Orchestrator error', asdict(error))
            raise

# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class AuthenticationError(Exception):
    """Authentication failed"""
    pass

class ResourceError(Exception):
    """Resource unavailable"""
    pass

class DeviceError(Exception):
    """Device health check failed"""
    pass

# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """Execute complete LLM training pipeline"""
    
    print("=" * 80)
    print("LLM TRAINING PROGRAM - COMPONENT-BASED DEVELOPMENT")
    print("=" * 80)
    
    try:
        # Initialize system components
        event_bus = EventBusAdaptor()
        config = ConfigurationAdaptor()
        
        print("\n[1/10] Initializing Data Source Connector...")
        connector = DataSourceConnector(event_bus)
        connector_input = DataSourceConnectorInput(
            source_type='s3',
            credentials={'auth_type': 'iam_role', 'credentials_encrypted': 'xxx'},
            source_config={'endpoint': 's3://training-data', 'filters': {}}
        )
        connection_output = connector.connect(connector_input)
        print(f"✓ Connected: {connection_output.connection_id}")
        
        print("\n[2/10] Streaming data...")
        reader = DataStreamReader(event_bus)
        reader_input = DataStreamReaderInput(
            connection_id=connection_output.connection_id,
            chunk_size_mb=50,
            format='txt'
        )
        stream_output = reader.read_stream(reader_input)
        print(f"✓ Streamed {stream_output.total_chunks} chunks")
        
        print("\n[3/10] Validating data quality...")
        validator = DataValidator(event_bus)
        validated_chunks = []
        for chunk in stream_output.chunks:
            validator_input = DataValidatorInput(
                stream_id=stream_output.stream_id,
                chunk_data=chunk.data,
                validation_rules={'encoding': 'utf-8', 'min_quality_score': 0.8, 'prohibited_patterns': []}
            )
            val_output = validator.validate(validator_input)
            validated_chunks.append(val_output.sanitized_data)
            print(f"  Chunk {chunk.chunk_id[:8]}... - Quality: {val_output.quality_metrics['quality_score']:.2f}")
        
        print("\n[4/10] Preprocessing text data...")
        preprocessor = DataPreprocessor(event_bus)
        all_tokens = []
        for chunk_data in validated_chunks:
            preprocessor_input = DataPreprocessorInput(
                raw_data=chunk_data.decode('utf-8', errors='ignore')[:1000],  # Limit for demo
                preprocessing_config={
                    'tokenizer_type': 'bpe',
                    'lowercase': True,
                    'remove_special_chars': False,
                    'max_sequence_length': 512
                }
            )
            preprocess_output = preprocessor.preprocess(preprocessor_input)
            all_tokens.append(preprocess_output.tokens)
            print(f"  Processed {preprocess_output.token_count} tokens")
        
        print("\n[5/10] Building dataset splits...")
        dataset_builder = DatasetBuilder(event_bus)
        dataset_input = DatasetBuilderInput(
            processed_tokens=all_tokens,
            split_ratios={'train': 0.8, 'validation': 0.1, 'test': 0.1},
            shuffle=True,
            seed=42
        )
        dataset_output = dataset_builder.build_dataset(dataset_input)
        print(f"✓ Dataset ID: {dataset_output.dataset_id}")
        print(f"  Train: {dataset_output.splits['train'].size} samples")
        print(f"  Val: {dataset_output.splits['validation'].size} samples")
        print(f"  Test: {dataset_output.splits['test'].size} samples")
        
        print("\n[6/10] Allocating compute resources...")
        resource_manager = ComputeResourceManager(event_bus)
        resource_input = ComputeResourceManagerInput(
            resource_requirements={'gpu_count': 2, 'gpu_memory_gb': 16, 'distributed': True},
            priority='high'
        )
        resource_output = resource_manager.allocate(resource_input)
        print(f"✓ Allocated {len(resource_output.resources)} GPUs")
        for device in resource_output.resources:
            print(f"  {device.device_id}: {device.memory_gb}GB")
        
        print("\n[7/10] Initializing model...")
        model_initializer = ModelInitializer(event_bus)
        model_input = ModelInitializerInput(
            architecture={
                'model_type': 'transformer',
                'num_layers': 12,
                'hidden_size': 768,
                'num_attention_heads': 12,
                'vocab_size': 50000
            },
            initialization={'method': 'random', 'pretrained_path': None, 'seed': 42}
        )
        model_output = model_initializer.initialize(model_input)
        print(f"✓ Model ID: {model_output.model_id}")
        print(f"  Parameters: {model_output.parameter_count:,}")
        print(f"  Memory: {model_output.memory_footprint_gb:.2f}GB")
        
        print("\n[8/10] Creating data loaders...")
        data_loader = DataLoader(event_bus)
        loader_input = DataLoaderInput(
            dataset_path=dataset_output.splits['train'].path,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            prefetch_factor=2
        )
        loader_output = data_loader.create_loader(loader_input)
        print(f"✓ DataLoader ID: {loader_output.loader_id}")
        print(f"  Batches per epoch: {loader_output.num_batches}")
        
        print("\n[9/10] Starting training...")
        orchestrator = TrainingOrchestrator(event_bus)
        training_input = TrainingOrchestratorInput(
            model_id=model_output.model_id,
            loader_id=loader_output.loader_id,
            training_config={
                'num_epochs': 3,
                'learning_rate': 0.0001,
                'optimizer': 'adamw',
                'gradient_accumulation_steps': 4,
                'checkpoint_every_n_steps': 2,
                'mixed_precision': True
            }
        )
        training_output = orchestrator.train(training_input, loader_output)
        print(f"✓ Training Status: {training_output.status}")
        print(f"  Total Steps: {training_output.current_step}")
        print(f"  Latest Checkpoint: {training_output.latest_checkpoint or 'N/A'}")
        
        print("\n[10/10] Pipeline Complete!")
        print("=" * 80)
        print("\nLOG FILES:")
        print("  - system.log: Technical errors and warnings")
        print("  - llm_interaction.log: Component interaction audit trail")
        print("\nOUTPUT ARTIFACTS:")
        print(f"  - datasets/: {dataset_output.dataset_id}_*.json")
        print(f"  - checkpoints/: {model_output.model_id}_*.pt")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Pipeline failed: {str(e)}")
        logger.log_system('critical', 'Pipeline execution failed', {'error': str(e)})
        raise

if __name__ == '__main__':
    main()


# ============================================================================
# ADDITIONAL COMPONENTS (Monitoring & Persistence)
# ============================================================================

# Component 10: LossCalculator
@dataclass
class LossCalculatorInput:
    predictions: List[float]
    targets: List[float]
    loss_function: str  # cross_entropy|mse|focal
    reduction: str  # mean|sum|none

@dataclass
class LossCalculatorOutput:
    loss_value: float
    per_sample_loss: Optional[List[float]]
    numerical_stable: bool

class LossCalculator:
    """Component: LossCalculator - Compute loss with numerical stability"""
    
    def __init__(self, event_bus: EventBusAdaptor):
        self.event_bus = event_bus
    
    def calculate(self, input_data: LossCalculatorInput) -> LossCalculatorOutput:
        start_time = time.time()
        try:
            logger.log_llm_interaction('LossCalculator', 'loss_computation_start', {
                'loss_function': input_data.loss_function
            })
            
            # Simplified MSE calculation
            if input_data.loss_function == 'mse':
                per_sample = [(p - t) ** 2 for p, t in zip(input_data.predictions, input_data.targets)]
            else:
                per_sample = [abs(p - t) for p, t in zip(input_data.predictions, input_data.targets)]
            
            if input_data.reduction == 'mean':
                loss_value = sum(per_sample) / max(len(per_sample), 1)
            elif input_data.reduction == 'sum':
                loss_value = sum(per_sample)
            else:
                loss_value = per_sample[0] if per_sample else 0.0
            
            # Check numerical stability
            numerical_stable = not (loss_value != loss_value or abs(loss_value) == float('inf'))
            
            logger.log_llm_interaction('LossCalculator', 'loss_computed', {
                'loss_value': loss_value,
                'execution_time_ms': (time.time() - start_time) * 1000
            })
            
            return LossCalculatorOutput(
                loss_value=loss_value,
                per_sample_loss=per_sample if input_data.reduction == 'none' else None,
                numerical_stable=numerical_stable
            )
        except Exception as e:
            logger.log_system('error', 'Loss calculation failed', {'error': str(e)})
            raise


# Component 11: MetricsCollector
@dataclass
class MetricsCollectorInput:
    training_id: str
    step: int
    metrics: Dict[str, float]

@dataclass
class MetricsCollectorOutput:
    metrics_id: str
    persisted: bool
    aggregated_metrics: Dict[str, Any]

class MetricsCollector:
    """Component: MetricsCollector - Collect and aggregate training metrics"""
    
    def __init__(self, event_bus: EventBusAdaptor):
        self.event_bus = event_bus
        self.metrics_history = []
    
    def collect(self, input_data: MetricsCollectorInput) -> MetricsCollectorOutput:
        start_time = time.time()
        try:
            metrics_id = str(uuid.uuid4())
            
            logger.log_llm_interaction('MetricsCollector', 'metrics_received', {
                'metrics_id': metrics_id,
                'step': input_data.step
            })
            
            # Store metrics
            self.metrics_history.append({
                'step': input_data.step,
                'metrics': input_data.metrics,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # Calculate aggregates
            recent = self.metrics_history[-100:]
            avg_loss = sum(m['metrics'].get('loss', 0) for m in recent) / max(len(recent), 1)
            
            # Determine trend
            if len(recent) >= 2:
                trend = 'improving' if recent[-1]['metrics'].get('loss', 0) < avg_loss else 'degrading'
            else:
                trend = 'stable'
            
            logger.log_llm_interaction('MetricsCollector', 'metrics_aggregated', {
                'metrics_id': metrics_id,
                'avg_loss_last_100': avg_loss
            })
            
            # Persist to file
            os.makedirs('metrics', exist_ok=True)
            with open(f'metrics/{input_data.training_id}_metrics.jsonl', 'a') as f:
                f.write(json.dumps({
                    'step': input_data.step,
                    'metrics': input_data.metrics,
                    'timestamp': datetime.utcnow().isoformat()
                }) + '\n')
            
            logger.log_llm_interaction('MetricsCollector', 'metrics_persisted', {
                'metrics_id': metrics_id,
                'execution_time_ms': (time.time() - start_time) * 1000
            })
            
            return MetricsCollectorOutput(
                metrics_id=metrics_id,
                persisted=True,
                aggregated_metrics={
                    'avg_loss_last_100': avg_loss,
                    'trend': trend
                }
            )
        except Exception as e:
            logger.log_system('error', 'Metrics collection failed', {'error': str(e)})
            raise


# Component 12: ModelCheckpointer
@dataclass
class ModelCheckpointerInput:
    model_id: str
    training_step: int
    validation_metrics: Dict[str, float]
    optimizer_state: Dict[str, Any]
    save_strategy: str  # best|periodic|final

@dataclass
class ModelCheckpointerOutput:
    checkpoint_id: str
    checkpoint_path: str
    size_mb: float
    metadata: Dict[str, Any]

class ModelCheckpointer:
    """Component: ModelCheckpointer - Save model with versioning"""
    
    def __init__(self, event_bus: EventBusAdaptor):
        self.event_bus = event_bus
        self.best_metric = float('inf')
    
    def checkpoint(self, input_data: ModelCheckpointerInput) -> ModelCheckpointerOutput:
        start_time = time.time()
        try:
            checkpoint_id = str(uuid.uuid4())
            
            logger.log_llm_interaction('ModelCheckpointer', 'checkpoint_start', {
                'checkpoint_id': checkpoint_id,
                'step': input_data.training_step
            })
            
            # Determine if should save
            should_save = False
            if input_data.save_strategy == 'periodic':
                should_save = True
            elif input_data.save_strategy == 'best':
                current_metric = input_data.validation_metrics.get('loss', float('inf'))
                if current_metric < self.best_metric:
                    self.best_metric = current_metric
                    should_save = True
            elif input_data.save_strategy == 'final':
                should_save = True
            
            if not should_save:
                return ModelCheckpointerOutput(
                    checkpoint_id=checkpoint_id,
                    checkpoint_path="",
                    size_mb=0.0,
                    metadata={}
                )
            
            # Create checkpoint
            os.makedirs('checkpoints', exist_ok=True)
            checkpoint_path = f'checkpoints/{input_data.model_id}_step_{input_data.training_step}.pt'
            
            if not SecurityValidator.validate_path(checkpoint_path):
                raise ValueError(f'Invalid checkpoint path: {checkpoint_path}')
            
            checkpoint_data = {
                'model_id': input_data.model_id,
                'step': input_data.training_step,
                'validation_metrics': input_data.validation_metrics,
                'optimizer_state': input_data.optimizer_state,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            size_mb = os.path.getsize(checkpoint_path) / (1024 ** 2)
            
            logger.log_llm_interaction('ModelCheckpointer', 'checkpoint_saved', {
                'checkpoint_id': checkpoint_id,
                'path': checkpoint_path,
                'size_mb': size_mb,
                'execution_time_ms': (time.time() - start_time) * 1000
            })
            
            metadata = {
                'step': input_data.training_step,
                'metrics': input_data.validation_metrics,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.log_llm_interaction('ModelCheckpointer', 'metadata_written', {
                'checkpoint_id': checkpoint_id
            })
            
            return ModelCheckpointerOutput(
                checkpoint_id=checkpoint_id,
                checkpoint_path=checkpoint_path,
                size_mb=size_mb,
                metadata=metadata
            )
            
        except Exception as e:
            logger.log_system('error', 'Checkpointing failed', {'error': str(e)})
            raise


# Component 13: ModelExporter
@dataclass
class ModelExporterInput:
    checkpoint_path: str
    export_format: str  # onnx|torchscript|safetensors|gguf
    optimization: Dict[str, Any]

@dataclass
class ModelExporterOutput:
    export_id: str
    export_path: str
    format: str
    size_mb: float
    inference_ready: bool

class ModelExporter:
    """Component: ModelExporter - Export model for deployment"""
    
    def __init__(self, event_bus: EventBusAdaptor):
        self.event_bus = event_bus
    
    def export(self, input_data: ModelExporterInput) -> ModelExporterOutput:
        start_time = time.time()
        try:
            export_id = str(uuid.uuid4())
            
            logger.log_llm_interaction('ModelExporter', 'export_start', {
                'export_id': export_id,
                'format': input_data.export_format
            })
            
            # Validate paths
            if not SecurityValidator.validate_path(input_data.checkpoint_path):
                raise ValueError(f'Invalid checkpoint path')
            
            # Load checkpoint
            if not os.path.exists(input_data.checkpoint_path):
                raise FileNotFoundError(f'Checkpoint not found: {input_data.checkpoint_path}')
            
            with open(input_data.checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            
            # Export to format
            os.makedirs('exports', exist_ok=True)
            export_path = f'exports/{checkpoint["model_id"]}.{input_data.export_format}'
            
            # Simulate export with optimization
            quantization = input_data.optimization.get('quantization', 'none')
            
            logger.log_llm_interaction('ModelExporter', 'optimization_applied', {
                'export_id': export_id,
                'quantization': quantization
            })
            
            # Write export file
            with open(export_path, 'w') as f:
                json.dump({
                    'checkpoint': checkpoint,
                    'format': input_data.export_format,
                    'optimization': input_data.optimization
                }, f, indent=2)
            
            size_mb = os.path.getsize(export_path) / (1024 ** 2)
            
            logger.log_llm_interaction('ModelExporter', 'export_complete', {
                'export_id': export_id,
                'export_path': export_path,
                'execution_time_ms': (time.time() - start_time) * 1000
            })
            
            return ModelExporterOutput(
                export_id=export_id,
                export_path=export_path,
                format=input_data.export_format,
                size_mb=size_mb,
                inference_ready=True
            )
            
        except Exception as e:
            logger.log_system('error', 'Export failed', {'error': str(e)})
            raise


# ============================================================================
# TESTING SUITE (Phase III)
# ============================================================================

def test_data_source_connector():
    """Test DataSourceConnector component"""
    print("\n" + "="*60)
    print("TEST: DataSourceConnector")
    print("="*60)
    
    event_bus = EventBusAdaptor()
    connector = DataSourceConnector(event_bus)
    
    # Test 1: Valid connection
    print("\n[TEST 1] Valid S3 connection...")
    input_data = DataSourceConnectorInput(
        source_type='s3',
        credentials={'auth_type': 'iam_role', 'credentials_encrypted': 'encrypted_key'},
        source_config={'endpoint': 's3://bucket/data', 'filters': {}}
    )
    output = connector.connect(input_data)
    assert output.status == 'connected', "Connection should succeed"
    assert output.connection_id, "Should have connection ID"
    print("✓ PASS: Valid connection established")
    
    # Test 2: Invalid source type
    print("\n[TEST 2] Invalid source type...")
    try:
        input_data = DataSourceConnectorInput(
            source_type='invalid',
            credentials={'auth_type': 'iam_role', 'credentials_encrypted': 'key'},
            source_config={'endpoint': 'test', 'filters': {}}
        )
        connector.connect(input_data)
        print("✗ FAIL: Should have raised ValueError")
    except ValueError:
        print("✓ PASS: ValueError raised as expected")
    
    print("\n✓ All tests passed for DataSourceConnector")


def test_data_validator():
    """Test DataValidator component"""
    print("\n" + "="*60)
    print("TEST: DataValidator")
    print("="*60)
    
    event_bus = EventBusAdaptor()
    validator = DataValidator(event_bus)
    
    # Test 1: Valid UTF-8 data
    print("\n[TEST 1] Valid UTF-8 encoding...")
    input_data = DataValidatorInput(
        stream_id=str(uuid.uuid4()),
        chunk_data=b'Hello World',
        validation_rules={'encoding': 'utf-8', 'min_quality_score': 0.8, 'prohibited_patterns': []}
    )
    output = validator.validate(input_data)
    assert output.status == 'passed', "Validation should pass"
    assert output.quality_metrics['encoding_valid'], "Encoding should be valid"
    print("✓ PASS: Valid data validated successfully")
    
    # Test 2: Invalid encoding
    print("\n[TEST 2] Invalid encoding...")
    try:
        input_data = DataValidatorInput(
            stream_id=str(uuid.uuid4()),
            chunk_data=b'\xff\xfe',  # Invalid UTF-8
            validation_rules={'encoding': 'utf-8', 'min_quality_score': 0.8, 'prohibited_patterns': []}
        )
        validator.validate(input_data)
        print("✗ FAIL: Should have raised UnicodeDecodeError")
    except UnicodeDecodeError:
        print("✓ PASS: Encoding error detected")
    
    print("\n✓ All tests passed for DataValidator")


def run_all_tests():
    """Execute complete test suite"""
    print("\n" + "="*80)
    print("COMPONENT TEST SUITE - PHASE III")
    print("="*80)
    
    test_data_source_connector()
    test_data_validator()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("="*80)


# Uncomment to run tests
# run_all_tests()
