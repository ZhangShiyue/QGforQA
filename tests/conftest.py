import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
import pytest
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path)

@pytest.fixture
def sample_text_file(temp_dir: Path) -> Path:
    """Create a sample text file for testing."""
    file_path = temp_dir / "sample.txt"
    file_path.write_text("This is a sample text for testing purposes.")
    return file_path

@pytest.fixture
def sample_json_file(temp_dir: Path) -> Path:
    """Create a sample JSON file for testing."""
    file_path = temp_dir / "sample.json"
    data = {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "context": "France is a country in Europe. Its capital is Paris."
    }
    file_path.write_text(json.dumps(data, indent=2))
    return file_path

@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration dictionary."""
    return {
        "model_name": "bert-base-uncased",
        "batch_size": 32,
        "learning_rate": 2e-5,
        "num_epochs": 3,
        "max_seq_length": 512,
        "output_dir": "/tmp/test_output",
        "cache_dir": "/tmp/test_cache",
        "do_train": True,
        "do_eval": True,
        "do_predict": False,
    }

@pytest.fixture
def mock_tokenizer_config() -> Dict[str, Any]:
    """Provide mock tokenizer configuration."""
    return {
        "do_lower_case": True,
        "max_len": 512,
        "vocab_file": None,
    }

@pytest.fixture
def sample_qa_data() -> list:
    """Provide sample question-answering data."""
    return [
        {
            "id": "1",
            "question": "What is machine learning?",
            "context": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "answer": "a subset of artificial intelligence",
            "answer_start": 20
        },
        {
            "id": "2",
            "question": "Who invented Python?",
            "context": "Python was created by Guido van Rossum and first released in 1991.",
            "answer": "Guido van Rossum",
            "answer_start": 23
        }
    ]

@pytest.fixture
def sample_qg_data() -> list:
    """Provide sample question generation data."""
    return [
        {
            "context": "The Eiffel Tower is located in Paris, France.",
            "answer": "Paris, France",
            "question": "Where is the Eiffel Tower located?"
        },
        {
            "context": "Python is a high-level programming language.",
            "answer": "high-level programming language",
            "question": "What type of language is Python?"
        }
    ]

@pytest.fixture
def mock_model_weights(temp_dir: Path) -> Path:
    """Create a mock model weights file."""
    weights_dir = temp_dir / "model_weights"
    weights_dir.mkdir()
    
    config_file = weights_dir / "config.json"
    config_file.write_text(json.dumps({
        "model_type": "bert",
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 12
    }))
    
    return weights_dir

@pytest.fixture
def clean_environment(monkeypatch):
    """Clean environment variables for testing."""
    env_vars_to_clean = [
        "CUDA_VISIBLE_DEVICES",
        "TF_CPP_MIN_LOG_LEVEL",
        "PYTHONPATH"
    ]
    
    for var in env_vars_to_clean:
        monkeypatch.delenv(var, raising=False)
    
    monkeypatch.setenv("TF_CPP_MIN_LOG_LEVEL", "3")

@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(42)
    np.random.seed(42)
    
    try:
        import tensorflow as tf
        tf.random.set_seed(42)
    except ImportError:
        pass

@pytest.fixture
def capture_logs():
    """Capture and return logs during test execution."""
    import logging
    from io import StringIO
    
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    
    logger = logging.getLogger()
    original_handlers = logger.handlers[:]
    logger.handlers = [handler]
    
    yield log_capture
    
    logger.handlers = original_handlers

@pytest.fixture
def mock_http_response(monkeypatch):
    """Mock HTTP responses for testing."""
    def mock_response(url, **kwargs):
        class MockResponse:
            def __init__(self, json_data, status_code):
                self.json_data = json_data
                self.status_code = status_code
                self.text = json.dumps(json_data) if json_data else ""
            
            def json(self):
                return self.json_data
            
            def raise_for_status(self):
                if self.status_code >= 400:
                    raise Exception(f"HTTP {self.status_code}")
        
        if "api/model" in url:
            return MockResponse({"model": "test-model", "status": "ready"}, 200)
        elif "api/predict" in url:
            return MockResponse({"prediction": "test prediction"}, 200)
        else:
            return MockResponse(None, 404)
    
    return mock_response

@pytest.fixture
def performance_timer():
    """Measure and report test performance."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.elapsed = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            if self.start_time:
                self.elapsed = time.time() - self.start_time
                return self.elapsed
            return None
    
    return Timer()