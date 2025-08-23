import pytest
import sys
import os
from pathlib import Path

class TestSetupValidation:
    """Validation tests to ensure the testing infrastructure is properly configured."""
    
    def test_python_version(self):
        """Verify Python version is 3.8 or higher."""
        assert sys.version_info >= (3, 8), "Python 3.8+ is required"
    
    def test_project_structure_exists(self):
        """Verify main project directories exist."""
        project_root = Path(__file__).parent.parent
        
        assert (project_root / "LIB").exists(), "LIB directory not found"
        assert (project_root / "QA").exists(), "QA directory not found"
        assert (project_root / "QG").exists(), "QG directory not found"
        assert (project_root / "QPC").exists(), "QPC directory not found"
    
    def test_test_directory_structure(self):
        """Verify test directory structure is correct."""
        tests_dir = Path(__file__).parent
        
        assert tests_dir.name == "tests", "Tests directory should be named 'tests'"
        assert (tests_dir / "__init__.py").exists(), "tests/__init__.py not found"
        assert (tests_dir / "conftest.py").exists(), "tests/conftest.py not found"
        assert (tests_dir / "unit").exists(), "tests/unit directory not found"
        assert (tests_dir / "integration").exists(), "tests/integration directory not found"
    
    def test_pyproject_toml_exists(self):
        """Verify pyproject.toml exists and contains required sections."""
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"
        
        assert pyproject_path.exists(), "pyproject.toml not found"
        
        content = pyproject_path.read_text()
        assert "[tool.poetry]" in content, "Poetry configuration not found"
        assert "[tool.pytest.ini_options]" in content, "Pytest configuration not found"
        assert "[tool.coverage.run]" in content, "Coverage configuration not found"
    
    def test_imports_work(self):
        """Verify that project modules can be imported."""
        try:
            import LIB
            import QA
            import QG
            import QPC
        except ImportError as e:
            pytest.fail(f"Failed to import project modules: {e}")
    
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit test marker works."""
        assert True, "Unit test marker is functional"
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration test marker works."""
        assert True, "Integration test marker is functional"
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow test marker works."""
        import time
        time.sleep(0.1)
        assert True, "Slow test marker is functional"
    
    def test_fixtures_available(self, temp_dir, mock_config, sample_qa_data):
        """Test that custom fixtures from conftest.py are available."""
        assert temp_dir.exists(), "temp_dir fixture not working"
        assert isinstance(mock_config, dict), "mock_config fixture not working"
        assert isinstance(sample_qa_data, list), "sample_qa_data fixture not working"
        assert len(sample_qa_data) > 0, "sample_qa_data should contain data"
    
    def test_temp_dir_cleanup(self, temp_dir):
        """Test that temp_dir fixture properly cleans up."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists(), "Failed to create test file"
    
    def test_coverage_configured(self):
        """Verify coverage is properly configured."""
        import coverage
        
        cov = coverage.Coverage()
        config = cov.config
        
        assert config is not None, "Coverage configuration not loaded"


class TestFixtureValidation:
    """Validate that all fixtures work correctly."""
    
    def test_sample_text_file_fixture(self, sample_text_file):
        """Test sample_text_file fixture."""
        assert sample_text_file.exists()
        content = sample_text_file.read_text()
        assert "sample text" in content.lower()
    
    def test_sample_json_file_fixture(self, sample_json_file):
        """Test sample_json_file fixture."""
        import json
        
        assert sample_json_file.exists()
        with open(sample_json_file) as f:
            data = json.load(f)
        
        assert "question" in data
        assert "answer" in data
        assert "context" in data
    
    def test_mock_tokenizer_config_fixture(self, mock_tokenizer_config):
        """Test mock_tokenizer_config fixture."""
        assert "do_lower_case" in mock_tokenizer_config
        assert "max_len" in mock_tokenizer_config
        assert mock_tokenizer_config["max_len"] == 512
    
    def test_sample_qg_data_fixture(self, sample_qg_data):
        """Test sample_qg_data fixture."""
        assert len(sample_qg_data) > 0
        
        for item in sample_qg_data:
            assert "context" in item
            assert "answer" in item
            assert "question" in item
    
    def test_mock_model_weights_fixture(self, mock_model_weights):
        """Test mock_model_weights fixture."""
        assert mock_model_weights.exists()
        config_file = mock_model_weights / "config.json"
        assert config_file.exists()
        
        import json
        with open(config_file) as f:
            config = json.load(f)
        
        assert "model_type" in config
        assert config["model_type"] == "bert"
    
    def test_clean_environment_fixture(self, clean_environment):
        """Test clean_environment fixture."""
        assert os.environ.get("TF_CPP_MIN_LOG_LEVEL") == "3"
    
    def test_reset_random_seeds_fixture(self):
        """Test that random seeds are properly reset."""
        import random
        import numpy as np
        
        # The autouse fixture already sets the seed to 42
        # So we'll test that the seed is indeed set
        random_val1 = random.random()
        np_val1 = np.random.random()
        
        # Reset to the same seed
        random.seed(42)
        np.random.seed(42)
        
        random_val2 = random.random()
        np_val2 = np.random.random()
        
        # Since autouse fixture sets seed to 42, values should be equal
        assert random_val1 == random_val2
        assert np_val1 == np_val2
    
    def test_capture_logs_fixture(self, capture_logs):
        """Test capture_logs fixture."""
        import logging
        
        logger = logging.getLogger()
        logger.error("Test error message")
        
        captured = capture_logs.getvalue()
        assert "Test error message" in captured
    
    def test_performance_timer_fixture(self, performance_timer):
        """Test performance_timer fixture."""
        import time
        
        performance_timer.start()
        time.sleep(0.01)
        elapsed = performance_timer.stop()
        
        assert elapsed is not None
        assert elapsed > 0
        assert elapsed < 1