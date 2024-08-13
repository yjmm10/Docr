"""Test config"""
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import os

@pytest.fixture
def mock_path() -> Path:
    """Mock a path, and clean when unit test done."""
    with TemporaryDirectory() as temp_path:
        yield Path(temp_path)

@pytest.fixture(scope="session", autouse=True)
def create_output_directory():
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)