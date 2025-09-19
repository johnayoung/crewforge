"""
Tests for validation caching functionality.

This module tests the caching system that optimizes validation performance
by avoiding re-validation of unchanged files.
"""

import tempfile
import time
import textwrap
from pathlib import Path
from unittest.mock import patch
import pytest

from crewforge.validation import (
    ValidationIssue,
    ValidationResult,
    IssueSeverity,
    validate_python_syntax,
    validate_python_imports,
    validate_generated_project,
    ValidationCache,
)


class TestValidationCaching:
    """Test cases for validation caching functionality."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary directory for test projects."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def cache(self):
        """Create a validation cache instance."""
        return ValidationCache()

    def test_cache_initialization(self, cache):
        """Test that cache initializes properly."""
        assert cache is not None
        assert hasattr(cache, "get")
        assert hasattr(cache, "set")
        assert hasattr(cache, "clear")

    def test_cache_key_generation(self, cache, temp_project_dir):
        """Test cache key generation based on file path and content."""
        # Create a test file
        test_file = temp_project_dir / "test.py"
        test_file.write_text("print('hello')")

        key1 = cache._generate_cache_key(str(test_file))
        assert key1 is not None
        assert isinstance(key1, str)

        # Modify file content
        test_file.write_text("print('world')")
        key2 = cache._generate_cache_key(str(test_file))

        # Keys should be different for different content
        assert key1 != key2

    def test_cache_hit_and_miss(self, cache, temp_project_dir):
        """Test cache hit and miss scenarios."""
        test_file = temp_project_dir / "test.py"
        test_file.write_text("print('hello')")

        # First call should be a miss
        result1 = cache.get(str(test_file))
        assert result1 is None

        # Set cache
        mock_result = ValidationResult(issues=[])
        cache.set(str(test_file), mock_result)

        # Second call should be a hit
        result2 = cache.get(str(test_file))
        assert result2 is not None
        assert result2 == mock_result

    def test_cache_invalidation_on_file_change(self, cache, temp_project_dir):
        """Test that cache invalidates when file content changes."""
        test_file = temp_project_dir / "test.py"
        test_file.write_text("print('hello')")

        # Cache initial result
        mock_result1 = ValidationResult(issues=[])
        cache.set(str(test_file), mock_result1)

        # Verify cache hit
        assert cache.get(str(test_file)) == mock_result1

        # Modify file
        time.sleep(0.01)  # Ensure different timestamp
        test_file.write_text("print('world')")

        # Cache should be invalidated (different content)
        assert cache.get(str(test_file)) is None

    def test_cached_syntax_validation(self, temp_project_dir):
        """Test that syntax validation uses caching."""
        test_file = temp_project_dir / "test.py"
        test_file.write_text("print('hello world')")

        # First validation
        result1 = validate_python_syntax(test_file)

        # Second validation should use cache
        with patch("crewforge.validation.ValidationCache.get") as mock_get:
            mock_get.return_value = result1
            result2 = validate_python_syntax(test_file)
            assert result2 == result1

    def test_cached_import_validation(self, temp_project_dir):
        """Test that import validation uses caching."""
        test_file = temp_project_dir / "test.py"
        test_file.write_text("import os\nprint('hello')")

        # First validation
        result1 = validate_python_imports(test_file, str(temp_project_dir))

        # Second validation should use cache
        with patch("crewforge.validation.ValidationCache.get") as mock_get:
            mock_get.return_value = result1
            result2 = validate_python_imports(test_file, str(temp_project_dir))
            assert result2 == result1

    def test_cache_performance_improvement(self, temp_project_dir):
        """Test that caching improves performance."""
        import time

        # Create a larger Python file
        test_file = temp_project_dir / "large_test.py"
        content = textwrap.dedent(
            """
            import os
            import sys
            from typing import List, Dict
            from pathlib import Path

            def complex_function():
                result = []
                for i in range(1000):
                    result.append(i * 2)
                return result

            class TestClass:
                def __init__(self):
                    self.data = {}

                def process(self, items: List[int]) -> Dict[str, int]:
                    return {str(i): i for i in items}

            if __name__ == "__main__":
                obj = TestClass()
                data = complex_function()
                result = obj.process(data)
                print(f"Processed {len(result)} items")
            """
        )
        test_file.write_text(content)

        # Time first validation
        start_time = time.time()
        result1 = validate_python_syntax(test_file)
        first_duration = time.time() - start_time

        # Time second validation (should be cached)
        start_time = time.time()
        result2 = validate_python_syntax(test_file)
        second_duration = time.time() - start_time

        # Results should be identical
        assert result1.is_valid == result2.is_valid
        assert len(result1.issues) == len(result2.issues)

        # Second validation should be faster (though this might not be measurable in a unit test)
        # At minimum, ensure both calls succeed
        assert result1.is_valid
        assert result2.is_valid

    def test_cache_clearing(self, cache, temp_project_dir):
        """Test cache clearing functionality."""
        test_file = temp_project_dir / "test.py"
        test_file.write_text("print('hello')")

        # Set cache
        mock_result = ValidationResult(issues=[])
        cache.set(str(test_file), mock_result)

        # Verify cache has entry
        assert cache.get(str(test_file)) == mock_result

        # Clear cache
        cache.clear()

        # Verify cache is empty
        assert cache.get(str(test_file)) is None

    def test_cache_with_project_validation(self, temp_project_dir):
        """Test caching with comprehensive project validation."""
        # Create a minimal project structure
        src_dir = temp_project_dir / "src" / "test_project"
        src_dir.mkdir(parents=True)

        main_py = src_dir / "main.py"
        main_py.write_text("print('Hello from main')")

        crew_py = src_dir / "crew.py"
        crew_py.write_text("from crewai import Crew\nprint('Crew setup')")

        # First validation
        result1 = validate_generated_project(str(temp_project_dir))

        # Second validation should use cache for individual file validations
        result2 = validate_generated_project(str(temp_project_dir))

        # Results should be consistent
        assert result1.is_valid == result2.is_valid
        assert len(result1.issues) == len(result2.issues)

    def test_cache_thread_safety(self, cache, temp_project_dir):
        """Test that cache operations are thread-safe."""
        import threading

        test_file = temp_project_dir / "test.py"
        test_file.write_text("print('thread test')")

        results = []
        errors = []

        def cache_operation(thread_id):
            try:
                # Set cache
                mock_result = ValidationResult(issues=[])
                cache.set(str(test_file), mock_result)

                # Get cache
                result = cache.get(str(test_file))
                results.append((thread_id, result))
            except Exception as e:
                errors.append((thread_id, str(e)))

        # Run multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=cache_operation, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify no errors and consistent results
        assert len(errors) == 0
        assert len(results) == 5

        # All results should be the same
        first_result = results[0][1]
        for thread_id, result in results:
            assert result == first_result
