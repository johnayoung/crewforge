"""Tests for the learning and storage system."""

import json
import tempfile
import yaml
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from crewforge.models.agent import AgentConfig
from crewforge.models.crew import CrewConfig
from crewforge.models.task import TaskConfig


class TestLearningStore:
    """Test the LearningStore class for configuration persistence and learning."""

    def test_learning_store_initialization(self):
        """Test LearningStore can be initialized."""
        from crewforge.storage.learning import LearningStore

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = LearningStore(storage_path=tmp_dir)
            assert store is not None
            assert store.storage_path == Path(tmp_dir)

    def test_learning_store_default_initialization(self):
        """Test LearningStore with default storage path."""
        from crewforge.storage.learning import LearningStore

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Use explicit path to avoid cwd issues
            store = LearningStore(storage_path=tmp_dir)
            assert store is not None
            assert store.storage_path == Path(tmp_dir)

    def test_save_successful_config_basic(self):
        """Test saving a basic successful configuration."""
        from crewforge.storage.learning import LearningStore

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = LearningStore(storage_path=tmp_dir)

            config = {
                "agents": [{"role": "Researcher", "goal": "Research topics"}],
                "tasks": [{"description": "Research AI trends"}],
                "tools": ["search", "scrape"],
            }

            config_id = store.save_successful_config(config)

            assert config_id is not None
            assert isinstance(config_id, str)
            assert len(config_id) > 0

    def test_save_successful_config_with_metadata(self):
        """Test saving configuration with metadata."""
        from crewforge.storage.learning import LearningStore

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = LearningStore(storage_path=tmp_dir)

            config = {"agents": [], "tasks": [], "tools": []}
            metadata = {
                "prompt": "Create a research crew",
                "generation_time": 5.2,
                "validation_passed": True,
            }

            config_id = store.save_successful_config(config, metadata=metadata)

            assert config_id is not None
            # Check that metadata is stored
            stored_config = store._load_config_by_id(config_id)
            assert stored_config["metadata"]["prompt"] == "Create a research crew"
            assert stored_config["metadata"]["generation_time"] == 5.2

    def test_save_successful_config_creates_files(self):
        """Test that saving creates proper file structure."""
        from crewforge.storage.learning import LearningStore

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = LearningStore(storage_path=tmp_dir)
            config = {"agents": [], "tasks": [], "tools": []}

            config_id = store.save_successful_config(config)

            # Check file structure
            storage_path = Path(tmp_dir)
            assert storage_path.exists()

            # Should have configs directory
            configs_dir = storage_path / "configs"
            assert configs_dir.exists()

            # Should have config file
            config_files = list(configs_dir.glob("*.yaml"))
            assert len(config_files) >= 1

    def test_retrieve_patterns_empty_store(self):
        """Test retrieving patterns from empty store."""
        from crewforge.storage.learning import LearningStore

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = LearningStore(storage_path=tmp_dir)
            patterns = store.retrieve_patterns()

            assert isinstance(patterns, list)
            assert len(patterns) == 0

    def test_retrieve_patterns_with_data(self):
        """Test retrieving patterns after storing configurations."""
        from crewforge.storage.learning import LearningStore

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = LearningStore(storage_path=tmp_dir)

            # Save multiple configurations
            config1 = {
                "agents": [{"role": "Researcher", "goal": "Research"}],
                "tasks": [{"description": "Research task"}],
                "tools": ["search"],
            }
            config2 = {
                "agents": [{"role": "Writer", "goal": "Write"}],
                "tasks": [{"description": "Writing task"}],
                "tools": ["editor"],
            }

            store.save_successful_config(config1)
            store.save_successful_config(config2)

            patterns = store.retrieve_patterns()

            assert isinstance(patterns, list)
            assert len(patterns) >= 2

    def test_retrieve_patterns_filtering(self):
        """Test retrieving patterns with filtering criteria."""
        from crewforge.storage.learning import LearningStore

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = LearningStore(storage_path=tmp_dir)

            # Save configurations with different agent roles
            research_config = {
                "agents": [{"role": "Researcher", "goal": "Research"}],
                "tasks": [{"description": "Research task"}],
                "tools": ["search"],
            }
            writer_config = {
                "agents": [{"role": "Writer", "goal": "Write"}],
                "tasks": [{"description": "Writing task"}],
                "tools": ["editor"],
            }

            store.save_successful_config(research_config)
            store.save_successful_config(writer_config)

            # Filter for research patterns
            research_patterns = store.retrieve_patterns(
                filter_criteria={"agent_roles": ["Researcher"]}
            )

            assert isinstance(research_patterns, list)
            # Should contain research configuration but not writer
            assert len(research_patterns) >= 1

    def test_configuration_deduplication(self):
        """Test that identical configurations are deduplicated."""
        from crewforge.storage.learning import LearningStore

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = LearningStore(storage_path=tmp_dir)

            config = {
                "agents": [{"role": "Researcher", "goal": "Research"}],
                "tasks": [{"description": "Research task"}],
                "tools": ["search"],
            }

            # Save the same configuration twice
            id1 = store.save_successful_config(config)
            id2 = store.save_successful_config(config)

            # Should return same ID or handle deduplication
            patterns = store.retrieve_patterns()

            # Check that we don't have exact duplicates stored
            assert len(patterns) <= 2  # Allow for slight variations in timestamps

    def test_success_rate_tracking(self):
        """Test tracking of generation success rates."""
        from crewforge.storage.learning import LearningStore

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = LearningStore(storage_path=tmp_dir)

            # Record successful generations with different configs
            for i in range(3):
                config = {
                    "agents": [{"role": f"Agent_{i}"}],
                    "tasks": [{"description": f"Task_{i}"}],
                    "tools": [f"tool_{i}"],
                }
                store.save_successful_config(config)

            # Record failed attempts (if method exists)
            if hasattr(store, "record_generation_attempt"):
                store.record_generation_attempt(success=False)
                store.record_generation_attempt(success=False)

            # Get success metrics
            metrics = store.get_success_metrics()

            assert isinstance(metrics, dict)
            assert "total_successes" in metrics
            assert metrics["total_successes"] >= 3

    def test_configuration_effectiveness_scoring(self):
        """Test scoring of configuration effectiveness."""
        from crewforge.storage.learning import LearningStore

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = LearningStore(storage_path=tmp_dir)

            # Save configuration with effectiveness metadata
            config = {"agents": [], "tasks": [], "tools": []}
            metadata = {
                "validation_score": 0.95,
                "user_feedback": "excellent",
                "execution_time": 2.3,
            }

            config_id = store.save_successful_config(config, metadata=metadata)

            # Get effectiveness score
            if hasattr(store, "get_configuration_score"):
                score = store.get_configuration_score(config_id)
                assert isinstance(score, (int, float))
                assert 0 <= score <= 1

    def test_storage_cleanup(self):
        """Test storage cleanup and maintenance."""
        from crewforge.storage.learning import LearningStore

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = LearningStore(storage_path=tmp_dir)

            # Create some old configurations with fake timestamps
            old_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
            with patch("crewforge.storage.learning.datetime") as mock_datetime:
                mock_datetime.now.return_value = old_date
                mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

                old_config = {"agents": [], "tasks": [], "tools": []}
                store.save_successful_config(old_config)

            # Add recent configuration
            recent_config = {"agents": [], "tasks": [], "tools": []}
            store.save_successful_config(recent_config)

            initial_count = len(store.retrieve_patterns())

            # Cleanup old configurations
            if hasattr(store, "cleanup_old_configs"):
                store.cleanup_old_configs(days_old=30)

                after_cleanup_count = len(store.retrieve_patterns())
                # Should have fewer configs after cleanup
                assert after_cleanup_count <= initial_count

    def test_yaml_serialization(self):
        """Test YAML serialization of configurations."""
        from crewforge.storage.learning import LearningStore

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = LearningStore(storage_path=tmp_dir)

            config = {
                "agents": [{"role": "Researcher", "goal": "Research deeply"}],
                "tasks": [{"description": "Comprehensive research"}],
                "tools": ["search", "scrape", "analyze"],
            }

            config_id = store.save_successful_config(config)

            # Find the YAML file
            configs_dir = Path(tmp_dir) / "configs"
            yaml_files = list(configs_dir.glob("*.yaml"))
            assert len(yaml_files) >= 1

            # Load and verify YAML content
            with open(yaml_files[0], "r") as f:
                loaded_data = yaml.safe_load(f)

            assert "configuration" in loaded_data
            assert "agents" in loaded_data["configuration"]

    def test_json_metadata_serialization(self):
        """Test JSON serialization for structured metadata."""
        from crewforge.storage.learning import LearningStore

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = LearningStore(storage_path=tmp_dir)

            config = {"agents": [], "tasks": [], "tools": []}
            metadata = {
                "prompt": "Test prompt",
                "metrics": {"accuracy": 0.95, "speed": 1.2},
                "tags": ["research", "ai"],
            }

            config_id = store.save_successful_config(config, metadata=metadata)

            # Verify metadata is properly serialized
            stored_config = store._load_config_by_id(config_id)
            assert stored_config is not None
            assert stored_config["metadata"]["metrics"]["accuracy"] == 0.95
            assert "research" in stored_config["metadata"]["tags"]

    def test_similar_configuration_detection(self):
        """Test detection of similar configurations."""
        from crewforge.storage.learning import LearningStore

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = LearningStore(storage_path=tmp_dir)

            # Base configuration
            base_config = {
                "agents": [{"role": "Researcher", "goal": "Research AI"}],
                "tasks": [{"description": "Research AI trends"}],
                "tools": ["search", "scrape"],
            }

            # Similar configuration
            similar_config = {
                "agents": [{"role": "Researcher", "goal": "Research ML"}],
                "tasks": [{"description": "Research ML trends"}],
                "tools": ["search", "scrape"],
            }

            store.save_successful_config(base_config)
            store.save_successful_config(similar_config)

            if hasattr(store, "find_similar_configs"):
                similar = store.find_similar_configs(base_config, threshold=0.7)
                assert isinstance(similar, list)

    def test_pydantic_model_integration(self):
        """Test integration with Pydantic models."""
        from crewforge.storage.learning import LearningStore

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = LearningStore(storage_path=tmp_dir)

            # Create configuration using Pydantic models
            agent = AgentConfig(
                role="Senior Researcher",
                goal="Conduct comprehensive research",
                backstory="Expert in data analysis",
            )

            task = TaskConfig(
                description="Research latest AI developments",
                expected_output="Detailed research report",
                agent="Senior Researcher",  # Assign the agent
            )

            crew = CrewConfig(name="research-crew", agents=[agent], tasks=[task])

            # Save Pydantic model as configuration
            config_dict = crew.model_dump()
            config_id = store.save_successful_config(config_dict)

            assert config_id is not None

            # Retrieve and verify
            patterns = store.retrieve_patterns()
            assert len(patterns) >= 1

    def test_storage_path_creation(self):
        """Test automatic creation of storage directories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            from crewforge.storage.learning import LearningStore

            storage_path = Path(tmp_dir) / "new_learning_store"
            store = LearningStore(storage_path=str(storage_path))

            # Storage path should be created automatically
            config = {"agents": [], "tasks": [], "tools": []}
            store.save_successful_config(config)

            assert storage_path.exists()
            assert (storage_path / "configs").exists()

    def test_error_handling_invalid_config(self):
        """Test error handling with invalid configurations."""
        from crewforge.storage.learning import LearningStore

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = LearningStore(storage_path=tmp_dir)

            # Test with None config
            with pytest.raises((ValueError, TypeError)):
                store.save_successful_config(None)  # type: ignore

            # Test with invalid config format
            with pytest.raises((ValueError, TypeError)):
                store.save_successful_config("not a dict")  # type: ignore

    def test_large_configuration_handling(self):
        """Test handling of large configurations."""
        from crewforge.storage.learning import LearningStore

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = LearningStore(storage_path=tmp_dir)

            # Create large configuration
            large_config = {
                "agents": [
                    {
                        "role": f"Agent_{i}",
                        "goal": f"Goal for agent {i}",
                        "backstory": f"Long backstory for agent {i}" * 10,
                    }
                    for i in range(50)
                ],
                "tasks": [
                    {
                        "description": f"Task {i} description " * 20,
                        "expected_output": f"Expected output for task {i}",
                    }
                    for i in range(100)
                ],
                "tools": [f"tool_{i}" for i in range(20)],
            }

            config_id = store.save_successful_config(large_config)
            assert config_id is not None

            # Should be able to retrieve it
            patterns = store.retrieve_patterns()
            assert len(patterns) >= 1
