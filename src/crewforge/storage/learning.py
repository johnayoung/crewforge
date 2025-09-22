"""Learning and storage system for CrewForge configurations."""

import hashlib
import json
import shutil
import yaml
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class ConfigurationMetrics(BaseModel):
    """Model for tracking configuration effectiveness metrics."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    total_successes: int = Field(
        default=0,
        description="Total number of successful configurations stored",
        ge=0,
    )
    total_attempts: int = Field(
        default=0,
        description="Total number of generation attempts (including failures)",
        ge=0,
    )
    average_generation_time: Optional[float] = Field(
        default=None,
        description="Average time to generate configurations (seconds)",
        ge=0,
    )
    average_validation_score: Optional[float] = Field(
        default=None,
        description="Average validation score for stored configurations",
        ge=0,
        le=1,
    )


class StoredConfiguration(BaseModel):
    """Model for stored configuration with metadata."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",  # Allow extra fields for configuration flexibility
    )

    configuration_id: str = Field(
        ...,
        description="Unique identifier for the configuration",
        min_length=1,
    )
    configuration: Dict[str, Any] = Field(
        ...,
        description="The actual configuration data",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Associated metadata for the configuration",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the configuration was stored",
    )
    hash_signature: str = Field(
        ...,
        description="Hash signature for deduplication",
        min_length=1,
    )


class LearningStore:
    """Learning and storage system for successful CrewAI configurations.

    Provides configuration persistence, pattern recognition, and effectiveness
    tracking to improve future generations.
    """

    def __init__(self, storage_path: Optional[Union[str, Path]] = None):
        """Initialize the learning store.

        Args:
            storage_path: Path to store configurations. Defaults to './learning_store'
        """
        if storage_path is None:
            try:
                storage_path = Path.cwd() / "learning_store"
            except FileNotFoundError:
                # Fallback for test environments where cwd might not exist
                storage_path = Path("/tmp") / "learning_store"

        self.storage_path = Path(storage_path)
        self._ensure_storage_structure()
        self._metrics_file = self.storage_path / "metrics.json"
        self._initialize_metrics()

    def _ensure_storage_structure(self) -> None:
        """Create the storage directory structure."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        (self.storage_path / "configs").mkdir(exist_ok=True)
        (self.storage_path / "patterns").mkdir(exist_ok=True)
        (self.storage_path / "backups").mkdir(exist_ok=True)

    def _initialize_metrics(self) -> None:
        """Initialize or load existing metrics."""
        if not self._metrics_file.exists():
            metrics = ConfigurationMetrics()
            self._save_metrics(metrics)

    def _load_metrics(self) -> ConfigurationMetrics:
        """Load metrics from storage."""
        if self._metrics_file.exists():
            try:
                with open(self._metrics_file, "r") as f:
                    data = json.load(f)
                return ConfigurationMetrics(**data)
            except (json.JSONDecodeError, ValueError):
                # Fallback to default metrics if file is corrupted
                return ConfigurationMetrics()
        return ConfigurationMetrics()

    def _save_metrics(self, metrics: ConfigurationMetrics) -> None:
        """Save metrics to storage."""
        with open(self._metrics_file, "w") as f:
            json.dump(metrics.model_dump(), f, indent=2)

    def _generate_config_hash(self, config: Dict[str, Any]) -> str:
        """Generate a hash signature for configuration deduplication.

        Args:
            config: Configuration dictionary

        Returns:
            Hash signature string
        """
        # Create a normalized JSON string for consistent hashing
        config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(config_json.encode("utf-8")).hexdigest()

    def _generate_config_id(self, config: Dict[str, Any]) -> str:
        """Generate a unique ID for the configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Unique configuration ID
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        config_hash = self._generate_config_hash(config)[:8]
        return f"config_{timestamp}_{config_hash}"

    def _get_config_filepath(self, config_id: str) -> Path:
        """Get the filepath for a configuration ID.

        Args:
            config_id: Configuration ID

        Returns:
            Path to configuration file
        """
        return self.storage_path / "configs" / f"{config_id}.yaml"

    def save_successful_config(
        self, config: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save a successful configuration for future reference.

        Args:
            config: Configuration dictionary to save
            metadata: Optional metadata about the configuration

        Returns:
            Configuration ID for the saved configuration

        Raises:
            ValueError: If config is invalid
            TypeError: If config is not a dictionary
        """
        if config is None:
            raise ValueError("Configuration cannot be None")

        if not isinstance(config, dict):
            raise TypeError("Configuration must be a dictionary")

        if not config:
            raise ValueError("Configuration cannot be empty")

        # Generate IDs and hash
        config_id = self._generate_config_id(config)
        config_hash = self._generate_config_hash(config)

        # Check for duplicates
        if self._is_duplicate_config(config_hash):
            # Return existing config ID or update with new metadata
            existing_id = self._find_existing_config_id(config_hash)
            if existing_id:
                return existing_id

        # Prepare metadata
        if metadata is None:
            metadata = {}

        metadata.update(
            {
                "stored_at": datetime.now(timezone.utc).isoformat(),
                "config_size": len(json.dumps(config)),
            }
        )

        # Create stored configuration
        stored_config = StoredConfiguration(
            configuration_id=config_id,
            configuration=config,
            metadata=metadata,
            hash_signature=config_hash,
        )

        # Save to YAML file
        config_file = self._get_config_filepath(config_id)
        with open(config_file, "w") as f:
            yaml.dump(stored_config.model_dump(), f, default_flow_style=False, indent=2)

        # Update metrics
        self._update_success_metrics(metadata)

        return config_id

    def _is_duplicate_config(self, config_hash: str) -> bool:
        """Check if a configuration with this hash already exists.

        Args:
            config_hash: Hash signature of the configuration

        Returns:
            True if duplicate exists
        """
        configs_dir = self.storage_path / "configs"
        for config_file in configs_dir.glob("*.yaml"):
            try:
                with open(config_file, "r") as f:
                    data = yaml.safe_load(f)
                if data and data.get("hash_signature") == config_hash:
                    return True
            except (yaml.YAMLError, KeyError):
                continue
        return False

    def _find_existing_config_id(self, config_hash: str) -> Optional[str]:
        """Find existing configuration ID for a hash.

        Args:
            config_hash: Hash signature

        Returns:
            Configuration ID if found, None otherwise
        """
        configs_dir = self.storage_path / "configs"
        for config_file in configs_dir.glob("*.yaml"):
            try:
                with open(config_file, "r") as f:
                    data = yaml.safe_load(f)
                if data and data.get("hash_signature") == config_hash:
                    return data.get("configuration_id")
            except (yaml.YAMLError, KeyError):
                continue
        return None

    def _update_success_metrics(self, metadata: Dict[str, Any]) -> None:
        """Update success metrics with new configuration data.

        Args:
            metadata: Metadata from the saved configuration
        """
        metrics = self._load_metrics()
        metrics.total_successes += 1

        # Update generation time average
        if "generation_time" in metadata:
            generation_time = metadata["generation_time"]
            if metrics.average_generation_time is None:
                metrics.average_generation_time = generation_time
            else:
                # Running average
                total_time = metrics.average_generation_time * (
                    metrics.total_successes - 1
                )
                metrics.average_generation_time = (
                    total_time + generation_time
                ) / metrics.total_successes

        # Update validation score average
        if "validation_score" in metadata:
            validation_score = metadata["validation_score"]
            if 0 <= validation_score <= 1:
                if metrics.average_validation_score is None:
                    metrics.average_validation_score = validation_score
                else:
                    # Running average
                    total_score = metrics.average_validation_score * (
                        metrics.total_successes - 1
                    )
                    metrics.average_validation_score = (
                        total_score + validation_score
                    ) / metrics.total_successes

        self._save_metrics(metrics)

    def retrieve_patterns(
        self,
        filter_criteria: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve stored configuration patterns.

        Args:
            filter_criteria: Optional criteria to filter patterns
            limit: Optional limit on number of patterns returned

        Returns:
            List of configuration patterns
        """
        patterns = []
        configs_dir = self.storage_path / "configs"

        if not configs_dir.exists():
            return patterns

        # Load all configurations
        for config_file in configs_dir.glob("*.yaml"):
            try:
                with open(config_file, "r") as f:
                    data = yaml.safe_load(f)

                if data and self._matches_filter(data, filter_criteria):
                    patterns.append(
                        {
                            "configuration_id": data.get("configuration_id"),
                            "configuration": data.get("configuration", {}),
                            "metadata": data.get("metadata", {}),
                            "created_at": data.get("created_at"),
                            "hash_signature": data.get("hash_signature"),
                        }
                    )

            except (yaml.YAMLError, KeyError):
                continue

        # Sort by creation date (most recent first)
        patterns.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        # Apply limit if specified
        if limit is not None:
            patterns = patterns[:limit]

        return patterns

    def _matches_filter(
        self, config_data: Dict[str, Any], filter_criteria: Optional[Dict[str, Any]]
    ) -> bool:
        """Check if configuration matches filter criteria.

        Args:
            config_data: Configuration data to check
            filter_criteria: Filter criteria

        Returns:
            True if matches filter
        """
        if filter_criteria is None:
            return True

        config = config_data.get("configuration", {})

        # Filter by agent roles
        if "agent_roles" in filter_criteria:
            required_roles = set(filter_criteria["agent_roles"])
            config_roles = set()

            for agent in config.get("agents", []):
                if isinstance(agent, dict) and "role" in agent:
                    config_roles.add(agent["role"])

            if not required_roles.intersection(config_roles):
                return False

        # Filter by tools
        if "tools" in filter_criteria:
            required_tools = set(filter_criteria["tools"])
            config_tools = set(config.get("tools", []))

            if not required_tools.intersection(config_tools):
                return False

        # Filter by metadata criteria
        if "metadata" in filter_criteria:
            metadata = config_data.get("metadata", {})
            for key, value in filter_criteria["metadata"].items():
                if metadata.get(key) != value:
                    return False

        return True

    def _load_config_by_id(self, config_id: str) -> Optional[Dict[str, Any]]:
        """Load a configuration by its ID.

        Args:
            config_id: Configuration ID

        Returns:
            Configuration data if found, None otherwise
        """
        config_file = self._get_config_filepath(config_id)

        if not config_file.exists():
            return None

        try:
            with open(config_file, "r") as f:
                return yaml.safe_load(f)
        except yaml.YAMLError:
            return None

    def get_success_metrics(self) -> Dict[str, Any]:
        """Get success metrics for the learning store.

        Returns:
            Dictionary containing success metrics
        """
        metrics = self._load_metrics()
        metrics_dict = metrics.model_dump()

        # Calculate success rate
        if metrics.total_attempts > 0:
            metrics_dict["success_rate"] = (
                metrics.total_successes / metrics.total_attempts
            )
        else:
            metrics_dict["success_rate"] = 1.0 if metrics.total_successes > 0 else 0.0

        return metrics_dict

    def record_generation_attempt(self, success: bool = True) -> None:
        """Record a generation attempt (successful or failed).

        Args:
            success: Whether the generation was successful
        """
        metrics = self._load_metrics()
        metrics.total_attempts += 1

        if success:
            metrics.total_successes += 1

        self._save_metrics(metrics)

    def get_configuration_score(self, config_id: str) -> Optional[float]:
        """Get effectiveness score for a configuration.

        Args:
            config_id: Configuration ID

        Returns:
            Effectiveness score (0-1) if available, None otherwise
        """
        config_data = self._load_config_by_id(config_id)
        if not config_data:
            return None

        metadata = config_data.get("metadata", {})

        # Calculate score based on available metrics
        score_components = []

        # Validation score (if available)
        if "validation_score" in metadata:
            validation_score = metadata["validation_score"]
            if 0 <= validation_score <= 1:
                score_components.append(validation_score)

        # User feedback score
        if "user_feedback" in metadata:
            feedback = metadata["user_feedback"].lower()
            if feedback == "excellent":
                score_components.append(1.0)
            elif feedback == "good":
                score_components.append(0.8)
            elif feedback == "average":
                score_components.append(0.6)
            elif feedback == "poor":
                score_components.append(0.3)

        # Performance score (inverse of execution time)
        if "execution_time" in metadata:
            exec_time = metadata["execution_time"]
            if exec_time > 0:
                # Normalize execution time (assume 10 seconds is baseline)
                time_score = max(0.1, min(1.0, 10.0 / exec_time))
                score_components.append(time_score)

        # Return average of available components
        if score_components:
            return sum(score_components) / len(score_components)

        return None

    def find_similar_configs(
        self, target_config: Dict[str, Any], threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Find configurations similar to the target.

        Args:
            target_config: Configuration to find similarities for
            threshold: Similarity threshold (0-1)

        Returns:
            List of similar configurations
        """
        similar_configs = []
        all_patterns = self.retrieve_patterns()

        for pattern in all_patterns:
            similarity = self._calculate_similarity(
                target_config, pattern["configuration"]
            )

            if similarity >= threshold:
                pattern_with_score = pattern.copy()
                pattern_with_score["similarity_score"] = similarity
                similar_configs.append(pattern_with_score)

        # Sort by similarity score (highest first)
        similar_configs.sort(key=lambda x: x["similarity_score"], reverse=True)

        return similar_configs

    def _calculate_similarity(
        self, config1: Dict[str, Any], config2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two configurations.

        Args:
            config1: First configuration
            config2: Second configuration

        Returns:
            Similarity score (0-1)
        """
        similarity_scores = []

        # Compare agents
        agents1 = config1.get("agents", [])
        agents2 = config2.get("agents", [])

        if agents1 or agents2:
            agent_similarity = self._calculate_agent_similarity(agents1, agents2)
            similarity_scores.append(agent_similarity)

        # Compare tasks
        tasks1 = config1.get("tasks", [])
        tasks2 = config2.get("tasks", [])

        if tasks1 or tasks2:
            task_similarity = self._calculate_task_similarity(tasks1, tasks2)
            similarity_scores.append(task_similarity)

        # Compare tools
        tools1 = set(config1.get("tools", []))
        tools2 = set(config2.get("tools", []))

        if tools1 or tools2:
            if tools1 and tools2:
                tool_similarity = len(tools1.intersection(tools2)) / len(
                    tools1.union(tools2)
                )
            else:
                tool_similarity = 0.0
            similarity_scores.append(tool_similarity)

        # Return average similarity
        if similarity_scores:
            return sum(similarity_scores) / len(similarity_scores)

        return 0.0

    def _calculate_agent_similarity(
        self, agents1: List[Dict[str, Any]], agents2: List[Dict[str, Any]]
    ) -> float:
        """Calculate similarity between agent lists.

        Args:
            agents1: First agent list
            agents2: Second agent list

        Returns:
            Similarity score (0-1)
        """
        if not agents1 and not agents2:
            return 1.0
        if not agents1 or not agents2:
            return 0.0

        # Extract roles
        roles1 = {agent.get("role", "") for agent in agents1}
        roles2 = {agent.get("role", "") for agent in agents2}

        if roles1 and roles2:
            return len(roles1.intersection(roles2)) / len(roles1.union(roles2))

        return 0.0

    def _calculate_task_similarity(
        self, tasks1: List[Dict[str, Any]], tasks2: List[Dict[str, Any]]
    ) -> float:
        """Calculate similarity between task lists.

        Args:
            tasks1: First task list
            tasks2: Second task list

        Returns:
            Similarity score (0-1)
        """
        if not tasks1 and not tasks2:
            return 1.0
        if not tasks1 or not tasks2:
            return 0.0

        # Simple comparison based on task count similarity
        count_similarity = min(len(tasks1), len(tasks2)) / max(len(tasks1), len(tasks2))

        return count_similarity

    def cleanup_old_configs(self, days_old: int = 30) -> int:
        """Clean up old configurations.

        Args:
            days_old: Remove configurations older than this many days

        Returns:
            Number of configurations removed
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_old)
        removed_count = 0

        configs_dir = self.storage_path / "configs"
        if not configs_dir.exists():
            return removed_count

        for config_file in configs_dir.glob("*.yaml"):
            try:
                with open(config_file, "r") as f:
                    data = yaml.safe_load(f)

                if data and "created_at" in data:
                    created_at_str = data["created_at"]
                    if isinstance(created_at_str, str):
                        try:
                            created_at = datetime.fromisoformat(
                                created_at_str.replace("Z", "+00:00")
                            )
                            if created_at < cutoff_date:
                                # Move to backup before deletion
                                backup_path = (
                                    self.storage_path / "backups" / config_file.name
                                )
                                shutil.move(str(config_file), str(backup_path))
                                removed_count += 1
                        except ValueError:
                            # Skip if date parsing fails
                            continue

            except (yaml.YAMLError, KeyError, OSError):
                continue

        return removed_count
