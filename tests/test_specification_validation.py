"""
Tests for specification validation and completeness checking system.

This module tests the validation system that ensures parsed project specifications
are complete, valid, and suitable for CrewAI project generation.
"""

import pytest
from typing import Dict, Any, List

from crewforge.validation import (
    SpecificationValidator,
    ValidationError,
    ValidationResult,
    ValidationIssue,
    IssueSeverity,
)


class TestSpecificationValidator:
    """Test cases for SpecificationValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a SpecificationValidator instance for testing."""
        return SpecificationValidator()

    @pytest.fixture
    def valid_spec(self) -> Dict[str, Any]:
        """Create a valid project specification for testing."""
        return {
            "project_name": "research-team",
            "project_description": "A research team for analyzing market trends",
            "agents": [
                {
                    "role": "Research Analyst",
                    "goal": "Gather and analyze market data from various sources",
                    "backstory": "Expert data analyst with 10 years experience in market research",
                    "tools": ["web_search", "data_analyzer", "report_generator"],
                },
                {
                    "role": "Content Writer",
                    "goal": "Create compelling reports based on research findings",
                    "backstory": "Professional writer specializing in business content",
                    "tools": ["document_writer", "grammar_checker"],
                },
            ],
            "tasks": [
                {
                    "description": "Research current market trends in technology sector",
                    "expected_output": "Comprehensive report with data visualizations and insights",
                    "agent": "Research Analyst",
                },
                {
                    "description": "Write executive summary based on research findings",
                    "expected_output": "2-page executive summary in PDF format",
                    "agent": "Content Writer",
                },
            ],
            "dependencies": ["crewai", "pandas", "requests", "matplotlib"],
        }

    def test_valid_specification_passes_validation(self, validator, valid_spec):
        """Test that a valid specification passes all validation checks."""
        result = validator.validate(valid_spec)

        assert result.is_valid is True
        assert len(result.issues) == 0
        assert result.errors == []
        assert result.warnings == []
        assert result.info_messages == []

    def test_missing_required_field_fails_validation(self, validator, valid_spec):
        """Test that missing required fields are detected."""
        # Remove required field
        del valid_spec["project_name"]

        result = validator.validate(valid_spec)

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "project_name" in result.errors[0].message
        assert result.errors[0].severity == IssueSeverity.ERROR

    def test_empty_agents_list_fails_validation(self, validator, valid_spec):
        """Test that empty agents list is detected as error."""
        valid_spec["agents"] = []

        result = validator.validate(valid_spec)

        assert result.is_valid is False
        assert any("agents" in error.message.lower() for error in result.errors)

    def test_empty_tasks_list_fails_validation(self, validator, valid_spec):
        """Test that empty tasks list is detected as error."""
        valid_spec["tasks"] = []

        result = validator.validate(valid_spec)

        assert result.is_valid is False
        assert any("task" in error.message.lower() for error in result.errors)

    def test_agent_missing_required_fields(self, validator, valid_spec):
        """Test that agents with missing required fields are detected."""
        # Remove required field from agent
        del valid_spec["agents"][0]["goal"]

        result = validator.validate(valid_spec)

        assert result.is_valid is False
        assert any("goal" in error.message for error in result.errors)

    def test_task_missing_required_fields(self, validator, valid_spec):
        """Test that tasks with missing required fields are detected."""
        # Remove required field from task
        del valid_spec["tasks"][0]["expected_output"]

        result = validator.validate(valid_spec)

        assert result.is_valid is False
        assert any("expected_output" in error.message for error in result.errors)

    def test_invalid_project_name_format(self, validator, valid_spec):
        """Test that invalid project name formats are detected."""
        test_cases = [
            "Project With Spaces",
            "project_with_underscores",
            "ProjectCamelCase",
            "UPPERCASE",
            "123-starts-with-number",
            "special-chars!@#",
            "",
        ]

        for invalid_name in test_cases:
            valid_spec["project_name"] = invalid_name
            result = validator.validate(valid_spec)

            assert (
                result.is_valid is False
            ), f"Should reject project name: {invalid_name}"
            assert any(
                "project" in error.message.lower() and "name" in error.message.lower()
                for error in result.errors
            )

    def test_valid_project_name_formats(self, validator, valid_spec):
        """Test that valid project name formats are accepted."""
        test_cases = [
            "simple-project",
            "multi-word-project",
            "project-123",
            "a",  # single character
            "very-long-project-name-with-many-words",
        ]

        for valid_name in test_cases:
            valid_spec["project_name"] = valid_name
            result = validator.validate(valid_spec)

            # Should not have project name errors (might have other issues)
            project_name_errors = [
                e
                for e in result.errors
                if "project" in e.message.lower() and "name" in e.message.lower()
            ]
            assert (
                len(project_name_errors) == 0
            ), f"Should accept project name: {valid_name}"

    def test_task_agent_reference_validation(self, validator, valid_spec):
        """Test that task agent references are validated against existing agents."""
        # Reference non-existent agent
        valid_spec["tasks"][0]["agent"] = "Non-Existent Agent"

        result = validator.validate(valid_spec)

        assert result.is_valid is False
        assert any("Non-Existent Agent" in error.message for error in result.errors)

    def test_missing_crewai_dependency(self, validator, valid_spec):
        """Test that missing 'crewai' dependency is detected."""
        valid_spec["dependencies"] = ["pandas", "requests"]  # Missing crewai

        result = validator.validate(valid_spec)

        assert result.is_valid is False
        assert any("crewai" in error.message.lower() for error in result.errors)

    def test_duplicate_agent_roles_warning(self, validator, valid_spec):
        """Test that duplicate agent roles generate warnings."""
        # Add duplicate agent role
        duplicate_agent = valid_spec["agents"][0].copy()
        valid_spec["agents"].append(duplicate_agent)

        result = validator.validate(valid_spec)

        # Should have warning about duplicate roles
        assert len(result.warnings) > 0
        assert any(
            "duplicate" in warning.message.lower() for warning in result.warnings
        )

    def test_unused_agent_warning(self, validator, valid_spec):
        """Test that unused agents generate warnings."""
        # Add agent not used in any task
        unused_agent = {
            "role": "Unused Agent",
            "goal": "Do nothing",
            "backstory": "Not used anywhere",
            "tools": [],
        }
        valid_spec["agents"].append(unused_agent)

        result = validator.validate(valid_spec)

        # Should have warning about unused agent
        assert len(result.warnings) > 0
        assert any("unused" in warning.message.lower() for warning in result.warnings)

    def test_empty_tools_list_warning(self, validator, valid_spec):
        """Test that agents with empty tools list generate warnings."""
        valid_spec["agents"][0]["tools"] = []

        result = validator.validate(valid_spec)

        # Should still be valid but with warnings
        assert len(result.warnings) > 0
        assert any("tools" in warning.message.lower() for warning in result.warnings)

    def test_long_field_values_warning(self, validator, valid_spec):
        """Test that overly long field values generate warnings."""
        # Make backstory very long
        valid_spec["agents"][0]["backstory"] = "A" * 1000

        result = validator.validate(valid_spec)

        # Should have warning about length
        assert len(result.warnings) > 0
        assert any(
            "long" in warning.message.lower() or "length" in warning.message.lower()
            for warning in result.warnings
        )

    def test_single_agent_single_task_valid(self, validator):
        """Test minimal valid specification with one agent and one task."""
        minimal_spec = {
            "project_name": "minimal-project",
            "project_description": "A minimal project for testing",
            "agents": [
                {
                    "role": "Worker",
                    "goal": "Complete the task",
                    "backstory": "A reliable worker",
                    "tools": ["basic_tool"],
                }
            ],
            "tasks": [
                {
                    "description": "Complete the main task",
                    "expected_output": "Task completion report",
                    "agent": "Worker",
                }
            ],
            "dependencies": ["crewai"],
        }

        result = validator.validate(minimal_spec)
        assert result.is_valid is True

    def test_multiple_validation_issues(self, validator):
        """Test specification with multiple validation issues."""
        problematic_spec = {
            "project_name": "Invalid Name With Spaces",  # Error
            "project_description": "",  # Error
            "agents": [
                {
                    "role": "Agent 1",
                    # Missing goal, backstory, tools
                },
                {
                    "role": "Agent 1",  # Duplicate role - Warning
                    "goal": "Do something",
                    "backstory": "Some backstory",
                    "tools": [],  # Empty tools - Warning
                },
            ],
            "tasks": [
                {
                    "description": "Do something",
                    "expected_output": "Something done",
                    "agent": "Non-Existent Agent",  # Error
                }
            ],
            "dependencies": ["pandas"],  # Missing crewai - Error
        }

        result = validator.validate(problematic_spec)

        assert result.is_valid is False
        assert len(result.errors) >= 4  # Multiple errors expected
        assert len(result.warnings) >= 2  # Multiple warnings expected

    def test_validation_with_none_input(self, validator):
        """Test validation with None input."""
        result = validator.validate(None)

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert (
            "null" in result.errors[0].message.lower()
            or "none" in result.errors[0].message.lower()
        )

    def test_validation_with_invalid_type(self, validator):
        """Test validation with non-dictionary input."""
        result = validator.validate("not a dictionary")

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "dictionary" in result.errors[0].message.lower()

    def test_get_completeness_score(self, validator, valid_spec):
        """Test completeness score calculation."""
        result = validator.validate(valid_spec)
        score = result.completeness_score

        assert 0 <= score <= 1.0
        assert score == 1.0  # Valid spec should have perfect score

    def test_completeness_score_with_missing_optional_fields(
        self, validator, valid_spec
    ):
        """Test completeness score with missing optional elements."""
        # Remove some dependencies (optional extras)
        valid_spec["dependencies"] = ["crewai"]

        result = validator.validate(valid_spec)
        score = result.completeness_score

        assert score < 1.0  # Should be less than perfect due to minimal dependencies


class TestValidationResult:
    """Test cases for ValidationResult class."""

    def test_validation_result_creation(self):
        """Test ValidationResult can be created with issues."""
        issues = [
            ValidationIssue(
                severity=IssueSeverity.ERROR,
                message="Test error",
                field_path="test.field",
            ),
            ValidationIssue(
                severity=IssueSeverity.WARNING,
                message="Test warning",
                field_path="test.other",
            ),
        ]

        result = ValidationResult(issues=issues, completeness_score=0.8)

        assert result.is_valid is False  # Has errors
        assert len(result.errors) == 1
        assert len(result.warnings) == 1
        assert len(result.info_messages) == 0
        assert result.completeness_score == 0.8

    def test_validation_result_only_warnings(self):
        """Test ValidationResult with only warnings is still valid."""
        issues = [
            ValidationIssue(
                severity=IssueSeverity.WARNING,
                message="Test warning",
                field_path="test.field",
            )
        ]

        result = ValidationResult(issues=issues, completeness_score=1.0)

        assert result.is_valid is True  # Only warnings, no errors
        assert len(result.errors) == 0
        assert len(result.warnings) == 1

    def test_validation_result_str_representation(self):
        """Test string representation of ValidationResult."""
        issues = [
            ValidationIssue(
                severity=IssueSeverity.ERROR,
                message="Test error",
                field_path="test.field",
            )
        ]

        result = ValidationResult(issues=issues, completeness_score=0.5)
        result_str = str(result)

        assert "ERROR" in result_str
        assert "Test error" in result_str
        assert "50" in result_str and "%" in result_str


class TestValidationIssue:
    """Test cases for ValidationIssue class."""

    def test_validation_issue_creation(self):
        """Test ValidationIssue can be created with required fields."""
        issue = ValidationIssue(
            severity=IssueSeverity.ERROR,
            message="Test message",
            field_path="test.field",
        )

        assert issue.severity == IssueSeverity.ERROR
        assert issue.message == "Test message"
        assert issue.field_path == "test.field"

    def test_validation_issue_str_representation(self):
        """Test string representation of ValidationIssue."""
        issue = ValidationIssue(
            severity=IssueSeverity.WARNING,
            message="Test warning",
            field_path="agents[0].tools",
        )

        issue_str = str(issue)
        assert "WARNING" in issue_str
        assert "Test warning" in issue_str
        assert "agents[0].tools" in issue_str


class TestIssueSeverity:
    """Test cases for IssueSeverity enum."""

    def test_severity_levels_exist(self):
        """Test that all expected severity levels exist."""
        assert IssueSeverity.ERROR
        assert IssueSeverity.WARNING
        assert IssueSeverity.INFO

    def test_severity_ordering(self):
        """Test that severity levels can be compared."""
        assert IssueSeverity.ERROR > IssueSeverity.WARNING
        assert IssueSeverity.WARNING > IssueSeverity.INFO


class TestAmbiguityDetection:
    """Test cases for ambiguity detection in specifications."""

    @pytest.fixture
    def validator(self):
        """Create a SpecificationValidator instance for testing."""
        return SpecificationValidator()

    @pytest.fixture
    def base_spec(self) -> Dict[str, Any]:
        """Create a base specification for ambiguity testing."""
        return {
            "project_name": "test-project",
            "project_description": "A test project",
            "agents": [],
            "tasks": [],
            "dependencies": ["crewai"],
        }

    def test_detect_vague_agent_roles(self, validator, base_spec):
        """Test detection of vague agent roles."""
        spec = base_spec.copy()
        spec["agents"] = [
            {
                "role": "agent",  # Vague role
                "goal": "Do things",
                "backstory": "Generic agent that helps",
                "tools": ["tool1"],
            },
            {
                "role": "assistant",  # Vague role
                "goal": "Assist with tasks",
                "backstory": "General assistant",
                "tools": ["tool2"],
            },
            {
                "role": "Research Analyst",  # Specific role - should not trigger
                "goal": "Analyze market data",
                "backstory": "Expert in market analysis",
                "tools": ["web_search"],
            },
        ]
        spec["tasks"] = [
            {
                "description": "Task 1",
                "expected_output": "Output 1",
                "agent": "agent",
            }
        ]

        result = validator.validate(spec)

        # Should detect ambiguity in vague agent roles
        ambiguity_warnings = [
            issue
            for issue in result.issues
            if "vague" in issue.message.lower() or "ambiguous" in issue.message.lower()
        ]

        assert len(ambiguity_warnings) >= 2  # Should detect "agent" and "assistant"

        # Check specific vague roles are flagged
        vague_role_messages = [issue.message for issue in ambiguity_warnings]
        assert any("agent" in msg.lower() for msg in vague_role_messages)
        assert any("assistant" in msg.lower() for msg in vague_role_messages)

    def test_detect_unclear_task_descriptions(self, validator, base_spec):
        """Test detection of unclear or too-generic task descriptions."""
        spec = base_spec.copy()
        spec["agents"] = [
            {
                "role": "Worker",
                "goal": "Work on tasks",
                "backstory": "A worker",
                "tools": ["tool1"],
            }
        ]
        spec["tasks"] = [
            {
                "description": "Do work",  # Too vague
                "expected_output": "Work done",  # Too vague
                "agent": "Worker",
            },
            {
                "description": "Task",  # Too short
                "expected_output": "Result",  # Too short
                "agent": "Worker",
            },
            {
                "description": "Research and analyze the market trends in the technology sector for Q4 2023",  # Specific - should not trigger
                "expected_output": "Detailed market analysis report with key findings and recommendations",
                "agent": "Worker",
            },
        ]

        result = validator.validate(spec)

        # Should detect ambiguity in vague task descriptions
        ambiguity_warnings = [
            issue
            for issue in result.issues
            if (
                "vague" in issue.message.lower()
                or "ambiguous" in issue.message.lower()
                or "unclear" in issue.message.lower()
                or "generic" in issue.message.lower()
            )
        ]

        assert len(ambiguity_warnings) >= 2  # Should detect multiple vague tasks

    def test_detect_vague_project_description(self, validator, base_spec):
        """Test detection of vague project descriptions."""
        spec = base_spec.copy()
        spec["project_description"] = "A project"  # Too vague
        spec["agents"] = [
            {
                "role": "Agent",
                "goal": "Do something",
                "backstory": "Agent backstory",
                "tools": ["tool1"],
            }
        ]
        spec["tasks"] = [
            {
                "description": "Task description",
                "expected_output": "Task output",
                "agent": "Agent",
            }
        ]

        result = validator.validate(spec)

        # Should detect vague project description
        ambiguity_warnings = [
            issue
            for issue in result.issues
            if "vague" in issue.message.lower() or "ambiguous" in issue.message.lower()
        ]

        assert len(ambiguity_warnings) >= 1
        assert any(
            "project_description" in issue.field_path for issue in ambiguity_warnings
        )

    def test_detect_mismatched_agent_tools(self, validator, base_spec):
        """Test detection of tools that don't match agent roles."""
        spec = base_spec.copy()
        spec["agents"] = [
            {
                "role": "Content Writer",
                "goal": "Write articles",
                "backstory": "Professional writer",
                "tools": [
                    "web_scraper",
                    "database_query",
                ],  # Tools don't match writing role
            },
            {
                "role": "Data Analyst",
                "goal": "Analyze data",
                "backstory": "Expert analyst",
                "tools": ["data_analyzer", "visualization_tool"],  # Good match
            },
        ]
        spec["tasks"] = [
            {
                "description": "Write content",
                "expected_output": "Articles",
                "agent": "Content Writer",
            }
        ]

        result = validator.validate(spec)

        # Should detect tool-role mismatch
        mismatch_warnings = [
            issue
            for issue in result.issues
            if (
                "mismatch" in issue.message.lower()
                or "doesn't match" in issue.message.lower()
                or "may not match" in issue.message.lower()
                or "not match" in issue.message.lower()
            )
        ]

        assert len(mismatch_warnings) >= 1

    def test_detect_ambiguous_agent_goals(self, validator, base_spec):
        """Test detection of ambiguous agent goals."""
        spec = base_spec.copy()
        spec["agents"] = [
            {
                "role": "Helper",
                "goal": "Help",  # Too vague
                "backstory": "Helpful agent",
                "tools": ["tool1"],
            },
            {
                "role": "Analyzer",
                "goal": "Do analysis",  # Vague
                "backstory": "Analysis expert",
                "tools": ["analyzer"],
            },
            {
                "role": "Research Specialist",
                "goal": "Conduct comprehensive market research to identify emerging trends and opportunities",  # Specific - good
                "backstory": "Market research expert",
                "tools": ["research_tool"],
            },
        ]
        spec["tasks"] = [
            {
                "description": "Task",
                "expected_output": "Output",
                "agent": "Helper",
            }
        ]

        result = validator.validate(spec)

        # Should detect vague goals
        ambiguity_warnings = [
            issue
            for issue in result.issues
            if (
                "vague" in issue.message.lower() or "ambiguous" in issue.message.lower()
            )
            and "goal" in issue.field_path
        ]

        assert len(ambiguity_warnings) >= 2  # Should detect both vague goals

    def test_no_false_positives_for_specific_content(self, validator, base_spec):
        """Test that specific, detailed content doesn't trigger ambiguity detection."""
        spec = base_spec.copy()
        spec["project_description"] = (
            "Advanced AI-powered market research system for analyzing technology sector trends and consumer behavior patterns"
        )
        spec["agents"] = [
            {
                "role": "Senior Market Research Analyst",
                "goal": "Conduct comprehensive market analysis using advanced statistical methods to identify emerging trends in the technology sector",
                "backstory": "PhD in Economics with 15 years of experience in market research and data analysis at Fortune 500 companies",
                "tools": [
                    "advanced_web_scraper",
                    "statistical_analyzer",
                    "trend_predictor",
                    "report_generator",
                ],
            }
        ]
        spec["tasks"] = [
            {
                "description": "Research and analyze current market trends in the artificial intelligence and machine learning sector, focusing on enterprise adoption rates and key technological innovations",
                "expected_output": "Comprehensive 50-page market analysis report including statistical charts, trend forecasts, and strategic recommendations for market entry",
                "agent": "Senior Market Research Analyst",
            }
        ]

        result = validator.validate(spec)

        # Should not detect any ambiguity warnings for this specific content
        ambiguity_warnings = [
            issue
            for issue in result.issues
            if (
                "vague" in issue.message.lower()
                or "ambiguous" in issue.message.lower()
                or "unclear" in issue.message.lower()
                or "generic" in issue.message.lower()
            )
        ]

        assert len(ambiguity_warnings) == 0

    def test_detect_multiple_ambiguities_comprehensive(self, validator, base_spec):
        """Test comprehensive ambiguity detection across multiple specification areas."""
        spec = base_spec.copy()
        spec["project_description"] = "Some project"  # Vague
        spec["agents"] = [
            {
                "role": "agent",  # Vague role
                "goal": "help",  # Vague goal
                "backstory": "helpful",  # Vague backstory
                "tools": ["tool"],  # Generic tool name
            }
        ]
        spec["tasks"] = [
            {
                "description": "task",  # Vague description
                "expected_output": "output",  # Vague output
                "agent": "agent",
            }
        ]

        result = validator.validate(spec)

        # Should detect multiple types of ambiguity
        ambiguity_issues = [
            issue
            for issue in result.issues
            if (
                "vague" in issue.message.lower()
                or "ambiguous" in issue.message.lower()
                or "unclear" in issue.message.lower()
                or "generic" in issue.message.lower()
            )
        ]

        # Should detect ambiguities in multiple areas
        assert len(ambiguity_issues) >= 5  # Multiple ambiguous elements

        # Check that different field paths are covered
        field_paths = {issue.field_path.split(".")[0] for issue in ambiguity_issues}
        assert "project_description" in field_paths or any(
            "agents" in path for path in field_paths
        )
