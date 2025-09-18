"""
Interactive Clarifier for CrewForge.

This module provides intelligent question generation to resolve ambiguities
and gather missing information from CrewAI project specifications.
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from crewforge.llm import LLMClient, LLMError
from crewforge.validation import ValidationResult, ValidationIssue, IssueSeverity


class QuestionType(Enum):
    """Types of clarification questions."""

    MISSING_FIELD = "missing_field"
    VAGUE_CONTENT = "vague_content"
    TOOL_MISMATCH = "tool_mismatch"
    INCOMPLETE_SPEC = "incomplete_spec"


@dataclass
class Question:
    """Represents a clarification question for the user."""

    question_type: QuestionType
    field: str
    question: str
    context: str
    suggestions: List[str]


class ClarificationError(Exception):
    """Exception raised when clarification fails."""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


class InteractiveClarifier:
    """
    Interactive clarifier for resolving ambiguous or incomplete specifications.

    This component analyzes validation results and generates targeted questions
    to help users provide missing or clarify vague information in their
    CrewAI project specifications.
    """

    QUESTION_GENERATION_PROMPT = """You are a CrewAI project specification expert helping users create better project configurations.

Analyze the following validation issues and generate targeted clarification questions to help resolve them.

Validation Issues:
{issues_text}

For each issue that requires user input, generate a JSON object with these fields:
- type: One of "missing_field", "vague_content", "tool_mismatch", "incomplete_spec"
- field: The field path that needs clarification (e.g., "agents[0].role")
- question: A clear, specific question for the user
- context: Brief explanation of why this information is needed
- suggestions: Array of 2-4 example values or options

Only generate questions for ERROR and WARNING level issues that require user input.
Skip INFO level issues and issues that are purely structural validation.

Return your response as a valid JSON array of question objects."""

    def __init__(self, llm_client: LLMClient):
        """
        Initialize the Interactive Clarifier.

        Args:
            llm_client: LLM client for generating questions
        """
        self.llm_client = llm_client
        self.conversation_history: List[Dict[str, str]] = []

    async def generate_questions(
        self, validation_result: ValidationResult
    ) -> List[Question]:
        """
        Generate targeted questions based on validation issues.

        Args:
            validation_result: Result from specification validation

        Returns:
            List of clarification questions for the user

        Raises:
            ClarificationError: If question generation fails
        """
        # Filter issues that need clarification (ERROR and WARNING only)
        clarifiable_issues = [
            issue
            for issue in validation_result.issues
            if issue.severity in [IssueSeverity.ERROR, IssueSeverity.WARNING]
        ]

        if not clarifiable_issues:
            return []

        # Format issues for the LLM prompt
        issues_text = self._format_issues_for_prompt(clarifiable_issues)

        # Generate the prompt
        prompt = self.QUESTION_GENERATION_PROMPT.format(issues_text=issues_text)

        try:
            # Call LLM to generate questions
            response = await self.llm_client.complete(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.3,  # Lower temperature for more focused questions
            )

            # Parse JSON response
            questions_data = json.loads(response.strip())

            # Convert to Question objects
            questions = []
            for q_data in questions_data:
                try:
                    question = Question(
                        question_type=QuestionType(q_data["type"]),
                        field=q_data["field"],
                        question=q_data["question"],
                        context=q_data["context"],
                        suggestions=q_data["suggestions"],
                    )
                    questions.append(question)
                except (KeyError, ValueError, TypeError) as e:
                    # Skip malformed question data but continue with others
                    continue

            return questions

        except LLMError as e:
            raise ClarificationError(
                f"Failed to generate clarification questions: {str(e)}", e
            )
        except json.JSONDecodeError as e:
            raise ClarificationError(
                "Invalid JSON response from question generation", e
            )
        except Exception as e:
            raise ClarificationError(
                f"Unexpected error during question generation: {str(e)}", e
            )

    def _format_issues_for_prompt(self, issues: List[ValidationIssue]) -> str:
        """
        Format validation issues for the LLM prompt.

        Args:
            issues: List of validation issues to format

        Returns:
            Formatted string representation of issues
        """
        formatted_issues = []

        for issue in issues:
            formatted_issues.append(
                f"- [{issue.severity.value.upper()}] {issue.field_path}: {issue.message}"
            )

        return "\n".join(formatted_issues)

    def add_to_history(self, role: str, message: str) -> None:
        """
        Add a message to the conversation history.

        Args:
            role: Role of the speaker ("user" or "assistant")
            message: The message content
        """
        self.conversation_history.append({"role": role, "message": message})

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []

    def get_conversation_context(self) -> str:
        """
        Get conversation history formatted for LLM context.

        Returns:
            Formatted conversation history
        """
        if not self.conversation_history:
            return ""

        context_lines = []
        for entry in self.conversation_history:
            context_lines.append(f"{entry['role']}: {entry['message']}")

        return "\n".join(context_lines)
