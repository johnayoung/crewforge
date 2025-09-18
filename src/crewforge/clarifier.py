"""
Interactive Clarifier for CrewForge.

This module provides intelligent question generation to resolve ambiguities
and gather missing information from CrewAI project specifications.
"""

import json
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from crewforge.llm import LLMClient, LLMError
from crewforge.validation import ValidationResult, ValidationIssue, IssueSeverity


class LLMProvider(Enum):
    """Supported LLM providers compatible with liteLLM."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"
    SAMBA_NOVA = "sambanova"


class ProjectStructure(Enum):
    """Project structure preferences."""

    STANDARD = "standard"  # Standard CrewAI structure
    FLAT = "flat"  # Minimal folder structure
    DETAILED = "detailed"  # Extended structure with more folders


class NamingConvention(Enum):
    """Agent naming conventions."""

    DESCRIPTIVE = "descriptive"  # ResearchAnalyst, ContentWriter
    ROLE_BASED = "role_based"  # Researcher, Writer
    FUNCTIONAL = "functional"  # DataGatherer, ReportGenerator


@dataclass
class UserPreferences:
    """Configuration preferences collected from user."""

    # LLM and API preferences
    preferred_llm_provider: Optional[LLMProvider] = None
    model_temperature: Optional[float] = None  # 0.0-1.0
    max_tokens_per_response: Optional[int] = None

    # Project structure preferences
    project_structure: Optional[ProjectStructure] = None
    naming_convention: Optional[NamingConvention] = None
    include_examples: Optional[bool] = None

    # CrewAI specific preferences
    max_agents_per_crew: Optional[int] = None
    max_tasks_per_agent: Optional[int] = None
    prefer_sequential_execution: Optional[bool] = None

    # Tool integration preferences
    include_web_search: Optional[bool] = None
    include_file_operations: Optional[bool] = None
    include_data_analysis: Optional[bool] = None

    # Output and documentation preferences
    verbose_output: Optional[bool] = None
    include_detailed_documentation: Optional[bool] = None
    generate_readme: Optional[bool] = None

    # User experience preferences
    interactive_mode: Optional[bool] = None  # Ask for clarification during generation
    auto_install_dependencies: Optional[bool] = None
    create_virtual_environment: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert preferences to dictionary format."""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if field_value is not None:
                if isinstance(field_value, Enum):
                    result[field_name] = field_value.value
                else:
                    result[field_name] = field_value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPreferences":
        """Create UserPreferences from dictionary."""
        preferences = cls()

        # Handle enum fields
        enum_fields = {
            "preferred_llm_provider": LLMProvider,
            "project_structure": ProjectStructure,
            "naming_convention": NamingConvention,
        }

        for field_name, value in data.items():
            if hasattr(preferences, field_name):
                if field_name in enum_fields and value is not None:
                    try:
                        setattr(preferences, field_name, enum_fields[field_name](value))
                    except ValueError:
                        # Skip invalid enum values
                        continue
                else:
                    setattr(preferences, field_name, value)

        return preferences

    def merge(self, other: "UserPreferences") -> "UserPreferences":
        """Merge with another preferences object, with other taking precedence."""
        merged_data = self.to_dict()
        merged_data.update(other.to_dict())
        return UserPreferences.from_dict(merged_data)


class PreferenceCategory(Enum):
    """Categories of preferences for organized collection."""

    LLM_SETUP = "llm_setup"
    PROJECT_STRUCTURE = "project_structure"
    CREW_CONFIGURATION = "crew_configuration"
    TOOL_INTEGRATION = "tool_integration"
    OUTPUT_OPTIONS = "output_options"
    USER_EXPERIENCE = "user_experience"


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
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


class ConversationState(Enum):
    """Current state of the conversation flow."""

    INITIAL = "initial"
    ASKING_QUESTIONS = "asking_questions"
    PROCESSING_RESPONSES = "processing_responses"
    GENERATING_FOLLOWUP = "generating_followup"
    COMPLETED = "completed"


class ResponseType(Enum):
    """Types of user responses to questions."""

    DIRECT_ANSWER = "direct_answer"
    PARTIAL_ANSWER = "partial_answer"
    CLARIFICATION_REQUEST = "clarification_request"
    SKIP_QUESTION = "skip_question"


@dataclass
class UserResponse:
    """Represents a user's response to a clarification question."""

    question_id: str
    response_type: ResponseType
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    confidence_level: Optional[float] = None


@dataclass
class ConversationTurn:
    """Represents one complete turn in the conversation."""

    turn_number: int
    questions_asked: List[Question]
    user_responses: List[UserResponse]
    specification_updates: Dict[str, Any]
    follow_up_needed: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConversationContext:
    """Complete context for a multi-turn clarification conversation."""

    session_id: str
    initial_specification: Dict[str, Any]
    current_specification: Dict[str, Any]
    user_preferences: UserPreferences = field(default_factory=UserPreferences)
    conversation_state: ConversationState = ConversationState.INITIAL
    turns: List[ConversationTurn] = field(default_factory=list)
    pending_questions: List[Question] = field(default_factory=list)
    resolved_questions: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

    @property
    def current_turn_number(self) -> int:
        """Get the current turn number."""
        return len(self.turns) + 1

    @property
    def total_questions_asked(self) -> int:
        """Get total number of questions asked across all turns."""
        return sum(len(turn.questions_asked) for turn in self.turns)

    @property
    def total_responses_received(self) -> int:
        """Get total number of responses received."""
        return sum(len(turn.user_responses) for turn in self.turns)

    def has_pending_questions(self) -> bool:
        """Check if there are unresolved questions."""
        return len(self.pending_questions) > 0

    def is_conversation_complete(self) -> bool:
        """Check if the conversation is complete."""
        return self.conversation_state == ConversationState.COMPLETED or (
            not self.has_pending_questions() and len(self.turns) > 0
        )


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

    PREFERENCE_COLLECTION_PROMPTS = {
        PreferenceCategory.LLM_SETUP: """Let me help you configure your LLM preferences for this CrewAI project.

Current specification suggests the following AI tasks: {tasks_summary}

Which LLM provider would you prefer to use?
1. OpenAI (GPT-3.5/GPT-4) - Most reliable, good for complex reasoning
2. Anthropic (Claude) - Excellent for analysis and writing tasks
3. Google (Gemini) - Strong at code generation and structured tasks
4. Groq (Llama/Mixtral) - Fast inference, good for simpler tasks
5. SambaNova - Efficient for text processing tasks

Please choose (1-5) or specify your preference: """,
        PreferenceCategory.PROJECT_STRUCTURE: """I'll help you set up the project structure for optimal organization.

Based on your project type, I can create:
1. Standard - CrewAI default structure with src/, config/, tools/ folders
2. Flat - Minimal structure for simple projects (all files in root)
3. Detailed - Extended structure with examples/, docs/, tests/ folders

For agent naming, would you prefer:
A. Descriptive names (ResearchAnalyst, ContentWriter)
B. Role-based names (Researcher, Writer) 
C. Functional names (DataGatherer, ReportGenerator)

Structure preference (1-3): 
Naming convention (A-C): """,
        PreferenceCategory.CREW_CONFIGURATION: """Let me configure your CrewAI crew settings for optimal performance.

For your project with {agent_count} agents and {task_count} tasks:

Maximum agents per crew (recommended 3-5): {max_agents_suggestion}
Maximum tasks per agent (recommended 1-3): {max_tasks_suggestion}

Execution preference:
1. Sequential - Tasks run one after another (more predictable)
2. Hierarchical - Manager agent delegates tasks (more flexible)

Would you like sequential execution? (y/n): """,
        PreferenceCategory.TOOL_INTEGRATION: """I can integrate various tools to enhance your agents' capabilities.

Based on your project requirements, these tools might be useful:

Web Search & Research:
- Include web search capabilities? (y/n)
- Include web scraping tools? (y/n)

File & Data Operations:
- Include file read/write operations? (y/n)  
- Include data analysis tools? (y/n)
- Include database connections? (y/n)

Communication & Integration:
- Include email/messaging tools? (y/n)
- Include API integration tools? (y/n)

Please specify your preferences: """,
        PreferenceCategory.OUTPUT_OPTIONS: """Let me configure the output and documentation preferences.

Documentation Generation:
- Generate detailed README with setup instructions? (y/n)
- Include code examples and usage guides? (y/n)
- Create API documentation for custom tools? (y/n)

Output Verbosity:
- Enable verbose logging during execution? (y/n)
- Include performance metrics in output? (y/n)
- Show intermediate step results? (y/n)

Please specify your preferences: """,
        PreferenceCategory.USER_EXPERIENCE: """Finally, let me configure your development workflow preferences.

Development Environment:
- Automatically install dependencies with uv? (y/n)
- Create virtual environment for project? (y/n)
- Set up pre-commit hooks for code quality? (y/n)

Interactive Features:
- Enable interactive mode (ask for clarification during generation)? (y/n)
- Auto-validate generated code before completion? (y/n)
- Include example usage in generated project? (y/n)

Please specify your preferences: """,
    }

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

    # Multi-turn conversation methods

    async def start_clarification_session(
        self, specification: Dict[str, Any]
    ) -> ConversationContext:
        """
        Start a new multi-turn clarification session.

        Args:
            specification: Initial project specification

        Returns:
            New conversation context
        """
        session_id = str(uuid.uuid4())

        context = ConversationContext(
            session_id=session_id,
            initial_specification=specification.copy(),
            current_specification=specification.copy(),
            conversation_state=ConversationState.INITIAL,
        )

        return context

    async def ask_next_questions(
        self, context: ConversationContext, max_questions: int = 3
    ) -> List[Question]:
        """
        Ask the next batch of questions in the conversation.

        Args:
            context: Current conversation context
            max_questions: Maximum number of questions to ask at once

        Returns:
            List of questions to ask the user
        """
        # Create a validation result from the current specification state
        # This would normally come from the SpecificationValidator
        # For now, we'll create a mock validation to generate questions

        validation_issues = self._analyze_specification_gaps(
            context.current_specification
        )
        validation_result = ValidationResult(
            issues=validation_issues,
            completeness_score=self._calculate_completeness(
                context.current_specification
            ),
        )

        # Generate context-aware questions
        questions = await self._generate_context_aware_questions(
            validation_result, context, max_questions
        )

        # Update conversation state
        context.conversation_state = ConversationState.ASKING_QUESTIONS
        context.pending_questions.extend(questions)
        context.last_activity = datetime.now()

        return questions

    async def start_preference_collection(
        self,
        context: ConversationContext,
        categories: Optional[List[PreferenceCategory]] = None,
    ) -> ConversationContext:
        """
        Start preference collection as part of the conversation flow.

        Args:
            context: Current conversation context
            categories: Specific categories to collect (all if None)

        Returns:
            Updated conversation context with collected preferences
        """
        # Collect preferences using LLM for intelligent defaults
        collected_preferences = await self.collect_user_preferences(context, categories)

        # Update context with collected preferences
        updated_context = self.update_context_preferences(
            context, collected_preferences
        )

        # Generate preference confirmation questions
        preference_questions = self._generate_preference_confirmation_questions(
            collected_preferences, categories or list(PreferenceCategory)
        )

        # Add preference questions to pending questions
        updated_context.pending_questions.extend(preference_questions)
        updated_context.conversation_state = ConversationState.ASKING_QUESTIONS

        return updated_context

    def _generate_preference_confirmation_questions(
        self, preferences: UserPreferences, categories: List[PreferenceCategory]
    ) -> List[Question]:
        """
        Generate questions to confirm and refine collected preferences.

        Args:
            preferences: Collected preferences to confirm
            categories: Categories that were processed

        Returns:
            List of confirmation questions
        """
        questions = []

        for category in categories:
            # Create a summary of preferences for this category
            category_prefs = self._extract_category_preferences(preferences, category)
            if not category_prefs:
                continue

            # Generate confirmation question
            question = Question(
                question_type=QuestionType.INCOMPLETE_SPEC,
                field=f"preferences.{category.value}",
                question=f"I've suggested these {category.value.replace('_', ' ')} preferences: {self._format_preferences_summary(category_prefs)}. Would you like to modify any of these? (y/n)",
                context=f"Confirming AI-suggested preferences for {category.value}",
                suggestions=["y", "n", "keep defaults", "customize"],
            )
            questions.append(question)

        return questions

    def _extract_category_preferences(
        self, preferences: UserPreferences, category: PreferenceCategory
    ) -> Dict[str, Any]:
        """
        Extract preferences relevant to a specific category.

        Args:
            preferences: Full preferences object
            category: Category to extract

        Returns:
            Dictionary of category-specific preferences
        """
        category_fields = {
            PreferenceCategory.LLM_SETUP: [
                "preferred_llm_provider",
                "model_temperature",
                "max_tokens_per_response",
            ],
            PreferenceCategory.PROJECT_STRUCTURE: [
                "project_structure",
                "naming_convention",
                "include_examples",
            ],
            PreferenceCategory.CREW_CONFIGURATION: [
                "max_agents_per_crew",
                "max_tasks_per_agent",
                "prefer_sequential_execution",
            ],
            PreferenceCategory.TOOL_INTEGRATION: [
                "include_web_search",
                "include_file_operations",
                "include_data_analysis",
            ],
            PreferenceCategory.OUTPUT_OPTIONS: [
                "verbose_output",
                "include_detailed_documentation",
                "generate_readme",
            ],
            PreferenceCategory.USER_EXPERIENCE: [
                "interactive_mode",
                "auto_install_dependencies",
                "create_virtual_environment",
            ],
        }

        fields = category_fields.get(category, [])
        result = {}

        for field in fields:
            value = getattr(preferences, field, None)
            if value is not None:
                result[field] = value

        return result

    def _format_preferences_summary(self, category_prefs: Dict[str, Any]) -> str:
        """
        Format preferences for user-friendly display.

        Args:
            category_prefs: Dictionary of preferences

        Returns:
            Formatted string summary
        """
        if not category_prefs:
            return "none"

        formatted_items = []
        for key, value in category_prefs.items():
            # Convert enum values to readable format
            if hasattr(value, "value"):
                display_value = value.value.replace("_", " ").title()
            elif isinstance(value, bool):
                display_value = "Yes" if value else "No"
            else:
                display_value = str(value)

            # Convert field names to readable format
            display_key = key.replace("_", " ").title()
            formatted_items.append(f"{display_key}: {display_value}")

        return "; ".join(formatted_items)

    async def process_answers(
        self, context: ConversationContext, answers: List[UserResponse]
    ) -> ConversationContext:
        """
        Process user answers and update the conversation context.

        Args:
            context: Current conversation context
            answers: List of user responses

        Returns:
            Updated conversation context
        """
        context.conversation_state = ConversationState.PROCESSING_RESPONSES

        # Process each response and update specification
        specification_updates = {}
        for response in answers:
            updates = await self._process_single_response(response, context)
            specification_updates.update(updates)

            # Mark question as resolved
            if response.question_id not in context.resolved_questions:
                context.resolved_questions.append(response.question_id)

        # Update current specification
        context.current_specification.update(specification_updates)

        # Create a new conversation turn
        turn = ConversationTurn(
            turn_number=context.current_turn_number,
            questions_asked=context.pending_questions.copy(),
            user_responses=answers,
            specification_updates=specification_updates,
            follow_up_needed=self._needs_follow_up(answers),
        )

        context.turns.append(turn)
        context.pending_questions = []  # Clear pending questions
        context.last_activity = datetime.now()

        return context

    async def check_completion(
        self, context: ConversationContext
    ) -> tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if clarification is complete.

        Args:
            context: Current conversation context

        Returns:
            Tuple of (is_complete, final_specification)
        """
        # Check if we have enough information for a complete specification
        completeness_score = self._calculate_completeness(context.current_specification)

        if completeness_score >= 0.7 and not context.has_pending_questions():
            context.conversation_state = ConversationState.COMPLETED
            return True, context.current_specification

        return False, None

    # Helper methods for multi-turn functionality

    def _analyze_specification_gaps(
        self, specification: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """
        Analyze specification for gaps that need clarification.

        Args:
            specification: Current specification to analyze

        Returns:
            List of validation issues found
        """
        issues = []

        # Check for missing required fields
        if not specification.get("project_description"):
            issues.append(
                ValidationIssue(
                    IssueSeverity.ERROR,
                    "Project description is missing",
                    "project_description",
                )
            )

        if not specification.get("agents"):
            issues.append(
                ValidationIssue(
                    IssueSeverity.ERROR, "At least one agent is required", "agents"
                )
            )

        # Check for vague agent roles
        agents = specification.get("agents", [])
        for i, agent in enumerate(agents):
            if isinstance(agent, dict) and agent.get("role") in [
                "assistant",
                "helper",
                "agent",
            ]:
                issues.append(
                    ValidationIssue(
                        IssueSeverity.WARNING,
                        f"Agent role '{agent.get('role')}' is too vague",
                        f"agents[{i}].role",
                    )
                )

        return issues

    def _calculate_completeness(self, specification: Dict[str, Any]) -> float:
        """
        Calculate completeness score for a specification.

        Args:
            specification: Specification to score

        Returns:
            Completeness score between 0.0 and 1.0
        """
        required_fields = ["project_description", "agents"]
        optional_fields = ["tasks", "tools", "dependencies"]

        score = 0.0

        # Check required fields (70% of score)
        for field in required_fields:
            if specification.get(field):
                score += 0.35

        # Check optional fields (30% of score)
        for field in optional_fields:
            if specification.get(field):
                score += 0.1

        return min(score, 1.0)

    async def _generate_context_aware_questions(
        self,
        validation_result: ValidationResult,
        context: ConversationContext,
        max_questions: int,
    ) -> List[Question]:
        """
        Generate questions that are aware of the conversation context.

        Args:
            validation_result: Validation result to base questions on
            context: Current conversation context
            max_questions: Maximum number of questions to generate

        Returns:
            List of context-aware questions
        """
        # Use the existing generate_questions method but enhance with context
        base_questions = await self.generate_questions(validation_result)

        # Enhance questions with conversation context
        context_enhanced_questions = []
        for question in base_questions[:max_questions]:
            # Add context from previous conversation turns
            enhanced_context = self._build_enhanced_context(question, context)

            enhanced_question = Question(
                question_type=question.question_type,
                field=question.field,
                question=question.question,
                context=enhanced_context,
                suggestions=question.suggestions,
            )

            context_enhanced_questions.append(enhanced_question)

        return context_enhanced_questions

    def _build_enhanced_context(
        self, question: Question, context: ConversationContext
    ) -> str:
        """
        Build enhanced context for a question based on conversation history.

        Args:
            question: Original question
            context: Conversation context

        Returns:
            Enhanced context string
        """
        base_context = question.context

        # Add context from previous turns if relevant
        if context.turns:
            # Look for related previous discussions
            for turn in context.turns[-2:]:  # Look at last 2 turns
                for response in turn.user_responses:
                    if self._is_related_to_question(response, question):
                        base_context += (
                            f" You previously mentioned: {response.content[:50]}..."
                        )
                        break

        return base_context

    def _is_related_to_question(
        self, response: UserResponse, question: Question
    ) -> bool:
        """
        Check if a previous response is related to the current question.

        Args:
            response: Previous user response
            question: Current question

        Returns:
            True if related, False otherwise
        """
        # Simple heuristic - check if question field is mentioned in response
        field_parts = question.field.lower().split(".")
        response_lower = response.content.lower()

        for part in field_parts:
            if part in response_lower:
                return True

        return False

    async def _process_single_response(
        self, response: UserResponse, context: ConversationContext
    ) -> Dict[str, Any]:
        """
        Process a single user response and extract specification updates.

        Args:
            response: User response to process
            context: Current conversation context

        Returns:
            Dictionary of specification updates
        """
        updates = {}

        if response.response_type == ResponseType.DIRECT_ANSWER:
            # Parse the response based on the question it's answering
            question = self._find_question_by_id(response.question_id, context)
            if question:
                updates = self._extract_specification_updates(
                    response.content, question
                )

        return updates

    def _find_question_by_id(
        self, question_id: str, context: ConversationContext
    ) -> Optional[Question]:
        """
        Find a question by its ID in the conversation context.

        Args:
            question_id: ID of the question to find
            context: Conversation context to search

        Returns:
            Question if found, None otherwise
        """
        # Search in pending questions
        for question in context.pending_questions:
            if question.id == question_id:
                return question

        # Search in previous turns
        for turn in context.turns:
            for question in turn.questions_asked:
                if question.id == question_id:
                    return question

        return None

    def _extract_specification_updates(
        self, response_content: str, question: Question
    ) -> Dict[str, Any]:
        """
        Extract specification updates from a response.

        Args:
            response_content: User's response content
            question: The question being answered

        Returns:
            Dictionary of specification updates
        """
        updates = {}

        # Simple extraction based on question field
        if question.field == "project_description":
            updates["project_description"] = response_content
        elif question.field == "agents":
            # Parse agent mentions from response
            agents = self._parse_agents_from_response(response_content)
            if agents:
                updates["agents"] = agents
        elif question.field.startswith("agents[") and "role" in question.field:
            # Update specific agent role
            updates = self._update_agent_role(response_content, question.field)

        return updates

    def _parse_agents_from_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse agent information from user response.

        Args:
            response: User response containing agent information

        Returns:
            List of agent dictionaries
        """
        agents = []
        response_lower = response.lower()

        # Look for common agent types
        agent_keywords = {
            "research": "researcher",
            "write": "writer",
            "content": "content_creator",
            "social": "social_media_manager",
            "analysis": "analyst",
            "market": "market_researcher",
        }

        for keyword, role in agent_keywords.items():
            if keyword in response_lower:
                agents.append({"role": role, "type": "agent"})

        return agents

    def _update_agent_role(self, response: str, field_path: str) -> Dict[str, Any]:
        """
        Update a specific agent's role based on field path.

        Args:
            response: User response with new role information
            field_path: Field path like "agents[0].role"

        Returns:
            Dictionary with agent updates
        """
        # For now, return a simple update structure
        # In a real implementation, this would parse the field path
        # and create the appropriate nested update
        return {"agent_role_update": response}

    def _needs_follow_up(self, responses: List[UserResponse]) -> bool:
        """
        Determine if follow-up questions are needed based on responses.

        Args:
            responses: List of user responses

        Returns:
            True if follow-up questions are needed
        """
        for response in responses:
            if response.response_type in [
                ResponseType.PARTIAL_ANSWER,
                ResponseType.CLARIFICATION_REQUEST,
            ]:
                return True

        return False

    async def collect_user_preferences(
        self,
        context: ConversationContext,
        categories: Optional[List[PreferenceCategory]] = None,
    ) -> UserPreferences:
        """
        Collect user preferences through guided prompts.

        Args:
            context: Current conversation context with project specification
            categories: Specific preference categories to collect (all if None)

        Returns:
            UserPreferences object with collected preferences

        Raises:
            ClarificationError: If preference collection fails
        """
        if categories is None:
            categories = list(PreferenceCategory)

        preferences = UserPreferences()

        for category in categories:
            try:
                category_preferences = await self._collect_category_preferences(
                    context, category
                )
                preferences = preferences.merge(category_preferences)
            except Exception as e:
                # Log error but continue with other categories
                continue

        return preferences

    async def _collect_category_preferences(
        self, context: ConversationContext, category: PreferenceCategory
    ) -> UserPreferences:
        """
        Collect preferences for a specific category.

        Args:
            context: Current conversation context
            category: Category to collect preferences for

        Returns:
            UserPreferences with category-specific preferences set
        """
        preferences = UserPreferences()

        # Get base prompt for this category
        if category not in self.PREFERENCE_COLLECTION_PROMPTS:
            return preferences

        base_prompt = self.PREFERENCE_COLLECTION_PROMPTS[category]

        # Customize prompt with context-specific information
        prompt = self._customize_preference_prompt(base_prompt, context, category)

        try:
            # Use LLM to suggest intelligent defaults
            suggestion_prompt = f"""
            Based on this project specification: {json.dumps(context.current_specification, indent=2)}
            
            Suggest optimal configuration preferences for the {category.value} category.
            Consider the project type, complexity, and mentioned requirements.
            
            Return suggestions as a JSON object with preference field names as keys.
            Only include fields you're confident about. Use null for uncertain preferences.
            
            Example format:
            {{
                "preferred_llm_provider": "openai",
                "max_agents_per_crew": 3,
                "include_examples": true
            }}
            """

            # Get intelligent defaults from LLM
            response = await self.llm_client.complete(
                prompt=suggestion_prompt, max_tokens=500, temperature=0.3
            )

            # Parse LLM suggestions
            try:
                suggested_preferences = json.loads(response.strip())
                preferences = UserPreferences.from_dict(suggested_preferences)
            except json.JSONDecodeError:
                # Fall back to defaults if parsing fails
                preferences = self._get_default_preferences_for_category(category)

        except LLMError:
            # Fall back to hardcoded defaults
            preferences = self._get_default_preferences_for_category(category)

        return preferences

    def _customize_preference_prompt(
        self,
        base_prompt: str,
        context: ConversationContext,
        category: PreferenceCategory,
    ) -> str:
        """
        Customize preference collection prompt with context-specific information.

        Args:
            base_prompt: Base template prompt
            context: Current conversation context
            category: Preference category being collected

        Returns:
            Customized prompt string
        """
        spec = context.current_specification

        # Extract context information
        agents = spec.get("agents", [])
        tasks = spec.get("tasks", [])

        # Build context replacements
        replacements = {
            "tasks_summary": self._summarize_tasks(tasks),
            "agent_count": str(len(agents)),
            "task_count": str(len(tasks)),
            "max_agents_suggestion": str(min(5, max(3, len(agents)))),
            "max_tasks_suggestion": str(
                min(3, max(1, len(tasks) // len(agents) if agents else 1))
            ),
        }

        # Apply replacements
        customized_prompt = base_prompt
        for placeholder, value in replacements.items():
            customized_prompt = customized_prompt.replace(f"{{{placeholder}}}", value)

        return customized_prompt

    def _summarize_tasks(self, tasks: List[Dict[str, Any]]) -> str:
        """
        Create a brief summary of tasks for prompt customization.

        Args:
            tasks: List of task specifications

        Returns:
            Brief task summary string
        """
        if not tasks:
            return "no specific tasks defined"

        task_types = []
        for task in tasks:
            description = task.get("description", "").lower()
            if "research" in description:
                task_types.append("research")
            elif "write" in description or "content" in description:
                task_types.append("content creation")
            elif "analy" in description:
                task_types.append("analysis")
            elif "social" in description:
                task_types.append("social media")
            else:
                task_types.append("general task")

        # Remove duplicates and create summary
        unique_types = list(dict.fromkeys(task_types))
        if len(unique_types) <= 2:
            return " and ".join(unique_types)
        else:
            return f"{', '.join(unique_types[:-1])}, and {unique_types[-1]}"

    def _get_default_preferences_for_category(
        self, category: PreferenceCategory
    ) -> UserPreferences:
        """
        Get sensible default preferences for a category.

        Args:
            category: Preference category

        Returns:
            UserPreferences with category defaults
        """
        defaults_map = {
            PreferenceCategory.LLM_SETUP: UserPreferences(
                preferred_llm_provider=LLMProvider.OPENAI,
                model_temperature=0.3,
                max_tokens_per_response=2000,
            ),
            PreferenceCategory.PROJECT_STRUCTURE: UserPreferences(
                project_structure=ProjectStructure.STANDARD,
                naming_convention=NamingConvention.DESCRIPTIVE,
                include_examples=True,
            ),
            PreferenceCategory.CREW_CONFIGURATION: UserPreferences(
                max_agents_per_crew=5,
                max_tasks_per_agent=2,
                prefer_sequential_execution=True,
            ),
            PreferenceCategory.TOOL_INTEGRATION: UserPreferences(
                include_web_search=True,
                include_file_operations=True,
                include_data_analysis=False,
            ),
            PreferenceCategory.OUTPUT_OPTIONS: UserPreferences(
                verbose_output=False,
                include_detailed_documentation=True,
                generate_readme=True,
            ),
            PreferenceCategory.USER_EXPERIENCE: UserPreferences(
                interactive_mode=True,
                auto_install_dependencies=True,
                create_virtual_environment=True,
            ),
        }

        return defaults_map.get(category, UserPreferences())

    def update_context_preferences(
        self, context: ConversationContext, new_preferences: UserPreferences
    ) -> ConversationContext:
        """
        Update conversation context with new user preferences.

        Args:
            context: Current conversation context
            new_preferences: New preferences to merge

        Returns:
            Updated conversation context
        """
        # Merge preferences
        updated_preferences = context.user_preferences.merge(new_preferences)

        # Create updated context
        updated_context = replace(
            context, user_preferences=updated_preferences, last_activity=datetime.now()
        )

        return updated_context
