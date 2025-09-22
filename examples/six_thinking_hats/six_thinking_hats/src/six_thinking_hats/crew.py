from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class SixThinkingHats():
    """Six Thinking Hats crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools    @agent
    def cto_advisor(self) -> Agent:
        return Agent(
            config=self.agents_config['cto_advisor'],  # type: ignore[index]
            verbose=True        )
    @agent
    def cmo_advisor(self) -> Agent:
        return Agent(
            config=self.agents_config['cmo_advisor'],  # type: ignore[index]
            verbose=True        )
    @agent
    def cfo_advisor(self) -> Agent:
        return Agent(
            config=self.agents_config['cfo_advisor'],  # type: ignore[index]
            verbose=True        )
    @agent
    def cpo_advisor(self) -> Agent:
        return Agent(
            config=self.agents_config['cpo_advisor'],  # type: ignore[index]
            verbose=True        )
    @agent
    def vc_advisor(self) -> Agent:
        return Agent(
            config=self.agents_config['vc_advisor'],  # type: ignore[index]
            verbose=True        )
    @agent
    def intelligent_moderator(self) -> Agent:
        return Agent(
            config=self.agents_config['intelligent_moderator'],  # type: ignore[index]
            verbose=True        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task    @task
    def gather_initial_startup_idea_de_task(self) -> Task:
        return Task(
            config=self.tasks_config['gather_initial_startup_idea_de_task'],  # type: ignore[index]        )
    @task
    def analyze_the_technical_feasibil_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_the_technical_feasibil_task'],  # type: ignore[index]        )
    @task
    def evaluate_the_market_potential_task(self) -> Task:
        return Task(
            config=self.tasks_config['evaluate_the_market_potential_task'],  # type: ignore[index]        )
    @task
    def assess_the_financial_viability_task(self) -> Task:
        return Task(
            config=self.tasks_config['assess_the_financial_viability_task'],  # type: ignore[index]        )
    @task
    def evaluate_the_operational_and_h_task(self) -> Task:
        return Task(
            config=self.tasks_config['evaluate_the_operational_and_h_task'],  # type: ignore[index]        )
    @task
    def provide_a_venture_capital_pers_task(self) -> Task:
        return Task(
            config=self.tasks_config['provide_a_venture_capital_pers_task'],  # type: ignore[index]        )
    @task
    def facilitate_a_structured_debate_task(self) -> Task:
        return Task(
            config=self.tasks_config['facilitate_a_structured_debate_task'],  # type: ignore[index]        )
    @task
    def compile_a_final_multi_dimensio_task(self) -> Task:
        return Task(
            config=self.tasks_config['compile_a_final_multi_dimensio_task'],  # type: ignore[index]        )

    @crew
    def crew(self) -> Crew:
        """Creates the Six Thinking Hats crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,    # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical,  # Uncomment for hierarchical process
            # For hierarchical process, see:
            # https://docs.crewai.com/concepts/processes#hierarchical_process
        )