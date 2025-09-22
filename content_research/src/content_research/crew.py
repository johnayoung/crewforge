from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import Listfrom .tools.custom_tool import {'name': 'SerperDevTool', 'reason': "The SerperDevTool is selected for the 'web_search' capability as it provides web search capabilities using Google Search."}, {'name': 'CodeInterpreterTool', 'reason': "The CodeInterpreterTool is selected for the 'data_analysis' capability as it can execute Python code and scripts, which can be used for data analysis."}
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class ContentResearch():
    """Content Research crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools    @agent
    def content_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['content_researcher'],  # type: ignore[index]
            verbose=True        )

    @agent
    def data_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['data_analyst'],  # type: ignore[index]
            verbose=True        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task    @task
    def identify_main_competitors_in_t_task(self) -> Task:
        return Task(
            config=self.tasks_config['identify_main_competitors_in_t_task'],  # type: ignore[index]        )

    @task
    def find_articles_published_by_the_task(self) -> Task:
        return Task(
            config=self.tasks_config['find_articles_published_by_the_task'],  # type: ignore[index]        )

    @task
    def extract_key_information_from_e_task(self) -> Task:
        return Task(
            config=self.tasks_config['extract_key_information_from_e_task'],  # type: ignore[index]        )

    @task
    def analyze_the_frequency_of_artic_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_the_frequency_of_artic_task'],  # type: ignore[index]        )

    @task
    def analyze_the_main_topics_covere_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_the_main_topics_covere_task'],  # type: ignore[index]        )

    @task
    def compare_the_content_strategies_task(self) -> Task:
        return Task(
            config=self.tasks_config['compare_the_content_strategies_task'],  # type: ignore[index]        )

    @crew
    def crew(self) -> Crew:
        """Creates the Content Research crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,    # Automatically created by the @task decorator
            process=Process.sequential,  # In case you want to use a hierarchical process: https://docs.crewai.com/concepts/processes#hierarchical_process
            verbose=True,
            # process=Process.hierarchical, # In case you want to use a hierarchical process: https://docs.crewai.com/concepts/processes#hierarchical_process
        )