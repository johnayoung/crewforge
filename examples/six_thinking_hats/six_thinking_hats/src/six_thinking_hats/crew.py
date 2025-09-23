from crewai.core import CrewBase, CrewAgent, CrewTask
from crewai.config import load_agents_config, load_tasks_config

@CrewBase
class SixThinkingHats:
    def __init__(self):
        # Load configurations for agents and tasks
        self.agents_config = load_agents_config('path/to/agents_config.yaml')
        self.tasks_config = load_tasks_config('path/to/tasks_config.yaml')

    def cto_advisor(self) -> CrewAgent:
        """Provide technical insights and evaluate the feasibility of startup ideas."""
        agent_config = self.agents_config.get('cto_advisor')
        return CrewAgent(name='CTO Advisor', config=agent_config)

    def cmo_advisor(self) -> CrewAgent:
        """Assess market potential and marketing strategies."""
        agent_config = self.agents_config.get('cmo_advisor')
        return CrewAgent(name='CMO Advisor', config=agent_config)

    def cfo_advisor(self) -> CrewAgent:
        """Evaluate financial viability and potential ROI."""
        agent_config = self.agents_config.get('cfo_advisor')
        return CrewAgent(name='CFO Advisor', config=agent_config)

    def cpo_advisor(self) -> CrewAgent:
        """Analyze product development strategy."""
        agent_config = self.agents_config.get('cpo_advisor')
        return CrewAgent(name='CPO Advisor', config=agent_config)

    def vc_advisor(self) -> CrewAgent:
        """Provide insights from a venture capital perspective."""
        agent_config = self.agents_config.get('vc_advisor')
        return CrewAgent(name='VC Advisor', config=agent_config)

    def six_thinking_hats_facilitator(self) -> CrewAgent:
        """Guide the evaluation process using the Six Thinking Hats methodology."""
        agent_config = self.agents_config.get('six_thinking_hats_facilitator')
        return CrewAgent(name='Six Thinking Hats Facilitator', config=agent_config)

    def initiate_session(self) -> CrewTask:
        """Initiate the Six Thinking Hats session."""
        task_config = self.tasks_config.get('initiate_session')
        return CrewTask(name='Initiate Session', config=task_config)

    def white_hat_session(self) -> CrewTask:
        """Conduct the White Hat session to gather factual information."""
        task_config = self.tasks_config.get('white_hat_session')
        return CrewTask(name='White Hat Session', config=task_config)

    def red_hat_session(self) -> CrewTask:
        """Conduct the Red Hat session to explore emotional responses."""
        task_config = self.tasks_config.get('red_hat_session')
        return CrewTask(name='Red Hat Session', config=task_config)

    def black_hat_session(self) -> CrewTask:
        """Conduct the Black Hat session to identify risks and challenges."""
        task_config = self.tasks_config.get('black_hat_session')
        return CrewTask(name='Black Hat Session', config=task_config)

    def yellow_hat_session(self) -> CrewTask:
        """Conduct the Yellow Hat session to highlight benefits and opportunities."""
        task_config = self.tasks_config.get('yellow_hat_session')
        return CrewTask(name='Yellow Hat Session', config=task_config)

    def green_hat_session(self) -> CrewTask:
        """Conduct the Green Hat session to brainstorm creative solutions."""
        task_config = self.tasks_config.get('green_hat_session')
        return CrewTask(name='Green Hat Session', config=task_config)

    def blue_hat_session(self) -> CrewTask:
        """Conduct the Blue Hat session to summarize findings and outline next steps."""
        task_config = self.tasks_config.get('blue_hat_session')
        return CrewTask(name='Blue Hat Session', config=task_config)

    def crew(self) -> None:
        """Assemble the crew with agents and tasks."""
        agents = [
            self.cto_advisor(),
            self.cmo_advisor(),
            self.cfo_advisor(),
            self.cpo_advisor(),
            self.vc_advisor(),
            self.six_thinking_hats_facilitator()
        ]

        tasks = [
            self.initiate_session(),
            self.white_hat_session(),
            self.red_hat_session(),
            self.black_hat_session(),
            self.yellow_hat_session(),
            self.green_hat_session(),
            self.blue_hat_session()
        ]

        # Orchestrate the agents and tasks
        for agent in agents:
            print(f"Agent {agent.name} is ready.")

        for task in tasks:
            print(f"Task {task.name} is scheduled.")