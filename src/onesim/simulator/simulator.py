import json
from typing import Dict, Any
from .agent_factory import AgentFactory
from .sim_env import BasicSimEnv
from onesim.events import EventBus,Event,Scheduler
from .rel_manager import SocialNetworkManager


class Simulator:
    def __init__(self, config_file: str):
        self.config = self.load_config(config_file)
        self.agent_factory = AgentFactory(self.config)
        self.environment = BasicSimEnv(self.config)
        self.event_bus = EventBus()
        self.social_network_manager = SocialNetworkManager()
 
        # 模拟器状态
        self.agents = []
        self.running = False

    def load_config(self, config_file: str) -> Dict[str, Any]:
        """加载配置文件并返回配置字典"""
        with open(config_file, 'r') as f:
            return json.load(f)

    def initialize(self):
        """初始化环境和代理"""
        # 使用 AgentFactory 创建代理
        self.agents = self.agent_factory.create_agents()
        self.social_network_manager.register_agents(self.agents)
        # 将所有代理加入环境
        self.environment.add_agents(self.agents)
        self.running = True

    def run(self, steps: int = 10):
        """运行模拟器，执行一定步数的交互"""
        if not self.running:
            raise RuntimeError("Simulator has not been initialized. Call initialize() first.")

        for step in range(steps):
            print(f"Step {step + 1}/{steps}")
            self.environment.step(self.event_manager)

    def stop(self):
        """停止模拟器"""
        self.running = False
        print("Simulation stopped.")
