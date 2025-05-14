"""
配置管理路由
处理系统配置、模型配置和环境设置
"""
import os
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator
from typing import Dict, Any, List, Optional
from loguru import logger
import onesim
from onesim.config import get_component_registry
from backend.models.config import ConfigOptions, ProfileCountRequest, ProfileCountResponse, SaveConfigRequest, SaveConfigResponse
from backend.models.simulation import AgentInfo


router = APIRouter(
    prefix="/config",
    tags=["config"],
)



# 存储用户配置
USER_CONFIGS = {}

# 模型配置路径
MODEL_CONFIG_PATH = os.path.join(os.getcwd(), 
                                "config", "model_config.json")

# 默认配置路径
DEFAULT_CONFIG_PATH = os.path.join(os.getcwd(), 
                                  "config", "config.json")

def load_default_config():
    """加载默认配置"""
    try:
        if os.path.exists(DEFAULT_CONFIG_PATH):
            with open(DEFAULT_CONFIG_PATH, 'r') as f:
                default_config = json.load(f)
                logger.info(f"已加载默认配置: {DEFAULT_CONFIG_PATH}")
                return default_config
        else:
            logger.warning(f"默认配置文件不存在: {DEFAULT_CONFIG_PATH}")
            return {}
    except Exception as e:
        logger.error(f"加载默认配置出错: {str(e)}")
        return {}

# 加载默认配置
DEFAULT_CONFIG = load_default_config()

def generate_default_portrait(agent_types):
    """为代理类型生成默认的头像值（1-5之间）"""
    portrait = {}
    # 根据代理类型索引使用1-5之间的不同值
    for i, agent_type in enumerate(agent_types):
        # 根据代理类型索引循环使用1-5的值
        portrait[agent_type] = (i % 5) + 1
    return portrait

@router.get("/options", response_model=ConfigOptions)
def get_config_options(env_name: str):
    """获取场景的配置选项"""
    if not env_name:
        raise HTTPException(status_code=400, detail="需要env_name查询参数")
    
    # 检查场景是否存在
    base_path = os.path.abspath(os.getcwd())
    scenes_root = os.path.join(base_path, "envs")
    scene_path = os.path.join(scenes_root, env_name)
    
    if not os.path.exists(scene_path):
        raise HTTPException(status_code=404, detail=f"环境 '{env_name}' 不存在")
    
    # 动态获取可用的规划策略
    try:
        import importlib
        import inspect
        from onesim.planning import __all__ as planning_all
        from onesim.planning.base import PlanningBase
        
        # 过滤掉基类，只获取具体的规划类
        planning_options = ["None"]  # 首先添加None选项
        for class_name in planning_all:
            module = importlib.import_module("onesim.planning")
            cls = getattr(module, class_name)
            if inspect.isclass(cls) and cls is not PlanningBase and issubclass(cls, PlanningBase):
                planning_options.append(class_name)
    except Exception as e:
        logger.error(f"加载规划选项出错: {e}")
        # 回退到默认选项
        planning_options = ["None", "BDIPlanning", "COTPlanning", "TOMPlanning"]
    
    # 动态获取可用的记忆策略
    try:
        import importlib
        import inspect
        from onesim.memory.strategy import __all__ as strategy_all
        from onesim.memory.strategy.strategy import MemoryStrategy
        
        # 过滤掉基类，只获取具体的策略类
        memory_options = ["None"]  # 首先添加None选项
        for class_name in strategy_all:
            try:
                module = importlib.import_module("onesim.memory.strategy")
                cls = getattr(module, class_name)
                # 检查是否为类、不是基类本身、并且是MemoryStrategy的子类
                if inspect.isclass(cls) and cls is not MemoryStrategy and issubclass(cls, MemoryStrategy):
                    memory_options.append(class_name)
            except (ImportError, AttributeError) as e:
                logger.error(f"检查内存策略类 {class_name} 出错: {e}")
    except Exception as e:
        logger.error(f"加载内存策略选项出错: {e}")
        # 回退到默认选项
        memory_options = ["None", "ListStrategy", "ShortLongStrategy"]
    
    # 从scene_info.json获取可用的代理类型
    scene_info_path = os.path.join(scene_path, "scene_info.json")
    agent_types = {}
    portrait = {}
    
    if os.path.exists(scene_info_path):
        try:
            with open(scene_info_path, 'r', encoding='utf-8') as f:
                scene_info = json.load(f)
                
            if "agent_types" in scene_info:
                agent_types = scene_info["agent_types"]
                
            # 从scene_info.json获取头像（如果存在）
            if "portrait" in scene_info:
                portrait = scene_info["portrait"]
            else:
                # 如果未找到，生成默认头像
                portrait = generate_default_portrait(agent_types.keys())
        except Exception as e:
            logger.error(f"加载scene_info.json出错: {e}")
    
    # 为每个代理类型创建配置文件选项
    profile_options = {}
    for agent_type in agent_types:
        # 检查默认配置文件路径
        profile_path = os.path.join(scene_path, "profile", "data", f"{agent_type}.json")
        count = 0
        
        if os.path.exists(profile_path):
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    profiles = json.load(f)
                    count = len(profiles)
            except Exception as e:
                logger.error(f"加载{agent_type}的配置文件数据出错: {e}")
        
        # 获取此代理类型的头像值
        portrait_value = portrait.get(agent_type, 1)  # 若未指定则默认为1
        
        profile_options[agent_type] = {
            "count": 1,  # 默认为1或如果配置文件较少则更少
            "max_count": count,  # 添加max_count字段以指示最大可用配置文件
            "portrait": portrait_value  # 添加来自scene_info.json的头像值或默认值
        }
    
    # 创建环境选项
    environment_options = {
        "name": env_name,  # 固定，不可更改
        "mode": "round",  # 默认模式
        "modes": ["round", "tick"],  # 可用模式
        "max_rounds": 3  # 默认最大回合数
    }
    
    # 创建代理选项
    agent_options = {
        "profile": profile_options,
        "planning": planning_options,
        "memory": memory_options
    }
    
    # 从model_config.json读取模型选项
    model_options = {
        "chat": [],
        "embedding": []
    }
    chat_model_names = set()
    embedding_model_names = set()
    if os.path.exists(MODEL_CONFIG_PATH):
        try:
            with open(MODEL_CONFIG_PATH, 'r', encoding='utf-8') as f:
                model_config = json.load(f)
                
            # 提取聊天模型选项
            if "chat" in model_config:
                model_options["chat"] = list({model.get("model_name", "") for model in model_config["chat"]})
            
            # 提取嵌入模型选项
            if "embedding" in model_config:
                model_options["embedding"] = [model.get("model_name", "") for model in model_config["embedding"]]
        except Exception as e:
            logger.error(f"加载model_config.json出错: {e}")
    
    config_options = {
        "environment": environment_options,
        "agent": agent_options,
        "model": model_options
    }
    
    return config_options

@router.post("/save", response_model=SaveConfigResponse)
async def save_config(data: SaveConfigRequest):
    """
    保存场景的配置并初始化环境和代理。
    从base config.json读取，应用来自请求的更改，并将合并的配置保存到内存中。
    然后准备好环境和代理的初始化参数，以便后续进行模拟。
    """
    env_name = data.env_name
    user_config = data.config
    
    # 检查场景是否存在
    scenes_root = os.path.join(os.getcwd(), "envs")
    scene_path = os.path.join(scenes_root, env_name)
    
    if not os.path.exists(scene_path):
        raise HTTPException(status_code=404, detail=f"环境 '{env_name}' 不存在")
    
    # 从默认配置开始
    merged_config = json.loads(json.dumps(DEFAULT_CONFIG))
    
    # 如果用户配置中存在则更新模拟器设置
    # if "simulator" in user_config:
    #     if "simulator" not in merged_config:
    #         merged_config["simulator"] = {}
        
    if "environment" in user_config:
        if "environment" not in merged_config["simulator"]:
            merged_config["simulator"]["environment"] = {}
        
        # 更新环境设置，保留名称
        env_config = user_config["environment"]
        merged_config["simulator"]["environment"].update(env_config)
        # 确保环境名称设置正确
        merged_config["simulator"]["environment"]["name"] = env_name
    
    # 如果在用户配置中存在则更新代理设置
    if "agent" in user_config:
        if "agent" not in merged_config:
            merged_config["agent"] = {}
        
        # 更新配置文件设置
        if "profile" in user_config["agent"]:
            if "profile" not in merged_config["agent"]:
                merged_config["agent"]["profile"] = {}
            
            # 为用户配置中的每个代理类型更新配置文件计数
            for agent_type, profile_data in user_config["agent"]["profile"].items():
                if agent_type not in merged_config["agent"]["profile"]:
                    merged_config["agent"]["profile"][agent_type] = {}
                
                # 更新计数但保留其他设置
                if "count" in profile_data:
                    merged_config["agent"]["profile"][agent_type]["count"] = profile_data["count"]
        
        # 更新规划设置
        if "planning" in user_config["agent"]:
            if user_config["agent"]["planning"]=="None":
                merged_config["agent"]["planning"] = None
            else:
                merged_config["agent"]["planning"] = user_config["agent"]["planning"]
        
        # 更新内存设置
        if "memory" in user_config["agent"]:
            if "memory" not in merged_config["agent"]:
                merged_config["agent"]["memory"] = {}
            if user_config["agent"]["memory"] == "None":
                merged_config["agent"]["memory"] = None
            else:
                merged_config["agent"]["memory"]["strategy"] = user_config["agent"]["memory"]
    
    # 如果在用户配置中存在则更新模型设置
    if "model" in user_config:
        if "model" not in merged_config:
            merged_config["model"] = {}
        
        # 更新聊天模型选择（必需）
        if "chat" in user_config["model"]:
            chat_model = user_config["model"]["chat"]
            if not chat_model:
                raise HTTPException(status_code=400, detail="需要聊天模型选择")
            merged_config["model"]["chat"] = chat_model
        
        # 更新嵌入模型选择（可选）
        if "embedding" in user_config["model"]:
            merged_config["model"]["embedding"] = user_config["model"]["embedding"]
    
    try:
        # 仅在内存中存储用于当前模拟
        USER_CONFIGS[env_name] = merged_config
        
        return SaveConfigResponse(
            success=True,
            message=f"已保存环境'{env_name}'的配置"
        )
    except Exception as e:
        logger.error(f"保存配置出错: {e}")
        raise HTTPException(status_code=500, detail=f"保存配置失败: {str(e)}")

# async def initialize_simulation(env_name: str, model_name: str = None) -> dict:
#     """
#     初始化模拟环境和相关组件。
#     参考main.py和onesim.__init__.py的初始化流程，整合到config模块中。
    
#     Args:
#         env_name: 环境名称
#         model_name: 可选的模型名称，如果未提供则从配置获取
        
#     Returns:
#         初始化状态的字典
#     """
#     # 检查环境是否存在
#     scenes_root = os.path.join(os.getcwd(), "envs")
#     env_path = os.path.join(scenes_root, env_name)
    
#     if not os.path.exists(env_path):
#         raise HTTPException(status_code=404, detail=f"环境 '{env_name}' 不存在")
    
#     try:
#         # 获取配置 - 使用内存中的配置或默认配置
#         if env_name in USER_CONFIGS:
#             logger.info(f"使用内存中的环境配置: {env_name}")
#             config_data = USER_CONFIGS[env_name]
#         else:
#             logger.info(f"使用默认环境配置: {env_name}")
#             config_data = DEFAULT_CONFIG
        
#         # 确保配置包含必要的环境信息
#         config_data['env_name'] = env_name
#         config_data['env_path'] = env_path
        
#         # 如果未提供模型名称，尝试从配置获取
#         if not model_name and "model" in config_data:
#             if "chat" in config_data["model"]:
#                 model_name = config_data["model"]["chat"]
#                 logger.info(f"从配置获取模型: {model_name}")
        
#         # 确保基础组件已初始化
#         components_to_init = ["model"]
        
#         # 添加其他可能需要的组件
#         if config_data.get("monitor", {}).get("enabled", False):
#             components_to_init.append("monitor")
        
#         if config_data.get("database", {}).get("enabled", False):
#             components_to_init.append("database")
            
#         if config_data.get("distribution", {}).get("enabled", False):
#             components_to_init.append("distribution")
            
#         # 初始化必要的OneSim组件
#         from onesim import init, COMPONENT_MODEL, COMPONENT_MONITOR, COMPONENT_DATABASE, COMPONENT_DISTRIBUTION
        
#         # 转换组件名称为常量
#         component_map = {
#             "model": COMPONENT_MODEL,
#             "monitor": COMPONENT_MONITOR,
#             "database": COMPONENT_DATABASE,
#             "distribution": COMPONENT_DISTRIBUTION
#         }
        
#         # 准备初始化配置
#         init_components = [component_map[c] for c in components_to_init if c in component_map]
        
#         # 如果有模型名称，在模型配置字典中添加
       
        
#         # 初始化组件并获取配置
#         config = await onesim.init(
#             components=init_components,
#             config_dict=config_data,  # 直接传入配置字典
#             model_config_path=MODEL_CONFIG_PATH,
#             #model_config_dict=model_config_dict
#         )
        
        
#         # 加载模型
#         model_config_name = await load_model_for_simulation(model_name)
        
#         # 加载SimEnv类定义
#         import sys
#         import importlib.util
        
#         if scenes_root not in sys.path:
#             sys.path.append(scenes_root)
            
#         module_name = f"{env_name}.code.SimEnv"
#         try:
#             sim_env_module = importlib.import_module(module_name)
#             if not hasattr(sim_env_module, "SimEnv"):
#                 raise AttributeError(f"模块 {module_name} 不包含名为 'SimEnv' 的类")
            
#             SimEnv = getattr(sim_env_module, "SimEnv")
#             logger.info(f"已加载环境类: {SimEnv.__name__}")
#         except Exception as e:
#             logger.error(f"加载环境类错误: {e}")
#             raise Exception(f"无法加载环境类: {str(e)}")
        
#         # 初始化代理
#         from onesim.simulator import AgentFactory
#         from onesim.utils.work_graph import WorkGraph
#         from onesim.events import get_event_bus
        
#         # 创建代理工厂
#         agent_factory = AgentFactory(
#             simulator_config=config.simulator_config,
#             model_config_name=model_config_name,
#             env_path=env_path,
#             agent_config=config.agent_config
#         )
        
#         # 创建代理
#         logger.info("创建代理实例")
#         agents = await agent_factory.create_agents()
        
#         # 构建工作流图
#         logger.info("构建工作流图")
#         actions_path = os.path.join(env_path, "actions.json")
#         events_path = os.path.join(env_path, "events.json")
        
#         # 解析操作和事件定义
#         from onesim.config import parse_json
#         actions = parse_json(actions_path)
#         events = parse_json(events_path)
        
#         # 创建工作流图并获取起始/结束节点
#         work_graph = WorkGraph(actions, events)
#         start_agent_types = work_graph.get_start_agent_types()
#         end_agent_types = work_graph.get_end_agent_types()
        
#         start_agent_ids = agent_factory.get_agent_profile_ids(start_agent_types)
#         end_agent_ids = agent_factory.get_agent_profile_ids(end_agent_types)
        
#         # 添加环境关系
#         for agent_type, ids in end_agent_ids.items():
#             for agent_id in ids:
#                 agent_factory.add_env_relationship(agent_id)
        
#         # 获取事件总线
#         event_bus = get_event_bus()
        
#         # 为分布式场景做检查
#         is_distributed = False
#         registry = get_component_registry()
#         if registry.is_initialized(COMPONENT_DISTRIBUTION):
#             node = registry.get_instance(COMPONENT_DISTRIBUTION)
#             is_distributed = True
#             logger.info(f"检测到分布式模式: {node.role}")
        
#         # 初始化数据跟踪ID（如果启用数据库）
#         trail_id = None
#         if registry.is_initialized(COMPONENT_DATABASE):
#             try:
#                 from onesim.data import ScenarioManager, TrailManager
#                 import time
#                 import uuid
                
#                 # 创建或获取情景ID
#                 scenario_mgr = ScenarioManager()
#                 env_config = config.simulator_config.environment
                
#                 # 尝试找到现有场景
#                 scenarios = await scenario_mgr.get_scenario_by_name(name=env_name, exact_match=True)
#                 scenario_id = None
                
#                 if scenarios and len(scenarios) > 0:
#                     for scenario in scenarios:
#                         if scenario['name'] == env_name:
#                             scenario_id = scenario['scenario_id']
#                             logger.info(f"使用现有场景ID {scenario_id} for {env_name}")
#                             break
                
#                 if scenario_id is None:
#                     # 创建新场景
#                     scenario_id = await scenario_mgr.create_scenario(
#                         name=env_name,
#                         folder_path=f"envs/{env_name}",
#                         description=env_config.get('description', f"Simulation scenario for {env_name}"),
#                         tags={
#                             "domain": env_config.get('domain', ''), 
#                             "version": env_config.get('version', '1.0')
#                         }
#                     )
#                     logger.info(f"创建新场景ID {scenario_id} for {env_name}")
                
#                 # 创建trail
#                 trail_mgr = TrailManager()
#                 trail_name = f"{env_name}_run_{time.strftime('%Y%m%d_%H%M%S')}"
#                 trail_id = await trail_mgr.create_trail(
#                     scenario_id=scenario_id,
#                     name=trail_name,
#                     description=f"Simulation run for {env_name}",
#                     config=config.simulator_config.to_dict()
#                 )
#                 logger.info(f"创建数据跟踪ID {trail_id} 用于数据存储")
#             except Exception as e:
#                 logger.error(f"初始化数据存储错误: {e}, 继续而不存储数据")
        
#         # 创建simulation_id
#         import uuid
#         simulation_id = str(uuid.uuid4())
        
#         # 存储这些对象以便simulation模块使用
#         from app.routers.simulation import SIMULATION_REGISTRY
        
#         # 在全局注册表中存储代理工厂和代理 - 与simulation.py中的格式保持一致
#         SIMULATION_REGISTRY[env_name] = {
#             "agent_factory": agent_factory,
#             "agents": agents,
#             "initialized": True,
#             "running": False,
#             "config": config_data,
#             "event_bus": event_bus,
#             "work_graph": work_graph,
#             "start_agent_ids": start_agent_ids,
#             "end_agent_ids": end_agent_ids,
#             "SimEnv": SimEnv,
#             "env_path": env_path,
#             "end_events": work_graph.get_end_events(),
#             "simulation_id": simulation_id,
#             "trail_id": trail_id,
#             # 添加状态信息
#             "status": "initialized",
#             "metrics": {},
#             "step": 0,
#             "start_time": None,
#             "pause_time": None,
#             "events": []
#         }
        
#         created_agents=[]
#         for agent_type in agents:
#             for agent_id, agent in agents[agent_type].items():
#                 created_agents.append(AgentInfo(
#                     id=agent_id,
#                     type=agent_type,
#                     profile=agent.get_profile(include_private=True)
#                 ))
#         # 返回初始化状态
#         result = {
#             "env_name": env_name,
#             "config_applied": True,
#             "agents": created_agents,
#             "agent_count": sum(len(agents[agent_type]) for agent_type in agents),
#             "is_distributed": is_distributed,
#             "trail_id": trail_id,
#             "components_initialized": {
#                 component: registry.is_initialized(component_map[component])
#                 for component in components_to_init if component in component_map
#             },
#             "workflow": {
#                 "start_agent_types": start_agent_types,
#                 "end_agent_types": end_agent_types,
#                 "start_agent_ids": start_agent_ids,
#                 "end_agent_ids": end_agent_ids,
#                 "end_events": work_graph.get_end_events()
#             },
#             "ready_for_simulation": True
#         }
        
#         logger.info(f"环境 '{env_name}' 初始化成功")
#         return result
        
#     except Exception as e:
#         logger.error(f"初始化模拟环境出错: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"初始化模拟环境失败: {str(e)}")

# async def load_model_for_simulation(model_name: str = None) -> str:
#     """
#     加载模型并返回模型配置名称。这个函数抽离出来是为了使config.py和simulation.py可以共享相同的模型加载逻辑。
    
#     Args:
#         model_name: 要加载的模型名称
        
#     Returns:
#         model_config_name: 加载好的模型配置名称
#     """
#     from app.utils.model_management import load_model_if_needed
    
#     # 如果没有指定模型名称，使用负载均衡器
#     if not model_name:
#         return "chat_load_balancer"
    
#     # 尝试加载模型
#     try:
#         model = await load_model_if_needed(model_name=model_name, category="chat")
#         model_config_name = model.config_name
#         logger.info(f"为模拟加载模型: {model_config_name}")
#         return model_config_name
#     except Exception as e:
#         logger.error(f"加载模型错误: {e}，使用默认配置")
#         return "chat_load_balancer"


# @router.post("/initialize_simulation")
# async def initialize_simulation_endpoint(env_name: str, model_name: Optional[str] = None):
#     """初始化模拟环境和相关组件的端点"""
#     try:
#         result = await initialize_simulation(env_name, model_name)
#         return result
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"初始化模拟出错: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"初始化模拟失败: {str(e)}")

@router.get("/get", response_model=Dict[str, Any])
def get_config(env_name: str):
    """获取特定环境的配置"""
    # 首先检查内存中的配置
    if env_name in USER_CONFIGS:
        return USER_CONFIGS[env_name]
    
    # 如果不在内存中，尝试加载默认配置
    scenes_root = os.path.join(os.getcwd(), "envs")
    scene_path = os.path.join(scenes_root, env_name)
    config_path = os.path.join(scene_path, "config", "simulator_config.json")
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except Exception as e:
            logger.error(f"加载配置出错: {e}")
    
    # 如果没有特定配置，返回默认配置
    return DEFAULT_CONFIG

# 获取模型配置
@router.get("/models", response_model=Dict[str, List[str]])
def get_models():
    """获取所有可用模型"""
    model_options = {
        "chat": [],
        "embedding": []
    }
    
    if os.path.exists(MODEL_CONFIG_PATH):
        try:
            with open(MODEL_CONFIG_PATH, 'r', encoding='utf-8') as f:
                model_config = json.load(f)
                
            # 提取聊天模型选项
            if "chat" in model_config:
                model_options["chat"] = [model.get("model_name", "") for model in model_config["chat"]]
            
            # 提取嵌入模型选项
            if "embedding" in model_config:
                model_options["embedding"] = [model.get("model_name", "") for model in model_config["embedding"]]
        except Exception as e:
            logger.error(f"加载model_config.json出错: {e}")
    
    return model_options 