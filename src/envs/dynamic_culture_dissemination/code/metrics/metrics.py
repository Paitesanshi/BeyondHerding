# -*- coding: utf-8 -*-
"""
文化传播模型的监控指标计算模块 - 适配新的Schema结构
"""

from typing import Dict, Any, List, Optional, Union, Callable
import math
from collections import Counter, defaultdict
from loguru import logger
from onesim.monitor.utils import (
    safe_get, safe_number, safe_list, safe_sum, 
    safe_avg, safe_max, safe_min, safe_count, log_metric_error
)


def calculate_cultural_distribution(data: Dict[str, Any]) -> Any:
    """
    计算指标: cultural_distribution
    描述: 追踪随时间变化的各种文化特征的分布，可视化文化传播模式
    可视化类型: line
    更新频率: 5 秒
    
    根据新的Schema，分别计算5个文化维度的分布
    """
    try:
        # 验证输入数据
        if not data or not isinstance(data, dict):
            log_metric_error("cultural_distribution", ValueError("无效的数据输入"), {"data": data})
            return {}

        # 文化维度列表
        dimensions = [
            "music_preference",
            "culinary_preference", 
            "fashion_style", 
            "political_orientation", 
            "leisure_activity"
        ]
        
        results = {}
        
        # 对每个文化维度计算分布
        for dimension in dimensions:
            # 获取所有agent的该维度偏好
            dimension_values = safe_list(safe_get(data, dimension, []))
            
            # 过滤掉None值
            valid_values = [value for value in dimension_values if value]
            
            # 如果没有有效数据，跳过该维度
            if not valid_values:
                continue
            
            # 计算每种值的数量
            value_counts = Counter(valid_values)
            
            # 保存该维度的分布数据
            results[dimension] = dict(value_counts)
        
        return results
    
    except Exception as e:
        log_metric_error("cultural_distribution", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return {}


def calculate_cultural_homogeneity(data: Dict[str, Any]) -> Any:
    """
    计算指标: cultural_homogeneity
    描述: 测量整个网络的文化同质性，范围从0(完全多样化)到1(完全同质化)
    可视化类型: line
    更新频率: 5 秒
    
    对5个维度分别计算同质性，然后取平均值
    """
    try:
        # 验证输入数据
        if not data or not isinstance(data, dict):
            log_metric_error("cultural_homogeneity", ValueError("无效的数据输入"), {"data": data})
            return 0
        
        # 文化维度列表
        dimensions = [
            "music_preference",
            "culinary_preference", 
            "fashion_style", 
            "political_orientation", 
            "leisure_activity"
        ]
        
        # 计算每个维度的同质性指数
        homogeneity_indices = []
        
        for dimension in dimensions:
            # 获取所有agent的该维度偏好
            dimension_values = safe_list(safe_get(data, dimension, []))
            
            # 过滤掉None值和空值
            valid_values = [value for value in dimension_values if value]
            
            # 如果没有有效数据，跳过该维度
            if not valid_values:
                continue
            
            # 计算每种值的数量
            value_counts = Counter(valid_values)
            
            # 计算最常见值的使用比例
            total_agents = len(valid_values)
            most_common_value, most_common_count = value_counts.most_common(1)[0]
            dimension_homogeneity = most_common_count / total_agents
            
            homogeneity_indices.append(dimension_homogeneity)
        
        # 如果没有收集到任何维度的同质性指数，返回0
        if not homogeneity_indices:
            return 0
        
        # 返回所有维度同质性指数的平均值
        return sum(homogeneity_indices) / len(homogeneity_indices)
    
    except Exception as e:
        log_metric_error("cultural_homogeneity", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return 0


def calculate_adoption_rate(data: Dict[str, Any]) -> Any:
    """
    计算指标: adoption_rate
    描述: 每轮推荐导致文化特征采纳的百分比
    可视化类型: line
    更新频率: 5 秒
    
    返回一个0到1之间的数值，表示当前轮次的采纳率
    """
    try:
        # 验证输入数据
        if not data or not isinstance(data, dict):
            log_metric_error("adoption_rate", ValueError("无效的数据输入"), {"data": data})
            return 0
        
        # 获取当前轮次
        current_round = safe_number(safe_get(data, "round_number", 0))
        
        # 获取所有agent的采纳历史
        adoption_histories = safe_list(safe_get(data, "adoption_history", []))
        
        # 如果没有采纳历史，返回0
        if not adoption_histories:
            return 0
        
        # 统计当前轮次的采纳情况
        total_recommendations = 0
        successful_adoptions = 0
        
        for agent_history in adoption_histories:
            if not isinstance(agent_history, list):
                continue
                
            for entry in agent_history:
                if not isinstance(entry, dict):
                    continue
                    
                entry_round = safe_number(entry.get("round", -1))
                
                # 只统计当前轮次的数据
                if entry_round == current_round:
                    total_recommendations += 1
                    if entry.get("adopted", False):
                        successful_adoptions += 1
        
        # 计算采纳率
        if total_recommendations == 0:
            return 0
            
        adoption_rate = successful_adoptions / total_recommendations
        return adoption_rate
    
    except Exception as e:
        log_metric_error("adoption_rate", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return 0


def calculate_cultural_regions(data: Dict[str, Any]) -> Any:
    """
    计算指标: cultural_regions
    描述: 共享相同文化特征的相连agent群体的数量
    可视化类型: line
    更新频率: 5 秒
    
    返回一个字典，键为维度名称，值为该维度上的文化区域数量
    """
    try:
        # 验证输入数据
        if not data or not isinstance(data, dict):
            log_metric_error("cultural_regions", ValueError("无效的数据输入"), {"data": data})
            return {}
        
        # 文化维度列表
        dimensions = [
            "music_preference",
            "culinary_preference", 
            "fashion_style", 
            "political_orientation", 
            "leisure_activity"
        ]
        
        # 获取关系数据
        relationships = safe_list(safe_get(data, "agent_relationships", []))
        
        # 构建邻接表
        adjacency = defaultdict(list)
        for rel in relationships:
            if not isinstance(rel, dict):
                continue
                
            source = rel.get("source_id")
            target = rel.get("target_id")
            
            if source and target:
                adjacency[source].append(target)
                adjacency[target].append(source)  # 假设关系是双向的
        
        results = {}
        
        # 对每个维度计算文化区域
        for dimension in dimensions:
            # 获取所有agent的该维度偏好
            agent_values = {}
            dimension_values = safe_get(data, dimension, {})
            
            if not isinstance(dimension_values, dict):
                continue
            
            for agent_id, value in dimension_values.items():
                if value:  # 只考虑有效值
                    agent_values[agent_id] = value
            
            # 使用BFS搜索标识文化区域
            visited = set()
            region_count = 0
            
            for agent_id in agent_values:
                if agent_id in visited:
                    continue
                    
                # 发现新区域
                value = agent_values[agent_id]
                region_count += 1
                
                # BFS标记该区域的所有agent
                queue = [agent_id]
                visited.add(agent_id)
                
                while queue:
                    current = queue.pop(0)
                    
                    for neighbor in adjacency[current]:
                        if neighbor not in visited and agent_values.get(neighbor) == value:
                            visited.add(neighbor)
                            queue.append(neighbor)
            
            results[dimension] = region_count
        
        # 还可以计算总体文化区域的平均数
        if results:
            results["average"] = sum(results.values()) / len(results)
        
        return results
    
    except Exception as e:
        log_metric_error("cultural_regions", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return {}


def calculate_influence_distribution(data: Dict[str, Any]) -> Any:
    """
    计算指标: influence_distribution
    描述: 可视化哪些agent成功地影响其他人采纳了他们的文化特征
    可视化类型: bar
    更新频率: 5 秒
    
    返回一个字典，其中键是agent ID，值是该agent成功影响他人的次数
    """
    try:
        # 验证输入数据
        if not data or not isinstance(data, dict):
            log_metric_error("influence_distribution", ValueError("无效的数据输入"), {"data": data})
            return {}
        
        # 获取所有agent的采纳历史
        adoption_histories = safe_list(safe_get(data, "adoption_history", []))
        
        # 统计每个agent的影响力
        influence_counts = Counter()
        
        for agent_history in adoption_histories:
            if not isinstance(agent_history, list):
                continue
                
            for entry in agent_history:
                if not isinstance(entry, dict):
                    continue
                    
                # 只统计成功采纳的情况
                if entry.get("adopted", False):
                    recommender = entry.get("recommender")
                    if recommender:
                        influence_counts[recommender] += 1
        
        # 返回影响力分布
        return dict(influence_counts)
    
    except Exception as e:
        log_metric_error("influence_distribution", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return {}


def calculate_dimension_influence(data: Dict[str, Any]) -> Any:
    """
    计算指标: dimension_influence
    描述: 分析哪些文化维度更容易被传播和采纳
    可视化类型: bar
    更新频率: 5 秒
    
    返回一个字典，其中键是文化维度，值是该维度被采纳的次数
    """
    try:
        # 验证输入数据
        if not data or not isinstance(data, dict):
            log_metric_error("dimension_influence", ValueError("无效的数据输入"), {"data": data})
            return {}
        
        # 文化维度列表
        dimensions = [
            "music_preference",
            "culinary_preference", 
            "fashion_style", 
            "political_orientation", 
            "leisure_activity"
        ]
        
        # 获取所有agent的采纳历史
        adoption_histories = safe_list(safe_get(data, "adoption_history", []))
        
        # 统计每个维度的采纳次数
        dimension_counts = Counter()
        
        for agent_history in adoption_histories:
            if not isinstance(agent_history, list):
                continue
                
            for entry in agent_history:
                if not isinstance(entry, dict):
                    continue
                    
                # 只统计成功采纳的情况
                if entry.get("adopted", False):
                    dimension = entry.get("dimension")
                    if dimension in dimensions:
                        dimension_counts[dimension] += 1
        
        # 返回维度影响力分布
        return dict(dimension_counts)
    
    except Exception as e:
        log_metric_error("dimension_influence", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return {}


# 指标函数字典，用于查找
METRIC_FUNCTIONS = {
    'calculate_cultural_distribution': calculate_cultural_distribution,
    'calculate_cultural_homogeneity': calculate_cultural_homogeneity,
    'calculate_adoption_rate': calculate_adoption_rate,
    'calculate_cultural_regions': calculate_cultural_regions,
    'calculate_influence_distribution': calculate_influence_distribution,
    'calculate_dimension_influence': calculate_dimension_influence,
}


def get_metric_function(function_name: str) -> Optional[Callable]:
    """
    根据函数名获取对应的指标计算函数
    
    Args:
        function_name: 函数名
        
    Returns:
        指标计算函数或None
    """
    return METRIC_FUNCTIONS.get(function_name)


def test_metric_function(function_name: str, test_data: Dict[str, Any]) -> Any:
    """
    测试指标计算函数
    
    Args:
        function_name: 函数名
        test_data: 测试数据
        
    Returns:
        指标计算结果
    """
    func = get_metric_function(function_name)
    if func is None:
        raise ValueError(f"找不到指标函数: {function_name}")
    
    try:
        result = func(test_data)
        print(f"指标 {function_name} 计算结果: {result}")
        return result
    except Exception as e:
        log_metric_error(function_name, e, {"test_data": test_data})
        raise


def generate_test_data() -> Dict[str, Any]:
    """
    生成用于测试的示例数据，适配新的Schema结构
    
    Returns:
        示例数据字典
    """
    return {
        # 环境变量
        "round_number": 5,
        
        # 文化维度数据 - 新的扁平化结构
        "music_preference": {
            "agent1": "Classical",
            "agent2": "Rock",
            "agent3": "Classical",
            "agent4": "Pop",
            "agent5": "Electronic",
            "agent6": "Folk/Traditional",
            "agent7": "Classical",
            "agent8": "Rock",
            "agent9": "Pop",
            "agent10": "Classical"
        },
        "culinary_preference": {
            "agent1": "Traditional Local",
            "agent2": "International",
            "agent3": "Traditional Local",
            "agent4": "Fast Food",
            "agent5": "Vegetarian/Organic",
            "agent6": "Gourmet/Experimental",
            "agent7": "Traditional Local",
            "agent8": "International",
            "agent9": "Fast Food",
            "agent10": "Traditional Local"
        },
        "fashion_style": {
            "agent1": "Formal/Professional",
            "agent2": "Casual/Relaxed",
            "agent3": "Formal/Professional",
            "agent4": "Athletic/Sporty",
            "agent5": "Artistic/Alternative",
            "agent6": "Luxury/High-fashion",
            "agent7": "Formal/Professional",
            "agent8": "Casual/Relaxed",
            "agent9": "Athletic/Sporty",
            "agent10": "Formal/Professional"
        },
        "political_orientation": {
            "agent1": "Conservative",
            "agent2": "Progressive",
            "agent3": "Conservative",
            "agent4": "Centrist/Moderate",
            "agent5": "Libertarian",
            "agent6": "Green/Environmental",
            "agent7": "Conservative",
            "agent8": "Progressive",
            "agent9": "Centrist/Moderate",
            "agent10": "Conservative"
        },
        "leisure_activity": {
            "agent1": "Sports/Fitness",
            "agent2": "Arts/Creative Pursuits",
            "agent3": "Sports/Fitness",
            "agent4": "Social Gatherings",
            "agent5": "Media/Entertainment",
            "agent6": "Nature/Outdoors",
            "agent7": "Sports/Fitness",
            "agent8": "Arts/Creative Pursuits",
            "agent9": "Social Gatherings",
            "agent10": "Sports/Fitness"
        },
        
        # 采纳历史数据
        "adoption_history": [
            [
                {"round": 5, "adopted": True, "recommender": "agent1", "dimension": "music_preference"},
                {"round": 3, "adopted": False, "recommender": "agent2", "dimension": "fashion_style"}
            ],
            [
                {"round": 5, "adopted": False, "recommender": "agent3", "dimension": "culinary_preference"}
            ],
            [
                {"round": 4, "adopted": True, "recommender": "agent1", "dimension": "political_orientation"},
                {"round": 5, "adopted": True, "recommender": "agent7", "dimension": "leisure_activity"}
            ]
        ],
        
        # 关系数据
        "agent_relationships": [
            {"source_id": "agent1", "target_id": "agent2"},
            {"source_id": "agent1", "target_id": "agent3"},
            {"source_id": "agent2", "target_id": "agent4"},
            {"source_id": "agent3", "target_id": "agent5"},
            {"source_id": "agent4", "target_id": "agent6"},
            {"source_id": "agent5", "target_id": "agent7"},
            {"source_id": "agent6", "target_id": "agent8"},
            {"source_id": "agent7", "target_id": "agent9"},
            {"source_id": "agent8", "target_id": "agent10"},
            {"source_id": "agent9", "target_id": "agent1"}
        ]
    }


def test_all_metrics(test_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    测试所有指标函数
    
    Args:
        test_data: 测试数据，如果为None则使用生成的示例数据
        
    Returns:
        测试结果字典，键为函数名，值为计算结果或错误信息
    """
    if test_data is None:
        test_data = generate_test_data()
        
    results = {}
    for func_name, func in METRIC_FUNCTIONS.items():
        try:
            result = func(test_data)
            results[func_name] = result
        except Exception as e:
            results[func_name] = f"ERROR: {str(e)}"
            log_metric_error(func_name, e, {"test_data": test_data})
    
    return results


# 如果直接运行此模块，执行所有指标的测试
if __name__ == "__main__":
    
    print("生成测试数据...")
    test_data = generate_test_data()
    
    print("测试所有指标函数...")
    results = test_all_metrics(test_data)
    
    print("\n测试结果:")
    for func_name, result in results.items():
        print(f"{func_name}: {result}")