# -*- coding: utf-8 -*-
"""
自动生成的监控指标计算模块
"""

from typing import Dict, Any, List, Optional, Union, Callable
import math
from loguru import logger
from onesim.monitor.utils import (
    safe_get, safe_number, safe_list, safe_sum, 
    safe_avg, safe_max, safe_min, safe_count, log_metric_error
)


from typing import Dict, Any
from onesim.monitor.utils import (
    safe_get, safe_number, safe_list, safe_sum, safe_avg, log_metric_error
)

def Average_Conformity_Tendency(data: Dict[str, Any]) -> Any:
    """
    计算指标: Average Conformity Tendency
    描述: Measures the average tendency of individual agents to conform within the system, providing insight into overall conformity behavior.
    可视化类型: bar
    更新频率: 5 秒
    
    Args:
        data: 包含所有变量的数据字典，注意agent变量是列表形式
        
    Returns:
        根据可视化类型返回不同格式的结果:
        - line: 返回单个数值
        - bar/pie: 返回字典，键为分类，值为对应数值
        
    注意:
        此函数处理各种异常情况，包括None值、空列表和类型错误等
    """
    try:
        # Check if data is a valid dictionary
        if not data or not isinstance(data, dict):
            log_metric_error("Average Conformity Tendency", ValueError("Invalid data input"), {"data": data})
            return {}

        # Retrieve the list of conformity tendencies from IndividualAgent
        conformity_tendencies = safe_list(safe_get(data, "conformity_tendency", []))

        # Convert all values in the list to numbers, treating None as zero
        conformity_tendencies = [safe_number(value, default=0) for value in conformity_tendencies]

        # Calculate the average conformity tendency
        average_conformity_tendency = safe_avg(conformity_tendencies, default=0)

        # Prepare the result as a dictionary for bar visualization
        result = {"Average Conformity Tendency": average_conformity_tendency}

        return result
    except Exception as e:
        log_metric_error("Average Conformity Tendency", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return {}

from typing import Dict, Any
from onesim.monitor.utils import (
    safe_get, safe_number, log_metric_error
)

def System_Social_Pressure(data: Dict[str, Any]) -> Any:
    """
    计算指标: System Social Pressure
    描述: Tracks the level of social pressure in the environment, which influences agent decision-making and conformity.
    可视化类型: line
    更新频率: 5 秒
    
    Args:
        data: 包含所有变量的数据字典，注意agent变量是列表形式
        
    Returns:
        根据可视化类型返回不同格式的结果:
        - line: 返回单个数值
        - bar/pie: 返回字典，键为分类，值为对应数值
        
    注意:
        此函数处理各种异常情况，包括None值、空列表和类型错误等
    """
    try:
        # Validate input data
        if not data or not isinstance(data, dict):
            log_metric_error("System Social Pressure", ValueError("Invalid data input"), {"data": data})
            return 0
        
        # Extract social_pressure from environment variables
        social_pressure = safe_get(data.get("environment", {}), "social_pressure", None)
        
        # Convert social_pressure to a number, defaulting to 0 if None or invalid type
        social_pressure_value = safe_number(social_pressure, default=0)
        
        # Return the calculated social pressure value for line visualization
        return social_pressure_value

    except Exception as e:
        log_metric_error("System Social Pressure", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return 0

def Opinion_Leader_Influence_Strength_Distribution(data: Dict[str, Any]) -> Any:
    """
    计算指标: Opinion Leader Influence Strength Distribution
    描述: Analyzes the distribution of influence strength among opinion leaders, indicating their potential impact on group dynamics.
    可视化类型: pie
    更新频率: 5 秒
    
    Args:
        data: 包含所有变量的数据字典，注意agent变量是列表形式
        
    Returns:
        根据可视化类型返回不同格式的结果:
        - line: 返回单个数值
        - bar/pie: 返回字典，键为分类，值为对应数值
        
    注意:
        此函数处理各种异常情况，包括None值、空列表和类型错误等
    """
    from onesim.monitor.utils import (
        safe_get, safe_list, safe_sum, log_metric_error
    )

    try:
        # Validate input data
        if not isinstance(data, dict):
            log_metric_error("Opinion Leader Influence Strength Distribution", ValueError("Invalid data input"), {"data": data})
            return {}

        print('!-' * 50)
        print(data)

        # Extract influence_strength values from OpinionLeaderAgent
        # influence_strengths = safe_get(data, 'OpinionLeaderAgent', [])
        # influence_strengths = safe_list(influence_strengths)
        # print(influence_strengths)

        # # Filter out None values and ensure all elements are numbers
        # valid_influence_strengths = [v for v in influence_strengths if isinstance(v, (int, float)) and v is not None]

        # print(valid_influence_strengths)

        # # If the list is empty after filtering, return an empty distribution
        # if not valid_influence_strengths:
        #     return {}

        # # Aggregate influence strengths
        # total_strength = safe_sum(valid_influence_strengths)
        
        # # Handle division by zero scenario
        # if total_strength == 0:
        #     return {}
        valid_influence_strengths = data['influence_strength']
        total_strength = safe_sum(valid_influence_strengths)

        # Calculate proportional values for pie chart
        distribution = {
            f"Leader {i+1}": strength / total_strength
            for i, strength in enumerate(valid_influence_strengths)
        }

        print(distribution)

        return distribution

    except Exception as e:
        log_metric_error("Opinion Leader Influence Strength Distribution", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return {}

# 指标函数字典，用于查找
METRIC_FUNCTIONS = {
    'Average_Conformity_Tendency': Average_Conformity_Tendency,
    'System_Social_Pressure': System_Social_Pressure,
    'Opinion_Leader_Influence_Strength_Distribution': Opinion_Leader_Influence_Strength_Distribution,
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
    生成用于测试的示例数据
    
    Returns:
        示例数据字典
    """
    # 创建一个包含常见数据类型和边界情况的测试数据字典
    return {
        # 环境变量示例
        "total_steps": 100,
        "current_time": 3600,
        "resource_pool": 1000,
        
        # 正常代理变量示例（列表）
        "agent_health": [100, 90, 85, 70, None, 60],
        "agent_resources": [50, 40, 30, 20, 10, None],
        "agent_age": [10, 20, 30, 40, 50, 60],
        
        # 边界情况
        "empty_list": [],
        "none_value": None,
        "zero_value": 0,
        
        # 错误类型示例
        "should_be_list_but_single": 42,
        "invalid_number": "not_a_number",
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
