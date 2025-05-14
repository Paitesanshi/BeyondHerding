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
    safe_get,
    safe_number,
    safe_list,
    safe_avg,
    log_metric_error
)

def average_credibility_score(data: Dict[str, Any]) -> Any:
    """
    计算指标: average_credibility_score
    描述: Measures the average credibility score of information as evaluated by Ordinary Users. This reflects the overall perception of information reliability in the system.
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
            log_metric_error("average_credibility_score", ValueError("Invalid data input"), {"data": data})
            return 0.0
        logger.info(f"credibility_score data: {data}")
        credibility_scores = safe_list(safe_get(data, "credibility_score", []))

        # Calculate the average credibility score
        average_score = safe_avg(credibility_scores, 0.0)

        # Return result for line visualization type
        return average_score

    except Exception as e:
        log_metric_error("average_credibility_score", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return 0.0

from typing import Dict, Any
from onesim.monitor.utils import (
    safe_get,
    safe_list,
    safe_count,
    log_metric_error
)

def opinion_expression_rate(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculates the opinion expression rate for Ordinary Users.
    This is the proportion of users who have expressed an opinion.

    Args:
        data (Dict[str, Any]): Input data containing agent variables.

    Returns:
        Dict[str, float]: A dictionary suitable for pie chart visualization
                          with "active" and "inactive" categories.
    """
    try:
        # Validate input data
        if not data or not isinstance(data, dict):
            log_metric_error("opinion_expression_rate", ValueError("Invalid data input"), {"data": data})
            return {"active": 0, "inactive": 1}

        # Retrieve the list of expressed_opinion values for Ordinary Users
        expressed_opinions = safe_list(safe_get(data, "expressed_opinion", []))

        logger.info(f"expressed_opinions: {expressed_opinions}")
        # Count the number of active users (non-None and non-empty string opinions)
        active_users_count = safe_count(expressed_opinions, predicate=lambda x: isinstance(x, str) and x.strip() != "")

        # Total number of users
        total_users_count = len(expressed_opinions)

        # Handle division by zero (no users)
        if total_users_count == 0:
            return {"active": 0, "inactive": 1}

        # Calculate active and inactive proportions
        active_rate = active_users_count / total_users_count
        inactive_rate = 1 - active_rate

        # Return results in a format suitable for pie chart visualization
        return {"active": active_rate, "inactive": inactive_rate}

    except Exception as e:
        # Log any unexpected errors
        log_metric_error("opinion_expression_rate", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return {"active": 0, "inactive": 1}

from typing import Dict, Any
from onesim.monitor.utils import (
    safe_get, safe_list, safe_count, safe_avg, log_metric_error
)

def fact_checking_rumor_detection_rate(data: Dict[str, Any]) -> Dict[str, int]:
    """
    Measures the average number of rumors detected by Fact-Check Organizations.
    Returns a dictionary mapping organization categories to average rumor counts.
    """
    try:
        # Validate input data
        if not data or not isinstance(data, dict):
            log_metric_error("fact_checking_rumor_detection_rate", ValueError("Invalid data input"), {"data": data})
            return {}

        detected_rumors = safe_list(safe_get(data, "detected_rumors", []))
        rumor_counts = []
        for rumor in detected_rumors:
            valid_rumor_count = safe_count(rumor, predicate=lambda x: x is not None)
            rumor_counts.append(valid_rumor_count)

        # Calculate average rumor detection rate
        average_rumor_count = safe_avg(rumor_counts, default=0)

        # Return result in bar chart format (dictionary mapping categories to values)
        return {"FactCheckOrganizations": average_rumor_count}

    except Exception as e:
        # Log any unexpected errors
        log_metric_error("fact_checking_rumor_detection_rate", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return {"FactCheckOrganizations": 0}

# 指标函数字典，用于查找
METRIC_FUNCTIONS = {
    'average_credibility_score': average_credibility_score,
    'opinion_expression_rate': opinion_expression_rate,
    'fact_checking_rumor_detection_rate': fact_checking_rumor_detection_rate,
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
