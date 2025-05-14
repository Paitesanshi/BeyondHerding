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
from onesim.monitor.utils import safe_get, safe_list, safe_count, log_metric_error

def Prosecution_Decision_Rate(data: Dict[str, Any]) -> Any:
    """
    计算指标: Prosecution Decision Rate
    描述: Measures the proportion of prosecution decisions made by prosecutors, indicating the level of activity and decision-making within the prosecution process.
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
    try:
        # Validate the input data
        if not data or not isinstance(data, dict):
            log_metric_error("Prosecution Decision Rate", ValueError("Invalid data input"), {"data": data})
            return {}

        # Retrieve prosecution decisions safely
        prosecution_decisions = safe_list(safe_get(data, "prosecution_decision", []))
        logger.info(f"prosecution_decisions: {prosecution_decisions}")
        # Handle None values by treating them as 'undecided'
        valid_decisions = [decision if decision is not None else 'undecided' for decision in prosecution_decisions]

        # Count the number of 'proceed' decisions
        prosecute_count = safe_count(valid_decisions, lambda x: x == 'proceed')

        # Count the total number of decisions excluding 'undecided'
        total_decisions = safe_count(valid_decisions, lambda x: x != 'undecided')

        # Calculate the prosecution decision rate
        if total_decisions == 0:
            prosecution_decision_rate = 0.0
        else:
            prosecution_decision_rate = prosecute_count / total_decisions

        # Return the result in appropriate format for pie visualization
        return {"Prosecute": prosecution_decision_rate, "Other": 1 - prosecution_decision_rate}
    
    except Exception as e:
        log_metric_error("Prosecution Decision Rate", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return {}

from typing import Dict, Any
from onesim.monitor.utils import safe_get, safe_list, safe_avg, log_metric_error

def Average_Evidence_Quality(data: Dict[str, Any]) -> Any:
    """
    计算指标: Average Evidence Quality
    描述: Calculates the average quality of evidence available to prosecutors, reflecting the strength of the case being built.
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
        # Check if required variables exist and validate input data
        if not data or not isinstance(data, dict):
            log_metric_error("Average Evidence Quality", ValueError("Invalid data input"), {"data": data})
            return {}

        # Retrieve the list of evidence quality values for Prosecutors
        evidence_quality_values = safe_list(safe_get(data, "evidence_quality", []))

        # Filter out None values and ensure all elements are numbers
        valid_evidence_quality = [float(value) for value in evidence_quality_values]

        # Calculate the average using the safe_avg utility function
        average_quality = safe_avg(valid_evidence_quality, default=0.0)

        # Return the result in the appropriate format for a bar chart
        return {"Average Evidence Quality": average_quality}

    except Exception as e:
        log_metric_error("Average Evidence Quality", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return {}

from typing import Dict, Any
from onesim.monitor.utils import (
    safe_get,
    safe_list,
    safe_count,
    log_metric_error
)

def Jury_Verdict_Distribution(data: Dict[str, Any]) -> Any:
    """
    计算指标: Jury Verdict Distribution
    描述: Shows the distribution of verdicts made by juries, providing insight into the outcomes of trials and the decision-making tendencies of juries.
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
        # Ensure the data is a valid dictionary
        if not data or not isinstance(data, dict):
            log_metric_error("Jury Verdict Distribution", ValueError("Invalid data input"), {"data": data})
            return {}

        # Extract the list of verdicts from the data
        verdicts = safe_list(safe_get(data, "verdict", []))

        # Initialize a dictionary to count occurrences of each verdict
        verdict_distribution = {}

        # Count each verdict, excluding None values
        for verdict in verdicts:
            if verdict is not None and isinstance(verdict, str):
                if verdict not in verdict_distribution:
                    verdict_distribution[verdict] = 0
                verdict_distribution[verdict] += 1

        # If the verdict distribution is empty, return zero counts for common verdict types
        if not verdict_distribution:
            verdict_distribution = {"guilty": 0, "not guilty": 0, "undecided": 0}

        return verdict_distribution

    except Exception as e:
        log_metric_error("Jury Verdict Distribution", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return {}

# 指标函数字典，用于查找
METRIC_FUNCTIONS = {
    'Prosecution_Decision_Rate': Prosecution_Decision_Rate,
    'Average_Evidence_Quality': Average_Evidence_Quality,
    'Jury_Verdict_Distribution': Jury_Verdict_Distribution,
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
