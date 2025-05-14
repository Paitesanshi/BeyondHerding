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
from onesim.monitor.utils import safe_get, safe_number, safe_list, safe_sum, log_metric_error

def Family_Education_Investment_Ratio(data: Dict[str, Any]) -> Any:
    """
    计算指标: Family Education Investment Ratio
    描述: This metric measures the ratio of resources allocated to education versus total collective resources within families, indicating the prioritization of educational investments.
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
        # Validate data input
        if not data or not isinstance(data, dict):
            log_metric_error("Family Education Investment Ratio", ValueError("Invalid data input"), {"data": data})
            return {}

        # Retrieve and validate the list of collective resources and education investments
        collective_resources_list = safe_list(safe_get(data, "collective_resources", []))
        education_investment_list = safe_list(safe_get(data, "education_investment", []))

        # Check if lists are empty
        if not collective_resources_list or not education_investment_list:
            log_metric_error("Family Education Investment Ratio", ValueError("Empty lists for required variables"), {
                "collective_resources_list": collective_resources_list,
                "education_investment_list": education_investment_list
            })
            return {}

        # Initialize total variables
        total_collective_resources = 0
        total_education_investment = 0

        # Calculate ratios and aggregate totals
        for collective_resources, education_investment in zip(collective_resources_list, education_investment_list):
            # Safely convert values to numbers
            collective_resources = safe_number(collective_resources, None)
            education_investment = safe_number(education_investment, None)

            # Skip invalid entries
            if collective_resources is None or education_investment is None or collective_resources == 0:
                continue

            # Aggregate totals
            total_collective_resources += collective_resources
            total_education_investment += education_investment

        # Calculate the overall ratio
        if total_collective_resources == 0:
            log_metric_error("Family Education Investment Ratio", ZeroDivisionError("Total collective resources is zero"), {
                "total_collective_resources": total_collective_resources,
                "total_education_investment": total_education_investment
            })
            return {}

        # Return the ratio in pie chart format as a proportion
        return {
            "Education Investment": total_education_investment / total_collective_resources,
            "Other Resources": 1 - (total_education_investment / total_collective_resources)
        }

    except Exception as e:
        log_metric_error("Family Education Investment Ratio", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return {}

from onesim.monitor.utils import safe_get, safe_list, safe_count, log_metric_error


def Student_Candidate_Selection_Effectiveness(data: Dict[str, Any]) -> Any:
    """
    计算指标: Student Candidate Selection Effectiveness
    描述: This metric evaluates the effectiveness of employers in selecting candidates by comparing the number of candidates evaluated versus those hired.
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
            raise ValueError("Invalid data input")

        # Retrieve candidates_list and hiring_decision
        candidates_list = safe_list(safe_get(data, "all_applications_received", []))
        hiring_decisions = safe_list(safe_get(data, "all_selected_students", []))

        # print('-' * 50)
        # print(candidates_list)
        # print(hiring_decisions)

        # Count candidates and successful hiring decisions
        total_candidates, successful_hires = 0, 0
        for x in candidates_list:
            total_candidates += len(x)
        for x in hiring_decisions:
            successful_hires += len(x)
            
        # total_candidates = safe_count(candidates_list)
        # successful_hires = safe_count(hiring_decisions)

        # print('-' * 50)
        # print(candidates)
        # print(successful_hires)

        # # Calculate effectiveness ratio
        if total_candidates == 0:
            return 0.0  # Avoid division by zero
        
        effectiveness_ratio = successful_hires / total_candidates

        # return effectiveness_ratio
        return {"Average Effectiveness Ratio": effectiveness_ratio}
        # return successful_hires

    except Exception as e:
        log_metric_error("Student Candidate Selection Effectiveness", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return 0.0


# 指标函数字典，用于查找
METRIC_FUNCTIONS = {
    'Family_Education_Investment_Ratio': Family_Education_Investment_Ratio,
    'Student_Candidate_Selection_Effectiveness': Student_Candidate_Selection_Effectiveness,
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
