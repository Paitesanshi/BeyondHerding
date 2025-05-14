# -*- coding: utf-8 -*-
"""
自动生成的监控指标计算模块
"""

import json
from typing import Dict, Any, List, Optional, Union, Callable
import math
from loguru import logger
from onesim.monitor.utils import (
    safe_get, safe_number, safe_list, safe_sum, 
    safe_avg, safe_max, safe_min, safe_count, log_metric_error
)


def Average_Worker_Performance(data: Dict[str, Any]) -> Any:
    """
    计算指标: Average Worker Performance
    描述: Measures the average performance of WorkerAgents based on their task completion and performance adjustment statuses.
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
    from onesim.monitor.utils import (
        safe_get, safe_number, safe_list, safe_sum, safe_avg, log_metric_error
    )

    try:
        # Safely retrieve the 'worker_performance' list from the data
        worker_performance_data = safe_get(data, 'worker_performance', default=None)

        # logger.info(f"Worker performance data: {worker_performance_data}")
        # Ensure the retrieved data is a list
        worker_performance_list = safe_list(worker_performance_data)

        all_performances = []
        for manager_item in worker_performance_list:
            for performance_item in manager_item:
                if performance_item['performance'] is not None and isinstance(performance_item['performance'], (int, float)):
                    all_performances.append(performance_item['performance'])

        # Calculate the average of valid performance values
        average_performance = safe_avg(all_performances, default=0)

        # Return the result in the required format for a bar visualization
        return {"Average Worker Performance": average_performance}

    except Exception as e:
        log_metric_error("Average Worker Performance", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return {"Average Worker Performance": 0}

from typing import Dict, Any
from onesim.monitor.utils import (
    safe_get,
    safe_number,
    safe_list,
    safe_sum,
    safe_avg,
    safe_max,
    safe_min,
    safe_count,
    log_metric_error
)

def Task_Allocation_Effectiveness(data: Dict[str, Any]) -> Any:
    """
    计算指标: Task Allocation Effectiveness
    描述: Evaluates how effectively tasks are allocated by comparing task allocation status with worker performance.
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
        # Check if required variables exist and validate input data
        if not data or not isinstance(data, dict):
            log_metric_error("Task Allocation Effectiveness", ValueError("Invalid data input"), {"data": data})
            return {"Effective": 0, "Ineffective": 0}
        
        # logger.info(f"Task Allocation Effectiveness data: {data}")
        # Retrieve the task_allocation_status and worker_performance lists
        task_allocation_status_list = safe_list(safe_get(data, "task_allocation_status", []))
        worker_performance_list = safe_list(safe_get(data, "worker_performance", []))
        
        # Validate that both lists have the same length
        if len(task_allocation_status_list) != len(worker_performance_list):
            log_metric_error("Task Allocation Effectiveness", ValueError("Mismatched list lengths"), {
                "task_allocation_status_length": len(task_allocation_status_list),
                "worker_performance_length": len(worker_performance_list)
            })
            return {"Effective": 0, "Ineffective": 0}
        
        # Define a high performance threshold
        high_performance_threshold = 75  # Example threshold
        
        # Calculate the number of successful allocations
        successful_allocations = 0
        total_allocations = 0
        
        for allocation_status, performance in zip(task_allocation_status_list, worker_performance_list):
            try:
                performance_value = safe_number(performance[0]["performance"], default=None)
                if allocation_status is not None and performance_value is not None:
                    total_allocations += 1
                    if performance_value >= high_performance_threshold:
                        successful_allocations += 1
            except Exception as e:
                log_metric_error("Task Allocation Effectiveness", e, {
                    "allocation_status": allocation_status,
                    "performance": performance
                })
                continue
        
        # Calculate the effectiveness proportion
        if total_allocations == 0:
            return {"Effective": "N/A", "Ineffective": "N/A"}
        
        effectiveness_proportion = successful_allocations / total_allocations
        
        # Return result in appropriate format for pie chart
        return {
            "Effective": effectiveness_proportion,
            "Ineffective": 1 - effectiveness_proportion
        }
    
    except Exception as e:
        log_metric_error("Task Allocation Effectiveness", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return {"Effective": 0, "Ineffective": 0}

from typing import Dict, Any
from onesim.monitor.utils import (
    safe_get, safe_list, safe_count, log_metric_error
)

def Incentive_Plan_Utilization(data: Dict[str, Any]) -> Any:
    """
    计算指标: Incentive Plan Utilization
    描述: Assesses how well the incentive plans are being utilized by comparing the number of incentives planned versus those applied.
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
            log_metric_error("Incentive Plan Utilization", ValueError("Invalid data input"), {"data": data})
            return 0
        
        logger.info(f"Incentive Plan Utilization data: {data}")
        # Retrieve and validate 'incentive_plan' data
        incentive_plan_data = safe_list(safe_get(data, "ManagerAgent", {}).get("profile.incentive_plan", []))
        num_incentives_planned = safe_count(incentive_plan_data)

        # Retrieve and validate 'performance_adjustment_status' data
        performance_adjustment_data = safe_list(safe_get(data, "WorkerAgent", {}).get("profile.performance_adjustment_status", []))
        num_incentives_applied = safe_count(performance_adjustment_data, lambda x: x is not None and x != '')

        # Calculate the utilization ratio
        if num_incentives_planned == 0:
            utilization_ratio = 0
        else:
            utilization_ratio = num_incentives_applied / num_incentives_planned
        
        return utilization_ratio

    except Exception as e:
        log_metric_error("Incentive Plan Utilization", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return 0

# 指标函数字典，用于查找
METRIC_FUNCTIONS = {
    'Average_Worker_Performance': Average_Worker_Performance,
    'Task_Allocation_Effectiveness': Task_Allocation_Effectiveness,
    'Incentive_Plan_Utilization': Incentive_Plan_Utilization,
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
