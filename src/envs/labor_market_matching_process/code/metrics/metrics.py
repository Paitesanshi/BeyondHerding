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


def average_job_seeker_experience(data: Dict[str, Any]) -> Any:
    """
    计算指标: average_job_seeker_experience
    描述: Measures the average years of experience among job seekers, providing insights into the overall experience level of the job-seeking population.
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
        safe_get, safe_list, safe_sum, safe_count, log_metric_error
    )
    
    try:
        # Validate the input data structure
        if not data or not isinstance(data, dict):
            log_metric_error("average_job_seeker_experience", ValueError("Invalid data input"), {"data": data})
            return {}

        # Extract the experience data from JobSeeker agents
        experience_list = safe_list(safe_get(data, "experience", []))
        # job_seekers = safe_get(data, "JobSeeker", [])
        # experience_list = []
        
        # for job_seeker in safe_list(job_seekers):
        #     experience = safe_get(job_seeker, "experience", None)
        #     if isinstance(experience, int):
        #         experience_list.append(experience)
        
        # Calculate the average experience
        total_experience = safe_sum(experience_list)
        count_experience = safe_count(experience_list)
        
        # Handle division by zero
        average_experience = total_experience / count_experience if count_experience > 0 else 0
        
        # Return result as a dictionary for bar visualization
        return {"Average Experience": average_experience}
    
    except Exception as e:
        log_metric_error("average_job_seeker_experience", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return {}

from typing import Dict, Any
from onesim.monitor.utils import (
    safe_get,
    safe_list,
    safe_count,
    log_metric_error
)

def job_application_success_rate(data: Dict[str, Any]) -> Any:
    """
    计算指标: job_application_success_rate
    描述: Measures the proportion of job applications that result in successful negotiations, indicating the efficiency of job matching and negotiation processes.
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
            log_metric_error("job_application_success_rate", ValueError("Invalid data input"), {"data": data})
            return {}

        # logger.info(f"data: {data}")
        # Retrieve and validate the applications_submitted and negotiation_outcomes
        applications_submitted = safe_list(safe_get(data, "applications_submitted", []))
        negotiation_outcomes = safe_list(safe_get(data, "negotiation_outcomes", []))

        # Check if the lists are valid
        if not applications_submitted or not negotiation_outcomes:
            log_metric_error("job_application_success_rate", ValueError("Missing or invalid input lists"), {
                "applications_submitted": applications_submitted,
                "negotiation_outcomes": negotiation_outcomes
            })
            return {"Success": 0.0, "Failure": 1.0}

        # Count the total number of applications submitted
        total_applications = safe_count(applications_submitted)

        # Count the number of successful negotiation outcomes
        successful_negotiations = safe_count(negotiation_outcomes, lambda outcome: outcome is not None and outcome)

        # Calculate success rate
        if total_applications == 0:
            success_rate = 0.0
        else:
            success_rate = successful_negotiations / total_applications

        # Return the result in a format suitable for a pie chart
        return {"Success": success_rate, "Failure": 1.0 - success_rate}

    except Exception as e:
        log_metric_error("job_application_success_rate", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return {"Success": 0.0, "Failure": 1.0}

def employer_screening_efficiency(data: Dict[str, Any]) -> Any:
    """
    计算指标: employer_screening_efficiency
    描述: Evaluates the efficiency of employers in screening candidates by comparing the number of screened candidates to the number of posted jobs.
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
        safe_get, safe_list, safe_sum, log_metric_error
    )
    
    try:
        logger.info(f"Screendata: {data}")
        # Validate input data
        if not data or not isinstance(data, dict):
            log_metric_error("employer_screening_efficiency", ValueError("Invalid data input"), {"data": data})
            return {}

        # Retrieve and validate screened_candidates and posted_jobs
        screened_candidates_list = safe_list(safe_get(data, "screened_candidates"))
        posted_jobs_list = safe_list(safe_get(data, "posted_jobs"))

        # Calculate total number of screened candidates
        total_screened_candidates = len([item for sublist in screened_candidates_list for item in sublist])

        # Calculate total number of posted jobs
        total_posted_jobs = len([item for sublist in posted_jobs_list for item in sublist])

        # Handle division by zero scenario
        if total_posted_jobs == 0:
            log_metric_error("employer_screening_efficiency", ZeroDivisionError("Total posted jobs is zero"), {
                "total_screened_candidates": total_screened_candidates,
                "total_posted_jobs": total_posted_jobs
            })
            efficiency = 0.0
        else:
            efficiency = total_screened_candidates / total_posted_jobs

        # Return result in appropriate format for bar visualization
        return {"employer_screening_efficiency": efficiency}

    except Exception as e:
        log_metric_error("employer_screening_efficiency", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return {}

# 指标函数字典，用于查找
METRIC_FUNCTIONS = {
    'average_job_seeker_experience': average_job_seeker_experience,
    'job_application_success_rate': job_application_success_rate,
    'employer_screening_efficiency': employer_screening_efficiency,
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
