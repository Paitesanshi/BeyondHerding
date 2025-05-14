from fastapi import APIRouter, HTTPException, Response
from typing import Dict, Any, Optional, List
import os
import json
import time
import threading
import pandas as pd
from datetime import datetime
from loguru import logger
import mlflow

from backend.models.training import (
    FeedbackExportRequest, FeedbackExportResponse,
    TrainingRequest, TrainingResponse
)
from backend.utils.model_management import load_model_if_needed

# 全局变量
TRAINING_STATUS = {}
TRAINING_THREAD = None

router = APIRouter(
    tags=["training"],
    prefix="/training"
)

@router.post("/export", response_model=FeedbackExportResponse)
async def export_training_data(request: FeedbackExportRequest):
    """导出训练数据"""
    env_name = request.env_name
    session_id = request.session_id
    start_date = request.start_date
    end_date = request.end_date
    limit = request.limit or 100
    offset = request.offset or 0
    
    # 构建查询条件
    query_conditions = []
    query_params = {}
    
    if env_name:
        query_conditions.append("env_name = :env_name")
        query_params["env_name"] = env_name
    
    if session_id:
        query_conditions.append("session_id = :session_id")
        query_params["session_id"] = session_id
    
    if start_date:
        try:
            start_timestamp = datetime.fromisoformat(start_date).timestamp()
            query_conditions.append("timestamp >= :start_timestamp")
            query_params["start_timestamp"] = start_timestamp
        except ValueError:
            raise HTTPException(status_code=400, detail="无效的开始日期格式，应为ISO格式 (YYYY-MM-DDTHH:MM:SS)")
    
    if end_date:
        try:
            end_timestamp = datetime.fromisoformat(end_date).timestamp()
            query_conditions.append("timestamp <= :end_timestamp")
            query_params["end_timestamp"] = end_timestamp
        except ValueError:
            raise HTTPException(status_code=400, detail="无效的结束日期格式，应为ISO格式 (YYYY-MM-DDTHH:MM:SS)")
    
    # 构建查询字符串
    query = " AND ".join(query_conditions) if query_conditions else ""
    
    # 模拟从数据库获取数据
    # 在实际应用中，应该使用数据库查询
    data = []
    total = 0
    
    # 假设我们有一些测试数据
    test_data = []
    if env_name:
        # 修改：统一从datasets目录获取决策数据
        datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "datasets")
        env_dir = os.path.join(datasets_dir, env_name)
        
        # 查找最新的决策数据文件
        decision_files = []
        if os.path.exists(env_dir) and os.path.isdir(env_dir):
            for file in os.listdir(env_dir):
                if file.startswith("decisions_") and file.endswith(".json"):
                    decision_files.append(os.path.join(env_dir, file))
        
        # 按修改时间排序，获取最新的文件
        if decision_files:
            decision_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            latest_decision_file = decision_files[0]
            
            try:
                with open(latest_decision_file, 'r') as f:
                    test_data = json.load(f)
                    # 添加环境名称
                    for item in test_data:
                        item["env_name"] = env_name
            except Exception as e:
                logger.error(f"加载决策数据错误: {str(e)}")
    
    # 过滤数据
    if query:
        # 注意：这里是简化的过滤逻辑，实际应用中应使用数据库查询
        filtered_data = []
        for item in test_data:
            match = True
            for condition in query_conditions:
                key, value = condition.split(" = ")
                value = query_params[value[1:]]  # 移除冒号
                if key in item and item[key] != value:
                    match = False
                    break
            if match:
                filtered_data.append(item)
        data = filtered_data[offset:offset+limit]
        total = len(filtered_data)
    else:
        data = test_data[offset:offset+limit]
        total = len(test_data)
    
    return FeedbackExportResponse(
        success=True,
        message="成功导出训练数据",
        data=data,
        total=total,
        limit=limit,
        offset=offset
    )

@router.get("/environments", response_model=Dict[str, Any])
async def get_feedback_environments():
    """获取反馈环境列表"""
    environments = []
    
    # 获取所有环境
    envs_path = "./envs"
    if os.path.exists(envs_path) and os.path.isdir(envs_path):
        for env_name in os.listdir(envs_path):
            env_path = os.path.join(envs_path, env_name)
            if os.path.isdir(env_path):
                # 检查是否有决策数据
                decisions_path = os.path.join(env_path, "data", "decisions.json")
                has_decisions = os.path.exists(decisions_path)
                
                # 检查元数据
                metadata_path = os.path.join(env_path, "metadata.json")
                metadata = {}
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    except Exception as e:
                        logger.error(f"加载元数据错误: {str(e)}")
                
                # 添加环境信息
                environments.append({
                    "name": env_name,
                    "display_name": metadata.get("name", env_name),
                    "description": metadata.get("description", ""),
                    "has_decisions": has_decisions
                })
    
    return {
        "environments": environments,
        "count": len(environments)
    }

@router.get("/sessions", response_model=Dict[str, Any])
async def get_feedback_sessions():
    """获取反馈会话列表"""
    sessions = []
    
    # 获取所有会话
    # 注意：在实际应用中，应该从数据库中获取会话列表
    # 这里我们使用模拟数据
    sessions_path = "./sessions"
    if os.path.exists(sessions_path) and os.path.isdir(sessions_path):
        for session_id in os.listdir(sessions_path):
            session_path = os.path.join(sessions_path, session_id)
            if os.path.isdir(session_path):
                # 检查会话信息
                info_path = os.path.join(session_path, "info.json")
                info = {}
                if os.path.exists(info_path):
                    try:
                        with open(info_path, 'r') as f:
                            info = json.load(f)
                    except Exception as e:
                        logger.error(f"加载会话信息错误: {str(e)}")
                
                # 添加会话信息
                sessions.append({
                    "session_id": session_id,
                    "created_at": info.get("created_at", ""),
                    "env_name": info.get("env_name", ""),
                    "description": info.get("description", "")
                })
    
    return {
        "sessions": sessions,
        "count": len(sessions)
    }

@router.get("/download")
async def download_training_data(env_name: Optional[str] = None, session_id: Optional[str] = None, format: str = "json"):
    """下载训练数据"""
    if not env_name and not session_id:
        raise HTTPException(status_code=400, detail="必须提供env_name或session_id")
    
    # 获取数据
    request = FeedbackExportRequest(env_name=env_name, session_id=session_id, limit=1000)
    response = await export_training_data(request)
    
    if not response.success:
        raise HTTPException(status_code=500, detail=response.message)
    
    data = response.data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 根据格式返回数据
    if format == "json":
        return Response(
            content=json.dumps(data, ensure_ascii=False, indent=2),
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename=training_data_{env_name}_{timestamp}.json"
            }
        )
    elif format == "csv":
        # 将数据转换为DataFrame
        df = pd.DataFrame(data)
        csv_data = df.to_csv(index=False)
        
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=training_data_{env_name}_{timestamp}.csv"
            }
        )
    else:
        raise HTTPException(status_code=400, detail=f"不支持的格式: {format}")

@router.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    """训练模型"""
    global TRAINING_THREAD, TRAINING_STATUS
    
    model_name = request.model_name
    dataset_path = request.dataset_path
    tuning_mode = request.tuning_mode
    experiment_name = request.experiment_name or f"{model_name}_{tuning_mode}_{int(time.time())}"
    
    # 检查数据集是否存在
    if not os.path.exists(dataset_path):
        # 尝试从datasets目录获取数据集
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        potential_path = os.path.join(base_path, "datasets", dataset_path)
        if os.path.exists(potential_path):
            dataset_path = potential_path
        else:
            raise HTTPException(status_code=404, detail=f"数据集不存在: {dataset_path}")
    
    # 初始化训练状态
    run_id = f"run_{int(time.time())}"
    TRAINING_STATUS[run_id] = {
        "status": "initializing",
        "progress": 0.0,
        "message": "正在初始化训练...",
        "start_time": time.time(),
        "end_time": None,
        "model_name": model_name,
        "dataset_path": dataset_path,
        "tuning_mode": tuning_mode,
        "experiment_name": experiment_name
    }
    
    # 设置MLflow跟踪URI
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # 创建实验
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
        else:
            experiment_id = mlflow.create_experiment(experiment_name)
    except Exception as e:
        logger.error(f"创建MLflow实验错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建MLflow实验错误: {str(e)}")
    
    def run_training():
        global TRAINING_STATUS
        
        try:
            # 更新状态
            TRAINING_STATUS[run_id]["status"] = "training"
            TRAINING_STATUS[run_id]["message"] = "训练已开始..."
            
            # 导入训练模块
            from llm_tuning.codes.tune_llm import run_tuning
            
            # 使用MLflow记录训练过程
            with mlflow.start_run(experiment_id=experiment_id, run_name=run_id) as run:
                # 记录参数
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("tuning_mode", tuning_mode)
                mlflow.log_param("dataset_path", dataset_path)
                
                # 更新进度
                def progress_callback(progress):
                    TRAINING_STATUS[run_id]["progress"] = progress
                    TRAINING_STATUS[run_id]["message"] = f"训练进度: {progress*100:.1f}%"
                    mlflow.log_metric("progress", progress)
                
                # 开始训练
                try:
                    # 调用训练函数，传入数据集路径
                    model_id = run_tuning(
                        tuning_mode=tuning_mode,
                        llm_path=model_name,
                        experiment_name=experiment_name,
                        tracking_uri=mlflow_tracking_uri,
                        devices="0",
                        dataset_path=dataset_path
                    )
                    
                    # 完成训练
                    TRAINING_STATUS[run_id]["status"] = "completed"
                    TRAINING_STATUS[run_id]["progress"] = 1.0
                    TRAINING_STATUS[run_id]["message"] = "训练已完成"
                    TRAINING_STATUS[run_id]["end_time"] = time.time()
                    TRAINING_STATUS[run_id]["mlflow_run_id"] = run.info.run_id
                    TRAINING_STATUS[run_id]["model_id"] = model_id
                    
                    # 记录模型ID
                    if model_id:
                        mlflow.log_param("model_id", model_id)
                except Exception as train_error:
                    logger.error(f"训练过程错误: {str(train_error)}")
                    TRAINING_STATUS[run_id]["status"] = "failed"
                    TRAINING_STATUS[run_id]["message"] = f"训练失败: {str(train_error)}"
                    TRAINING_STATUS[run_id]["end_time"] = time.time()
                    raise train_error
        except Exception as e:
            logger.error(f"训练错误: {str(e)}")
            TRAINING_STATUS[run_id]["status"] = "failed"
            TRAINING_STATUS[run_id]["message"] = f"训练失败: {str(e)}"
            TRAINING_STATUS[run_id]["end_time"] = time.time()
    
    # 启动训练线程
    def thread_func():
        # 这个额外的函数是为了捕获线程中的所有异常
        try:
            run_training()
        except Exception as e:
            logger.error(f"训练线程错误: {str(e)}")
            TRAINING_STATUS[run_id]["status"] = "failed"
            TRAINING_STATUS[run_id]["message"] = f"训练线程错误: {str(e)}"
            TRAINING_STATUS[run_id]["end_time"] = time.time()
    
    TRAINING_THREAD = threading.Thread(target=thread_func)
    TRAINING_THREAD.start()
    
    # 构建MLflow实验URL
    experiment_url = f"{mlflow_tracking_uri}/#/experiments/{experiment_id}"
    
    return TrainingResponse(
        success=True,
        message="训练已启动",
        tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
        experiment_url=experiment_url,
        run_id=run_id
    )

@router.get("/datasets", response_model=Dict[str, List[str]])
async def get_available_datasets():
    """获取可用数据集"""
    datasets = {
        "json": [],
        "csv": []
    }
    
    # 修改: 统一从项目根目录下的datasets目录查找数据集
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    dataset_dirs = [os.path.join(base_path, "datasets")]
    
    for dir_path in dataset_dirs:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.startswith("decisions_") and file.endswith(".json"):
                        # 使用相对于项目根目录的路径
                        rel_path = os.path.relpath(os.path.join(root, file), base_path)
                        datasets["json"].append(rel_path)
                    elif file.endswith(".csv"):
                        rel_path = os.path.relpath(os.path.join(root, file), base_path)
                        datasets["csv"].append(rel_path)
    
    return datasets

@router.get("/models", response_model=List[str])
async def get_available_base_models():
    """获取可用基础模型"""
    # 注意：在实际应用中，这些值应该从配置文件中加载
    return [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-3.5-turbo",
        "claude-3-sonnet",
        "claude-3-opus",
        "llama-3-8b",
        "llama-3-70b",
        "mixtral-8x7b",
        "mistral-7b",
        "qwen-7b",
        "qwen-14b"
    ]

@router.get("/training_status", response_model=Dict[str, Any])
async def get_training_status(experiment_name: Optional[str] = None, run_id: Optional[str] = None):
    """获取训练状态"""
    global TRAINING_STATUS
    
    # 如果提供了run_id，返回特定运行的状态
    if run_id and run_id in TRAINING_STATUS:
        return TRAINING_STATUS[run_id]
    
    # 如果提供了experiment_name，返回该实验的所有运行
    if experiment_name:
        experiment_runs = {}
        for r_id, status in TRAINING_STATUS.items():
            if status.get("experiment_name") == experiment_name:
                experiment_runs[r_id] = status
        
        if not experiment_runs:
            raise HTTPException(status_code=404, detail=f"未找到实验: {experiment_name}")
        
        return {
            "experiment_name": experiment_name,
            "runs": experiment_runs
        }
    
    # 返回所有训练状态
    return {
        "runs": TRAINING_STATUS
    } 