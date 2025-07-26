#!/usr/bin/env python3
"""
LLM Testing Script for OneSim

This script provides comprehensive LLM testing capabilities with support for:
- Multiple choice questions
- Fill-in-the-blank questions  
- Multiple model configurations
- Detailed performance analysis
- Structured result storage

"""

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from dataclasses_json import dataclass_json
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from onesim.models import ModelManager, UserMessage, get_model


@dataclass_json
@dataclass
class TestConfiguration:
    """Test configuration parameters."""
    dataset_file: str = "1000.csv"
    model_configs: Optional[List[str]] = None
    max_questions: Optional[int] = None
    output_dir: str = "test_results"
    save_detailed: bool = True
    save_summary: bool = True
    question_types: Set[str] = field(default_factory=lambda: {"multiple_choice", "fill_blank"})
    timeout_seconds: float = 30.0
    max_concurrent: int = 10
    use_async: bool = True
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.max_questions is not None and self.max_questions <= 0:
            raise ValueError("max_questions must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_concurrent <= 0:
            raise ValueError("max_concurrent must be positive")


@dataclass_json
@dataclass
class QuestionResult:
    """Result of testing a single question."""
    model_config: str
    question_type: str
    question: str
    expected_answer: str
    model_answer: str
    is_correct: bool
    response_time: float
    confidence_score: Optional[float] = None
    full_response: str = ""
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "model_config": self.model_config,
            "question_type": self.question_type,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "model_answer": self.model_answer,
            "is_correct": self.is_correct,
            "response_time": self.response_time,
            "confidence_score": self.confidence_score,
            "full_response": self.full_response,
            "error": self.error,
            "timestamp": self.timestamp
        }


class LLMTester:
    """LLM testing utility class.
    
    This class provides comprehensive testing capabilities for language models
    with support for multiple question types, concurrent processing, and detailed
    performance analytics.
    """
    
    def __init__(self, config: TestConfiguration):
        """
        Initialize LLM tester.
        
        Args:
            config: Test configuration parameters
        """
        self.config = config
        self.project_root = Path(__file__).parent.parent
        self.config_path = self.project_root / "config" / "config.json"
        self.model_config_path = self.project_root / "config" / "model_config.json"
        self.datasets_path = self.project_root / "datasets"
        self.results_dir = self.project_root / config.output_dir
        
        # Ensure results directory exists
        self.results_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Load configurations
        self.main_config = self._load_main_config()
        self._init_model_manager()
        
        # Test results storage
        self.test_results: List[QuestionResult] = []
        
        # Concurrency control
        self.semaphore = asyncio.Semaphore(config.max_concurrent) if config.use_async else None
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_file = self.results_dir / "llm_test.log"
        logger.add(
            log_file,
            rotation="10 MB",
            compression="zip",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
        )
        logger.info("LLM Tester initialized")
        
    def _load_main_config(self) -> Dict[str, Any]:
        """Load main configuration file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Loaded main configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load main configuration: {e}")
            raise
    
    def _init_model_manager(self) -> bool:
        """Initialize model manager.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            with open(self.model_config_path, 'r', encoding='utf-8') as f:
                model_config = json.load(f)
            
            model_manager = ModelManager.get_instance()
            model_manager.load_model_configs(model_config)
            
            chat_models_count = len(model_config.get('chat', []))
            embedding_models_count = len(model_config.get('embedding', []))
            
            logger.info(f"Model manager initialized with {chat_models_count} chat models and {embedding_models_count} embedding models")
            self._list_available_models()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize model manager: {e}")
            return False
    
    def _list_available_models(self) -> None:
        """List available models."""
        try:
            with open(self.model_config_path, 'r', encoding='utf-8') as f:
                model_config = json.load(f)
            
            chat_models = model_config.get('chat', [])
            embedding_models = model_config.get('embedding', [])
            
            if chat_models:
                logger.info("Available Chat Models:")
                for i, model in enumerate(chat_models, 1):
                    logger.info(f"  {i}. {model['config_name']} ({model['provider']}: {model['model_name']})")
            
            if embedding_models:
                logger.info("Available Embedding Models:")
                for i, model in enumerate(embedding_models, 1):
                    logger.info(f"  {i}. {model['config_name']} ({model['provider']}: {model['model_name']})")
                
        except Exception as e:
            logger.error(f"Failed to list available models: {e}")
    
    def load_dataset(self, dataset_file: str) -> Optional[pd.DataFrame]:
        """
        Load dataset from CSV file.
        
        Args:
            dataset_file: Name of the dataset file
            
        Returns:
            DataFrame if successful, None otherwise
        """
        dataset_path = self.datasets_path / dataset_file
        
        try:
            df = pd.read_csv(dataset_path)
            logger.info(f"Loaded dataset from {dataset_path}")
            logger.info(f"Dataset contains {len(df)} rows with columns: {list(df.columns)}")
            
            # Validate required columns
            required_columns = {'Question', 'Answer'}
            if not required_columns.issubset(df.columns):
                missing = required_columns - set(df.columns)
                logger.error(f"Dataset missing required columns: {missing}")
                return None
                
            return df
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return None
    
    def test_single_question(
        self, 
        model_config_name: str, 
        question: str, 
        expected_answer: str,
        question_type: str = "multiple_choice",
        choices: Optional[str] = None
    ) -> QuestionResult:
        """
        Test a single question with the specified model.
        
        Args:
            model_config_name: Name of the model configuration
            question: The question text
            expected_answer: The correct answer
            question_type: Type of question (multiple_choice, fill_blank)
            choices: Available choices for multiple choice questions
            
        Returns:
            QuestionResult: Test result with performance metrics
        """
        try:
            model = get_model(config_name=model_config_name)
            
            # Build prompt based on question type
            prompt = self._build_prompt(question, question_type, choices)
            formatted_prompt = model.format(UserMessage(content=prompt))
            
            start_time = time.time()
            response = model(formatted_prompt)
            end_time = time.time()
            
            response_time = end_time - start_time
            # Use .text attribute instead of .content for consistency
            response_text = getattr(response, 'text', '') or getattr(response, 'content', '')
            model_answer = self._extract_answer(response_text, question_type)
            is_correct = self._evaluate_answer(model_answer, expected_answer, question_type)
            
            return QuestionResult(
                model_config=model_config_name,
                question_type=question_type,
                question=question,
                expected_answer=expected_answer,
                model_answer=model_answer,
                is_correct=is_correct,
                response_time=response_time,
                full_response=response_text[:1000]  # Truncate for storage
            )
            
        except Exception as e:
            logger.error(f"Failed to test question with model {model_config_name}: {e}")
            return QuestionResult(
                model_config=model_config_name,
                question_type=question_type,
                question=question,
                expected_answer=expected_answer,
                model_answer="",
                is_correct=False,
                response_time=0.0,
                error=str(e)
            )
    
    async def test_single_question_async(
        self, 
        model_config_name: str, 
        question: str, 
        expected_answer: str,
        question_type: str = "multiple_choice",
        choices: Optional[str] = None
    ) -> QuestionResult:
        """
        Test a single question with the specified model (asynchronous).
        
        Args:
            model_config_name: Name of the model configuration
            question: The question text
            expected_answer: The correct answer
            question_type: Type of question (multiple_choice, fill_blank)
            choices: Available choices for multiple choice questions
            
        Returns:
            QuestionResult: Test result with performance metrics
        """
        async with self.semaphore:  # Control concurrency
            try:
                model = get_model(config_name=model_config_name)
                
                # Build prompt based on question type
                prompt = self._build_prompt(question, question_type, choices)
                formatted_prompt = model.format(UserMessage(content=prompt))
                
                start_time = time.time()
                
                # Use acall if available, otherwise fall back to sync in executor
                if hasattr(model, 'acall'):
                    response = await asyncio.wait_for(
                        model.acall(formatted_prompt),
                        timeout=self.config.timeout_seconds
                    )
                else:
                    # Run sync method in thread pool
                    loop = asyncio.get_event_loop()
                    response = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: model(formatted_prompt)),
                        timeout=self.config.timeout_seconds
                    )
                
                end_time = time.time()
                
                response_time = end_time - start_time
                # Use .text attribute instead of .content
                response_text = getattr(response, 'text', '') or getattr(response, 'content', '')
                model_answer = self._extract_answer(response_text, question_type)
                is_correct = self._evaluate_answer(model_answer, expected_answer, question_type)
                
                return QuestionResult(
                    model_config=model_config_name,
                    question_type=question_type,
                    question=question,
                    expected_answer=expected_answer,
                    model_answer=model_answer,
                    is_correct=is_correct,
                    response_time=response_time,
                    full_response=response_text[:1000]  # Truncate for storage
                )
                
            except asyncio.TimeoutError:
                logger.error(f"Timeout testing question with model {model_config_name}")
                return QuestionResult(
                    model_config=model_config_name,
                    question_type=question_type,
                    question=question,
                    expected_answer=expected_answer,
                    model_answer="",
                    is_correct=False,
                    response_time=self.config.timeout_seconds,
                    error="Timeout"
                )
            except Exception as e:
                logger.error(f"Failed to test question with model {model_config_name}: {e}")
                return QuestionResult(
                    model_config=model_config_name,
                    question_type=question_type,
                    question=question,
                    expected_answer=expected_answer,
                    model_answer="",
                    is_correct=False,
                    response_time=0.0,
                    error=str(e)
                )
    
    def _build_prompt(self, question: str, question_type: str, choices: Optional[str] = None) -> str:
        """Build prompt based on question type.
        
        Args:
            question: The question text
            question_type: Type of question (multiple_choice, fill_blank)
            choices: Available choices for multiple choice questions
            
        Returns:
            Formatted prompt string
        """
        if question_type == "multiple_choice" and choices:
            return f"""You are an expert in logical reasoning. Please carefully read the following multiple choice question and select the correct answer.

Question: {question}

Choices: {choices}

Please answer with only the letter (A, B, C, or D) without explanation.

Answer:"""
        
        elif question_type == "fill_blank":
            return f"""You are an expert in logical reasoning. Please carefully read the following fill-in-the-blank question and provide the correct answer.

Question: {question}

Please provide a concise and accurate answer without explanation.

Answer:"""
        
        else:
            # Generic prompt for other question types
            return f"""You are an expert in logical reasoning. Please carefully read the following question and provide the correct answer.

Question: {question}

Please provide a clear and concise answer.

Answer:"""
    
    def _extract_answer(self, response: str, question_type: str) -> str:
        """Extract answer from model response based on question type.
        
        Args:
            response: Raw model response
            question_type: Type of question to determine extraction method
            
        Returns:
            Extracted answer string
        """
        response = response.strip()
        
        if question_type == "multiple_choice":
            # Look for option letters (A, B, C, D)
            response_upper = response.upper()
            for option in ['A', 'B', 'C', 'D']:
                if option in response_upper:
                    return option
            
            # If no option found, return first word
            first_word = response.split()[0] if response.split() else "UNKNOWN"
            return first_word.upper()
            
        elif question_type == "fill_blank":
            # For fill-in-the-blank, extract the main answer
            # Remove common prefixes and clean the response
            clean_response = response.lower()
            prefixes_to_remove = [
                "the answer is", "answer:", "答案是", "答案：", 
                "the correct answer is", "correct answer:"
            ]
            
            for prefix in prefixes_to_remove:
                if clean_response.startswith(prefix):
                    clean_response = clean_response[len(prefix):].strip()
                    break
            
            # Remove punctuation and extra whitespace
            clean_response = re.sub(r'[^\w\s]', '', clean_response).strip()
            
            # Return first significant word/phrase (up to 50 characters)
            return clean_response[:50] if clean_response else "UNKNOWN"
            
        else:
            # Generic extraction for other question types
            return response[:100].strip() if response else "UNKNOWN"
    
    
    def _evaluate_answer(self, model_answer: str, expected_answer: str, question_type: str) -> bool:
        """Evaluate whether the model answer is correct.
        
        Args:
            model_answer: Answer provided by the model
            expected_answer: Correct answer
            question_type: Type of question for evaluation method
            
        Returns:
            True if answer is correct, False otherwise
        """
        if question_type == "multiple_choice":
            return model_answer.upper().strip() == expected_answer.upper().strip()
            
        elif question_type == "fill_blank":
            # For fill-in-the-blank, use more flexible matching
            model_lower = model_answer.lower().strip()
            expected_lower = expected_answer.lower().strip()
            
            # Exact match
            if model_lower == expected_lower:
                return True
                
            # Check if model answer contains expected answer
            if expected_lower in model_lower or model_lower in expected_lower:
                return True
                
            # Check similarity for common variations
            model_words = set(model_lower.split())
            expected_words = set(expected_lower.split())
            
            # If significant overlap in words, consider correct
            if len(model_words & expected_words) / max(len(expected_words), 1) >= 0.8:
                return True
                
            return False
            
        else:
            # Generic evaluation
            return model_answer.lower().strip() == expected_answer.lower().strip()
    
    def test_dataset(self) -> Optional[Dict[str, Any]]:
        """
        Test entire dataset with configured parameters.
        
        Returns:
            Dictionary containing test results and statistics, None if failed
        """
        if self.config.use_async:
            return asyncio.run(self.test_dataset_async())
        else:
            return self._test_dataset_sync()
    
    def _test_dataset_sync(self) -> Optional[Dict[str, Any]]:
        """
        Synchronous version of dataset testing.
        
        Returns:
            Dictionary containing test results and statistics, None if failed
        """
        df = self.load_dataset(self.config.dataset_file)
        if df is None:
            return None
        
        # Determine model configurations to test - Fix the --models parameter issue
        model_configs = self.config.model_configs
        if not model_configs:
            with open(self.model_config_path, 'r', encoding='utf-8') as f:
                model_config = json.load(f)
            model_configs = [model['config_name'] for model in model_config.get('chat', [])[:1]]
        
        logger.info(f"Starting synchronous test with models: {model_configs}")
        
        # Limit questions if specified
        if self.config.max_questions:
            df = df.head(self.config.max_questions)
        
        results = []
        total_questions = len(df) * len(model_configs)
        current_question = 0
        
        for _, row in df.iterrows():
            # Determine question type
            question_type = self._determine_question_type(row)
            if question_type not in self.config.question_types:
                continue
                
            for model_config_name in model_configs:
                current_question += 1
                logger.info(f"Testing question {current_question}/{total_questions} with {model_config_name}")
                
                result = self.test_single_question(
                    model_config_name=model_config_name,
                    question=row['Question'],
                    expected_answer=row['Answer'],
                    question_type=question_type,
                    choices=row.get('Choices', None)
                )
                results.append(result)
        
        # Calculate statistics
        stats = self._calculate_statistics(results)
        
        test_results = {
            'configuration': self.config.to_dict(),
            'results': [r.to_dict() for r in results],
            'statistics': stats,
            'metadata': {
                'test_date': datetime.now().isoformat(),
                'total_questions': len(results),
                'dataset_file': self.config.dataset_file
            }
        }
        
        return test_results
    
    async def test_dataset_async(self) -> Optional[Dict[str, Any]]:
        """
        Asynchronous version of dataset testing with concurrency control.
        
        Returns:
            Dictionary containing test results and statistics, None if failed
        """
        df = self.load_dataset(self.config.dataset_file)
        if df is None:
            return None
        
        # Determine model configurations to test - Fix the --models parameter issue  
        model_configs = self.config.model_configs
        if not model_configs:
            with open(self.model_config_path, 'r', encoding='utf-8') as f:
                model_config = json.load(f)
            model_configs = [model['config_name'] for model in model_config.get('chat', [])[:1]]
        
        logger.info(f"Starting asynchronous test with models: {model_configs} (max concurrent: {self.config.max_concurrent})")
        
        # Limit questions if specified
        if self.config.max_questions:
            df = df.head(self.config.max_questions)
        
        # Create tasks for all question-model combinations
        tasks = []
        
        for _, row in df.iterrows():
            # Determine question type
            question_type = self._determine_question_type(row)
            if question_type not in self.config.question_types:
                continue
                
            for model_config_name in model_configs:
                task = self.test_single_question_async(
                    model_config_name=model_config_name,
                    question=row['Question'],
                    expected_answer=row['Answer'],
                    question_type=question_type,
                    choices=row.get('Choices', None)
                )
                tasks.append(task)
        
        total_tasks = len(tasks)
        logger.info(f"Created {total_tasks} tasks for concurrent execution")
        
        # Process all tasks concurrently with progress tracking
        results = []
        completed = 0
        
        # Use asyncio.as_completed for progress tracking
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                results.append(result)
                completed += 1
                
                if completed % 10 == 0 or completed == total_tasks:
                    logger.info(f"Completed {completed}/{total_tasks} questions ({completed/total_tasks:.1%})")
                    
            except Exception as e:
                logger.error(f"Task failed: {e}")
                completed += 1
        
        # Calculate statistics
        stats = self._calculate_statistics(results)
        
        test_results = {
            'configuration': self.config.to_dict(),
            'results': [r.to_dict() for r in results],
            'statistics': stats,
            'metadata': {
                'test_date': datetime.now().isoformat(),
                'total_questions': len(results),
                'dataset_file': self.config.dataset_file,
                'execution_mode': 'async'
            }
        }
        
        return test_results
    
    def _determine_question_type(self, row: pd.Series) -> str:
        """Determine question type from dataset row.
        
        Args:
            row: Dataset row containing question information
            
        Returns:
            Question type string
        """
        # Check for explicit type column
        if 'Type' in row and pd.notna(row['Type']):
            question_type = str(row['Type']).lower()
            if 'choice' in question_type or 'multiple' in question_type:
                return "multiple_choice"
            elif 'fill' in question_type or 'blank' in question_type:
                return "fill_blank"
        
        # Infer from choices column
        if 'Choices' in row and pd.notna(row['Choices']):
            choices = str(row['Choices'])
            if any(option in choices.upper() for option in ['A.', 'B.', 'C.', 'D.']):
                return "multiple_choice"
        
        # Default to fill_blank if no clear indicators
        return "fill_blank"
    
    def _calculate_statistics(self, results: List[QuestionResult]) -> Dict[str, Any]:
        """Calculate comprehensive test statistics.
        
        Args:
            results: List of question results
            
        Returns:
            Dictionary containing detailed statistics
        """
        if not results:
            return {}
        
        # Overall statistics
        total_questions = len(results)
        successful_tests = [r for r in results if r.error is None]
        failed_tests = [r for r in results if r.error is not None]
        
        overall_stats = {
            'total_questions': total_questions,
            'successful_tests': len(successful_tests),
            'failed_tests': len(failed_tests),
            'success_rate': len(successful_tests) / total_questions if total_questions > 0 else 0
        }
        
        # Model-specific statistics
        model_stats = {}
        for result in successful_tests:
            model_name = result.model_config
            if model_name not in model_stats:
                model_stats[model_name] = {
                    'total': 0,
                    'correct': 0,
                    'total_time': 0.0,
                    'response_times': [],
                    'by_question_type': {}
                }
            
            stats = model_stats[model_name]
            stats['total'] += 1
            if result.is_correct:
                stats['correct'] += 1
            stats['total_time'] += result.response_time
            stats['response_times'].append(result.response_time)
            
            # Question type breakdown
            q_type = result.question_type
            if q_type not in stats['by_question_type']:
                stats['by_question_type'][q_type] = {'total': 0, 'correct': 0}
            stats['by_question_type'][q_type]['total'] += 1
            if result.is_correct:
                stats['by_question_type'][q_type]['correct'] += 1
        
        # Calculate derived metrics
        for model_name, stats in model_stats.items():
            stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            stats['avg_response_time'] = stats['total_time'] / stats['total'] if stats['total'] > 0 else 0
            
            # Response time statistics
            if stats['response_times']:
                stats['min_response_time'] = min(stats['response_times'])
                stats['max_response_time'] = max(stats['response_times'])
                stats['median_response_time'] = sorted(stats['response_times'])[len(stats['response_times'])//2]
            
            # Question type accuracy
            for q_type, type_stats in stats['by_question_type'].items():
                type_stats['accuracy'] = type_stats['correct'] / type_stats['total'] if type_stats['total'] > 0 else 0
        
        return {
            'overall': overall_stats,
            'by_model': model_stats
        }
    
    def print_results(self, test_results: Dict[str, Any]) -> None:
        """Print formatted test results to console.
        
        Args:
            test_results: Complete test results dictionary
        """
        stats = test_results.get('statistics', {})
        overall = stats.get('overall', {})
        by_model = stats.get('by_model', {})
        
        # Print header
        print("\n" + "="*80)
        print("LLM Testing Results Report")
        print("="*80)
        
        # Overall statistics
        print(f"\nOverall Results:")
        print(f"  Total Questions: {overall.get('total_questions', 0)}")
        print(f"  Successful Tests: {overall.get('successful_tests', 0)}")
        print(f"  Failed Tests: {overall.get('failed_tests', 0)}")
        print(f"  Success Rate: {overall.get('success_rate', 0):.2%}")
        
        # Model-specific results
        for model_name, model_stats in by_model.items():
            print(f"\nModel: {model_name}")
            print("-" * 50)
            print(f"  Questions Tested: {model_stats['total']}")
            print(f"  Correct Answers: {model_stats['correct']}")
            print(f"  Accuracy: {model_stats['accuracy']:.2%}")
            print(f"  Average Response Time: {model_stats['avg_response_time']:.2f}s")
            
            if 'min_response_time' in model_stats:
                print(f"  Response Time Range: {model_stats['min_response_time']:.2f}s - {model_stats['max_response_time']:.2f}s")
                print(f"  Median Response Time: {model_stats['median_response_time']:.2f}s")
            
            # Question type breakdown
            if 'by_question_type' in model_stats:
                print(f"  Question Type Breakdown:")
                for q_type, type_stats in model_stats['by_question_type'].items():
                    print(f"    {q_type.replace('_', ' ').title()}: {type_stats['correct']}/{type_stats['total']} ({type_stats['accuracy']:.2%})")
        
        print("\n" + "="*80)
    
    def save_results(self, test_results: Dict[str, Any]) -> bool:
        """Save test results to files.
        
        Args:
            test_results: Complete test results dictionary
            
        Returns:
            True if save successful, False otherwise
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Save detailed results if configured
            if self.config.save_detailed:
                detailed_file = self.results_dir / f"llm_test_detailed_{timestamp}.json"
                with open(detailed_file, 'w', encoding='utf-8') as f:
                    json.dump(test_results, f, ensure_ascii=False, indent=2)
                logger.info(f"Detailed results saved to {detailed_file}")
            
            # Save summary results if configured
            if self.config.save_summary:
                summary_data = {
                    'configuration': test_results['configuration'],
                    'statistics': test_results['statistics'],
                    'metadata': test_results['metadata']
                }
                summary_file = self.results_dir / f"llm_test_summary_{timestamp}.json"
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(summary_data, f, ensure_ascii=False, indent=2)
                logger.info(f"Summary results saved to {summary_file}")
            
            # Save CSV export for easy analysis
            self._save_csv_results(test_results, timestamp)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return False
    
    def _save_csv_results(self, test_results: Dict[str, Any], timestamp: str) -> None:
        """Save results in CSV format for easy analysis.
        
        Args:
            test_results: Complete test results dictionary
            timestamp: Timestamp string for filename
        """
        try:
            results_data = []
            for result_dict in test_results['results']:
                results_data.append({
                    'model_config': result_dict['model_config'],
                    'question_type': result_dict['question_type'],
                    'is_correct': result_dict['is_correct'],
                    'response_time': result_dict['response_time'],
                    'expected_answer': result_dict['expected_answer'],
                    'model_answer': result_dict['model_answer'],
                    'has_error': result_dict.get('error') is not None
                })
            
            if results_data:
                df = pd.DataFrame(results_data)
                csv_file = self.results_dir / f"llm_test_results_{timestamp}.csv"
                df.to_csv(csv_file, index=False)
                logger.info(f"CSV results saved to {csv_file}")
                
        except Exception as e:
            logger.error(f"Failed to save CSV results: {e}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="LLM Testing Script for OneSim",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default settings (async mode)
  python llm_test_script.py
  
  # Test specific models with limited questions
  python llm_test_script.py --models openai-gpt4o vllm-qwen --max-questions 50
  
  # High concurrency for fast testing
  python llm_test_script.py --max-concurrent 20
  
  # Synchronous mode for debugging
  python llm_test_script.py --no-async
  
  # Test only multiple choice questions
  python llm_test_script.py --question-types multiple_choice
  
  # Custom dataset and output directory
  python llm_test_script.py --dataset custom.csv --output-dir custom_results
"""
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="1000.csv",
        help="Dataset file name (default: 1000.csv)"
    )
    
    parser.add_argument(
        "--models",
        nargs="*",
        help="Model configuration names to test (default: use first available)"
    )
    
    parser.add_argument(
        "--max_questions",
        type=int,
        help="Maximum number of questions to test (default: all)",
        default=50
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_results",
        help="Output directory for results (default: test_results)"
    )
    
    parser.add_argument(
        "--question_types",
        nargs="*",
        choices=["multiple_choice", "fill_blank"],
        default=["multiple_choice", "fill_blank"],
        help="Question types to include (default: all types)"
    )
    
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout for model responses in seconds (default: 30.0)"
    )
    
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=10,
        help="Maximum concurrent requests (default: 10)"
    )
    
    parser.add_argument(
        "--no_async",
        action="store_true",
        help="Disable async processing (use synchronous mode)"
    )
    
    parser.add_argument(
        "--no_detailed",
        action="store_true",
        help="Skip saving detailed results"
    )
    
    parser.add_argument(
        "--no_summary",
        action="store_true",
        help="Skip saving summary results"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    
    return parser.parse_args()


def main() -> int:
    """Main entry point for the LLM testing script.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    args = parse_arguments()
    
    # Configure logging level
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    try:
        # Create test configuration
        config = TestConfiguration(
            dataset_file=args.dataset,
            model_configs=args.models,  # This should now work correctly
            max_questions=args.max_questions,
            output_dir=args.output_dir,
            save_detailed=not args.no_detailed,
            save_summary=not args.no_summary,
            question_types=set(args.question_types),
            timeout_seconds=args.timeout,
            max_concurrent=args.max_concurrent,
            use_async=not args.no_async
        )
        
        logger.info("Starting OneSim LLM Testing Script")
        logger.info(f"Execution mode: {'Async' if config.use_async else 'Sync'}")
        if config.use_async:
            logger.info(f"Max concurrent requests: {config.max_concurrent}")
        logger.info(f"Configuration: {config.to_dict()}")
        
        # Create tester instance
        tester = LLMTester(config)
        
        # Run tests
        test_results = tester.test_dataset()
        if test_results is None:
            logger.error("Testing failed")
            return 1
        
        tester.print_results(test_results)
        
        # Save results
        if not tester.save_results(test_results):
            logger.error("Failed to save results")
            return 1
        
        # Log performance summary
        stats = test_results.get('statistics', {})
        overall = stats.get('overall', {})
        total_questions = overall.get('total_questions', 0)
        successful_tests = overall.get('successful_tests', 0)
        
        logger.info(f"LLM testing completed successfully: {successful_tests}/{total_questions} tests passed")
        
        if config.use_async and total_questions > 0:
            # Estimate time saved by async processing
            by_model = stats.get('by_model', {})
            if by_model:
                avg_response_time = sum(model_stats.get('avg_response_time', 0) for model_stats in by_model.values()) / len(by_model)
                estimated_sync_time = total_questions * avg_response_time
                logger.info(f"Estimated time saved by async processing: {estimated_sync_time - (total_questions * avg_response_time / config.max_concurrent):.1f}s")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Testing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Testing failed with error: {e}")
        logger.debug("Full traceback:", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())