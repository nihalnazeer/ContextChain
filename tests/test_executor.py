import unittest
from unittest.mock import patch, MagicMock
import requests
from app.engine.executor import execute_pipeline, execute_single_task
from app.registry.version_manager import VersionManager
import copy

class TestExecutor(unittest.TestCase):
    def setUp(self):
        self.vm = VersionManager()
        try:
            self.client = self.vm.get_client()  # Get client with error handling
        except RuntimeError as e:
            self.client = None  # Fallback for testing if connection fails
            self.skipTest(str(e))
        self.db_name = "test_db"
        self.schema = copy.deepcopy(DEMO_SCHEMA)

    def tearDown(self):
        if self.client is not None:
            self.client[self.db_name]["task_results"].delete_many({})
            self.client[self.db_name]["trigger_logs"].delete_many({})
        self.vm.close()  # Close the managed MongoClient

    @patch("app.engine.executor.execute_local")
    @patch("app.engine.executor.execute_http")
    @patch("app.engine.executor.execute_llm")
    def test_execute_pipeline(self, mock_llm, mock_http, mock_local):
        if self.client is None:
            self.skipTest("MongoClient not available")
        # Mock task executions
        mock_http.return_value = {"status": "success", "output": {"data": "test"}, "task_id": 1}
        mock_local.return_value = {"status": "success", "output": "processed", "task_id": 2}
        mock_llm.return_value = {"status": "success", "output": "summary", "task_id": 3}

        execute_pipeline(self.client, self.db_name, self.schema)

        results = list(self.client[self.db_name]["task_results"].find())
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["task_id"], 1)
        self.assertEqual(results[1]["task_id"], 2)
        self.assertEqual(results[2]["task_id"], 3)

    @patch("app.engine.executor.execute_local")
    def test_execute_single_task(self, mock_local):
        if self.client is None:
            self.skipTest("MongoClient not available")
        mock_local.return_value = {"status": "success", "output": "processed", "task_id": 2}

        task = self.schema["tasks"][1]  # Task 2 (LOCAL)
        context = {1: {"status": "success", "output": "test_data"}}
        result = execute_single_task(self.client, self.db_name, self.schema, task, context)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["task_id"], 2)
        self.assertIn("output", result)

        result_doc = self.client[self.db_name]["task_results"].find_one({"task_id": 2})
        self.assertIsNotNone(result_doc)

    @patch("app.engine.executor.execute_http")
    def test_execute_single_task_with_retry(self, mock_http):
        if self.client is None:
            self.skipTest("MongoClient not available")
        # Ensure side_effect matches expected retries (e.g., 2 failures + 1 success)
        mock_http.side_effect = [Exception("Failed")] * 2 + [{"status": "success", "output": "data", "task_id": 1}]

        task = self.schema["tasks"][0]  # Task 1 (GET)
        result = execute_single_task(self.client, self.db_name, self.schema, task)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["task_id"], 1)

    @patch("app.engine.executor.execute_http")
    def test_execute_api_task(self, mock_http):
        if self.client is None:
            self.skipTest("MongoClient not available")
        # Define a schema with an API task and a dependent task
        api_schema = {
            "pipeline_id": "api_test_pipeline",
            "schema_version": "v1.0.0",
            "description": "Test pipeline for API task",
            "created_by": "test_user",
            "created_at": "2025-07-10T03:38:00Z",  # Updated to current time
            "tasks": [
                {
                    "task_id": 1,
                    "description": "Fetch API data",
                    "task_type": "GET",
                    "endpoint": "https://api.example.com/data",
                    "inputs": [],
                    "input_source": "https://api.example.com/source",
                    "output_collection": "task_results",
                    "wait_for_input": False
                },
                {
                    "task_id": 2,
                    "description": "Process API data",
                    "task_type": "LOCAL",
                    "endpoint": "path.to.process_api",
                    "inputs": [1],
                    "output_collection": "task_results",
                    "wait_for_input": False
                }
            ],
            "global_config": {
                "default_output_db": "test_db",
                "logging_level": "INFO",
                "retry_on_failure": True,
                "max_retries": 2,
                "allowed_task_types": ["GET", "LOCAL"],
                "allowed_domains": ["api.example.com"]
            },
            "metadata": {
                "tags": ["api", "test"],
                "pipeline_type": "api-pipeline",
                "linked_pipelines": []
            }
        }

        # Mock the API response
        mock_http.return_value = {
            "status": "success",
            "output": {"api_data": "sample_data"},
            "task_id": 1
        }

        # Execute the pipeline
        execute_pipeline(self.client, self.db_name, api_schema)

        # Verify results
        results = list(self.client[self.db_name]["task_results"].find())
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["task_id"], 1)
        self.assertEqual(results[0]["output"]["api_data"], "sample_data")
        self.assertEqual(results[1]["task_id"], 2)
        self.assertIn("output", results[1])  # Verify processing occurred

        # Optionally check MongoDB storage
        task1_doc = self.client[self.db_name]["task_results"].find_one({"task_id": 1})
        self.assertIsNotNone(task1_doc)
        self.assertEqual(task1_doc["output"]["api_data"], "sample_data")

    def test_execute_real_api_call(self):
        if self.client is None:
            self.skipTest("MongoClient not available")
        # Define a schema for a real API call (using a public test API)
        real_api_schema = {
            "pipeline_id": "real_api_test_pipeline",
            "schema_version": "v1.0.0",
            "description": "Test pipeline for real API call",
            "created_by": "test_user",
            "created_at": "2025-07-10T03:38:00Z",  # Current time
            "tasks": [
                {
                    "task_id": 1,
                    "description": "Fetch real API data",
                    "task_type": "GET",
                    "endpoint": "https://jsonplaceholder.typicode.com/posts/1",
                    "inputs": [],
                    "input_source": "https://jsonplaceholder.typicode.com",
                    "output_collection": "task_results",
                    "wait_for_input": False
                }
            ],
            "global_config": {
                "default_output_db": "test_db",
                "logging_level": "INFO",
                "retry_on_failure": True,
                "max_retries": 2,
                "allowed_task_types": ["GET"],
                "allowed_domains": ["jsonplaceholder.typicode.com"]
            },
            "metadata": {
                "tags": ["api", "real"],
                "pipeline_type": "api-pipeline",
                "linked_pipelines": []
            }
        }

        # Patch execute_http to use requests (unmocked for real call)
        with patch("app.engine.executor.execute_http") as mock_http:
            def real_http_call(task):
                try:
                    response = requests.get(task["endpoint"])
                    response.raise_for_status()  # Raise exception for bad status codes
                    return {
                        "status": "success",
                        "output": response.json(),
                        "task_id": task["task_id"]
                    }
                except requests.RequestException as e:
                    return {"status": "failure", "output": str(e), "task_id": task["task_id"]}

            mock_http.side_effect = real_http_call

            # Execute the pipeline
            execute_pipeline(self.client, self.db_name, real_api_schema)

            # Verify results
            results = list(self.client[self.db_name]["task_results"].find())
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["task_id"], 1)
            self.assertEqual(results[0]["status"], "success")
            self.assertIn("title", results[0]["output"])  # JSONPlaceholder posts have a "title" field

            task1_doc = self.client[self.db_name]["task_results"].find_one({"task_id": 1})
            self.assertIsNotNone(task1_doc)
            self.assertEqual(task1_doc["status"], "success")

# Demo schema (embedded for other tests)
DEMO_SCHEMA = {
    "pipeline_id": "demo_pipeline",
    "schema_version": "v1.0.0",
    "description": "Demo pipeline for testing executor and validator",
    "created_by": "test_user",
    "created_at": "2025-07-10T03:38:00Z",  # Updated to current time
    "tasks": [
        {
            "task_id": 1,
            "description": "Fetch test data",
            "task_type": "GET",
            "endpoint": "http://example.com/test",
            "inputs": [],
            "input_source": "http://example.com/source",
            "output_collection": "task_results",
            "wait_for_input": False
        },
        {
            "task_id": 2,
            "description": "Process fetched data",
            "task_type": "LOCAL",
            "endpoint": "path.to.process",
            "inputs": [1],
            "output_collection": "task_results",
            "wait_for_input": False
        },
        {
            "task_id": 3,
            "description": "Generate LLM summary",
            "task_type": "LLM",
            "endpoint": "llm.summary",
            "prompt_template": "Summarize: {input}",
            "inputs": [2],
            "output_collection": "task_results",
            "wait_for_input": False
        }
    ],
    "global_config": {
        "default_output_db": "test_db",
        "logging_level": "INFO",
        "retry_on_failure": True,
        "max_retries": 2,
        "allowed_task_types": ["GET", "LOCAL", "LLM"],
        "allowed_domains": ["example.com"]
    },
    "metadata": {
        "tags": ["test", "demo"],
        "pipeline_type": "fullstack-ai",
        "linked_pipelines": []
    }
}

if __name__ == "__main__":
    unittest.main()