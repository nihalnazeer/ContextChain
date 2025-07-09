# tests/test_validator.py
import unittest
from app.engine.validator import validate_schema
import copy  # Import copy to create deep copies

class TestValidator(unittest.TestCase):
    def setUp(self):
        # Create a fresh copy of DEMO_SCHEMA for each test
        self.schema = copy.deepcopy(DEMO_SCHEMA)

    def test_valid_schema(self):
        # Test with the demo schema
        try:
            validate_schema(self.schema)
            self.assertTrue(True)  # No exception means success
        except ValueError as e:
            self.fail(f"Validation failed unexpectedly: {str(e)}")

    def test_missing_pipeline_id(self):
        # Test with missing pipeline_id
        invalid_schema = copy.deepcopy(self.schema)
        del invalid_schema["pipeline_id"]
        with self.assertRaises(ValueError) as context:
            validate_schema(invalid_schema)
        self.assertEqual(str(context.exception), "Missing required schema fields: pipeline_id")

    def test_invalid_task_id(self):
        # Test with invalid task_id (negative)
        invalid_schema = copy.deepcopy(self.schema)
        invalid_schema["tasks"][0]["task_id"] = -1
        with self.assertRaises(ValueError) as context:
            validate_schema(invalid_schema)
        self.assertEqual(str(context.exception), "Task -1 must have a positive integer task_id")

    def test_invalid_dependency(self):
        # Test with non-existent input task
        invalid_schema = copy.deepcopy(self.schema)
        invalid_schema["tasks"][1]["inputs"] = [99]  # Non-existent task
        with self.assertRaises(ValueError) as context:
            validate_schema(invalid_schema)
        self.assertEqual(str(context.exception), "Task 2 references non-existent or future input task 99")

    def test_invalid_task_type(self):
        # Test with invalid task_type
        invalid_schema = copy.deepcopy(self.schema)
        invalid_schema["tasks"][0]["task_type"] = "INVALID"
        with self.assertRaises(ValueError) as context:
            validate_schema(invalid_schema)
        self.assertEqual(str(context.exception), "Task 1 task_type must be one of GET, POST, PUT, LLM, LOCAL")

    def test_missing_llm_prompt(self):
        # Test LLM task without prompt_template
        invalid_schema = copy.deepcopy(self.schema)
        del invalid_schema["tasks"][2]["prompt_template"]
        with self.assertRaises(ValueError) as context:
            validate_schema(invalid_schema)
        self.assertEqual(str(context.exception), "Task 3 (LLM) requires a non-empty prompt_template")

# Demo schema (embedded)
DEMO_SCHEMA = {
    "pipeline_id": "demo_pipeline",
    "schema_version": "v1.0.0",
    "description": "Demo pipeline for testing executor and validator",
    "created_by": "test_user",
    "created_at": "2025-07-09T23:41:00Z",  # Updated to current time
    "tasks": [
        {
            "task_id": 1,
            "description": "Fetch test data",
            "task_type": "GET",
            "endpoint": "http://example.com/test",
            "inputs": [],
            "input_source": "http://example.com/source",
            "output_collection": "task_results"
        },
        {
            "task_id": 2,
            "description": "Process fetched data",
            "task_type": "LOCAL",
            "endpoint": "path.to.process",
            "inputs": [1],
            "output_collection": "task_results"
        },
        {
            "task_id": 3,
            "description": "Generate LLM summary",
            "task_type": "LLM",
            "endpoint": "llm.summary",
            "prompt_template": "Summarize: {input}",
            "inputs": [2],
            "output_collection": "task_results"
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