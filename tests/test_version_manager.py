import unittest
from app.registry.version_manager import VersionManager

class TestVersionManager(unittest.TestCase):
    def setUp(self):
        self.vm = VersionManager()

    def test_get_version_supported(self):
        schema = {"schema_version": "v1.0.0"}
        self.assertEqual(self.vm.get_version(schema), "v1.0.0")

    def test_get_version_unsupported_fallback(self):
        schema = {"schema_version": "v2.0.0"}
        self.assertEqual(self.vm.get_version(schema), "v1.0.0")  # fallback

    def test_get_raw_version(self):
        schema = {"schema_version": "v9.9.9"}
        self.assertEqual(self.vm.get_raw_version(schema), "v9.9.9")

    def test_is_compatible_true(self):
        schema = {"schema_version": "v1.0.0"}
        self.assertTrue(self.vm.is_compatible(schema))

    def test_is_compatible_false(self):
        schema = {"schema_version": "v9.9.9"}
        self.assertFalse(self.vm.is_compatible(schema))

    def test_update_version_success(self):
        schema = {"schema_version": "v1.0.0"}
        updated = self.vm.update_version(schema, "v1.0.0")
        self.assertEqual(updated["schema_version"], "v1.0.0")

    def test_update_version_fail(self):
        schema = {"schema_version": "v1.0.0"}
        with self.assertRaises(ValueError):
            self.vm.update_version(schema, "v2.0.0")

if __name__ == '__main__':
    unittest.main()