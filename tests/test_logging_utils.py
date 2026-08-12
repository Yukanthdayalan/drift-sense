"""
Unit tests for the drift_sense.logging_utils module.
"""
import unittest
import logging
import io
import os
import tempfile
from unittest.mock import patch

from drift_sense.logging_utils import get_logger

class TestLoggingUtils(unittest.TestCase):
    
    def test_get_logger_creation(self):
        """Verify logger is created with correct name, level, and handlers."""
        logger = get_logger("test_module", level=logging.DEBUG)
        self.assertEqual(logger.name, "test_module")
        self.assertEqual(logger.level, logging.DEBUG)
        self.assertFalse(logger.propagate)
        self.assertEqual(len(logger.handlers), 1)
        self.assertIsInstance(logger.handlers[0], logging.StreamHandler)
        
    @patch('sys.stderr', new_callable=io.StringIO)
    def test_logger_output_format(self, mock_stderr):
        """Verify the structured formatting of the log output."""
        logger = get_logger("format_test", level=logging.INFO)
        logger.info("Test message")
        
        output = mock_stderr.getvalue()
        self.assertIn("INFO", output)
        self.assertIn("format_test", output)
        self.assertIn("Test message", output)
        
    def test_logger_file_handler(self):
        """Verify the logger correctly attaches and writes to a file handler."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            log_path = os.path.join(tmpdirname, "test.log")
            logger = get_logger("file_test", level=logging.INFO, log_file=log_path)
            
            self.assertEqual(len(logger.handlers), 2)
            
            logger.info("File test message")
            
            # Flush and close handlers to ensure write completes and file lock is released
            for handler in logger.handlers:
                handler.flush()
                handler.close()
                
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            self.assertIn("INFO", content)
            self.assertIn("file_test", content)
            self.assertIn("File test message", content)

if __name__ == '__main__':
    unittest.main()
