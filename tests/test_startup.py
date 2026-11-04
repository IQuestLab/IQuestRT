"""CPU tests for the deployment checks that run before weights load."""

import tempfile
import unittest
from unittest import mock


class BuildToolTest(unittest.TestCase):
    def test_missing_python_headers_fail_before_loading(self):
        from iquestrt.startup import check_build_tools

        with tempfile.TemporaryDirectory() as empty, \
                mock.patch("shutil.which", return_value="/usr/bin/tool"), \
                mock.patch("sysconfig.get_path", return_value=empty):
            with self.assertRaisesRegex(RuntimeError, "python3-dev"):
                check_build_tools()


if __name__ == "__main__":
    unittest.main()
