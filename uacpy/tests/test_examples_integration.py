"""
Integration tests for UACPY examples

Tests that key examples run without errors and produce expected outputs
"""

import pytest
import subprocess
import sys
import os
from pathlib import Path

# Examples directory
EXAMPLES_DIR = Path(__file__).parent.parent / 'examples'


class TestExamplesBasic:
    """Tests for basic examples (01-10)"""

    def test_example_helpers_import(self):
        """Test that example_helpers can be imported"""
        sys.path.insert(0, str(EXAMPLES_DIR))
        try:
            import example_helpers
            assert hasattr(example_helpers, 'create_example_report')
        finally:
            sys.path.remove(str(EXAMPLES_DIR))

    @pytest.mark.slow
    def test_example_01_runs(self):
        """Test example 01 runs without errors"""
        example_path = EXAMPLES_DIR / 'example_01_basic_shallow_water.py'
        if not example_path.exists():
            pytest.skip("Example 01 not found")

        # Set PYTHONPATH to find uacpy module
        env = os.environ.copy()
        env['PYTHONPATH'] = str(EXAMPLES_DIR.parent.parent)

        result = subprocess.run(
            [sys.executable, str(example_path)],
            cwd=str(EXAMPLES_DIR),
            capture_output=True,
            text=True,
            timeout=120,
            env=env
        )

        assert result.returncode == 0, f"Example 01 failed with error:\n{result.stderr}"
        assert "Example 1 complete" in result.stdout or "✓" in result.stdout


class TestExamplesImports:
    """Tests that examples have correct imports"""

    def test_example_helpers_only_in_01_10(self):
        """Verify example_helpers is only used in examples 01-10"""
        examples = list(EXAMPLES_DIR.glob('example_*.py'))

        for example_file in examples:
            # Extract example number
            name = example_file.stem
            if not name.startswith('example_'):
                continue

            try:
                num_str = name.split('_')[1]
                num = int(num_str[:2])  # Get first two digits
            except (IndexError, ValueError):
                continue

            with open(example_file, 'r') as f:
                content = f.read()

            uses_example_helpers = 'from example_helpers import' in content or \
                                 'import example_helpers' in content

            if num <= 10:
                # Examples 01-10 may use example_helpers OR official UACPY API
                # Both are acceptable
                pass
            else:
                # Examples 11+ should NOT use example_helpers
                assert not uses_example_helpers, \
                    f"{example_file.name} (11+) should not use example_helpers"


class TestExamplesDocumentation:
    """Tests for examples documentation"""

    def test_example_helpers_has_warnings(self):
        """Test example_helpers.py has proper warnings"""
        helpers = EXAMPLES_DIR / 'example_helpers.py'
        if not helpers.exists():
            pytest.skip("example_helpers.py not found")

        with open(helpers, 'r') as f:
            content = f.read()

        assert '⚠️ IMPORTANT' in content, "example_helpers.py missing warning"
        assert 'examples-only' in content.lower() or 'examples directory' in content.lower()
        assert 'quickplot' in content, "example_helpers.py should mention quickplot"


class TestExamplesOutputs:
    """Tests that examples create expected outputs"""

    def test_example_output_directory_exists(self):
        """Test output directory exists"""
        output_dir = EXAMPLES_DIR / 'output'
        assert output_dir.exists(), "examples/output directory not found"
        assert output_dir.is_dir(), "examples/output is not a directory"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
