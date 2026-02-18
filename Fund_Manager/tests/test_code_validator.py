"""
Tests for the code_validator module.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code_validator import (
    validate_syntax,
    validate_required_imports,
    validate_and_fix_code,
    extract_imports,
    extract_used_names,
    inject_missing_imports,
    remove_markdown_artifacts,
    ensure_code_starts_with_imports,
    check_datetime_import,
    create_strategy_template
)


class TestValidateSyntax:
    """Tests for the validate_syntax function."""

    def test_valid_code_passes(self):
        """Test that valid code passes syntax validation."""
        code = '''
def test():
    return True
'''
        is_valid, error = validate_syntax(code)
        assert is_valid is True
        assert error is None

    def test_syntax_error_detected(self):
        """Test that syntax errors are detected."""
        bad_code = '''
def broken(
    print("missing closing paren"
'''
        is_valid, error = validate_syntax(bad_code)
        assert is_valid is False
        assert "Syntax error" in error

    def test_indentation_error_detected(self):
        """Test that indentation errors are detected."""
        bad_code = '''
def test():
print("bad indent")
'''
        is_valid, error = validate_syntax(bad_code)
        assert is_valid is False

    def test_empty_code_is_valid(self):
        """Test that empty code is valid syntax."""
        is_valid, error = validate_syntax("")
        assert is_valid is True
        assert error is None


class TestExtractImports:
    """Tests for the extract_imports function."""

    def test_extract_import_statements(self):
        """Test extraction of import statements."""
        code = '''
import pandas as pd
import numpy as np
import os
'''
        imports = extract_imports(code)
        assert 'pandas' in imports
        assert 'numpy' in imports
        assert 'os' in imports

    def test_extract_from_imports(self):
        """Test extraction of from...import statements."""
        code = '''
from datetime import datetime, time
from queue import Queue
'''
        imports = extract_imports(code)
        assert 'datetime' in imports
        assert 'queue' in imports

    def test_empty_code_returns_empty_set(self):
        """Test that empty code returns empty set."""
        imports = extract_imports("")
        assert imports == set()

    def test_code_with_syntax_errors(self):
        """Test extraction from code with syntax errors (uses regex fallback)."""
        code = '''
import pandas as pd
def broken(
'''
        imports = extract_imports(code)
        assert 'pandas' in imports


class TestExtractUsedNames:
    """Tests for the extract_used_names function."""

    def test_extract_name_usage(self):
        """Test extraction of used names."""
        code = '''
import pandas as pd
df = pd.DataFrame()
'''
        names = extract_used_names(code)
        assert 'pd' in names
        assert 'df' in names

    def test_extract_module_prefixes(self):
        """Test extraction of module prefixes like np., os., etc."""
        code = '''
result = np.array([1, 2, 3])
path = os.path.join('a', 'b')
'''
        names = extract_used_names(code)
        assert 'np' in names
        assert 'os' in names


class TestValidateRequiredImports:
    """Tests for the validate_required_imports function."""

    def test_all_imports_present(self):
        """Test when all required imports are present."""
        code = '''
import pandas as pd
import numpy as np

df = pd.DataFrame()
arr = np.array([1, 2, 3])
'''
        is_valid, missing = validate_required_imports(code)
        assert is_valid is True
        assert len(missing) == 0

    def test_missing_pandas_detected(self):
        """Test detection of missing pandas import."""
        code = '''
df = pd.DataFrame()
'''
        is_valid, missing = validate_required_imports(code)
        assert is_valid is False
        assert 'pandas' in missing

    def test_missing_numpy_detected(self):
        """Test detection of missing numpy import."""
        code = '''
arr = np.array([1, 2, 3])
'''
        is_valid, missing = validate_required_imports(code)
        assert is_valid is False
        assert 'numpy' in missing

    def test_missing_warnings_detected(self):
        """Test detection of missing warnings import."""
        code = '''
warnings.filterwarnings('ignore')
'''
        is_valid, missing = validate_required_imports(code)
        assert is_valid is False
        assert 'warnings' in missing

    def test_missing_sys_detected(self):
        """Test detection of missing sys import."""
        code = '''
sys.path.insert(0, '/some/path')
'''
        is_valid, missing = validate_required_imports(code)
        assert is_valid is False
        assert 'sys' in missing

    def test_missing_os_detected(self):
        """Test detection of missing os import."""
        code = '''
path = os.path.join('a', 'b')
'''
        is_valid, missing = validate_required_imports(code)
        assert is_valid is False
        assert 'os' in missing


class TestInjectMissingImports:
    """Tests for the inject_missing_imports function."""

    def test_inject_single_import(self):
        """Test injecting a single missing import."""
        code = "df = pd.DataFrame()"
        fixed = inject_missing_imports(code, ['pandas'])
        assert 'import pandas as pd' in fixed
        assert 'df = pd.DataFrame()' in fixed

    def test_inject_multiple_imports(self):
        """Test injecting multiple missing imports."""
        code = "result = np.array([1, 2, 3])"
        fixed = inject_missing_imports(code, ['numpy', 'warnings'])
        assert 'import numpy as np' in fixed
        assert 'import warnings' in fixed

    def test_no_missing_imports(self):
        """Test with no missing imports."""
        code = "x = 1"
        fixed = inject_missing_imports(code, [])
        assert fixed == code


class TestRemoveMarkdownArtifacts:
    """Tests for the remove_markdown_artifacts function."""

    def test_removes_code_blocks(self):
        """Test removal of markdown code blocks."""
        code = '''```python
def test():
    return True
```'''
        cleaned = remove_markdown_artifacts(code)
        assert '```' not in cleaned
        assert 'def test():' in cleaned

    def test_removes_preamble_text(self):
        """Test removal of LLM preamble text."""
        code = '''Here's the code:

def test():
    return True'''
        cleaned = remove_markdown_artifacts(code)
        assert "Here's the code:" not in cleaned

    def test_plain_code_unchanged(self):
        """Test that plain code is mostly unchanged."""
        code = '''def test():
    return True'''
        cleaned = remove_markdown_artifacts(code)
        assert 'def test():' in cleaned
        assert 'return True' in cleaned


class TestEnsureCodeStartsWithImports:
    """Tests for the ensure_code_starts_with_imports function."""

    def test_code_starting_with_import(self):
        """Test code that already starts with import."""
        code = '''import pandas as pd

def test():
    pass'''
        result = ensure_code_starts_with_imports(code)
        assert result == code

    def test_code_starting_with_from(self):
        """Test code that starts with from...import."""
        code = '''from datetime import datetime

def test():
    pass'''
        result = ensure_code_starts_with_imports(code)
        assert result == code

    def test_code_with_leading_text(self):
        """Test code with leading non-code text."""
        code = '''Here is the code:

import pandas as pd

def test():
    pass'''
        result = ensure_code_starts_with_imports(code)
        assert result.strip().startswith('import pandas')


class TestCheckDatetimeImport:
    """Tests for the check_datetime_import function."""

    def test_correct_import_passes(self):
        """Test code with correct datetime.time import."""
        code = '''from datetime import time

start_time = time(9, 30)'''
        is_valid, fixed = check_datetime_import(code)
        assert is_valid is True

    def test_missing_time_import_fixed(self):
        """Test that missing datetime.time import is fixed."""
        code = '''start_time = time(9, 30)'''
        is_valid, fixed = check_datetime_import(code)
        assert is_valid is False
        assert 'from datetime import time' in fixed


class TestValidateAndFixCode:
    """Tests for the validate_and_fix_code function."""

    def test_valid_code_passes(self, sample_strategy_code):
        """Test that valid code passes validation."""
        is_valid, fixed_code, issues = validate_and_fix_code(sample_strategy_code)
        assert is_valid is True

    def test_fixes_markdown_artifacts(self):
        """Test that markdown artifacts are fixed."""
        code = '''```python
import pandas as pd

def test():
    return True
```'''
        is_valid, fixed_code, issues = validate_and_fix_code(code)
        assert is_valid is True
        assert '```' not in fixed_code
        assert any("markdown" in issue.lower() for issue in issues)

    def test_fixes_missing_imports(self):
        """Test that missing imports are added."""
        code = '''
def test():
    df = pd.DataFrame()
    return df
'''
        is_valid, fixed_code, issues = validate_and_fix_code(code)
        assert 'import pandas as pd' in fixed_code
        assert any("missing imports" in issue.lower() for issue in issues)

    def test_syntax_errors_not_fixed(self):
        """Test that syntax errors cannot be auto-fixed."""
        bad_code = '''
def broken(
    print("missing paren"
'''
        is_valid, fixed_code, issues = validate_and_fix_code(bad_code)
        assert is_valid is False
        assert any("syntax" in issue.lower() for issue in issues)

    def test_complex_valid_code(self):
        """Test validation of complex valid code."""
        complex_code = '''
import warnings
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

warnings.filterwarnings('ignore')

class Strategy:
    def __init__(self, params):
        self.params = params

    def calculate_signal(self, data):
        sma = data['close'].rolling(self.params['period']).mean()
        return np.where(data['close'] > sma, 1, -1)

def run_backtest(start_date, end_date):
    df = pd.DataFrame()
    strategy = Strategy({'period': 20})
    signals = strategy.calculate_signal(df)
    return signals

if __name__ == "__main__":
    run_backtest("2024-01-01", "2024-12-31")
'''
        is_valid, fixed_code, issues = validate_and_fix_code(complex_code)
        assert is_valid is True


class TestCreateStrategyTemplate:
    """Tests for the create_strategy_template function."""

    def test_default_template(self):
        """Test creating default template."""
        template = create_strategy_template()
        assert 'class GeneratedStrategy' in template
        assert 'import pandas as pd' in template
        assert 'import numpy as np' in template
        assert 'from datetime import datetime, time' in template

    def test_custom_class_name(self):
        """Test creating template with custom class name."""
        template = create_strategy_template("MyCustomStrategy")
        assert 'class MyCustomStrategy' in template
        assert 'class GeneratedStrategy' not in template

    def test_template_is_valid_python(self):
        """Test that generated template is valid Python."""
        template = create_strategy_template()
        is_valid, error = validate_syntax(template)
        assert is_valid is True
