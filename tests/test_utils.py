import pytest
import sys
sys.path.append('./src')
from utils import (
    create_new_file_with_prefix,
    get_files_with_prefix
)

def test_create_new_file_with_prefix():
    # Test case 1: Valid prefix and files
    prefix = "file"
    files = ["file1.txt", "file2.txt", "file2.csv"]
    expected_result = "file3"
    assert create_new_file_with_prefix(prefix, files) == expected_result

    # Test case 2: Empty prefix
    prefix = "file"
    files = ["file1.txt", "filedsf.txt"]
    expected_result = "file2"
    assert create_new_file_with_prefix(prefix, files) == expected_result

    # Test case 3: Empty files list
    prefix = "new"
    files = []
    expected_result = "new1"
    assert create_new_file_with_prefix(prefix, files) == expected_result

    # Test case 4: Invalid prefix and files
    prefix = "file"
    files = ["file1.txt", "file100.txt"]
    expected_result = "file101"
    assert create_new_file_with_prefix(prefix, files) == expected_result


def test_get_files_with_prefix():
    # Test case 1: Valid prefix and files
    prefix = "file"
    files = ["file1.txt", "file2.txt", "file2.csv"]
    expected_result = []
    assert get_files_with_prefix(prefix, files) == expected_result

    # Test case 2: Empty prefix
    prefix = "file"
    files = ["file1.txt", "filedsf.txt", "file.csv", "file.txt"]
    expected_result = ["file.csv", "file.txt"]
    assert get_files_with_prefix(prefix, files) == expected_result

    # Test case 3: Empty files list
    prefix = "new"
    files = []
    expected_result = []
    assert get_files_with_prefix(prefix, files) == expected_result

    # Test case 4: Invalid prefix and files
    prefix = "file1"
    files = ["file1.txt", "file100.txt"]
    expected_result = ["file1.txt"]
    assert get_files_with_prefix(prefix, files) == expected_result