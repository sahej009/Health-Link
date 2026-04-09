"""
Helper utilities for HealthLink.
General-purpose utility functions.
"""
import json
import hashlib
from datetime import datetime, date
from typing import Any, Dict, Optional


def generate_hash(text: str, algorithm: str = "sha256") -> str:
    """
    Generate hash of text.

    Args:
        text: Text to hash
        algorithm: Hash algorithm (md5, sha256, etc.)

    Returns:
        Hexadecimal hash string
    """
    hasher = hashlib.new(algorithm)
    hasher.update(text.encode('utf-8'))
    return hasher.hexdigest()


def sanitize_input(text: str, max_length: int = 5000) -> str:
    """
    Sanitize user input.

    Args:
        text: Input text
        max_length: Maximum allowed length

    Returns:
        Sanitized text
    """
    # Remove leading/trailing whitespace
    sanitized = text.strip()

    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    # Remove control characters
    sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char == '\n')

    return sanitized


def format_datetime(dt: datetime, format_string: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format datetime to string.

    Args:
        dt: Datetime object
        format_string: Format string

    Returns:
        Formatted datetime string
    """
    return dt.strftime(format_string)


def parse_datetime(date_string: str, format_string: str = "%Y-%m-%d %H:%M:%S") -> Optional[datetime]:
    """
    Parse datetime from string.

    Args:
        date_string: Date string
        format_string: Format string

    Returns:
        Datetime object or None if parsing fails
    """
    try:
        return datetime.strptime(date_string, format_string)
    except ValueError:
        return None


def to_json(obj: Any, indent: int = 2) -> str:
    """
    Convert object to JSON string.

    Handles datetime objects and Pydantic models.

    Args:
        obj: Object to convert
        indent: JSON indentation

    Returns:
        JSON string
    """
    def default_handler(x):
        if isinstance(x, (datetime, date)):
            return x.isoformat()
        if hasattr(x, 'model_dump'):  # Pydantic v2
            return x.model_dump()
        if hasattr(x, 'dict'):  # Pydantic v1
            return x.dict()
        raise TypeError(f"Object of type {type(x)} is not JSON serializable")

    return json.dumps(obj, default=default_handler, indent=indent)


def from_json(json_string: str) -> Any:
    """
    Parse JSON string to object.

    Args:
        json_string: JSON string

    Returns:
        Parsed object
    """
    return json.loads(json_string)


def dict_to_query_string(params: Dict[str, Any]) -> str:
    """
    Convert dictionary to URL query string.

    Args:
        params: Parameters dictionary

    Returns:
        Query string
    """
    pairs = [f"{key}={value}" for key, value in params.items() if value is not None]
    return "&".join(pairs)


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to append if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


def batch_items(items: list, batch_size: int) -> list:
    """
    Split list into batches.

    Args:
        items: List to batch
        batch_size: Size of each batch

    Returns:
        List of batches

    Example:
        >>> batch_items([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
    """
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    return batches


def merge_dicts(*dicts: Dict) -> Dict:
    """
    Merge multiple dictionaries.

    Later dictionaries override earlier ones.

    Args:
        *dicts: Dictionaries to merge

    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def get_nested_value(data: Dict, path: str, default: Any = None) -> Any:
    """
    Get value from nested dictionary using dot notation.

    Args:
        data: Dictionary to search
        path: Dot-separated path (e.g., "user.profile.name")
        default: Default value if path not found

    Returns:
        Value at path or default

    Example:
        >>> get_nested_value({"user": {"name": "John"}}, "user.name")
        'John'
    """
    keys = path.split('.')
    value = data

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value
