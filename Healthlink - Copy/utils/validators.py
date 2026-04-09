"""
Input validation utilities for HealthLink.
"""
import re
from typing import Tuple, Optional
from datetime import datetime


def validate_email(email: str) -> Tuple[bool, str]:
    """
    Validate email format.

    Args:
        email: Email address to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not email:
        return False, "Email is required"

    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

    if not re.match(pattern, email):
        return False, "Invalid email format"

    return True, ""


def validate_phone(phone: str) -> Tuple[bool, str]:
    """
    Validate phone number format.

    Accepts various formats: (123) 456-7890, 123-456-7890, 1234567890

    Args:
        phone: Phone number to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not phone:
        return False, "Phone number is required"

    # Remove common separators
    cleaned = re.sub(r'[\s\-\(\)]', '', phone)

    # Check if it's all digits and appropriate length
    if not cleaned.isdigit():
        return False, "Phone number must contain only digits"

    if len(cleaned) < 10 or len(cleaned) > 15:
        return False, "Phone number must be between 10 and 15 digits"

    return True, ""


def validate_date(date_string: str, date_format: str = "%Y-%m-%d") -> Tuple[bool, str]:
    """
    Validate date string format.

    Args:
        date_string: Date string to validate
        date_format: Expected date format

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not date_string:
        return False, "Date is required"

    try:
        datetime.strptime(date_string, date_format)
        return True, ""
    except ValueError:
        return False, f"Invalid date format. Expected {date_format}"


def validate_text_length(text: str, min_length: int = 0, max_length: int = 10000) -> Tuple[bool, str]:
    """
    Validate text length.

    Args:
        text: Text to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text:
        if min_length > 0:
            return False, f"Text must be at least {min_length} characters"
        return True, ""

    text_len = len(text.strip())

    if text_len < min_length:
        return False, f"Text must be at least {min_length} characters"

    if text_len > max_length:
        return False, f"Text must be no more than {max_length} characters"

    return True, ""


def validate_user_input(user_input: str) -> Tuple[bool, str]:
    """
    Validate user health input.

    Args:
        user_input: User's health concern description

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check minimum length
    is_valid, error = validate_text_length(user_input, min_length=10, max_length=5000)
    if not is_valid:
        return is_valid, error

    # Check for suspicious patterns (very basic)
    suspicious_patterns = [
        r"<script",
        r"javascript:",
        r"onclick=",
        r"onerror=",
    ]

    for pattern in suspicious_patterns:
        if re.search(pattern, user_input, re.IGNORECASE):
            return False, "Input contains potentially harmful content"

    return True, ""


def validate_rating(rating: float) -> Tuple[bool, str]:
    """
    Validate rating value.

    Args:
        rating: Rating value

    Returns:
        Tuple of (is_valid, error_message)
    """
    if rating < 0 or rating > 5:
        return False, "Rating must be between 0 and 5"

    return True, ""


def validate_required_fields(data: dict, required_fields: list) -> Tuple[bool, str]:
    """
    Validate that required fields are present in dictionary.

    Args:
        data: Dictionary to validate
        required_fields: List of required field names

    Returns:
        Tuple of (is_valid, error_message)
    """
    missing_fields = [field for field in required_fields if field not in data or not data[field]]

    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"

    return True, ""


def sanitize_sql_input(text: str) -> str:
    """
    Basic SQL injection prevention through input sanitization.

    Note: This is a basic implementation. Always use parameterized queries.

    Args:
        text: Input text

    Returns:
        Sanitized text
    """
    # Remove common SQL injection patterns
    dangerous_patterns = [
        r"--",
        r";",
        r"'",
        r'"',
        r"/*",
        r"*/",
        r"xp_",
        r"sp_",
        r"DROP",
        r"INSERT",
        r"DELETE",
        r"UPDATE",
        r"UNION",
    ]

    sanitized = text
    for pattern in dangerous_patterns:
        sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)

    return sanitized
