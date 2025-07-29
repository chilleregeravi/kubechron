"""
Retry utilities with exponential backoff for robust network operations.
Handles transient failures gracefully with configurable retry policies.
"""
import time
import asyncio
from typing import Any, Callable, Optional, Type, Tuple
from functools import wraps
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError, HTTPError
from config import settings
from logger import logger


class RetryableError(Exception):
    """Base exception for errors that should trigger a retry."""
    pass


class NonRetryableError(Exception):
    """Exception for errors that should not trigger a retry."""
    pass


# Define which HTTP status codes should be retried
RETRYABLE_STATUS_CODES = {500, 502, 503, 504, 429}  # Server errors and rate limits
NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404}  # Client errors


def is_retryable_error(exception: Exception) -> bool:
    """Determine if an error should trigger a retry attempt."""
    if isinstance(exception, NonRetryableError):
        return False
    
    if isinstance(exception, RetryableError):
        return True
    
    # Handle HTTP errors
    if isinstance(exception, HTTPError):
        if hasattr(exception, 'response') and exception.response is not None:
            status_code = exception.response.status_code
            return status_code in RETRYABLE_STATUS_CODES
    
    # Handle network-related errors (always retryable)
    if isinstance(exception, (ConnectionError, Timeout)):
        return True
    
    # Default to not retrying unknown errors
    return False


def retry_with_backoff(
    max_attempts: Optional[int] = None,
    initial_delay: Optional[float] = None,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts (uses config default if None)
        initial_delay: Initial delay between retries (uses config default if None)
        backoff_factor: Multiplier for delay between retries
        max_delay: Maximum delay between retries
        exceptions: Tuple of exception types to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = max_attempts or settings.retry_attempts
            delay = initial_delay or settings.retry_delay
            
            last_exception = None
            
            for attempt in range(attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if not is_retryable_error(e):
                        logger.warning(
                            "non_retryable_error",
                            function=func.__name__,
                            attempt=attempt + 1,
                            error=str(e),
                            error_type=type(e).__name__
                        )
                        raise
                    
                    if attempt == attempts - 1:  # Last attempt
                        logger.error(
                            "max_retries_exceeded",
                            function=func.__name__,
                            max_attempts=attempts,
                            final_error=str(e),
                            error_type=type(e).__name__
                        )
                        break
                    
                    # Calculate delay with exponential backoff
                    current_delay = min(delay * (backoff_factor ** attempt), max_delay)
                    
                    logger.warning(
                        "retrying_after_error",
                        function=func.__name__,
                        attempt=attempt + 1,
                        max_attempts=attempts,
                        delay=current_delay,
                        error=str(e),
                        error_type=type(e).__name__
                    )
                    
                    time.sleep(current_delay)
            
            # If we get here, all retries failed
            raise last_exception
        
        return wrapper
    return decorator


def retry_async_with_backoff(
    max_attempts: Optional[int] = None,
    initial_delay: Optional[float] = None,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Async version of retry_with_backoff decorator.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            attempts = max_attempts or settings.retry_attempts
            delay = initial_delay or settings.retry_delay
            
            last_exception = None
            
            for attempt in range(attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if not is_retryable_error(e):
                        logger.warning(
                            "non_retryable_error_async",
                            function=func.__name__,
                            attempt=attempt + 1,
                            error=str(e),
                            error_type=type(e).__name__
                        )
                        raise
                    
                    if attempt == attempts - 1:  # Last attempt
                        logger.error(
                            "max_retries_exceeded_async",
                            function=func.__name__,
                            max_attempts=attempts,
                            final_error=str(e),
                            error_type=type(e).__name__
                        )
                        break
                    
                    # Calculate delay with exponential backoff
                    current_delay = min(delay * (backoff_factor ** attempt), max_delay)
                    
                    logger.warning(
                        "retrying_after_error_async",
                        function=func.__name__,
                        attempt=attempt + 1,
                        max_attempts=attempts,
                        delay=current_delay,
                        error=str(e),
                        error_type=type(e).__name__
                    )
                    
                    await asyncio.sleep(current_delay)
            
            # If we get here, all retries failed
            raise last_exception
        
        return wrapper
    return decorator


@retry_with_backoff(exceptions=(RequestException,))
def robust_get(url: str, **kwargs) -> requests.Response:
    """
    Make a robust GET request with automatic retries.
    
    Args:
        url: URL to request
        **kwargs: Additional arguments to pass to requests.get
    
    Returns:
        requests.Response object
    
    Raises:
        NonRetryableError: For client errors that shouldn't be retried
        RetryableError: For server errors after max retries
    """
    # Set default timeout if not provided
    kwargs.setdefault('timeout', settings.request_timeout)
    
    try:
        response = requests.get(url, **kwargs)
        
        # Check for HTTP errors
        if response.status_code in NON_RETRYABLE_STATUS_CODES:
            raise NonRetryableError(f"HTTP {response.status_code}: {response.reason}")
        elif response.status_code in RETRYABLE_STATUS_CODES:
            raise RetryableError(f"HTTP {response.status_code}: {response.reason}")
        
        response.raise_for_status()  # Raise for any other HTTP errors
        
        logger.debug("successful_request", url=url, status_code=response.status_code)
        return response
        
    except requests.exceptions.Timeout as e:
        raise RetryableError(f"Request timeout: {e}")
    except requests.exceptions.ConnectionError as e:
        raise RetryableError(f"Connection error: {e}")
    except requests.exceptions.HTTPError as e:
        # Re-raise as appropriate error type based on status code
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
            if status_code in NON_RETRYABLE_STATUS_CODES:
                raise NonRetryableError(f"HTTP {status_code}: {e}")
            else:
                raise RetryableError(f"HTTP {status_code}: {e}")
        raise RetryableError(f"HTTP error: {e}")
    except requests.exceptions.RequestException as e:
        raise RetryableError(f"Request error: {e}") 