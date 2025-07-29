"""
Structured logging setup for kubechron data scraping.
Provides consistent logging across all modules with configurable formats and levels.
"""
import sys
import structlog
from typing import Any, Dict
from config import settings


def configure_logging() -> structlog.stdlib.BoundLogger:
    """Configure structured logging based on settings."""
    
    # Configure structlog processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    # Add appropriate renderer based on format setting
    if settings.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.processors.KeyValueRenderer())
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure stdlib logging with error handling
    import logging
    import atexit
    
    # Create a custom handler that doesn't fail on flush
    class SafeStreamHandler(logging.StreamHandler):
        def flush(self):
            try:
                super().flush()
            except (BrokenPipeError, ValueError):
                # Ignore flush errors during shutdown
                pass
    
    # Clear any existing handlers to avoid conflicts
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set up our safe handler
    handler = SafeStreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    
    logging.basicConfig(
        handlers=[handler],
        level=getattr(logging, settings.log_level),
        force=True
    )
    
    # Ensure clean shutdown
    def cleanup_logging():
        try:
            logging.shutdown()
        except:
            pass
    
    atexit.register(cleanup_logging)
    
    return structlog.get_logger("kubechron")


class MetricsLogger:
    """Logger for tracking scraping metrics and performance."""
    
    def __init__(self, logger: structlog.stdlib.BoundLogger):
        self.logger = logger
        self.metrics: Dict[str, Any] = {
            "pages_scraped": 0,
            "pages_failed": 0,
            "repos_processed": 0,
            "yaml_files_extracted": 0,
            "total_chunks_created": 0,
        }
    
    def increment_metric(self, metric_name: str, value: int = 1):
        """Increment a metric counter."""
        if metric_name in self.metrics:
            self.metrics[metric_name] += value
            self.logger.debug("metric_updated", metric=metric_name, value=self.metrics[metric_name])
    
    def log_metrics_summary(self):
        """Log a summary of all metrics."""
        self.logger.info("scraping_metrics_summary", **self.metrics)
    
    def reset_metrics(self):
        """Reset all metrics to zero."""
        self.metrics = {key: 0 for key in self.metrics.keys()}
        self.logger.info("metrics_reset")


# Global logger instance
logger = configure_logging()
metrics = MetricsLogger(logger) 