"""
Configuration management for kubernetes-copilot data scraping.
Uses environment variables with sensible defaults for production deployment.
"""
import os
from typing import List, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from pathlib import Path


class ScrapingSettings(BaseSettings):
    """Configuration settings for the data scraping pipeline."""
    
    # Output settings
    output_dir: str = Field(default="output", description="Directory for output files")
    dataset_filename: str = Field(default="k8s_dataset.jsonl", description="Output dataset filename")
    
    # Scraping behavior
    max_chunk_length: int = Field(default=800, description="Maximum text chunk length")
    request_timeout: int = Field(default=30, description="HTTP request timeout in seconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts for failed requests")
    retry_delay: float = Field(default=1.0, description="Delay between retries in seconds")
    
    # Rate limiting
    requests_per_second: float = Field(default=2.0, description="Maximum requests per second")
    concurrent_requests: int = Field(default=5, description="Maximum concurrent requests")
    
    # GitHub settings
    github_token: Optional[str] = Field(default=None, description="GitHub personal access token")
    target_repos: List[str] = Field(
        default=[
            "https://github.com/kubernetes/examples",
            "https://github.com/kelseyhightower/kubernetes-the-hard-way",
        ],
        description="List of GitHub repositories to scrape"
    )
    
    # Documentation scraping
    docs_base_url: str = Field(
        default="https://kubernetes.io/docs/",
        description="Base URL for Kubernetes documentation"
    )
    
    # Logging settings
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format: json or text")
    
    # Data validation
    validate_yaml: bool = Field(default=True, description="Validate YAML content")
    min_content_length: int = Field(default=50, description="Minimum content length to include")
    
    class Config:
        env_file = ".env"
        env_prefix = "K8S_COPILOT_"
        case_sensitive = False
    
    @validator('output_dir')
    def create_output_dir(cls, v):
        """Ensure output directory exists."""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.upper()
    
    @validator('log_format')
    def validate_log_format(cls, v):
        """Validate log format."""
        valid_formats = ['json', 'text']
        if v.lower() not in valid_formats:
            raise ValueError(f'Log format must be one of: {valid_formats}')
        return v.lower()
    
    @property
    def output_path(self) -> str:
        """Full path to output file."""
        return os.path.join(self.output_dir, self.dataset_filename)


# Global settings instance
settings = ScrapingSettings()
