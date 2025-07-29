"""
Utility functions for data processing and validation.
Enhanced with proper logging and configuration support.
"""
import re
import hashlib
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse, urljoin
from config import settings
from logger import logger, metrics


def clean_text(text: str) -> str:
    """
    Clean and normalize text content with improved handling.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text string
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove excessive whitespace and newlines
    text = re.sub(r'\n{3,}', '\n\n', text)  # Limit consecutive newlines
    text = re.sub(r'[ \t]{2,}', ' ', text)  # Limit consecutive spaces/tabs
    
    # Remove non-ASCII characters but preserve common unicode
    text = re.sub(r'[^\x00-\x7F\u00A0-\u024F\u1E00-\u1EFF]+', ' ', text)
    
    # Remove control characters except newline and tab
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Clean up excessive punctuation
    text = re.sub(r'[.]{3,}', '...', text)
    text = re.sub(r'[!]{2,}', '!!', text)
    text = re.sub(r'[?]{2,}', '??', text)
    
    return text.strip()


def validate_content_quality(content: str, content_type: str = "text") -> bool:
    """
    Validate if content meets quality thresholds.
    
    Args:
        content: Content to validate
        content_type: Type of content (text, yaml, etc.)
        
    Returns:
        True if content passes validation, False otherwise
    """
    if not content or not isinstance(content, str):
        logger.debug("content_validation_failed", reason="empty_or_invalid", content_type=content_type)
        return False
    
    # Check minimum length
    if len(content.strip()) < settings.min_content_length:
        logger.debug(
            "content_validation_failed",
            reason="too_short",
            content_type=content_type,
            length=len(content.strip()),
            min_length=settings.min_content_length
        )
        return False
    
    # Check for reasonable text-to-whitespace ratio
    non_whitespace_chars = len(re.sub(r'\s', '', content))
    if non_whitespace_chars == 0:
        logger.debug("content_validation_failed", reason="no_content", content_type=content_type)
        return False
    
    whitespace_ratio = (len(content) - non_whitespace_chars) / len(content)
    if whitespace_ratio > 0.8:  # More than 80% whitespace
        logger.debug(
            "content_validation_failed",
            reason="too_much_whitespace",
            content_type=content_type,
            whitespace_ratio=whitespace_ratio
        )
        return False
    
    # Additional validation for specific content types
    if content_type == "yaml":
        return validate_yaml_content(content)
    
    logger.debug("content_validation_passed", content_type=content_type, length=len(content))
    return True


def validate_yaml_content(content: str) -> bool:
    """
    Validate YAML content structure and completeness.
    
    Args:
        content: YAML content to validate
        
    Returns:
        True if YAML is valid and meaningful, False otherwise
    """
    if not settings.validate_yaml:
        return True
    
    try:
        import yaml
        
        # Try to parse YAML
        parsed = yaml.safe_load(content)
        
        # Check if parsing resulted in meaningful data
        if parsed is None:
            logger.debug("yaml_validation_failed", reason="empty_document")
            return False
        
        # Initialize Kubernetes resource indicators
        has_kind = False
        has_api_version = False
        
        # Check for common Kubernetes resource indicators
        if isinstance(parsed, dict):
            has_kind = 'kind' in parsed
            has_api_version = 'apiVersion' in parsed
            has_metadata = 'metadata' in parsed
            
            # If it looks like a Kubernetes resource but is missing key fields
            if (has_kind or has_api_version) and not (has_kind and has_api_version):
                logger.debug("yaml_validation_failed", reason="incomplete_k8s_resource")
                return False
            
            # Check for completely empty or trivial objects
            if len(parsed) < 2:
                logger.debug("yaml_validation_failed", reason="trivial_content")
                return False
        
        # Check for lists with actual content
        elif isinstance(parsed, list):
            if len(parsed) == 0:
                logger.debug("yaml_validation_failed", reason="empty_list")
                return False
        
        logger.debug("yaml_validation_passed", has_k8s_structure=has_kind and has_api_version if isinstance(parsed, dict) else False)
        return True
        
    except yaml.YAMLError as e:
        logger.debug("yaml_validation_failed", reason="parse_error", error=str(e))
        return False
    except Exception as e:
        logger.warning("yaml_validation_error", error=str(e), error_type=type(e).__name__)
        return False


def generate_content_hash(content: str) -> str:
    """
    Generate a stable hash for content deduplication.
    
    Args:
        content: Content to hash
        
    Returns:
        SHA-256 hash of the content
    """
    # Normalize content before hashing to catch near-duplicates
    normalized = clean_text(content).lower()
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def is_valid_url(url: str, allowed_domains: Optional[List[str]] = None) -> bool:
    """
    Validate if URL is properly formed and from allowed domains.
    
    Args:
        url: URL to validate
        allowed_domains: List of allowed domains (if None, any domain is allowed)
        
    Returns:
        True if URL is valid and allowed, False otherwise
    """
    try:
        parsed = urlparse(url)
        
        # Check basic URL structure
        if not all([parsed.scheme, parsed.netloc]):
            return False
        
        # Check protocol
        if parsed.scheme not in ['http', 'https']:
            return False
        
        # Check allowed domains
        if allowed_domains:
            domain = parsed.netloc.lower()
            if not any(domain.endswith(allowed) for allowed in allowed_domains):
                logger.debug("url_validation_failed", url=url, reason="domain_not_allowed")
                return False
        
        return True
        
    except Exception as e:
        logger.debug("url_validation_failed", url=url, error=str(e))
        return False


def extract_metadata_from_content(content: str, source_url: str = "") -> Dict[str, Any]:
    """
    Extract metadata from content for better dataset organization.
    
    Args:
        content: Content to analyze
        source_url: Source URL of the content
        
    Returns:
        Dictionary containing extracted metadata
    """
    metadata = {
        "content_length": len(content),
        "word_count": len(content.split()),
        "line_count": len(content.split('\n')),
        "content_hash": generate_content_hash(content),
        "source_url": source_url,
        "has_code_blocks": bool(re.search(r'```|`[^`]+`', content)),
        "has_yaml_structure": bool(re.search(r'^\s*(apiVersion|kind):\s*\S+', content, re.MULTILINE)),
        "has_kubernetes_terms": bool(re.search(r'\b(pod|service|deployment|namespace|configmap|secret)\b', content, re.IGNORECASE)),
    }
    
    # Extract title if present
    title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if title_match:
        metadata["title"] = title_match.group(1).strip()
    
    logger.debug("metadata_extracted", **{k: v for k, v in metadata.items() if k != "content_hash"})
    return metadata


class ContentDeduplicator:
    """Utility class for detecting and handling duplicate content."""
    
    def __init__(self):
        self.seen_hashes: set = set()
        self.duplicate_count = 0
    
    def is_duplicate(self, content: str) -> bool:
        """Check if content is a duplicate based on hash."""
        content_hash = generate_content_hash(content)
        if content_hash in self.seen_hashes:
            self.duplicate_count += 1
            logger.debug("duplicate_content_detected", content_hash=content_hash)
            return True
        
        self.seen_hashes.add(content_hash)
        return False
    
    def get_stats(self) -> Dict[str, int]:
        """Get deduplication statistics."""
        return {
            "unique_content_items": len(self.seen_hashes),
            "duplicates_found": self.duplicate_count
        }


# Global deduplicator instance
deduplicator = ContentDeduplicator()