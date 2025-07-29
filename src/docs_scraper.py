"""
Enhanced documentation scraper with robust error handling and data validation.
Scrapes Kubernetes documentation with production-grade reliability.
"""
import time
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

from config import settings
from logger import logger, metrics
from retry_utils import robust_get, NonRetryableError, RetryableError
from utils import clean_text, validate_content_quality, is_valid_url, extract_metadata_from_content, deduplicator


class DocsScrapingError(Exception):
    """Custom exception for documentation scraping errors."""
    pass


class DocumentationScraper:
    """Production-grade Kubernetes documentation scraper with robust error handling."""
    
    def __init__(self):
        self.base_url = settings.docs_base_url
        self.scraped_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        self._session_start_time = time.time()
        
        # Validate base URL
        if not is_valid_url(self.base_url, ["kubernetes.io"]):
            raise DocsScrapingError(f"Invalid base URL: {self.base_url}")
        
        logger.info("docs_scraper_initialized", base_url=self.base_url)
    
    def get_all_links(self, start_url: Optional[str] = None) -> List[str]:
        """
        Discover all documentation links from the main docs page.
        
        Args:
            start_url: Starting URL for link discovery (uses base_url if None)
            
        Returns:
            List of discovered documentation URLs
            
        Raises:
            DocsScrapingError: If link discovery fails completely
        """
        start_url = start_url or self.base_url
        
        logger.info("discovering_links", start_url=start_url)
        
        try:
            response = robust_get(start_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            links = set()
            processed_hrefs = set()
            
            for a_tag in soup.find_all("a", href=True):
                href = a_tag['href']
                
                # Skip already processed hrefs to improve efficiency
                if href in processed_hrefs:
                    continue
                processed_hrefs.add(href)
                
                # Filter for documentation links
                if self._is_valid_docs_link(href):
                    full_url = urljoin(self.base_url, href)
                    
                    # Validate the constructed URL
                    if is_valid_url(full_url, ["kubernetes.io"]):
                        links.add(full_url)
                    else:
                        logger.debug("invalid_url_skipped", url=full_url, href=href)
                else:
                    # Debug: log why links are being rejected
                    logger.debug("link_rejected", href=href, starts_with_docs=href.startswith("/docs") if href else False)
            
            link_list = list(links)
            logger.info("links_discovered", count=len(link_list), start_url=start_url)
            
            if len(link_list) == 0:
                logger.warning("no_links_found", start_url=start_url)
            
            return link_list
            
        except NonRetryableError as e:
            logger.error("link_discovery_failed_permanently", start_url=start_url, error=str(e))
            raise DocsScrapingError(f"Failed to discover links from {start_url}: {e}")
        except RetryableError as e:
            logger.error("link_discovery_failed_after_retries", start_url=start_url, error=str(e))
            raise DocsScrapingError(f"Failed to discover links from {start_url} after retries: {e}")
        except Exception as e:
            logger.error("link_discovery_unexpected_error", start_url=start_url, error=str(e), error_type=type(e).__name__)
            raise DocsScrapingError(f"Unexpected error during link discovery: {e}")
    
    def _is_valid_docs_link(self, href: str) -> bool:
        """
        Check if a link is a valid documentation link to scrape.
        
        Args:
            href: The href attribute from an anchor tag
            
        Returns:
            True if the link should be scraped, False otherwise
        """
        if not href:
            return False
        
        # Must be a docs link
        if not href.startswith("/docs"):
            return False
        
        # Skip anchors and fragments
        if "#" in href:
            return False
        
        # Skip known non-content paths (reduced filtering for now)
        skip_patterns = [
            "/docs/api/",
            "/docs/reference/generated/",
        ]
        
        # Allow directory indices for now to get more links
        return not any(pattern in href for pattern in skip_patterns)
    
    def scrape_page(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Scrape content from a single documentation page.
        
        Args:
            url: URL of the page to scrape
            
        Returns:
            Dictionary containing scraped content and metadata, or None if scraping failed
        """
        if url in self.scraped_urls:
            logger.debug("url_already_scraped", url=url)
            return None
        
        if url in self.failed_urls:
            logger.debug("url_previously_failed", url=url)
            return None
        
        logger.debug("scraping_page", url=url)
        
        try:
            response = robust_get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract main content
            content_element = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
            
            if not content_element:
                logger.warning("no_main_content_found", url=url)
                self.failed_urls.add(url)
                metrics.increment_metric("pages_failed")
                return None
            
            # Extract text content
            raw_text = content_element.get_text(separator="\n")
            cleaned_text = clean_text(raw_text)
            
            # Validate content quality
            if not validate_content_quality(cleaned_text, "text"):
                logger.debug("content_quality_validation_failed", url=url)
                self.failed_urls.add(url)
                metrics.increment_metric("pages_failed")
                return None
            
            # Check for duplicates
            if deduplicator.is_duplicate(cleaned_text):
                logger.debug("duplicate_content_skipped", url=url)
                self.scraped_urls.add(url)  # Mark as scraped to avoid retry
                return None
            
            # Extract metadata
            metadata = extract_metadata_from_content(cleaned_text, url)
            
            # Extract page title
            title_element = soup.find('title') or soup.find('h1')
            if title_element:
                metadata['page_title'] = clean_text(title_element.get_text())
            
            # Create result
            result = {
                "url": url,
                "text": cleaned_text,
                "metadata": metadata,
                "scraped_at": time.time(),
            }
            
            self.scraped_urls.add(url)
            metrics.increment_metric("pages_scraped")
            
            logger.debug("page_scraped_successfully", url=url, content_length=len(cleaned_text))
            return result
            
        except NonRetryableError as e:
            logger.warning("page_scraping_failed_permanently", url=url, error=str(e))
            self.failed_urls.add(url)
            metrics.increment_metric("pages_failed")
            return None
        except RetryableError as e:
            logger.warning("page_scraping_failed_after_retries", url=url, error=str(e))
            self.failed_urls.add(url)
            metrics.increment_metric("pages_failed")
            return None
        except Exception as e:
            logger.error("page_scraping_unexpected_error", url=url, error=str(e), error_type=type(e).__name__)
            self.failed_urls.add(url)
            metrics.increment_metric("pages_failed")
            return None
    
    def scrape_all_docs(self) -> List[Dict[str, Any]]:
        """
        Scrape all documentation pages with rate limiting and error handling.
        
        Returns:
            List of scraped page data dictionaries
        """
        logger.info("starting_docs_scraping", base_url=self.base_url)
        
        try:
            # Discover all links
            all_links = self.get_all_links()
            
            if not all_links:
                logger.error("no_links_to_scrape")
                return []
            
            logger.info("scraping_pages", total_pages=len(all_links))
            
            results = []
            request_interval = 1.0 / settings.requests_per_second
            
            for i, link in enumerate(all_links, 1):
                # Rate limiting
                if i > 1:  # Don't sleep before the first request
                    time.sleep(request_interval)
                
                # Log progress periodically
                if i % 10 == 0:
                    logger.info("scraping_progress", completed=i, total=len(all_links), url=link)
                
                # Scrape the page
                page_data = self.scrape_page(link)
                if page_data:
                    results.append(page_data)
            
            # Log final statistics
            scraping_duration = time.time() - self._session_start_time
            dedup_stats = deduplicator.get_stats()
            
            logger.info(
                "docs_scraping_completed",
                total_links=len(all_links),
                successful_pages=len(results),
                failed_pages=len(self.failed_urls),
                duration_seconds=scraping_duration,
                pages_per_second=len(all_links) / scraping_duration if scraping_duration > 0 else 0,
                **dedup_stats
            )
            
            metrics.log_metrics_summary()
            
            return results
            
        except DocsScrapingError:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            logger.error("docs_scraping_failed", error=str(e), error_type=type(e).__name__)
            raise DocsScrapingError(f"Documentation scraping failed: {e}")


def scrape_all_docs() -> List[Dict[str, Any]]:
    """
    Legacy function wrapper for backward compatibility.
    
    Returns:
        List of scraped documentation pages
    """
    scraper = DocumentationScraper()
    return scraper.scrape_all_docs()