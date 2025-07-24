"""
Enhanced GitHub repository scraper with robust error handling and data validation.
Extracts YAML files from Kubernetes-related repositories with production-grade reliability.
"""
import os
import tempfile
import time
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import shutil

from git import Repo, GitCommandError
import yaml

from config import settings
from logger import logger, metrics
from utils import validate_content_quality, extract_metadata_from_content, deduplicator


class GitHubScrapingError(Exception):
    """Custom exception for GitHub scraping errors."""
    pass


class GitHubScraper:
    """Production-grade GitHub repository scraper for YAML content extraction."""
    
    def __init__(self):
        self.target_repos = settings.target_repos
        self.github_token = settings.github_token
        self.processed_repos: Set[str] = set()
        self.failed_repos: Set[str] = set()
        self._session_start_time = time.time()
        
        # Validate configuration
        if not self.target_repos:
            raise GitHubScrapingError("No target repositories configured")
        
        logger.info("github_scraper_initialized", repo_count=len(self.target_repos))
    
    def _prepare_git_auth_url(self, repo_url: str) -> str:
        """
        Prepare repository URL with authentication if token is available.
        
        Args:
            repo_url: Original repository URL
            
        Returns:
            URL with authentication credentials if token is available
        """
        if not self.github_token:
            return repo_url
        
        # Handle GitHub URLs
        if "github.com" in repo_url:
            if repo_url.startswith("https://github.com/"):
                # Insert token into URL: https://token@github.com/...
                return repo_url.replace("https://github.com/", f"https://{self.github_token}@github.com/")
            elif repo_url.startswith("git@github.com:"):
                # Convert SSH to HTTPS with token
                repo_path = repo_url.replace("git@github.com:", "").replace(".git", "")
                return f"https://{self.github_token}@github.com/{repo_path}"
        
        return repo_url
    
    def _validate_yaml_file(self, file_path: str, content: str) -> bool:
        """
        Validate YAML file content and structure.
        
        Args:
            file_path: Path to the YAML file
            content: File content
            
        Returns:
            True if YAML is valid and should be included, False otherwise
        """
        # Basic content validation
        if not validate_content_quality(content, "yaml"):
            return False
        
        # Skip certain file patterns that are usually not useful
        skip_patterns = [
            "/.github/",
            "/test/",
            "/tests/",
            "test-",
            "example-secret",
            "example-configmap",
        ]
        
        if any(pattern in file_path.lower() for pattern in skip_patterns):
            logger.debug("yaml_file_skipped_by_pattern", file_path=file_path)
            return False
        
        # Check for duplicates
        if deduplicator.is_duplicate(content):
            logger.debug("duplicate_yaml_skipped", file_path=file_path)
            return False
        
        return True
    
    def clone_and_extract(self, repo_url: str) -> List[Dict[str, Any]]:
        """
        Clone a repository and extract YAML files with error handling.
        
        Args:
            repo_url: URL of the repository to clone
            
        Returns:
            List of dictionaries containing YAML file data
        """
        if repo_url in self.processed_repos:
            logger.debug("repo_already_processed", repo_url=repo_url)
            return []
        
        if repo_url in self.failed_repos:
            logger.debug("repo_previously_failed", repo_url=repo_url)
            return []
        
        logger.info("cloning_repository", repo_url=repo_url)
        
        yaml_snippets = []
        tmpdir = None
        
        try:
            # Create temporary directory
            tmpdir = tempfile.mkdtemp(prefix="k8s_scraper_")
            logger.debug("temp_dir_created", tmpdir=tmpdir)
            
            # Prepare authenticated URL
            auth_url = self._prepare_git_auth_url(repo_url)
            
            # Clone repository with depth limit for efficiency
            repo = Repo.clone_from(
                auth_url, 
                tmpdir,
                depth=1,  # Shallow clone for efficiency
                single_branch=True  # Only default branch
            )
            
            logger.info("repository_cloned", repo_url=repo_url, tmpdir=tmpdir)
            
            # Walk through files and extract YAML
            yaml_files_found = 0
            yaml_files_processed = 0
            
            for root, dirs, files in os.walk(tmpdir):
                # Skip hidden directories and common non-content directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__']]
                
                for file in files:
                    if file.endswith((".yaml", ".yml")):
                        yaml_files_found += 1
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, tmpdir)
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                            
                            # Validate YAML syntax
                            try:
                                yaml.safe_load(content)
                            except yaml.YAMLError as e:
                                logger.debug("invalid_yaml_skipped", 
                                           file_path=relative_path, 
                                           repo_url=repo_url, 
                                           error=str(e))
                                continue
                            
                            # Validate content quality
                            if not self._validate_yaml_file(relative_path, content):
                                continue
                            
                            # Extract metadata
                            metadata = extract_metadata_from_content(content, repo_url)
                            metadata.update({
                                "file_path": relative_path,
                                "repository": repo_url,
                                "file_type": "yaml",
                                "scraped_at": time.time(),
                            })
                            
                            yaml_snippet = {
                                "path": relative_path,
                                "content": content,
                                "metadata": metadata,
                                "repository": repo_url,
                            }
                            
                            yaml_snippets.append(yaml_snippet)
                            yaml_files_processed += 1
                            
                            logger.debug("yaml_file_processed", 
                                       file_path=relative_path,
                                       repo_url=repo_url,
                                       content_length=len(content))
                            
                        except (IOError, UnicodeDecodeError) as e:
                            logger.warning("yaml_file_read_error", 
                                         file_path=relative_path,
                                         repo_url=repo_url,
                                         error=str(e))
                            continue
                        except Exception as e:
                            logger.error("yaml_file_processing_error",
                                       file_path=relative_path,
                                       repo_url=repo_url,
                                       error=str(e),
                                       error_type=type(e).__name__)
                            continue
            
            self.processed_repos.add(repo_url)
            metrics.increment_metric("repos_processed")
            metrics.increment_metric("yaml_files_extracted", yaml_files_processed)
            
            logger.info("repository_processing_completed",
                       repo_url=repo_url,
                       yaml_files_found=yaml_files_found,
                       yaml_files_processed=yaml_files_processed,
                       yaml_files_extracted=len(yaml_snippets))
            
            return yaml_snippets
            
        except GitCommandError as e:
            error_msg = f"Git operation failed: {e}"
            logger.error("git_clone_failed", repo_url=repo_url, error=error_msg)
            self.failed_repos.add(repo_url)
            return []
        except OSError as e:
            error_msg = f"File system error: {e}"
            logger.error("filesystem_error", repo_url=repo_url, error=error_msg)
            self.failed_repos.add(repo_url)
            return []
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            logger.error("repo_processing_unexpected_error",
                        repo_url=repo_url,
                        error=error_msg,
                        error_type=type(e).__name__)
            self.failed_repos.add(repo_url)
            return []
        finally:
            # Clean up temporary directory
            if tmpdir and os.path.exists(tmpdir):
                try:
                    shutil.rmtree(tmpdir)
                    logger.debug("temp_dir_cleaned", tmpdir=tmpdir)
                except Exception as e:
                    logger.warning("temp_dir_cleanup_failed", tmpdir=tmpdir, error=str(e))
    
    def extract_from_all_repos(self) -> List[Dict[str, Any]]:
        """
        Extract YAML content from all configured repositories.
        
        Returns:
            List of all extracted YAML data
        """
        logger.info("starting_github_extraction", repo_count=len(self.target_repos))
        
        all_yaml_data = []
        
        for i, repo_url in enumerate(self.target_repos, 1):
            logger.info("processing_repository", 
                       repo_number=i,
                       total_repos=len(self.target_repos),
                       repo_url=repo_url)
            
            try:
                yaml_data = self.clone_and_extract(repo_url)
                all_yaml_data.extend(yaml_data)
                
                # Add small delay between repositories to be respectful
                if i < len(self.target_repos):
                    time.sleep(1.0)
                    
            except Exception as e:
                logger.error("repository_extraction_failed",
                           repo_url=repo_url,
                           error=str(e),
                           error_type=type(e).__name__)
                continue
        
        # Log final statistics
        extraction_duration = time.time() - self._session_start_time
        dedup_stats = deduplicator.get_stats()
        
        logger.info("github_extraction_completed",
                   total_repos=len(self.target_repos),
                   successful_repos=len(self.processed_repos),
                   failed_repos=len(self.failed_repos),
                   total_yaml_files=len(all_yaml_data),
                   duration_seconds=extraction_duration,
                   **dedup_stats)
        
        return all_yaml_data


def clone_and_extract(repo_url: str) -> List[Dict[str, Any]]:
    """
    Legacy function wrapper for backward compatibility.
    
    Args:
        repo_url: Repository URL to clone and extract from
        
    Returns:
        List of extracted YAML data
    """
    scraper = GitHubScraper()
    return scraper.clone_and_extract(repo_url)


def extract_all_repos() -> List[Dict[str, Any]]:
    """
    Extract YAML content from all configured repositories.
    
    Returns:
        List of all extracted YAML data
    """
    scraper = GitHubScraper()
    return scraper.extract_from_all_repos()