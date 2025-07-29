"""
Main orchestrator for kubechron data scraping pipeline.
Production-grade implementation with comprehensive error handling and monitoring.
"""
import sys
import time
from typing import List, Dict, Any

from config import settings
from logger import logger, metrics
from docs_scraper import DocumentationScraper, DocsScrapingError
from github_scraper import GitHubScraper, GitHubScrapingError
from parser import chunk_text
from dataset_formatter import format_dataset, DatasetFormattingError
from utils import deduplicator


class DataScrapingPipeline:
    """Production-grade data scraping pipeline with comprehensive error handling."""
    
    def __init__(self):
        self.start_time = time.time()
        logger.info("pipeline_initialized", 
                   output_path=settings.output_path,
                   chunk_length=settings.max_chunk_length)
    
    def run_docs_scraping(self) -> List[Dict[str, Any]]:
        """
        Run documentation scraping with error handling.
        
        Returns:
            List of scraped documentation data
        """
        logger.info("starting_docs_scraping_phase")
        
        try:
            scraper = DocumentationScraper()
            docs_data = scraper.scrape_all_docs()
            
            if not docs_data:
                logger.warning("no_docs_data_scraped")
                return []
            
            logger.info("docs_scraping_phase_completed", docs_count=len(docs_data))
            return docs_data
            
        except DocsScrapingError as e:
            logger.error("docs_scraping_phase_failed", error=str(e))
            # Continue with pipeline even if docs scraping fails
            return []
        except Exception as e:
            logger.error("docs_scraping_unexpected_error", 
                        error=str(e), 
                        error_type=type(e).__name__)
            return []
    
    def run_github_scraping(self) -> List[Dict[str, Any]]:
        """
        Run GitHub repository scraping with error handling.
        
        Returns:
            List of scraped YAML data from repositories
        """
        logger.info("starting_github_scraping_phase")
        
        try:
            scraper = GitHubScraper()
            yaml_data = scraper.extract_from_all_repos()
            
            if not yaml_data:
                logger.warning("no_yaml_data_scraped")
                return []
            
            logger.info("github_scraping_phase_completed", yaml_files_count=len(yaml_data))
            return yaml_data
            
        except GitHubScrapingError as e:
            logger.error("github_scraping_phase_failed", error=str(e))
            # Continue with pipeline even if GitHub scraping fails
            return []
        except Exception as e:
            logger.error("github_scraping_unexpected_error",
                        error=str(e),
                        error_type=type(e).__name__)
            return []
    
    def process_content_chunks(self, all_docs: List[Dict[str, Any]], yaml_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process all content into chunks with error handling.
        
        Args:
            all_docs: List of documentation data
            yaml_data: List of YAML data
            
        Returns:
            List of all processed data with chunks
        """
        logger.info("starting_chunking_phase", 
                   docs_count=len(all_docs),
                   yaml_count=len(yaml_data))
        
        processed_data = []
        chunking_errors = 0
        total_chunks = 0
        
        # Process documentation
        for i, doc in enumerate(all_docs):
            try:
                if doc and "text" in doc:
                    doc["chunks"] = chunk_text(doc["text"], settings.max_chunk_length)
                    total_chunks += len(doc["chunks"])
                    processed_data.append(doc)
                    
                    if (i + 1) % 50 == 0:  # Log progress every 50 docs
                        logger.info("docs_chunking_progress", processed=i+1, total=len(all_docs))
                        
            except Exception as e:
                chunking_errors += 1
                logger.warning("doc_chunking_failed", 
                             doc_url=doc.get("url", "unknown"),
                             error=str(e))
                continue
        
        # Process YAML files
        for i, yaml_file in enumerate(yaml_data):
            try:
                if yaml_file and "content" in yaml_file:
                    yaml_file["chunks"] = chunk_text(yaml_file["content"], settings.max_chunk_length)
                    total_chunks += len(yaml_file["chunks"])
                    processed_data.append(yaml_file)
                    
                    if (i + 1) % 20 == 0:  # Log progress every 20 YAML files
                        logger.info("yaml_chunking_progress", processed=i+1, total=len(yaml_data))
                        
            except Exception as e:
                chunking_errors += 1
                logger.warning("yaml_chunking_failed",
                             yaml_path=yaml_file.get("path", "unknown"),
                             repo=yaml_file.get("repository", "unknown"),
                             error=str(e))
                continue
        
        metrics.increment_metric("total_chunks_created", total_chunks)
        
        logger.info("chunking_phase_completed",
                   total_processed=len(processed_data),
                   total_chunks=total_chunks,
                   chunking_errors=chunking_errors)
        
        return processed_data
    
    def run_dataset_formatting(self, dataset: List[Dict[str, Any]]) -> bool:
        """
        Format and save the final dataset with error handling.
        
        Args:
            dataset: List of processed data to format
            
        Returns:
            True if formatting succeeded, False otherwise
        """
        logger.info("starting_dataset_formatting_phase", dataset_size=len(dataset))
        
        try:
            success = format_dataset(dataset, settings.output_path)
            
            if success:
                logger.info("dataset_formatting_completed", output_path=settings.output_path)
                return True
            else:
                logger.error("dataset_formatting_failed")
                return False
                
        except DatasetFormattingError as e:
            logger.error("dataset_formatting_phase_failed", error=str(e))
            return False
        except Exception as e:
            logger.error("dataset_formatting_unexpected_error",
                        error=str(e),
                        error_type=type(e).__name__)
            return False
    
    def run_complete_pipeline(self) -> bool:
        """
        Run the complete data scraping pipeline with comprehensive error handling.
        
        Returns:
            True if pipeline completed successfully, False otherwise
        """
        logger.info("starting_complete_pipeline")
        
        try:
            # Phase 1: Scrape documentation
            all_docs = self.run_docs_scraping()
            
            # Phase 2: Scrape GitHub repositories
            yaml_data = self.run_github_scraping()
            
            # Check if we have any data at all
            if not all_docs and not yaml_data:
                logger.error("no_data_scraped", 
                           reason="Both documentation and GitHub scraping failed or returned no data")
                return False
            
            # Phase 3: Process content into chunks
            dataset = self.process_content_chunks(all_docs, yaml_data)
            
            if not dataset:
                logger.error("no_processable_data", reason="Chunking phase produced no valid data")
                return False
            
            # Phase 4: Format and save dataset
            success = self.run_dataset_formatting(dataset)
            
            if not success:
                logger.error("pipeline_failed", reason="Dataset formatting failed")
                return False
            
            # Log final pipeline statistics
            pipeline_duration = time.time() - self.start_time
            dedup_stats = deduplicator.get_stats()
            
            logger.info("pipeline_completed_successfully",
                       total_duration_seconds=pipeline_duration,
                       docs_scraped=len(all_docs),
                       yaml_files_scraped=len(yaml_data),
                       final_dataset_size=len(dataset),
                       output_file=settings.output_path,
                       **dedup_stats)
            
            metrics.log_metrics_summary()
            
            return True
            
        except KeyboardInterrupt:
            logger.warning("pipeline_interrupted", reason="User interrupted execution")
            return False
        except Exception as e:
            logger.error("pipeline_unexpected_error",
                        error=str(e),
                        error_type=type(e).__name__)
            return False


def main():
    """Main entry point for the data scraping pipeline."""
    try:
        logger.info("kubernetes_copilot_starting", 
                   log_level=settings.log_level,
                   output_dir=settings.output_dir)
        
        pipeline = DataScrapingPipeline()
        success = pipeline.run_complete_pipeline()
        
        if success:
            print(f"[SUCCESS] Dataset created successfully at {settings.output_path}")
            logger.info("main_completed_successfully")
            sys.exit(0)
        else:
            print(f"[ERROR] Pipeline failed. Check logs for details.")
            logger.error("main_completed_with_errors")
            sys.exit(1)
            
    except Exception as e:
        logger.error("main_unexpected_error", 
                    error=str(e), 
                    error_type=type(e).__name__)
        print(f"[FATAL ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()