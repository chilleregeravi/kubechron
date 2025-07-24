"""
Dataset formatting utilities with robust error handling and validation.
Converts scraped data into instruction-tuning format for model training.
"""
import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from config import settings
from logger import logger, metrics


class DatasetFormattingError(Exception):
    """Custom exception for dataset formatting errors."""
    pass


def format_instruction_record(chunk: str, source_type: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Format a single content chunk into instruction-tuning format.
    
    Args:
        chunk: Text content chunk
        source_type: Type of source (docs, yaml)
        metadata: Optional metadata about the source
        
    Returns:
        Formatted instruction record
    """
    # Create base instruction based on source type
    if source_type == "yaml":
        instruction = "Explain the following Kubernetes YAML configuration:"
        if metadata and metadata.get("has_kubernetes_terms"):
            instruction = "Analyze and explain this Kubernetes resource configuration:"
    else:
        instruction = "Explain the following Kubernetes concept or documentation:"
        if metadata and metadata.get("title"):
            instruction = f"Explain the Kubernetes concept: {metadata['title']}"
    
    record = {
        "instruction": instruction,
        "input": "",
        "output": chunk,
    }
    
    # Add metadata if available
    if metadata:
        record["metadata"] = {
            "source_type": source_type,
            "content_length": len(chunk),
            "word_count": len(chunk.split()),
            **{k: v for k, v in metadata.items() if k not in ["content_hash"]}  # Exclude content_hash for size
        }
    
    return record


def validate_dataset_record(record: Dict[str, Any]) -> bool:
    """
    Validate a dataset record for completeness and quality.
    
    Args:
        record: Dataset record to validate
        
    Returns:
        True if record is valid, False otherwise
    """
    # Check required fields
    required_fields = ["instruction", "input", "output"]
    for field in required_fields:
        if field not in record or not isinstance(record[field], str):
            logger.debug("record_validation_failed", reason=f"missing_or_invalid_{field}")
            return False
    
    # Check output quality
    output = record["output"].strip()
    if len(output) < settings.min_content_length:
        logger.debug("record_validation_failed", reason="output_too_short", length=len(output))
        return False
    
    # Check for reasonable content
    if len(output.split()) < 5:  # At least 5 words
        logger.debug("record_validation_failed", reason="insufficient_words")
        return False
    
    return True


def format_dataset(dataset: List[Dict[str, Any]], output_file: str) -> bool:
    """
    Format complete dataset and save to file with comprehensive error handling.
    
    Args:
        dataset: List of scraped data items with chunks
        output_file: Path to output file
        
    Returns:
        True if formatting succeeded, False otherwise
        
    Raises:
        DatasetFormattingError: If formatting fails completely
    """
    if not dataset:
        raise DatasetFormattingError("Cannot format empty dataset")
    
    logger.info("starting_dataset_formatting", 
               dataset_size=len(dataset),
               output_file=output_file)
    
    try:
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Format records
        formatted_records = []
        formatting_errors = 0
        
        for item in dataset:
            try:
                chunks = item.get("chunks", [])
                if not chunks:
                    logger.debug("item_skipped_no_chunks", item_type=type(item.get("metadata", {}).get("source_type", "unknown")))
                    continue
                
                # Determine source type
                source_type = "yaml" if "content" in item else "docs"
                metadata = item.get("metadata", {})
                
                # Process each chunk
                for chunk in chunks:
                    if not chunk or not chunk.strip():
                        continue
                    
                    try:
                        record = format_instruction_record(chunk, source_type, metadata)
                        
                        # Validate record
                        if validate_dataset_record(record):
                            formatted_records.append(record)
                        else:
                            formatting_errors += 1
                            logger.debug("record_validation_failed_in_formatting")
                            
                    except Exception as e:
                        formatting_errors += 1
                        logger.warning("chunk_formatting_failed", 
                                     source_type=source_type,
                                     error=str(e))
                        continue
                        
            except Exception as e:
                formatting_errors += 1
                logger.warning("item_formatting_failed", 
                             item_keys=list(item.keys()) if isinstance(item, dict) else "not_dict",
                             error=str(e))
                continue
        
        if not formatted_records:
            raise DatasetFormattingError("No valid records could be formatted from dataset")
        
        # Write to file
        records_written = 0
        write_errors = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in formatted_records:
                try:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records_written += 1
                except Exception as e:
                    write_errors += 1
                    logger.warning("record_write_failed", error=str(e))
                    continue
        
        if records_written == 0:
            raise DatasetFormattingError("No records could be written to output file")
        
        # Log statistics
        file_size = os.path.getsize(output_file)
        logger.info("dataset_formatting_completed",
                   input_items=len(dataset),
                   formatted_records=len(formatted_records),
                   records_written=records_written,
                   formatting_errors=formatting_errors,
                   write_errors=write_errors,
                   output_file=output_file,
                   file_size_bytes=file_size)
        
        return True
        
    except DatasetFormattingError:
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        logger.error("dataset_formatting_unexpected_error",
                    error=str(e),
                    error_type=type(e).__name__)
        raise DatasetFormattingError(f"Unexpected error during dataset formatting: {e}")


def to_instruction_format(dataset: List[Dict[str, Any]], output_file: Optional[str] = None) -> bool:
    """
    Legacy function wrapper for backward compatibility.
    
    Args:
        dataset: List of scraped data items
        output_file: Optional output file path
        
    Returns:
        True if formatting succeeded, False otherwise
    """
    output_path = output_file or settings.output_path
    return format_dataset(dataset, output_path)