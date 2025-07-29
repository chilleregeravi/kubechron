"""
Text parsing and chunking utilities with configurable parameters.
Handles content segmentation for optimal dataset creation.
"""
from typing import List, Optional
import re

from config import settings
from logger import logger


def chunk_text(text: str, max_len: Optional[int] = None) -> List[str]:
    """
    Split text into chunks with intelligent boundary detection.
    
    Args:
        text: Text to be chunked
        max_len: Maximum length per chunk (uses config default if None)
        
    Returns:
        List of text chunks
    """
    if not text or not isinstance(text, str):
        logger.debug("chunk_text_invalid_input", text_type=type(text).__name__)
        return []
    
    max_length = max_len or settings.max_chunk_length
    
    # Clean and normalize text
    text = text.strip()
    if not text:
        return []
    
    # Split by lines for processing
    lines = text.split("\n")
    chunks = []
    current_chunk = []
    current_length = 0
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines but preserve paragraph breaks
        if not line:
            if current_chunk and not current_chunk[-1].endswith('\n'):
                current_chunk.append('')  # Add paragraph break
            continue
        
        line_length = len(line)
        
        # If adding this line would exceed max length, finalize current chunk
        if current_length + line_length + 1 > max_length and current_chunk:
            # Join current chunk and add to chunks
            chunk_text = '\n'.join(current_chunk).strip()
            if chunk_text:  # Only add non-empty chunks
                chunks.append(chunk_text)
            
            # Start new chunk
            current_chunk = []
            current_length = 0
        
        # Handle very long lines that exceed max_length on their own
        if line_length > max_length:
            # If we have accumulated content, save it first
            if current_chunk:
                chunk_text = '\n'.join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                current_chunk = []
                current_length = 0
            
            # Split the long line intelligently
            sub_chunks = _split_long_line(line, max_length)
            chunks.extend(sub_chunks)
        else:
            # Add line to current chunk
            current_chunk.append(line)
            current_length += line_length + 1  # +1 for newline
    
    # Handle remaining content
    if current_chunk:
        chunk_text = '\n'.join(current_chunk).strip()
        if chunk_text:
            chunks.append(chunk_text)
    
    # Filter out chunks that are too short to be useful
    min_chunk_length = max(10, settings.min_content_length // 2)  # Reasonable minimum
    filtered_chunks = [chunk for chunk in chunks if len(chunk.strip()) >= min_chunk_length]
    
    logger.debug("text_chunked", 
                original_length=len(text),
                chunks_created=len(filtered_chunks),
                chunks_filtered_out=len(chunks) - len(filtered_chunks),
                max_chunk_length=max_length)
    
    return filtered_chunks


def _split_long_line(line: str, max_length: int) -> List[str]:
    """
    Split a very long line into smaller chunks with intelligent boundaries.
    
    Args:
        line: Long line to split
        max_length: Maximum length per chunk
        
    Returns:
        List of line chunks
    """
    if len(line) <= max_length:
        return [line]
    
    chunks = []
    
    # Try to split on sentence boundaries first
    sentences = re.split(r'(?<=[.!?])\s+', line)
    
    current_chunk = ""
    for sentence in sentences:
        # If adding this sentence would exceed max length
        if len(current_chunk) + len(sentence) + 1 > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # If the sentence itself is too long, split it further
            if len(sentence) > max_length:
                word_chunks = _split_by_words(sentence, max_length)
                chunks.extend(word_chunks)
            else:
                current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Add remaining content
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def _split_by_words(text: str, max_length: int) -> List[str]:
    """
    Split text by words when other methods fail.
    
    Args:
        text: Text to split
        max_length: Maximum length per chunk
        
    Returns:
        List of word-based chunks
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word)
        
        # If adding this word would exceed max length
        if current_length + word_length + len(current_chunk) > max_length:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
        
        current_chunk.append(word)
        current_length += word_length
    
    # Add remaining words
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks