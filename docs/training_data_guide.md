# LLM Training Data Creation Guide

## Table of Contents
1. [Overview](#overview)
2. [Data Collection Methods](#data-collection-methods)
3. [Data Quality Requirements](#data-quality-requirements)
4. [Data Preparation Pipeline](#data-preparation-pipeline)
5. [Format Specifications](#format-specifications)
6. [Best Practices](#best-practices)
7. [Tools & Scripts](#tools--scripts)
8. [Common Issues & Solutions](#common-issues--solutions)

---

## Overview

### What Makes Good Training Data?

Training data for Large Language Models must be:
- **Diverse**: Multiple domains, styles, and perspectives
- **High-Quality**: Grammatically correct, factually accurate
- **Large-Scale**: Billions of tokens minimum for modern LLMs
- **Clean**: Free from personal information, toxic content, duplicates
- **Balanced**: Representative across domains and topics

### Training Data Volume Guidelines

| Model Size | Recommended Tokens | Storage (Compressed) |
|------------|-------------------|---------------------|
| Small (125M params) | 5-10B tokens | ~20-40 GB |
| Medium (1.3B params) | 50-100B tokens | ~200-400 GB |
| Large (7B params) | 300-500B tokens | ~1-2 TB |
| Very Large (70B+ params) | 1-2T tokens | ~4-8 TB |

---

## Data Collection Methods

### 1. Web Scraping

**Public Datasets:**
- **Common Crawl**: 250+ TB of web pages monthly
  - URL: https://commoncrawl.org/
  - Format: WARC files
  - Access: Free, S3 buckets
  
- **The Pile**: 825 GB curated dataset
  - Components: Books, GitHub, ArXiv, StackExchange, Wikipedia
  - URL: https://pile.eleuther.ai/
  
- **Wikipedia Dumps**: 20+ GB compressed
  - URL: https://dumps.wikimedia.org/
  - Format: XML
  - Languages: 300+ languages

**Web Scraping Best Practices:**
```python
# Example: Ethical web scraping
import requests
from bs4 import BeautifulSoup
import time
import json

def scrape_with_ethics(urls, output_file, delay=1.0):
    """
    Scrape websites ethically
    
    Args:
        urls: List of URLs to scrape
        output_file: Path to save scraped data
        delay: Delay between requests (respect robots.txt)
    """
    scraped_data = []
    
    for url in urls:
        try:
            # Check robots.txt first
            robots_url = f"{url.split('/')[0]}//{url.split('/')[2]}/robots.txt"
            
            # Respect rate limits
            time.sleep(delay)
            
            # Set user agent
            headers = {
                'User-Agent': 'Educational-Bot/1.0 (your-email@example.com)'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text
            text = soup.get_text(separator='\n', strip=True)
            
            scraped_data.append({
                'url': url,
                'text': text,
                'timestamp': time.time(),
                'length': len(text)
            })
            
            print(f"✓ Scraped: {url} ({len(text)} chars)")
            
        except Exception as e:
            print(f"✗ Failed: {url} - {str(e)}")
    
    # Save to JSONL format
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in scraped_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\nSaved {len(scraped_data)} documents to {output_file}")

# Usage
urls = [
    'https://en.wikipedia.org/wiki/Artificial_intelligence',
    'https://en.wikipedia.org/wiki/Machine_learning',
    # Add more URLs
]
scrape_with_ethics(urls, 'raw_data.jsonl', delay=2.0)
```

### 2. Academic & Research Sources

**ArXiv Papers:**
```bash
# Download ArXiv metadata
wget https://arxiv.org/help/bulk_data_s3

# Extract abstracts
python extract_arxiv.py --output papers.jsonl
```

**PubMed Central:**
- Access: 7+ million full-text articles
- Format: XML, PDF
- API: https://www.ncbi.nlm.nih.gov/pmc/tools/developers/

**GitHub Code:**
```bash
# Clone Google's BigQuery GitHub dataset
# Contains 3TB+ of open source code

# Filter by language and license
SELECT 
    content, 
    path, 
    repo_name
FROM `bigquery-public-data.github_repos.contents`
WHERE language = 'Python'
AND license = 'mit'
LIMIT 1000000
```

### 3. Books & Literature

**Project Gutenberg:**
- 70,000+ free books
- Pre-1928 (public domain)
- URL: https://www.gutenberg.org/

**Books3 (Caution: Copyright issues)**
- Part of The Pile dataset
- Consider legal implications

**OpenLibrary:**
- API access to metadata
- Some full texts available
- URL: https://openlibrary.org/developers/api

### 4. Conversational Data

**Reddit Comments:**
```python
# Using Pushshift Reddit API
import requests

def download_reddit_comments(subreddit, limit=1000):
    """Download Reddit comments for training"""
    url = f"https://api.pushshift.io/reddit/search/comment/"
    params = {
        'subreddit': subreddit,
        'size': limit,
        'sort': 'desc',
        'sort_type': 'created_utc'
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    comments = []
    for comment in data['data']:
        comments.append({
            'text': comment['body'],
            'subreddit': subreddit,
            'score': comment['score'],
            'created_utc': comment['created_utc']
        })
    
    return comments

# Usage
ml_comments = download_reddit_comments('MachineLearning', limit=10000)
```

**Twitter/X Archives:**
- Academic datasets available
- Requires API access
- Consider privacy and terms of service

### 5. Domain-Specific Data

**Medical:**
- MIMIC-III (clinical notes) - requires credentialing
- PubMed abstracts - public access

**Legal:**
- Court opinions (CourtListener)
- Legal briefs (public records)

**Financial:**
- SEC filings (EDGAR)
- Financial news archives

**Scientific:**
- ArXiv preprints
- PubMed Central articles
- Patent databases (USPTO)

---

## Data Quality Requirements

### Quality Metrics

| Metric | Threshold | Check Method |
|--------|-----------|--------------|
| Language detection accuracy | > 95% | langdetect library |
| Duplicate ratio | < 5% | MinHash/LSH |
| Toxicity score | < 0.3 | Perspective API |
| Gibberish score | < 0.2 | Custom classifier |
| Average sentence length | 10-30 words | Statistical analysis |
| Reading level | Grade 8-12 | Flesch-Kincaid |

### Data Quality Script

```python
import json
import hashlib
from typing import Dict, List
import re
from collections import Counter

class DataQualityChecker:
    """Comprehensive data quality validation"""
    
    def __init__(self):
        self.stats = {
            'total_docs': 0,
            'passed': 0,
            'failed_encoding': 0,
            'failed_length': 0,
            'failed_quality': 0,
            'duplicates': 0
        }
        self.seen_hashes = set()
    
    def check_encoding(self, text: str) -> bool:
        """Verify text is valid UTF-8"""
        try:
            text.encode('utf-8').decode('utf-8')
            return True
        except UnicodeDecodeError:
            return False
    
    def check_length(self, text: str, min_length: int = 100, 
                     max_length: int = 1000000) -> bool:
        """Verify text length is reasonable"""
        return min_length <= len(text) <= max_length
    
    def check_duplicate(self, text: str) -> bool:
        """Check for exact duplicates using hash"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.seen_hashes:
            return False
        self.seen_hashes.add(text_hash)
        return True
    
    def check_quality(self, text: str) -> Dict[str, float]:
        """Calculate quality metrics"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Count sentences
        sentences = re.split(r'[.!?]+', text)
        num_sentences = len([s for s in sentences if s.strip()])
        
        # Count words
        words = text.split()
        num_words = len(words)
        
        # Calculate metrics
        avg_word_length = sum(len(w) for w in words) / max(num_words, 1)
        avg_sentence_length = num_words / max(num_sentences, 1)
        
        # Character distribution
        char_counts = Counter(text.lower())
        total_chars = sum(char_counts.values())
        
        # Check for gibberish (too many rare characters)
        alpha_ratio = sum(char_counts[c] for c in 'abcdefghijklmnopqrstuvwxyz ') / max(total_chars, 1)
        
        # Quality score (0-1)
        quality_score = 0.0
        
        # Good average word length (4-7 chars)
        if 4 <= avg_word_length <= 7:
            quality_score += 0.3
        
        # Good sentence length (10-30 words)
        if 10 <= avg_sentence_length <= 30:
            quality_score += 0.3
        
        # Good alpha ratio (>0.85)
        if alpha_ratio > 0.85:
            quality_score += 0.4
        
        return {
            'quality_score': quality_score,
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'alpha_ratio': alpha_ratio,
            'num_words': num_words,
            'num_sentences': num_sentences
        }
    
    def validate_document(self, doc: Dict, min_quality: float = 0.6) -> tuple:
        """
        Validate a single document
        
        Returns:
            (is_valid: bool, metrics: dict)
        """
        self.stats['total_docs'] += 1
        text = doc.get('text', '')
        
        # Check encoding
        if not self.check_encoding(text):
            self.stats['failed_encoding'] += 1
            return False, {'reason': 'encoding_error'}
        
        # Check length
        if not self.check_length(text):
            self.stats['failed_length'] += 1
            return False, {'reason': 'invalid_length'}
        
        # Check duplicates
        if not self.check_duplicate(text):
            self.stats['duplicates'] += 1
            return False, {'reason': 'duplicate'}
        
        # Check quality
        quality_metrics = self.check_quality(text)
        if quality_metrics['quality_score'] < min_quality:
            self.stats['failed_quality'] += 1
            return False, {'reason': 'low_quality', **quality_metrics}
        
        self.stats['passed'] += 1
        return True, quality_metrics
    
    def process_file(self, input_file: str, output_file: str, 
                     min_quality: float = 0.6):
        """Process entire file and filter documents"""
        valid_docs = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    doc = json.loads(line)
                    is_valid, metrics = self.validate_document(doc, min_quality)
                    
                    if is_valid:
                        doc['quality_metrics'] = metrics
                        valid_docs.append(doc)
                    
                    if line_num % 1000 == 0:
                        print(f"Processed {line_num} documents...")
                        
                except Exception as e:
                    print(f"Error on line {line_num}: {str(e)}")
        
        # Save valid documents
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in valid_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        # Print statistics
        print("\n" + "="*60)
        print("DATA QUALITY REPORT")
        print("="*60)
        print(f"Total documents: {self.stats['total_docs']}")
        print(f"Passed: {self.stats['passed']} ({self.stats['passed']/max(self.stats['total_docs'],1)*100:.1f}%)")
        print(f"Failed - Encoding: {self.stats['failed_encoding']}")
        print(f"Failed - Length: {self.stats['failed_length']}")
        print(f"Failed - Quality: {self.stats['failed_quality']}")
        print(f"Failed - Duplicates: {self.stats['duplicates']}")
        print("="*60)

# Usage
checker = DataQualityChecker()
checker.process_file('raw_data.jsonl', 'clean_data.jsonl', min_quality=0.7)
```

---

## Data Preparation Pipeline

### Step-by-Step Process

```
┌─────────────────┐
│  Raw Data       │
│  Collection     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Deduplication  │ ◄── Remove exact/near duplicates
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Filtering      │ ◄── Remove low-quality, toxic content
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Normalization  │ ◄── Standardize encoding, whitespace
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Tokenization   │ ◄── Convert to token IDs
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Chunking       │ ◄── Split into training sequences
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Train/Val Split│ ◄── 99%/1% or 98%/2%
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Final Dataset  │
└─────────────────┘
```

### Complete Pipeline Script

```python
import json
import os
from pathlib import Path
from typing import List, Dict
import multiprocessing as mp
from functools import partial

class DataPreparationPipeline:
    """End-to-end data preparation for LLM training"""
    
    def __init__(self, output_dir: str = 'prepared_data'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def deduplicate(self, input_file: str, output_file: str):
        """Remove duplicate documents using MinHash"""
        from datasketch import MinHash, MinHashLSH
        
        print("[1/6] Deduplicating...")
        lsh = MinHashLSH(threshold=0.8, num_perm=128)
        seen_docs = []
        unique_docs = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                doc = json.loads(line)
                text = doc.get('text', '')
                
                # Create MinHash
                m = MinHash(num_perm=128)
                for word in text.split():
                    m.update(word.encode('utf-8'))
                
                # Check for duplicates
                result = lsh.query(m)
                if not result:
                    lsh.insert(f"doc_{idx}", m)
                    unique_docs.append(doc)
                    
                if (idx + 1) % 10000 == 0:
                    print(f"  Processed {idx + 1} docs, kept {len(unique_docs)}")
        
        # Save unique documents
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in unique_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        print(f"  ✓ Kept {len(unique_docs)} unique documents")
        return output_file
    
    def filter_quality(self, input_file: str, output_file: str):
        """Filter low-quality and toxic content"""
        print("[2/6] Filtering quality...")
        checker = DataQualityChecker()
        checker.process_file(input_file, output_file, min_quality=0.7)
        return output_file
    
    def normalize(self, input_file: str, output_file: str):
        """Normalize text formatting"""
        print("[3/6] Normalizing...")
        
        def normalize_text(text: str) -> str:
            # Remove multiple spaces
            text = ' '.join(text.split())
            # Remove multiple newlines
            text = re.sub(r'\n{3,}', '\n\n', text)
            # Fix common encoding issues
            text = text.replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
            return text
        
        normalized_docs = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                doc['text'] = normalize_text(doc.get('text', ''))
                normalized_docs.append(doc)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in normalized_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        print(f"  ✓ Normalized {len(normalized_docs)} documents")
        return output_file
    
    def tokenize(self, input_file: str, output_file: str, vocab_size: int = 50000):
        """Tokenize text into token IDs"""
        print("[4/6] Tokenizing...")
        
        # Build vocabulary (simplified - use sentencepiece/tiktoken in production)
        vocab = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
        word_freq = Counter()
        
        # Count words
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                words = doc.get('text', '').lower().split()
                word_freq.update(words)
        
        # Add most common words to vocab
        for word, _ in word_freq.most_common(vocab_size - 4):
            vocab[word] = len(vocab)
        
        # Tokenize documents
        tokenized_docs = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                words = doc.get('text', '').lower().split()
                token_ids = [vocab.get(w, vocab['<unk>']) for w in words]
                
                tokenized_docs.append({
                    'token_ids': token_ids,
                    'num_tokens': len(token_ids),
                    'metadata': doc.get('metadata', {})
                })
        
        # Save tokenized data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab': vocab,
                'documents': tokenized_docs
            }, f)
        
        # Save vocabulary separately
        vocab_file = self.output_dir / 'vocab.json'
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, indent=2)
        
        print(f"  ✓ Tokenized {len(tokenized_docs)} documents")
        print(f"  ✓ Vocabulary size: {len(vocab)}")
        return output_file
    
    def chunk_sequences(self, input_file: str, output_file: str, 
                        max_length: int = 2048):
        """Split documents into fixed-length training sequences"""
        print("[5/6] Chunking sequences...")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = data['documents']
        vocab = data['vocab']
        
        sequences = []
        for doc in documents:
            token_ids = doc['token_ids']
            
            # Split into chunks
            for i in range(0, len(token_ids), max_length):
                chunk = token_ids[i:i + max_length]
                
                # Pad if necessary
                if len(chunk) < max_length:
                    chunk += [vocab['<pad>']] * (max_length - len(chunk))
                
                sequences.append(chunk)
        
        # Save sequences
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'sequences': sequences,
                'seq_length': max_length,
                'num_sequences': len(sequences)
            }, f)
        
        print(f"  ✓ Created {len(sequences)} sequences of length {max_length}")
        return output_file
    
    def train_val_split(self, input_file: str, train_ratio: float = 0.99):
        """Split into training and validation sets"""
        print("[6/6] Creating train/val split...")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        sequences = data['sequences']
        total = len(sequences)
        train_size = int(total * train_ratio)
        
        # Shuffle sequences
        import random
        random.seed(42)
        random.shuffle(sequences)
        
        # Split
        train_sequences = sequences[:train_size]
        val_sequences = sequences[train_size:]
        
        # Save train set
        train_file = self.output_dir / 'train.json'
        with open(train_file, 'w') as f:
            json.dump({
                'sequences': train_sequences,
                'seq_length': data['seq_length'],
                'num_sequences': len(train_sequences)
            }, f)
        
        # Save validation set
        val_file = self.output_dir / 'validation.json'
        with open(val_file, 'w') as f:
            json.dump({
                'sequences': val_sequences,
                'seq_length': data['seq_length'],
                'num_sequences': len(val_sequences)
            }, f)
        
        print(f"  ✓ Train set: {len(train_sequences)} sequences")
        print(f"  ✓ Validation set: {len(val_sequences)} sequences")
        
        return str(train_file), str(val_file)
    
    def run(self, input_file: str, max_seq_length: int = 2048):
        """Execute complete pipeline"""
        print("\n" + "="*60)
        print("DATA PREPARATION PIPELINE")
        print("="*60 + "\n")
        
        # Step 1: Deduplicate
        dedup_file = self.output_dir / '01_dedup.jsonl'
        self.deduplicate(input_file, str(dedup_file))
        
        # Step 2: Filter
        filtered_file = self.output_dir / '02_filtered.jsonl'
        self.filter_quality(str(dedup_file), str(filtered_file))
        
        # Step 3: Normalize
        normalized_file = self.output_dir / '03_normalized.jsonl'
        self.normalize(str(filtered_file), str(normalized_file))
        
        # Step 4: Tokenize
        tokenized_file = self.output_dir / '04_tokenized.json'
        self.tokenize(str(normalized_file), str(tokenized_file))
        
        # Step 5: Chunk
        chunked_file = self.output_dir / '05_chunked.json'
        self.chunk_sequences(str(tokenized_file), str(chunked_file), max_seq_length)
        
        # Step 6: Split
        train_file, val_file = self.train_val_split(str(chunked_file))
        
        print("\n" + "="*60)
        print("✓ PIPELINE COMPLETE")
        print("="*60)
        print(f"\nOutput files:")
        print(f"  - Training: {train_file}")
        print(f"  - Validation: {val_file}")
        print(f"  - Vocabulary: {self.output_dir / 'vocab.json'}")
        print("="*60 + "\n")

# Usage
pipeline = DataPreparationPipeline(output_dir='training_data')
pipeline.run('raw_data.jsonl', max_seq_length=2048)
```

---

## Format Specifications

### Input Format (JSONL)

Each line is a JSON object:

```json
{
  "text": "The actual text content goes here...",
  "metadata": {
    "source": "wikipedia",
    "url": "https://en.wikipedia.org/wiki/Example",
    "timestamp": "2024-01-15T10:30:00Z",
    "language": "en",
    "domain": "science"
  }
}
```

### Processed Format (Training-Ready)

```json
{
  "sequences": [
    [101, 2023, 2003, 1037, 7953, ...],
    [101, 2178, 7953, 2003, 2182, ...]
  ],
  "seq_length": 2048,
  "num_sequences": 1000000,
  "vocabulary_size": 50000
}
```

### Metadata Schema

Recommended metadata fields:

```json
{
  "doc_id": "unique-identifier",
  "source": "source-name",
  "url": "original-url",
  "timestamp": "ISO-8601-datetime",
  "language": "en",
  "domain": "technology",
  "quality_score": 0.87,
  "word_count": 1523,
  "license": "CC-BY-SA"
}
```

---

## Best Practices

### 1. Data Diversity

**Ensure balanced representation:**
```python
# Check domain distribution
from collections import Counter

def analyze_domain_distribution(input_file):
    """Analyze data distribution across domains"""
    domains = []
    
    with open(input_file, 'r') as f:
        for line in f:
            doc = json.loads(line)
            domain = doc.get('metadata', {}).get('domain', 'unknown')
            domains.append(domain)
    
    distribution = Counter(domains)
    total = len(domains)
    
    print("Domain Distribution:")
    for domain, count in distribution.most_common():
        percentage = (count / total) * 100
        print(f"  {domain}: {count:,} ({percentage:.1f}%)")
    
    # Calculate diversity score (entropy)
    import math
    entropy = -sum((count/total) * math.log2(count/total) 
                   for count in distribution.values())
    print(f"\nDiversity Score (Entropy): {entropy:.2f}")
    print(f"Maximum possible: {math.log2(len(distribution)):.2f}")

analyze_domain_distribution('clean_data.jsonl')
```

### 2. Privacy Protection

**Remove Personal Identifiable Information (PII):**

```python
import re

def redact_pii(text):
    """Remove common PII patterns"""
    
    # Email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                  '[EMAIL]', text)
    
    # Phone numbers (US format)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    
    # Social Security Numbers
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
    
    # Credit card numbers
    text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD]', text)
    
    # IP addresses
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP]', text)
    
    # Dates (to reduce memorization of specific events)
    text = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '[DATE]', text)
    
    return text
```

### 3. License Compliance

**Track and respect licenses:**

```python
ALLOWED_LICENSES = [
    'CC0',
    'CC-BY',
    'CC-BY-SA',
    'MIT',
    'Apache-2.0',
    'Public Domain'
]

def filter_by_license(input_file, output_file):
    """Keep only documents with permissive licenses"""
    kept = 0
    removed = 0
    
    with open(output_file, 'w') as out:
        with open(input_file, 'r') as inp:
            for line in inp:
                doc = json.loads(line)
                license = doc.get('metadata', {}).get('license', 'unknown')
                
                if license in ALLOWED_LICENSES:
                    out.write(line)
                    kept += 1
                else:
                    removed += 1
    
    print(f"Kept: {kept}, Removed: {removed}")
```

### 4. Deduplication Strategies

**Near-duplicate detection:**

```python
from datasketch import MinHash, MinHashLSH

def find_near_duplicates(documents, threshold=0.8):
    """Find near-duplicate documents using LSH"""
    
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    duplicates = []
    unique_docs = []
    
    for idx, doc in enumerate(documents):
        # Create MinHash signature
        m = MinHash(num_perm=128)
        for word in doc['text'].split():
            m.update(word.encode('utf-8'))
        
        # Check for near-duplicates
        result = lsh.query(m)
        
        if result:
            duplicates.append({
                'doc_id': idx,
                'similar_to': result[0],
                'text_preview': doc['text'][:100]
            })
        else:
            lsh.insert(str(idx), m)
            unique_docs.append(doc)
    
    print(f"Found {len(duplicates)} near-duplicates")
    print(f"Keeping {len(unique_docs)} unique documents")
    
    return unique_docs, duplicates
```

### 5. Content Filtering

**Remove toxic/harmful content:**

```python
# Using simple keyword filtering (use ML models in production)
TOXIC_KEYWORDS = [
    'explicit-word-1', 'explicit-word-2',  # Add actual words
    # ... more keywords
]

def filter_toxic_content(text, threshold=3):
    """Filter content with toxic keywords"""
    text_lower = text.lower()
    toxic_count = sum(1 for word in TOXIC_KEYWORDS if word in text_lower)
    
    return toxic_count < threshold

def clean_dataset(input_file, output_file):
    """Remove toxic content from dataset"""
    kept = 0
    removed = 0
    
    with open(output_file, 'w') as out:
        with open(input_file, 'r') as inp:
            for line in inp:
                doc = json.loads(line)
                
                if filter_toxic_content(doc['text']):
                    out.write(line)
                    kept += 1
                else:
                    removed += 1
    
    print(f"Toxic content filter: Kept {kept}, Removed {removed}")
```

---

## Tools & Scripts

### Essential Libraries

```bash
# Install required packages
pip install datasketch         # MinHash deduplication
pip install langdetect          # Language detection
pip install sentencepiece       # Tokenization
pip install tiktoken            # OpenAI tokenizer
pip install datasets            # HuggingFace datasets
pip install beautifulsoup4      # Web scraping
pip install requests            # HTTP requests
pip install tqdm                # Progress bars
pip install pandas              # Data manipulation
pip install nltk                # NLP utilities
```

### Production-Grade Tokenizer Setup

```python
import sentencepiece as spm
import json

def train_tokenizer(input_file, vocab_size=50000, model_prefix='tokenizer'):
    """
    Train SentencePiece tokenizer on your data
    
    Args:
        input_file: Path to text file (one sentence per line)
        vocab_size: Target vocabulary size
        model_prefix: Output model name
    """
    
    # Extract text from JSONL
    texts = []
    with open(input_file, 'r') as f:
        for line in f:
            doc = json.loads(line)
            texts.append(doc['text'])
    
    # Write to temporary file
    temp_file = 'temp_train.txt'
    with open(temp_file, 'w', encoding='utf-8') as f:
        for text in texts:
            # One sentence per line
            sentences = text.split('.')
            for sent in sentences:
                sent = sent.strip()
                if sent:
                    f.write(sent + '\n')
    
    # Train SentencePiece
    spm.SentencePieceTrainer.train(
        input=temp_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type='bpe',  # or 'unigram', 'char', 'word'
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece='<pad>',
        unk_piece='<unk>',
        bos_piece='<bos>',
        eos_piece='<eos>',
        user_defined_symbols=['<mask>'],
    )
    
    print(f"✓ Tokenizer trained: {model_prefix}.model")
    print(f"✓ Vocabulary: {model_prefix}.vocab")
    
    # Test tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(f'{model_prefix}.model')
    
    test_text = "This is a test sentence for tokenization."
    tokens = sp.encode_as_pieces(test_text)
    ids = sp.encode_as_ids(test_text)
    
    print(f"\nTest tokenization:")
    print(f"Text: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"IDs: {ids}")
    
    # Clean up
    import os
    os.remove(temp_file)

# Usage
train_tokenizer('clean_data.jsonl', vocab_size=50000, model_prefix='my_tokenizer')
```

### Parallel Processing Script

```python
from multiprocessing import Pool, cpu_count
from functools import partial
import json

def process_document_batch(batch, processor_func):
    """Process a batch of documents"""
    results = []
    for doc in batch:
        try:
            result = processor_func(doc)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error processing document: {e}")
    return results

def parallel_process(input_file, output_file, processor_func, 
                     batch_size=1000, n_workers=None):
    """
    Process large files in parallel
    
    Args:
        input_file: Input JSONL file
        output_file: Output JSONL file
        processor_func: Function to process each document
        batch_size: Documents per batch
        n_workers: Number of workers (default: CPU count)
    """
    
    if n_workers is None:
        n_workers = cpu_count()
    
    print(f"Processing with {n_workers} workers...")
    
    # Read documents in batches
    batches = []
    current_batch = []
    
    with open(input_file, 'r') as f:
        for line in f:
            doc = json.loads(line)
            current_batch.append(doc)
            
            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []
        
        if current_batch:
            batches.append(current_batch)
    
    print(f"Created {len(batches)} batches")
    
    # Process in parallel
    with Pool(n_workers) as pool:
        process_func = partial(process_document_batch, processor_func=processor_func)
        results = pool.map(process_func, batches)
    
    # Flatten results and write
    total_processed = 0
    with open(output_file, 'w') as f:
        for batch_results in results:
            for doc in batch_results:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                total_processed += 1
    
    print(f"✓ Processed {total_processed} documents")

# Example processor function
def example_processor(doc):
    """Example: Clean and validate document"""
    text = doc.get('text', '')
    
    # Clean text
    text = text.strip()
    if len(text) < 100:  # Skip short documents
        return None
    
    # Update document
    doc['text'] = text
    doc['word_count'] = len(text.split())
    
    return doc

# Usage
parallel_process('raw_data.jsonl', 'processed_data.jsonl', 
                example_processor, batch_size=1000)
```

### Dataset Statistics Script

```python
import json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def analyze_dataset(input_file):
    """Generate comprehensive dataset statistics"""
    
    stats = {
        'total_docs': 0,
        'total_tokens': 0,
        'doc_lengths': [],
        'domains': Counter(),
        'languages': Counter(),
        'sources': Counter(),
    }
    
    print("Analyzing dataset...")
    
    with open(input_file, 'r') as f:
        for line in f:
            doc = json.loads(line)
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            
            # Count
            stats['total_docs'] += 1
            word_count = len(text.split())
            stats['total_tokens'] += word_count
            stats['doc_lengths'].append(word_count)
            
            # Metadata
            stats['domains'][metadata.get('domain', 'unknown')] += 1
            stats['languages'][metadata.get('language', 'unknown')] += 1
            stats['sources'][metadata.get('source', 'unknown')] += 1
    
    # Print report
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    print(f"\nTotal Documents: {stats['total_docs']:,}")
    print(f"Total Tokens: {stats['total_tokens']:,}")
    print(f"Average Document Length: {np.mean(stats['doc_lengths']):.1f} tokens")
    print(f"Median Document Length: {np.median(stats['doc_lengths']):.1f} tokens")
    print(f"Min Document Length: {min(stats['doc_lengths']):,} tokens")
    print(f"Max Document Length: {max(stats['doc_lengths']):,} tokens")
    
    print(f"\nTop 10 Domains:")
    for domain, count in stats['domains'].most_common(10):
        pct = (count / stats['total_docs']) * 100
        print(f"  {domain}: {count:,} ({pct:.1f}%)")
    
    print(f"\nLanguages:")
    for lang, count in stats['languages'].most_common():
        pct = (count / stats['total_docs']) * 100
        print(f"  {lang}: {count:,} ({pct:.1f}%)")
    
    print(f"\nTop 10 Sources:")
    for source, count in stats['sources'].most_common(10):
        pct = (count / stats['total_docs']) * 100
        print(f"  {source}: {count:,} ({pct:.1f}%)")
    
    print("="*60)
    
    # Generate histogram
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(stats['doc_lengths'], bins=50, edgecolor='black')
    plt.xlabel('Document Length (tokens)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Document Lengths')
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    top_domains = stats['domains'].most_common(15)
    domains, counts = zip(*top_domains)
    plt.barh(range(len(domains)), counts)
    plt.yticks(range(len(domains)), domains)
    plt.xlabel('Number of Documents')
    plt.title('Top 15 Domains')
    
    plt.tight_layout()
    plt.savefig('dataset_stats.png', dpi=300)
    print(f"\n✓ Saved visualization: dataset_stats.png")

# Usage
analyze_dataset('clean_data.jsonl')
```

---

## Common Issues & Solutions

### Issue 1: Memory Overflow

**Problem:** Dataset too large to fit in memory

**Solution:**
```python
def stream_process_large_file(input_file, output_file, processor_func):
    """Process files without loading entire dataset into memory"""
    
    with open(output_file, 'w') as out:
        with open(input_file, 'r') as inp:
            for i, line in enumerate(inp):
                try:
                    doc = json.loads(line)
                    processed = processor_func(doc)
                    
                    if processed:
                        out.write(json.dumps(processed, ensure_ascii=False) + '\n')
                    
                    if (i + 1) % 10000 == 0:
                        print(f"Processed {i + 1} documents...")
                        
                except Exception as e:
                    print(f"Error on line {i + 1}: {e}")
```

### Issue 2: Encoding Errors

**Problem:** Mixed encodings in source data

**Solution:**
```python
import chardet

def detect_and_convert_encoding(file_path):
    """Detect and convert file to UTF-8"""
    
    # Detect encoding
    with open(file_path, 'rb') as f:
        raw_data = f.read(100000)  # Sample first 100KB
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
    
    print(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
    
    # Convert to UTF-8
    output_file = file_path.replace('.txt', '_utf8.txt')
    
    with open(file_path, 'r', encoding=encoding, errors='ignore') as inp:
        with open(output_file, 'w', encoding='utf-8') as out:
            for line in inp:
                out.write(line)
    
    print(f"✓ Converted to UTF-8: {output_file}")
    return output_file
```

### Issue 3: Imbalanced Dataset

**Problem:** Some domains over-represented

**Solution:**
```python
def balance_dataset(input_file, output_file, max_per_domain=10000):
    """Balance dataset by limiting samples per domain"""
    
    domain_counts = Counter()
    balanced_docs = []
    
    with open(input_file, 'r') as f:
        for line in f:
            doc = json.loads(line)
            domain = doc.get('metadata', {}).get('domain', 'unknown')
            
            if domain_counts[domain] < max_per_domain:
                balanced_docs.append(doc)
                domain_counts[domain] += 1
    
    # Shuffle to mix domains
    import random
    random.shuffle(balanced_docs)
    
    # Write balanced dataset
    with open(output_file, 'w') as f:
        for doc in balanced_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    print(f"✓ Balanced dataset: {len(balanced_docs)} documents")
    print("\nDomain distribution:")
    for domain, count in domain_counts.most_common():
        print(f"  {domain}: {count}")
```

### Issue 4: Slow Processing

**Problem:** Processing takes too long

**Solutions:**

1. **Use faster JSON parser:**
```python
import orjson  # Faster than standard json

def fast_json_parse(input_file):
    """Use orjson for faster parsing"""
    with open(input_file, 'rb') as f:
        for line in f:
            doc = orjson.loads(line)
            # Process doc
```

2. **Batch processing:**
```python
def batch_process(items, batch_size=1000):
    """Process items in batches"""
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        # Process batch
        yield batch
```

3. **Use memory mapping for large files:**
```python
import mmap

def mmap_process(file_path):
    """Use memory mapping for efficient file access"""
    with open(file_path, 'r+b') as f:
        with mmap.mmap(f.fileno(), 0) as m:
            # Process memory-mapped file
            content = m.read()
```

### Issue 5: Duplicate URLs/Sources

**Problem:** Same content from different URLs

**Solution:**
```python
def deduplicate_by_url_pattern(input_file, output_file):
    """Remove duplicates based on URL patterns"""
    
    seen_patterns = set()
    unique_docs = []
    
    with open(input_file, 'r') as f:
        for line in f:
            doc = json.loads(line)
            url = doc.get('metadata', {}).get('url', '')
            
            # Extract domain + path (ignore query params)
            from urllib.parse import urlparse
            parsed = urlparse(url)
            pattern = f"{parsed.netloc}{parsed.path}"
            
            if pattern not in seen_patterns:
                seen_patterns.add(pattern)
                unique_docs.append(doc)
    
    with open(output_file, 'w') as f:
        for doc in unique_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    print(f"✓ Removed {len(seen_patterns) - len(unique_docs)} URL duplicates")
```

---

## Advanced Topics

### Multi-Modal Data (Text + Images)

For vision-language models:

```python
import base64
from PIL import Image
import io

def prepare_multimodal_data(text, image_path):
    """Prepare text-image pairs for training"""
    
    # Load and encode image
    with Image.open(image_path) as img:
        # Resize if needed
        img = img.resize((224, 224))
        
        # Convert to bytes
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        img_bytes = buffer.getvalue()
        
        # Base64 encode
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    
    # Create multimodal document
    return {
        'text': text,
        'image': img_b64,
        'image_format': 'jpeg',
        'image_size': [224, 224]
    }
```

### Instruction-Tuning Data

For instruction-following models:

```python
def create_instruction_data(prompt, response, system_prompt=None):
    """Format instruction-tuning examples"""
    
    return {
        'instruction': prompt,
        'response': response,
        'system': system_prompt or "You are a helpful assistant.",
        'format': 'instruction'
    }

# Example
instruction_data = create_instruction_data(
    prompt="What is the capital of France?",
    response="The capital of France is Paris.",
    system_prompt="You are a helpful geography assistant."
)
```

### Synthetic Data Generation

Using GPT models to generate training data:

```python
def generate_synthetic_examples(seed_examples, num_to_generate=1000):
    """
    Generate synthetic training examples
    Note: This is pseudocode - requires API access
    """
    
    synthetic_data = []
    
    for i in range(num_to_generate):
        # Use seed examples as few-shot prompt
        prompt = f"""Generate a similar example to these:

{seed_examples}

Generate example {i+1}:"""
        
        # Call LLM API (pseudocode)
        # response = llm_api.complete(prompt)
        # synthetic_data.append(response)
    
    return synthetic_data
```

---

## Checklist: Ready for Training?

Before starting training, verify:

- [ ] **Data Volume**: Sufficient tokens for model size
- [ ] **Quality Score**: Average quality > 0.7
- [ ] **Deduplication**: < 5% duplicates
- [ ] **Diversity**: Multiple domains represented
- [ ] **Toxicity**: Harmful content filtered
- [ ] **PII Removed**: No personal information
- [ ] **License Compliance**: All sources verified
- [ ] **Encoding**: UTF-8 throughout
- [ ] **Tokenization**: Vocabulary trained
- [ ] **Split Created**: Train/validation/test sets
- [ ] **Statistics Generated**: Know your data
- [ ] **Backups Created**: Data safely stored

---

## Summary

Creating training data for LLMs involves:

1. **Collection**: Gather diverse, high-quality sources
2. **Quality Control**: Filter, validate, clean
3. **Preprocessing**: Normalize, tokenize, chunk
4. **Organization**: Split and format for training
5. **Validation**: Verify quality and diversity

**Key Principles:**
- Quality over quantity
- Respect copyright and privacy
- Document everything
- Test at each stage
- Monitor diversity

**Next Steps:**
1. Collect raw data from approved sources
2. Run quality pipeline
3. Train tokenizer
4. Create final datasets
5. Begin training with small model first
6. Iterate based on results

Good luck with your LLM training! 🚀
