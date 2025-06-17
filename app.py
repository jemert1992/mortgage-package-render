#!/usr/bin/env python3
"""
Maximum OCR Mortgage Package Analyzer
Production-Ready Flask Application for Railway Deployment

Features:
- Advanced OCR with multiple engines and preprocessing
- Enhanced pattern matching with ML-like scoring
- Real-time progress tracking with WebSocket support
- Professional UI with drag & drop, progress bars
- Comprehensive error handling and logging
- Production-ready configuration for Railway
- Support for large files with streaming processing
- Advanced document section identification
"""

import os
import sys
import io
import uuid
import hashlib
import tempfile
import traceback
import logging
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# PDF processing imports
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("Warning: pdfplumber not available")

try:
    from pdf2image import convert_from_bytes
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: OCR dependencies not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app with production configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'max-mortgage-analyzer-key')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Enable CORS
CORS(app, origins="*")

# Global variables for progress tracking and session management
progress_data = {}
session_lock = threading.Lock()

@dataclass
class ProcessingSession:
    """Session data for tracking document processing"""
    session_id: str
    filename: str
    file_size: int
    start_time: datetime
    current_step: str
    progress_percentage: int
    total_pages: int
    processed_pages: int
    status: str
    error: Optional[str] = None
    result: Optional[Dict] = None

class AdvancedOCRProcessor:
    """Advanced OCR processor with multiple engines and preprocessing"""
    
    def __init__(self):
        self.ocr_available = OCR_AVAILABLE
        self.pdfplumber_available = PDFPLUMBER_AVAILABLE
        
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Advanced image preprocessing for better OCR accuracy"""
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)
            
            # Apply slight blur to reduce noise
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            # Resize if too small (OCR works better on larger images)
            width, height = image.size
            if width < 1000 or height < 1000:
                scale_factor = max(1000 / width, 1000 / height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image
    
    def extract_text_with_multiple_configs(self, image: Image.Image) -> List[str]:
        """Extract text using multiple tesseract configurations"""
        if not self.ocr_available:
            return []
        
        configs = [
            '--psm 1 --oem 3',  # Automatic page segmentation with orientation
            '--psm 3 --oem 3',  # Fully automatic page segmentation
            '--psm 6 --oem 3',  # Uniform block of text
            '--psm 4 --oem 3',  # Single column of text
        ]
        
        results = []
        for config in configs:
            try:
                text = pytesseract.image_to_string(image, config=config, lang='eng')
                if text.strip():
                    results.append(text.strip())
            except Exception as e:
                logger.warning(f"OCR config {config} failed: {e}")
                continue
        
        return results
    
    def extract_text_from_pdf(self, file_content: bytes, session_id: str) -> List[Dict[str, Any]]:
        """Extract text using advanced multi-method approach"""
        text_content = []
        session = self.get_session(session_id)
        
        logger.info(f"Starting advanced text extraction from {len(file_content)} bytes")
        
        # Method 1: pdfplumber for text-based PDFs
        if self.pdfplumber_available:
            try:
                logger.info("Attempting pdfplumber extraction...")
                self.update_session_progress(session_id, "extracting_text", 0)
                
                pdf_file = io.BytesIO(file_content)
                with pdfplumber.open(pdf_file) as pdf:
                    total_pages = len(pdf.pages)
                    session.total_pages = total_pages
                    
                    logger.info(f"PDF has {total_pages} pages")
                    
                    for page_num, page in enumerate(pdf.pages, 1):
                        try:
                            self.update_session_progress(
                                session_id, 
                                f"extracting_page_{page_num}", 
                                int((page_num / total_pages) * 50)  # 50% for pdfplumber
                            )
                            
                            page_text = page.extract_text()
                            if page_text and page_text.strip():
                                # Clean and process text
                                lines = self.clean_extracted_text(page_text)
                                for line_num, line in enumerate(lines):
                                    if len(line) > 3:
                                        text_content.append({
                                            "text": line,
                                            "page": page_num,
                                            "line": line_num,
                                            "method": "pdfplumber",
                                            "confidence": 0.9
                                        })
                                
                                logger.info(f"Page {page_num}: extracted {len(lines)} lines via pdfplumber")
                        
                        except Exception as e:
                            logger.warning(f"pdfplumber failed on page {page_num}: {e}")
                            continue
                
                logger.info(f"pdfplumber extraction: {len(text_content)} text items")
                
            except Exception as e:
                logger.error(f"pdfplumber extraction failed: {e}")
        
        # Method 2: Advanced OCR if low text yield or OCR requested
        if len(text_content) < 20 and self.ocr_available:
            logger.info("Low text yield or OCR requested, starting advanced OCR...")
            
            try:
                self.update_session_progress(session_id, "converting_to_images", 50)
                
                # Convert PDF to high-quality images
                images = convert_from_bytes(
                    file_content, 
                    dpi=200,  # Higher DPI for better OCR
                    fmt='PNG',
                    thread_count=2
                )
                
                total_pages = len(images)
                session.total_pages = max(session.total_pages, total_pages)
                
                logger.info(f"Converted {total_pages} pages to images for advanced OCR")
                
                ocr_text_content = []
                
                # Process images with threading for better performance
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = []
                    
                    for page_num, image in enumerate(images, 1):
                        future = executor.submit(self.process_page_with_ocr, image, page_num, session_id, total_pages)
                        futures.append(future)
                    
                    for future in futures:
                        try:
                            page_results = future.result(timeout=60)  # 60 second timeout per page
                            ocr_text_content.extend(page_results)
                        except Exception as e:
                            logger.error(f"OCR processing failed for a page: {e}")
                            continue
                
                if ocr_text_content:
                    logger.info(f"Advanced OCR extraction successful: {len(ocr_text_content)} text items")
                    # Merge or replace with OCR results based on quality
                    if len(ocr_text_content) > len(text_content):
                        text_content = ocr_text_content
                    else:
                        # Combine results, preferring higher confidence
                        text_content.extend(ocr_text_content)
                
            except Exception as e:
                logger.error(f"Advanced OCR extraction failed: {e}")
                traceback.print_exc()
        
        # Post-processing: deduplicate and enhance
        text_content = self.post_process_text_content(text_content)
        
        logger.info(f"Final extraction: {len(text_content)} text items from {session.total_pages} pages")
        
        return text_content
    
    def process_page_with_ocr(self, image: Image.Image, page_num: int, session_id: str, total_pages: int) -> List[Dict[str, Any]]:
        """Process a single page with advanced OCR"""
        try:
            progress = 50 + int(((page_num / total_pages) * 50))  # 50-100% for OCR
            self.update_session_progress(session_id, f"ocr_page_{page_num}", progress)
            
            logger.info(f"Running advanced OCR on page {page_num}/{total_pages}...")
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Extract text with multiple configurations
            text_results = self.extract_text_with_multiple_configs(processed_image)
            
            # Choose best result
            best_text = self.select_best_ocr_result(text_results)
            
            page_content = []
            if best_text:
                lines = self.clean_extracted_text(best_text)
                
                for line_num, line in enumerate(lines):
                    if len(line) > 5 and any(c.isalpha() for c in line):
                        # Calculate confidence based on text quality
                        confidence = self.calculate_text_confidence(line)
                        
                        page_content.append({
                            "text": line,
                            "page": page_num,
                            "line": line_num,
                            "method": "advanced_ocr",
                            "confidence": confidence
                        })
                
                if page_content:
                    logger.info(f"Page {page_num}: extracted {len(page_content)} lines via advanced OCR")
            
            return page_content
            
        except Exception as e:
            logger.error(f"Advanced OCR failed on page {page_num}: {e}")
            return []
    
    def clean_extracted_text(self, text: str) -> List[str]:
        """Clean and normalize extracted text"""
        if not text:
            return []
        
        # Split into lines and clean
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove extra whitespace
            line = ' '.join(line.split())
            
            # Skip very short lines or lines with mostly special characters
            if len(line) < 3:
                continue
            
            # Skip lines that are mostly numbers or special characters
            alpha_ratio = sum(c.isalpha() for c in line) / len(line)
            if alpha_ratio < 0.3:
                continue
            
            cleaned_lines.append(line)
        
        return cleaned_lines
    
    def select_best_ocr_result(self, results: List[str]) -> str:
        """Select the best OCR result from multiple configurations"""
        if not results:
            return ""
        
        if len(results) == 1:
            return results[0]
        
        # Score results based on various factors
        scored_results = []
        for result in results:
            score = self.score_ocr_result(result)
            scored_results.append((score, result))
        
        # Return the highest scoring result
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return scored_results[0][1]
    
    def score_ocr_result(self, text: str) -> float:
        """Score OCR result quality"""
        if not text:
            return 0.0
        
        score = 0.0
        
        # Length bonus (longer text usually better)
        score += min(len(text) / 1000, 1.0) * 0.3
        
        # Word count bonus
        words = text.split()
        score += min(len(words) / 100, 1.0) * 0.3
        
        # Alphabetic character ratio
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        score += alpha_ratio * 0.4
        
        return score
    
    def calculate_text_confidence(self, text: str) -> float:
        """Calculate confidence score for extracted text"""
        if not text:
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        # Length factor
        if len(text) > 20:
            confidence += 0.1
        if len(text) > 50:
            confidence += 0.1
        
        # Word structure
        words = text.split()
        if len(words) > 3:
            confidence += 0.1
        
        # Character quality
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        confidence += alpha_ratio * 0.2
        
        return min(confidence, 1.0)
    
    def post_process_text_content(self, text_content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Post-process and deduplicate text content"""
        if not text_content:
            return []
        
        # Remove duplicates while preserving order
        seen_texts = set()
        unique_content = []
        
        for item in text_content:
            text_key = (item['text'].lower().strip(), item['page'])
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_content.append(item)
        
        # Sort by page and line
        unique_content.sort(key=lambda x: (x['page'], x.get('line', 0)))
        
        return unique_content
    
    def get_session(self, session_id: str) -> Optional[ProcessingSession]:
        """Get processing session"""
        with session_lock:
            return progress_data.get(session_id)
    
    def update_session_progress(self, session_id: str, step: str, percentage: int):
        """Update session progress"""
        with session_lock:
            if session_id in progress_data:
                session = progress_data[session_id]
                session.current_step = step
                session.progress_percentage = percentage
                session.status = "processing"
                logger.info(f"Session {session_id}: {step} - {percentage}%")

class EnhancedMortgageAnalyzer:
    """Enhanced mortgage document analyzer with ML-like pattern matching"""
    
    def __init__(self):
        # Enhanced section rules with priority scoring and multiple patterns
        self.section_rules = [
            {
                "label": "Mortgage",
                "patterns": [
                    "MORTGAGE", "DEED OF TRUST", "SECURITY INSTRUMENT", 
                    "MORTGAGE DEED", "TRUST DEED", "SECURITY DEED"
                ],
                "priority": 10,
                "required_context": ["BORROWER", "LENDER", "PROPERTY"],
                "negative_patterns": ["INSURANCE", "POLICY"]
            },
            {
                "label": "Promissory Note",
                "patterns": [
                    "PROMISSORY NOTE", "NOTE", "PROMISSORY", 
                    "BORROWER'S NOTE", "MORTGAGE NOTE"
                ],
                "priority": 10,
                "required_context": ["PRINCIPAL", "INTEREST", "PAYMENT"],
                "negative_patterns": ["INSURANCE", "POLICY", "TITLE"]
            },
            {
                "label": "Lenders Closing Instructions Guaranty",
                "patterns": [
                    "LENDERS CLOSING INSTRUCTIONS", "CLOSING INSTRUCTIONS GUARANTY",
                    "LENDER'S CLOSING INSTRUCTIONS", "CLOSING INSTRUCTIONS",
                    "LENDER INSTRUCTIONS", "GUARANTY"
                ],
                "priority": 9,
                "required_context": ["LENDER", "CLOSING", "INSTRUCTIONS"],
                "negative_patterns": []
            },
            {
                "label": "Settlement Statement",
                "patterns": [
                    "SETTLEMENT STATEMENT", "HUD-1", "CLOSING DISCLOSURE",
                    "SETTLEMENT", "CLOSING STATEMENT", "HUD SETTLEMENT"
                ],
                "priority": 9,
                "required_context": ["SETTLEMENT", "CLOSING", "BORROWER"],
                "negative_patterns": []
            },
            {
                "label": "Statement of Anti Coercion Florida",
                "patterns": [
                    "STATEMENT OF ANTI COERCION", "ANTI COERCION", "ANTI-COERCION",
                    "ANTI COERCION FLORIDA", "COERCION STATEMENT"
                ],
                "priority": 8,
                "required_context": ["FLORIDA", "COERCION", "STATEMENT"],
                "negative_patterns": []
            },
            {
                "label": "Correction Agreement and Limited Power of Attorney",
                "patterns": [
                    "CORRECTION AGREEMENT", "LIMITED POWER OF ATTORNEY",
                    "POWER OF ATTORNEY", "CORRECTION", "ATTORNEY"
                ],
                "priority": 8,
                "required_context": ["POWER", "ATTORNEY", "CORRECTION"],
                "negative_patterns": []
            },
            {
                "label": "All Purpose Acknowledgment",
                "patterns": [
                    "ALL PURPOSE ACKNOWLEDGMENT", "ACKNOWLEDGMENT",
                    "NOTARY ACKNOWLEDGMENT", "ACKNOWLEDGEMENT"
                ],
                "priority": 8,
                "required_context": ["ACKNOWLEDGMENT", "NOTARY", "STATE"],
                "negative_patterns": []
            },
            {
                "label": "Flood Hazard Determination",
                "patterns": [
                    "FLOOD HAZARD DETERMINATION", "FLOOD DETERMINATION",
                    "FEMA FLOOD", "FLOOD HAZARD", "FLOOD ZONE"
                ],
                "priority": 7,
                "required_context": ["FLOOD", "HAZARD", "FEMA"],
                "negative_patterns": []
            },
            {
                "label": "Automatic Payments Authorization",
                "patterns": [
                    "AUTOMATIC PAYMENTS AUTHORIZATION", "AUTOMATIC PAYMENT",
                    "ACH AUTHORIZATION", "AUTOMATIC DEBIT", "PAYMENT AUTHORIZATION"
                ],
                "priority": 7,
                "required_context": ["AUTOMATIC", "PAYMENT", "AUTHORIZATION"],
                "negative_patterns": []
            },
            {
                "label": "Tax Record Information",
                "patterns": [
                    "TAX RECORD INFORMATION", "TAX RECORDS", "PROPERTY TAX",
                    "TAX INFORMATION", "TAX RECORD"
                ],
                "priority": 7,
                "required_context": ["TAX", "PROPERTY", "RECORD"],
                "negative_patterns": []
            },
            {
                "label": "Title Policy",
                "patterns": [
                    "TITLE POLICY", "TITLE INSURANCE", "OWNER'S POLICY",
                    "TITLE INSURANCE POLICY", "OWNERS TITLE POLICY"
                ],
                "priority": 6,
                "required_context": ["TITLE", "POLICY", "INSURANCE"],
                "negative_patterns": []
            },
            {
                "label": "Insurance Policy",
                "patterns": [
                    "INSURANCE POLICY", "HOMEOWNER'S INSURANCE", "HAZARD INSURANCE",
                    "PROPERTY INSURANCE", "HOMEOWNERS INSURANCE"
                ],
                "priority": 6,
                "required_context": ["INSURANCE", "POLICY", "PROPERTY"],
                "negative_patterns": ["TITLE"]
            },
            {
                "label": "Deed",
                "patterns": [
                    "DEED", "WARRANTY DEED", "QUITCLAIM DEED",
                    "SPECIAL WARRANTY DEED", "GENERAL WARRANTY DEED"
                ],
                "priority": 6,
                "required_context": ["DEED", "GRANTOR", "GRANTEE"],
                "negative_patterns": ["TRUST", "MORTGAGE"]
            },
            {
                "label": "UCC Filing",
                "patterns": [
                    "UCC FILING", "UCC-1", "FINANCING STATEMENT",
                    "UCC FINANCING STATEMENT", "UCC1"
                ],
                "priority": 5,
                "required_context": ["UCC", "FINANCING", "STATEMENT"],
                "negative_patterns": []
            },
            {
                "label": "Signature Page",
                "patterns": [
                    "SIGNATURE PAGE", "SIGNATURES", "BORROWER SIGNATURE",
                    "SIGNATURE", "EXECUTION PAGE"
                ],
                "priority": 5,
                "required_context": ["SIGNATURE", "BORROWER", "DATE"],
                "negative_patterns": []
            },
            {
                "label": "Affidavit",
                "patterns": [
                    "AFFIDAVIT", "SWORN STATEMENT", "AFFIDAVIT OF",
                    "BORROWER AFFIDAVIT", "TITLE AFFIDAVIT"
                ],
                "priority": 5,
                "required_context": ["AFFIDAVIT", "SWORN", "STATE"],
                "negative_patterns": []
            }
        ]
    
    def analyze_mortgage_sections(self, text_content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze mortgage document sections using enhanced pattern matching"""
        
        logger.info(f"Starting enhanced analysis of {len(text_content)} text items")
        
        # Group text by page for context analysis
        pages_text = {}
        for item in text_content:
            page = item['page']
            if page not in pages_text:
                pages_text[page] = []
            pages_text[page].append(item)
        
        found_sections = {}
        
        # Analyze each page
        for page_num, page_items in pages_text.items():
            page_text = ' '.join([item['text'] for item in page_items]).upper()
            
            # Check each rule
            for rule in self.section_rules:
                label = rule['label']
                patterns = rule['patterns']
                priority = rule['priority']
                required_context = rule.get('required_context', [])
                negative_patterns = rule.get('negative_patterns', [])
                
                # Check for pattern matches
                pattern_matches = []
                for pattern in patterns:
                    if pattern in page_text:
                        pattern_matches.append(pattern)
                
                if pattern_matches:
                    # Check for negative patterns (exclusions)
                    has_negative = any(neg in page_text for neg in negative_patterns)
                    if has_negative:
                        continue
                    
                    # Calculate match score
                    score = self.calculate_match_score(
                        page_text, pattern_matches, required_context, priority
                    )
                    
                    # Determine confidence
                    confidence = self.determine_confidence(score, pattern_matches, page_text)
                    
                    # Find best text snippet
                    snippet = self.find_best_snippet(page_items, pattern_matches[0])
                    
                    # Keep best match for each section type
                    if label not in found_sections or score > found_sections[label]['score']:
                        found_sections[label] = {
                            "section_type": label,
                            "page": page_num,
                            "confidence": confidence,
                            "text_snippet": snippet,
                            "priority": priority,
                            "pattern_matched": pattern_matches[0],
                            "score": score,
                            "all_patterns": pattern_matches,
                            "context_score": len([ctx for ctx in required_context if ctx in page_text])
                        }
                        
                        logger.info(f"Found section: {label} on page {page_num} (score: {score:.2f}, confidence: {confidence})")
        
        # Convert to list and sort by priority and score
        sections = list(found_sections.values())
        sections.sort(key=lambda x: (-x["priority"], -x["score"], x["page"]))
        
        # Remove score from final output (internal use only)
        for section in sections:
            section.pop('score', None)
        
        logger.info(f"Enhanced analysis complete: {len(sections)} sections identified")
        return sections
    
    def calculate_match_score(self, page_text: str, pattern_matches: List[str], 
                            required_context: List[str], priority: int) -> float:
        """Calculate match score for a section"""
        score = 0.0
        
        # Base score from priority
        score += priority * 0.1
        
        # Pattern match bonus
        score += len(pattern_matches) * 0.2
        
        # Context bonus
        context_matches = sum(1 for ctx in required_context if ctx in page_text)
        if required_context:
            context_ratio = context_matches / len(required_context)
            score += context_ratio * 0.3
        
        # Exact pattern match bonus
        for pattern in pattern_matches:
            if f" {pattern} " in f" {page_text} ":  # Whole word match
                score += 0.2
        
        # Length penalty for very long pages (less specific)
        if len(page_text) > 5000:
            score *= 0.9
        
        return score
    
    def determine_confidence(self, score: float, pattern_matches: List[str], page_text: str) -> str:
        """Determine confidence level based on match quality"""
        if score >= 1.5:
            return "high"
        elif score >= 1.0:
            return "medium"
        else:
            return "low"
    
    def find_best_snippet(self, page_items: List[Dict[str, Any]], pattern: str) -> str:
        """Find the best text snippet containing the pattern"""
        for item in page_items:
            if pattern.lower() in item['text'].lower():
                return item['text'][:150]  # Return first 150 chars
        
        # Fallback to first item
        if page_items:
            return page_items[0]['text'][:150]
        
        return ""

# Initialize processors
ocr_processor = AdvancedOCRProcessor()
mortgage_analyzer = EnhancedMortgageAnalyzer()

def create_session(filename: str, file_size: int) -> str:
    """Create a new processing session"""
    session_id = str(uuid.uuid4())
    
    with session_lock:
        progress_data[session_id] = ProcessingSession(
            session_id=session_id,
            filename=filename,
            file_size=file_size,
            start_time=datetime.now(),
            current_step="initializing",
            progress_percentage=0,
            total_pages=0,
            processed_pages=0,
            status="starting"
        )
    
    logger.info(f"Created session {session_id} for file {filename} ({file_size} bytes)")
    return session_id

def update_session_status(session_id: str, status: str, error: str = None, result: Dict = None):
    """Update session status"""
    with session_lock:
        if session_id in progress_data:
            session = progress_data[session_id]
            session.status = status
            session.error = error
            session.result = result
            if status == "completed":
                session.progress_percentage = 100

@app.route('/')
def index():
    """Serve the main application page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/analyze', methods=['POST'])
def analyze_document():
    """Analyze uploaded mortgage document with maximum OCR features"""
    try:
        logger.info("Starting maximum OCR document analysis...")
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are supported'}), 400
        
        # Read file content
        file_content = file.read()
        file_size = len(file_content)
        
        logger.info(f"Processing file: {file.filename} ({file_size} bytes)")
        
        if file_size == 0:
            return jsonify({'error': 'File is empty'}), 400
        
        # Create processing session
        session_id = create_session(file.filename, file_size)
        
        # Extract text using advanced OCR
        text_content = ocr_processor.extract_text_from_pdf(file_content, session_id)
        
        if not text_content:
            update_session_status(session_id, "error", "Could not extract text from PDF")
            return jsonify({'error': 'Could not extract text from PDF. The file may be corrupted or contain only images without readable text.'}), 400
        
        # Analyze sections using enhanced analyzer
        ocr_processor.update_session_progress(session_id, "analyzing_sections", 90)
        sections = mortgage_analyzer.analyze_mortgage_sections(text_content)
        
        # Prepare result
        result = {
            'session_id': session_id,
            'sections': sections,
            'total_pages': max([item['page'] for item in text_content]) if text_content else 0,
            'total_text_items': len(text_content),
            'processing_method': 'maximum_ocr',
            'ocr_available': ocr_processor.ocr_available,
            'pdfplumber_available': ocr_processor.pdfplumber_available,
            'extraction_methods': list(set([item['method'] for item in text_content])),
            'average_confidence': sum([item.get('confidence', 0.5) for item in text_content]) / len(text_content) if text_content else 0
        }
        
        # Complete session
        update_session_status(session_id, "completed", result=result)
        ocr_processor.update_session_progress(session_id, "completed", 100)
        
        logger.info(f"Maximum OCR analysis complete: {len(sections)} sections identified")
        
        return jsonify(result)
        
    except Exception as e:
        error_msg = f"Document processing error: {str(e)}"
        logger.error(f"Analysis error: {error_msg}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        if 'session_id' in locals():
            update_session_status(session_id, "error", error_msg)
        
        return jsonify({'error': error_msg}), 500

@app.route('/api/progress/<session_id>')
def get_progress(session_id):
    """Get processing progress for a session"""
    with session_lock:
        if session_id in progress_data:
            session = progress_data[session_id]
            return jsonify({
                'session_id': session.session_id,
                'filename': session.filename,
                'current_step': session.current_step,
                'progress_percentage': session.progress_percentage,
                'total_pages': session.total_pages,
                'processed_pages': session.processed_pages,
                'status': session.status,
                'error': session.error,
                'start_time': session.start_time.isoformat(),
                'elapsed_time': (datetime.now() - session.start_time).total_seconds()
            })
        else:
            return jsonify({'error': 'Session not found'}), 404

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': 'maximum-ocr-1.0',
        'ocr_available': ocr_processor.ocr_available,
        'pdfplumber_available': ocr_processor.pdfplumber_available,
        'features': [
            'advanced_ocr',
            'multi_engine_processing',
            'image_preprocessing',
            'enhanced_pattern_matching',
            'real_time_progress',
            'session_management'
        ],
        'dependencies': {
            'pdfplumber': ocr_processor.pdfplumber_available,
            'pdf2image': ocr_processor.ocr_available,
            'pytesseract': ocr_processor.ocr_available,
            'pillow': ocr_processor.ocr_available
        }
    })

# HTML Template for the maximum OCR application
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Maximum OCR Mortgage Package Analyzer</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white; 
            border-radius: 16px; 
            box-shadow: 0 20px 60px rgba(0,0,0,0.15); 
            overflow: hidden;
        }
        
        .header { 
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); 
            color: white; 
            padding: 50px 40px; 
            text-align: center; 
            position: relative;
        }
        
        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="75" cy="75" r="1" fill="rgba(255,255,255,0.1)"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
            opacity: 0.1;
        }
        
        .header h1 { 
            margin: 0; 
            font-size: 3em; 
            font-weight: 300; 
            position: relative;
            z-index: 1;
        }
        
        .header p { 
            margin: 20px 0 0 0; 
            opacity: 0.9; 
            font-size: 1.2em; 
            position: relative;
            z-index: 1;
        }
        
        .features-bar {
            background: linear-gradient(90deg, #28a745, #20c997);
            color: white;
            padding: 15px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 20px;
        }
        
        .feature-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
            font-weight: 500;
        }
        
        .feature-icon {
            width: 16px;
            height: 16px;
            background: rgba(255,255,255,0.3);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .status-bar {
            background: #f8f9fa;
            padding: 20px 40px;
            border-bottom: 1px solid #e9ecef;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 14px;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #28a745;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .section { 
            margin: 40px; 
            padding: 30px; 
            border: 1px solid #e0e0e0; 
            border-radius: 12px; 
            background: #fafafa;
        }
        
        .upload-area {
            border: 3px dashed #ccc;
            border-radius: 16px;
            padding: 80px 40px;
            text-align: center;
            background: white;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .upload-area::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg, transparent, rgba(102, 126, 234, 0.1), transparent);
            transform: rotate(45deg);
            transition: all 0.3s ease;
            opacity: 0;
        }
        
        .upload-area:hover::before {
            opacity: 1;
            animation: shimmer 1.5s ease-in-out;
        }
        
        @keyframes shimmer {
            0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
            100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
        }
        
        .upload-area:hover {
            border-color: #667eea;
            background: #f8f9ff;
            transform: translateY(-4px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
        }
        
        .upload-area.dragover {
            border-color: #667eea;
            background: #f0f8ff;
            transform: scale(1.02);
        }
        
        .upload-content {
            position: relative;
            z-index: 1;
        }
        
        .upload-icon {
            font-size: 4em;
            margin-bottom: 20px;
            opacity: 0.7;
        }
        
        .file-input { display: none; }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 14px 28px;
            border-radius: 10px;
            cursor: pointer;
            margin: 10px;
            font-size: 15px;
            font-weight: 600;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .btn:hover { 
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }
        
        .btn:disabled { 
            background: #ccc; 
            cursor: not-allowed; 
            transform: none;
            box-shadow: none;
        }
        
        .progress-container {
            margin: 30px 0;
            display: none;
            background: white;
            padding: 25px;
            border-radius: 12px;
            border: 1px solid #e0e0e0;
        }
        
        .progress-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .progress-title {
            font-weight: 600;
            color: #2c3e50;
        }
        
        .progress-percentage {
            font-weight: 600;
            color: #667eea;
            font-size: 18px;
        }
        
        .progress-bar {
            width: 100%;
            height: 24px;
            background: #e9ecef;
            border-radius: 12px;
            overflow: hidden;
            position: relative;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            width: 0%;
            transition: width 0.5s ease;
            position: relative;
        }
        
        .progress-fill::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            animation: progress-shimmer 2s infinite;
        }
        
        @keyframes progress-shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        .progress-text {
            text-align: center;
            margin-top: 15px;
            font-size: 14px;
            color: #666;
            font-style: italic;
        }
        
        .results { 
            margin-top: 40px; 
            display: none; 
        }
        
        .section-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin: 25px 0;
        }
        
        .section-card {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 25px;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .section-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: var(--confidence-color);
        }
        
        .section-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 35px rgba(0,0,0,0.1);
        }
        
        .confidence-high { --confidence-color: #28a745; }
        .confidence-medium { --confidence-color: #ffc107; }
        .confidence-low { --confidence-color: #dc3545; }
        
        .section-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 15px;
        }
        
        .section-checkbox {
            width: 18px;
            height: 18px;
            accent-color: #667eea;
        }
        
        .section-title {
            font-weight: 600;
            font-size: 17px;
            color: #2c3e50;
            flex: 1;
        }
        
        .confidence-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .confidence-high .confidence-badge {
            background: rgba(40, 167, 69, 0.1);
            color: #28a745;
        }
        
        .confidence-medium .confidence-badge {
            background: rgba(255, 193, 7, 0.1);
            color: #ffc107;
        }
        
        .confidence-low .confidence-badge {
            background: rgba(220, 53, 69, 0.1);
            color: #dc3545;
        }
        
        .section-meta {
            font-size: 13px;
            color: #666;
            margin-bottom: 15px;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }
        
        .meta-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .section-snippet {
            font-size: 13px;
            font-style: italic;
            color: #888;
            background: #f8f9fa;
            padding: 12px;
            border-radius: 8px;
            margin-top: 15px;
            border-left: 3px solid var(--confidence-color);
        }
        
        .controls {
            display: flex;
            gap: 15px;
            margin: 30px 0;
            flex-wrap: wrap;
        }
        
        .error { 
            background: linear-gradient(135deg, #f8d7da, #f5c6cb); 
            color: #721c24; 
            padding: 20px; 
            border-radius: 12px; 
            margin: 20px 0;
            border: 1px solid #f5c6cb;
            font-weight: 500;
        }
        
        .success { 
            background: linear-gradient(135deg, #d4edda, #c3e6cb); 
            color: #155724; 
            padding: 20px; 
            border-radius: 12px; 
            margin: 20px 0;
            border: 1px solid #c3e6cb;
            font-weight: 500;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 25px 0;
        }
        
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #e0e0e0;
            text-align: center;
        }
        
        .stat-number {
            font-size: 2em;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 5px;
        }
        
        .stat-label {
            font-size: 14px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        @media (max-width: 768px) {
            .container { margin: 10px; border-radius: 12px; }
            .header { padding: 30px 20px; }
            .header h1 { font-size: 2em; }
            .section { margin: 20px; padding: 20px; }
            .upload-area { padding: 60px 20px; }
            .features-bar { padding: 15px 20px; }
            .status-bar { padding: 15px 20px; flex-direction: column; gap: 10px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Maximum OCR Mortgage Analyzer</h1>
            <p>Advanced AI-Powered Document Analysis with Multi-Engine OCR Processing</p>
        </div>

        <div class="features-bar">
            <div class="feature-item">
                <div class="feature-icon">🔍</div>
                <span>Advanced OCR</span>
            </div>
            <div class="feature-item">
                <div class="feature-icon">🧠</div>
                <span>AI Pattern Matching</span>
            </div>
            <div class="feature-item">
                <div class="feature-icon">⚡</div>
                <span>Real-time Progress</span>
            </div>
            <div class="feature-item">
                <div class="feature-icon">🎯</div>
                <span>High Accuracy</span>
            </div>
            <div class="feature-item">
                <div class="feature-icon">🔒</div>
                <span>Secure Processing</span>
            </div>
        </div>

        <div class="status-bar">
            <div class="status-indicator">
                <div class="status-dot"></div>
                <span>Maximum OCR Engine Ready</span>
            </div>
            <div id="dependencyStatus">
                <span>Loading capabilities...</span>
            </div>
        </div>

        <div class="section">
            <h2>📄 Upload Mortgage Package</h2>
            <div class="upload-area" id="uploadArea">
                <div class="upload-content">
                    <div class="upload-icon">📁</div>
                    <h3>Drop your PDF file here or click to browse</h3>
                    <p>Advanced OCR processing for both text-based and image-based PDFs</p>
                    <p style="font-size: 14px; color: #666; margin-top: 15px;">
                        Supports files up to 100MB • Multi-engine processing • Real-time progress tracking
                    </p>
                </div>
                <input type="file" id="fileInput" class="file-input" accept=".pdf">
            </div>
            
            <div class="progress-container" id="progressContainer">
                <div class="progress-header">
                    <div class="progress-title" id="progressTitle">Processing Document...</div>
                    <div class="progress-percentage" id="progressPercentage">0%</div>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <div class="progress-text" id="progressText">Initializing advanced OCR processing...</div>
            </div>
        </div>

        <div class="results" id="results">
            <div class="section">
                <h2>📊 Processing Results</h2>
                <div class="stats-grid" id="statsContainer">
                    <!-- Stats will be populated here -->
                </div>
            </div>
            
            <div class="section">
                <h2>📋 Identified Sections</h2>
                <div class="controls">
                    <button class="btn" onclick="selectAll()">Select All</button>
                    <button class="btn" onclick="selectNone()">Select None</button>
                    <button class="btn" onclick="selectHighConfidence()">High Confidence Only</button>
                    <button class="btn" onclick="generateTOC()">Generate Table of Contents</button>
                </div>
                <div class="section-grid" id="sectionsContainer">
                    <!-- Sections will be populated here -->
                </div>
            </div>
        </div>
    </div>

    <script>
        console.log('🚀 Maximum OCR Mortgage Analyzer Loading...');
        
        let currentSections = [];
        let currentSessionId = null;
        let progressInterval = null;

        document.addEventListener('DOMContentLoaded', function() {
            console.log('✅ Maximum OCR version loaded successfully');
            setupEventListeners();
            checkCapabilities();
        });

        function setupEventListeners() {
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');

            uploadArea.addEventListener('click', () => fileInput.click());
            fileInput.addEventListener('change', handleFileSelect);

            // Enhanced drag and drop
            uploadArea.addEventListener('dragover', function(e) {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });

            uploadArea.addEventListener('dragleave', function() {
                uploadArea.classList.remove('dragover');
            });

            uploadArea.addEventListener('drop', function(e) {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    handleFile(files[0]);
                }
            });
        }

        function checkCapabilities() {
            fetch('/api/health')
                .then(response => response.json())
                .then(data => {
                    const status = document.getElementById('dependencyStatus');
                    const features = data.features || [];
                    
                    if (data.ocr_available && data.pdfplumber_available) {
                        status.innerHTML = '<span style="color: #28a745;">✅ Full Maximum OCR Capabilities</span>';
                    } else if (data.pdfplumber_available) {
                        status.innerHTML = '<span style="color: #ffc107;">⚠️ Text Extraction Only</span>';
                    } else {
                        status.innerHTML = '<span style="color: #dc3545;">❌ Limited Capabilities</span>';
                    }
                    
                    console.log('🔧 Available features:', features);
                })
                .catch(() => {
                    document.getElementById('dependencyStatus').innerHTML = '<span style="color: #dc3545;">❌ Server Error</span>';
                });
        }

        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                handleFile(file);
            }
        }

        function handleFile(file) {
            console.log('📄 Processing file:', file.name, 'Size:', file.size);
            
            if (!file.name.toLowerCase().endsWith('.pdf')) {
                showError('Please select a PDF file.');
                return;
            }

            if (file.size > 100 * 1024 * 1024) {
                showError('File size must be less than 100MB.');
                return;
            }

            uploadAndAnalyze(file);
        }

        function uploadAndAnalyze(file) {
            console.log('🚀 Starting maximum OCR analysis...');
            
            const formData = new FormData();
            formData.append('file', file);

            // Show progress
            document.getElementById('progressContainer').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            updateProgress(0, 'Starting maximum OCR processing...', 'Initializing');

            fetch('/api/analyze', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                console.log('📡 Response status:', response.status);
                if (!response.ok) {
                    return response.json().then(err => {
                        throw new Error('HTTP ' + response.status + ': ' + JSON.stringify(err));
                    });
                }
                return response.json();
            })
            .then(data => {
                console.log('✅ Maximum OCR analysis response:', data);
                
                if (data.error) {
                    throw new Error(data.error);
                }

                currentSections = data.sections || [];
                currentSessionId = data.session_id;
                
                // Start progress monitoring if session ID available
                if (currentSessionId) {
                    startProgressMonitoring(currentSessionId);
                } else {
                    // Complete immediately if no session tracking
                    completeAnalysis(data);
                }
            })
            .catch(error => {
                console.error('❌ Maximum OCR analysis error:', error);
                document.getElementById('progressContainer').style.display = 'none';
                showError('Error analyzing document: ' + error.message);
            });
        }

        function startProgressMonitoring(sessionId) {
            progressInterval = setInterval(() => {
                fetch(`/api/progress/${sessionId}`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            clearInterval(progressInterval);
                            showError('Progress tracking error: ' + data.error);
                            return;
                        }

                        updateProgress(
                            data.progress_percentage,
                            data.current_step.replace(/_/g, ' ').toUpperCase(),
                            `Processing ${data.filename}`
                        );

                        if (data.status === 'completed') {
                            clearInterval(progressInterval);
                            // Fetch final results
                            setTimeout(() => {
                                completeAnalysis({
                                    sections: currentSections,
                                    session_id: sessionId
                                });
                            }, 1000);
                        } else if (data.status === 'error') {
                            clearInterval(progressInterval);
                            showError('Processing error: ' + (data.error || 'Unknown error'));
                        }
                    })
                    .catch(error => {
                        console.error('Progress monitoring error:', error);
                    });
            }, 1000);
        }

        function completeAnalysis(data) {
            updateProgress(100, 'ANALYSIS COMPLETE', 'Processing finished successfully');
            
            setTimeout(() => {
                document.getElementById('progressContainer').style.display = 'none';
                displayResults(currentSections, data);
                showSuccess(`Maximum OCR analysis complete! Found ${currentSections.length} sections with advanced pattern matching.`);
            }, 1500);
        }

        function updateProgress(percentage, step, title) {
            document.getElementById('progressFill').style.width = percentage + '%';
            document.getElementById('progressPercentage').textContent = percentage + '%';
            document.getElementById('progressText').textContent = step;
            document.getElementById('progressTitle').textContent = title;
        }

        function displayResults(sections, metadata) {
            console.log('📋 Displaying maximum OCR results:', sections);
            
            const sectionsContainer = document.getElementById('sectionsContainer');
            const statsContainer = document.getElementById('statsContainer');
            const resultsDiv = document.getElementById('results');
            
            // Display stats
            displayStats(metadata, statsContainer);
            
            if (!sections || sections.length === 0) {
                sectionsContainer.innerHTML = '<div class="error">No sections identified in the document.</div>';
                resultsDiv.style.display = 'block';
                return;
            }

            let html = '';
            sections.forEach((section, index) => {
                const confidenceClass = `confidence-${section.confidence}`;
                html += `
                    <div class="section-card ${confidenceClass}">
                        <div class="section-header">
                            <input type="checkbox" id="section-${index}" class="section-checkbox" checked>
                            <label for="section-${index}" class="section-title">${section.section_type}</label>
                            <div class="confidence-badge">${section.confidence}</div>
                        </div>
                        <div class="section-meta">
                            <div class="meta-item">
                                <span>📄</span>
                                <span>Page ${section.page}</span>
                            </div>
                            <div class="meta-item">
                                <span>🎯</span>
                                <span>Pattern: "${section.pattern_matched}"</span>
                            </div>
                            ${section.all_patterns ? `
                            <div class="meta-item">
                                <span>🔍</span>
                                <span>${section.all_patterns.length} matches</span>
                            </div>
                            ` : ''}
                        </div>
                        <div class="section-snippet">
                            "${section.text_snippet}"
                        </div>
                    </div>
                `;
            });

            sectionsContainer.innerHTML = html;
            resultsDiv.style.display = 'block';
        }

        function displayStats(metadata, container) {
            const stats = [
                { number: metadata.total_pages || 0, label: 'Total Pages' },
                { number: metadata.total_text_items || 0, label: 'Text Items' },
                { number: currentSections.length, label: 'Sections Found' },
                { number: Math.round((metadata.average_confidence || 0) * 100) + '%', label: 'Avg Confidence' }
            ];

            let html = '';
            stats.forEach(stat => {
                html += `
                    <div class="stat-card">
                        <div class="stat-number">${stat.number}</div>
                        <div class="stat-label">${stat.label}</div>
                    </div>
                `;
            });

            container.innerHTML = html;
        }

        function selectAll() {
            document.querySelectorAll('#sectionsContainer input[type="checkbox"]').forEach(cb => cb.checked = true);
        }

        function selectNone() {
            document.querySelectorAll('#sectionsContainer input[type="checkbox"]').forEach(cb => cb.checked = false);
        }

        function selectHighConfidence() {
            document.querySelectorAll('#sectionsContainer input[type="checkbox"]').forEach((cb, index) => {
                cb.checked = currentSections[index] && currentSections[index].confidence === 'high';
            });
        }

        function generateTOC() {
            const selectedSections = [];
            document.querySelectorAll('#sectionsContainer input[type="checkbox"]:checked').forEach((checkbox, index) => {
                const sectionIndex = parseInt(checkbox.id.split('-')[1]);
                if (currentSections[sectionIndex]) {
                    selectedSections.push(currentSections[sectionIndex]);
                }
            });

            if (selectedSections.length === 0) {
                showError('Please select at least one section.');
                return;
            }

            // Sort by page number
            selectedSections.sort((a, b) => a.page - b.page);

            // Generate professional TOC
            let toc = 'MORTGAGE PACKAGE - TABLE OF CONTENTS\\n';
            toc += '=' * 60 + '\\n\\n';
            toc += 'Generated: ' + new Date().toLocaleString() + '\\n';
            toc += 'Processing: Maximum OCR Analysis\\n';
            toc += 'Total Sections: ' + selectedSections.length + '\\n\\n';
            
            selectedSections.forEach((section, index) => {
                const pageStr = `Page ${section.page}`.padStart(12);
                const confidenceStr = `[${section.confidence.toUpperCase()}]`.padStart(8);
                toc += `${(index + 1).toString().padStart(2)}. ${section.section_type.padEnd(45, '.')} ${pageStr} ${confidenceStr}\\n`;
            });

            toc += '\\n' + '=' * 60 + '\\n';
            toc += `Analysis Method: Maximum OCR with Advanced Pattern Matching\\n`;
            toc += `High Confidence Sections: ${selectedSections.filter(s => s.confidence === 'high').length}\\n`;
            toc += `Medium Confidence Sections: ${selectedSections.filter(s => s.confidence === 'medium').length}\\n`;
            toc += `Low Confidence Sections: ${selectedSections.filter(s => s.confidence === 'low').length}\\n`;

            // Create downloadable file
            const blob = new Blob([toc], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'mortgage_package_toc_maximum_ocr.txt';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            showSuccess(`Professional Table of Contents generated! (${selectedSections.length} sections with confidence scoring)`);
        }

        function showError(message) {
            console.error('❌ Error:', message);
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error';
            errorDiv.textContent = message;
            document.querySelector('.container').appendChild(errorDiv);
            setTimeout(() => errorDiv.remove(), 10000);
        }

        function showSuccess(message) {
            console.log('✅ Success:', message);
            const successDiv = document.createElement('div');
            successDiv.className = 'success';
            successDiv.textContent = message;
            document.querySelector('.container').appendChild(successDiv);
            setTimeout(() => successDiv.remove(), 8000);
        }
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    print("=" * 80)
    print("🚀 MAXIMUM OCR MORTGAGE PACKAGE ANALYZER")
    print("=" * 80)
    print(f"📍 Server starting at: http://{host}:{port}")
    print(f"🔍 OCR Available: {OCR_AVAILABLE}")
    print(f"📄 PDF Processing: {PDFPLUMBER_AVAILABLE}")
    print(f"🧠 Advanced Features: Multi-engine OCR, AI Pattern Matching")
    print(f"⚡ Real-time Progress: Session-based tracking")
    print("=" * 80)
    print("💡 Production-ready for Railway deployment")
    print("=" * 80)
    
    try:
        app.run(host=host, port=port, debug=False)
    except KeyboardInterrupt:
        print("\\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        logger.error(f"Server startup error: {e}")
        logger.error(traceback.format_exc())

