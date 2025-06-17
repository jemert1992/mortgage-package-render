#!/bin/bash

# Render Build Script for Maximum OCR Mortgage Analyzer
# This script installs system dependencies for OCR processing

echo "🔧 Installing OCR system dependencies for Render..."

# Update package list
apt-get update

# Install tesseract OCR engine and English language pack
apt-get install -y tesseract-ocr tesseract-ocr-eng

# Install poppler utilities for PDF processing
apt-get install -y poppler-utils

# Install additional image processing libraries
apt-get install -y libpoppler-cpp-dev

# Clean up to reduce image size
apt-get clean
rm -rf /var/lib/apt/lists/*

echo "✅ OCR dependencies installed successfully!"
echo "📋 Installed packages:"
echo "   - tesseract-ocr (OCR engine)"
echo "   - tesseract-ocr-eng (English language pack)"
echo "   - poppler-utils (PDF processing)"
echo "   - libpoppler-cpp-dev (PDF development libraries)"

# Verify installations
echo "🔍 Verifying installations..."
tesseract --version
pdftoppm -h | head -1

echo "🎉 Render build script completed successfully!"

