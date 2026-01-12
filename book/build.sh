#!/bin/bash
# System Design Patterns - Book Build Script
# Alternative to Makefile for easier cross-platform use

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/build"
BOOK_DIR="$PROJECT_ROOT/book"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check dependencies
check_deps() {
    echo "Checking dependencies..."
    
    if ! command -v pandoc &> /dev/null; then
        echo -e "${RED}ERROR: pandoc not found${NC}"
        echo "Install with:"
        echo "  macOS:  brew install pandoc"
        echo "  Ubuntu: sudo apt install pandoc"
        exit 1
    fi
    
    if ! command -v xelatex &> /dev/null; then
        echo -e "${YELLOW}WARNING: xelatex not found. PDF generation may fail.${NC}"
        echo "Install with:"
        echo "  macOS:  brew install --cask mactex"
        echo "  Ubuntu: sudo apt install texlive-xetex texlive-fonts-recommended"
    fi
    
    echo -e "${GREEN}Dependencies OK${NC}"
}

# Get list of chapters
get_chapters() {
    grep -v '^#' "$BOOK_DIR/chapters.txt" | grep -v '^$' | while read -r chapter; do
        echo "$PROJECT_ROOT/$chapter"
    done
}

# Build PDF
build_pdf() {
    echo "Building PDF book..."
    mkdir -p "$OUTPUT_DIR"
    
    cd "$PROJECT_ROOT"
    
    # Get chapters as array
    chapters=$(get_chapters | tr '\n' ' ')
    
    pandoc \
        --toc \
        --toc-depth=3 \
        --number-sections \
        --highlight-style=tango \
        --metadata-file="$BOOK_DIR/metadata.yaml" \
        --pdf-engine=xelatex \
        --variable=geometry:margin=1in \
        --variable=documentclass:book \
        --variable=classoption:11pt \
        --variable=colorlinks:true \
        --variable=linkcolor:NavyBlue \
        --variable=urlcolor:NavyBlue \
        -o "$OUTPUT_DIR/system-design-patterns.pdf" \
        $chapters
    
    echo -e "${GREEN}PDF created: $OUTPUT_DIR/system-design-patterns.pdf${NC}"
}

# Build EPUB
build_epub() {
    echo "Building EPUB book..."
    mkdir -p "$OUTPUT_DIR"
    
    cd "$PROJECT_ROOT"
    chapters=$(get_chapters | tr '\n' ' ')
    
    pandoc \
        --toc \
        --toc-depth=3 \
        --number-sections \
        --highlight-style=tango \
        --metadata-file="$BOOK_DIR/metadata.yaml" \
        -o "$OUTPUT_DIR/system-design-patterns.epub" \
        $chapters
    
    echo -e "${GREEN}EPUB created: $OUTPUT_DIR/system-design-patterns.epub${NC}"
}

# Build HTML (single page)
build_html() {
    echo "Building HTML book..."
    mkdir -p "$OUTPUT_DIR"
    
    cd "$PROJECT_ROOT"
    chapters=$(get_chapters | tr '\n' ' ')
    
    pandoc \
        --toc \
        --toc-depth=3 \
        --number-sections \
        --highlight-style=tango \
        --metadata-file="$BOOK_DIR/metadata.yaml" \
        --standalone \
        --self-contained \
        -o "$OUTPUT_DIR/system-design-patterns.html" \
        $chapters
    
    echo -e "${GREEN}HTML created: $OUTPUT_DIR/system-design-patterns.html${NC}"
}

# Show help
show_help() {
    echo "System Design Patterns - Book Build Script"
    echo ""
    echo "Usage: ./book/build.sh [command]"
    echo ""
    echo "Commands:"
    echo "  pdf      Build PDF book (requires xelatex)"
    echo "  epub     Build EPUB ebook"
    echo "  html     Build single-page HTML"
    echo "  all      Build all formats"
    echo "  check    Check dependencies"
    echo "  clean    Remove build directory"
    echo "  help     Show this help"
    echo ""
}

# Main
case "${1:-pdf}" in
    pdf)
        check_deps
        build_pdf
        ;;
    epub)
        check_deps
        build_epub
        ;;
    html)
        check_deps
        build_html
        ;;
    all)
        check_deps
        build_pdf
        build_epub
        build_html
        ;;
    check)
        check_deps
        ;;
    clean)
        rm -rf "$OUTPUT_DIR"
        echo "Cleaned build directory"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        show_help
        exit 1
        ;;
esac
