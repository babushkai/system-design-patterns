# System Design Patterns - Book Build System
# Requires: pandoc, texlive (or mactex on macOS)

BOOK_DIR = book
OUTPUT_DIR = build
METADATA = $(BOOK_DIR)/metadata.yaml
CHAPTERS_FILE = $(BOOK_DIR)/chapters.txt

# Output files
PDF_OUTPUT = $(OUTPUT_DIR)/system-design-patterns.pdf
EPUB_OUTPUT = $(OUTPUT_DIR)/system-design-patterns.epub

# Pandoc options
PANDOC_OPTS = --toc \
              --toc-depth=3 \
              --number-sections \
              --highlight-style=tango \
              --metadata-file=$(METADATA) \
              --resource-path=.:$(BOOK_DIR)

# PDF-specific options (using XeLaTeX for better font support)
PDF_OPTS = $(PANDOC_OPTS) \
           --pdf-engine=xelatex \
           --variable=geometry:margin=1in \
           --variable=documentclass:book \
           --variable=classoption:11pt \
           --variable=colorlinks:true \
           --variable=linkcolor:NavyBlue \
           --variable=urlcolor:NavyBlue

# EPUB-specific options
EPUB_OPTS = $(PANDOC_OPTS) \
            --epub-cover-image=$(BOOK_DIR)/cover.png

# Get chapters from file, filtering comments and empty lines
CHAPTERS = $(shell grep -v '^\#' $(CHAPTERS_FILE) | grep -v '^$$')

.PHONY: all pdf epub clean check-deps help

all: pdf

# Build PDF book
pdf: $(PDF_OUTPUT)

$(PDF_OUTPUT): $(CHAPTERS) $(METADATA) | $(OUTPUT_DIR)
	@echo "Building PDF book..."
	pandoc $(PDF_OPTS) -o $@ $(CHAPTERS)
	@echo "PDF created: $@"

# Build EPUB ebook
epub: $(EPUB_OUTPUT)

$(EPUB_OUTPUT): $(CHAPTERS) $(METADATA) | $(OUTPUT_DIR)
	@echo "Building EPUB book..."
	pandoc $(EPUB_OPTS) -o $@ $(CHAPTERS)
	@echo "EPUB created: $@"

# Create output directory
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

# Clean build artifacts
clean:
	rm -rf $(OUTPUT_DIR)
	@echo "Cleaned build directory"

# Check dependencies
check-deps:
	@echo "Checking dependencies..."
	@which pandoc > /dev/null || (echo "ERROR: pandoc not found. Install with: brew install pandoc" && exit 1)
	@which xelatex > /dev/null || (echo "ERROR: xelatex not found. Install with: brew install --cask mactex" && exit 1)
	@echo "All dependencies found!"

# Install dependencies (macOS)
install-deps:
	@echo "Installing dependencies..."
	brew install pandoc
	brew install --cask mactex
	@echo "Dependencies installed. You may need to restart your terminal."

# Quick build without full LaTeX (faster, less pretty)
pdf-quick: | $(OUTPUT_DIR)
	@echo "Building quick PDF (no LaTeX)..."
	pandoc $(PANDOC_OPTS) \
		--pdf-engine=wkhtmltopdf \
		-o $(OUTPUT_DIR)/system-design-patterns-quick.pdf \
		$(CHAPTERS)

# Build single chapter for testing
pdf-chapter: | $(OUTPUT_DIR)
	@if [ -z "$(CHAPTER)" ]; then \
		echo "Usage: make pdf-chapter CHAPTER=01-foundations/01-acid-transactions.md"; \
		exit 1; \
	fi
	pandoc $(PDF_OPTS) -o $(OUTPUT_DIR)/chapter-preview.pdf $(CHAPTER)
	@echo "Chapter preview created: $(OUTPUT_DIR)/chapter-preview.pdf"

# Word count statistics
stats:
	@echo "Word count by part:"
	@for dir in [0-9][0-9]-*/; do \
		words=$$(cat $$dir*.md 2>/dev/null | wc -w); \
		echo "  $$dir: $$words words"; \
	done
	@echo ""
	@echo "Total words: $$(cat [0-9][0-9]-*/*.md | wc -w)"
	@echo "Total files: $$(ls [0-9][0-9]-*/*.md | wc -l)"

# Help
help:
	@echo "System Design Patterns - Book Build System"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  pdf          Build PDF book (default)"
	@echo "  epub         Build EPUB ebook"
	@echo "  all          Build all formats"
	@echo "  clean        Remove build artifacts"
	@echo "  check-deps   Verify required tools are installed"
	@echo "  install-deps Install dependencies (macOS)"
	@echo "  pdf-quick    Build quick PDF without LaTeX"
	@echo "  pdf-chapter  Build single chapter (CHAPTER=path/to/file.md)"
	@echo "  stats        Show word count statistics"
	@echo "  help         Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make pdf"
	@echo "  make pdf-chapter CHAPTER=01-foundations/01-acid-transactions.md"
