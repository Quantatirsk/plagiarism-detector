# Readers Utils Package

This package contains reusable utilities for text processing and optimization that can be shared across different document parsers.

## ParagraphOptimizer

A universal text optimization engine that provides language-aware text processing capabilities.

### Features

- **Language Detection**: Automatic CJK (Chinese, Japanese, Korean) vs Latin language detection
- **Line Break Fixing**: Intelligent repair of PDF line breaks while preserving paragraph structure
- **Hyphen Handling**: Smart handling of hyphenated words split across lines (e.g., "Canadian-based")
- **Paragraph Processing**: Automatic paragraph boundary detection and formatting
- **Text Spacing**: Unified spacing optimization for better readability

### Usage

```python
from tools.readers.utils import ParagraphOptimizer

# Basic usage
optimizer = ParagraphOptimizer()

# Simple text optimization
text = "This is broken\ntext with\n\nhyphenated\nwords"
result = optimizer.optimize_text(text)

# Advanced usage with options
options = {
    'fix_line_breaks': True,
    'normalize_spacing': True
}
result = optimizer.optimize_text(text, options)

# Individual methods
is_cjk = optimizer.is_cjk_language("这是中文")
fixed_text = optimizer.fix_text_line_breaks(raw_text)
joined_lines = optimizer.join_paragraph_lines(['line1', 'line2'])
```

### Integration with PDFParser

The PDFParser now uses ParagraphOptimizer internally:

```python
from tools.readers.readers_pdf import PDFParser

parser = PDFParser()

# Text optimization enabled by default
text = parser.parse("document.pdf")

# Disable text optimization for raw extraction
raw_text = parser.parse("document.pdf", optimize_text=False)

# All existing methods still work (backward compatibility)
parser._fix_text_line_breaks(text)  # Delegates to optimizer
```

### Performance Benefits

- **Separation of Concerns**: PDF parsing logic separated from text optimization
- **Reusability**: Same optimizer can be used by other document parsers
- **Modularity**: Text processing can be tested and maintained independently
- **Flexibility**: Users can choose to disable optimization for performance

### Supported Languages

- **CJK Languages**: Chinese (Simplified/Traditional), Japanese, Korean
- **Latin Languages**: English, Spanish, French, German, etc.
- **Language-Aware Processing**: Different handling for CJK vs Latin text

### Migration Guide

For existing code using PDFParser methods directly:

#### Old Code (Still Works)
```python
parser = PDFParser()
fixed = parser._fix_text_line_breaks(text)
is_cjk = parser._is_cjk_language(text)
```

#### New Recommended Code
```python
parser = PDFParser()
# Access optimizer directly for better performance
fixed = parser.text_optimizer.fix_text_line_breaks(text)
is_cjk = parser.text_optimizer.is_cjk_language(text)

# Or use the unified optimize method
optimized = parser.text_optimizer.optimize_text(text)
```

### Future Enhancements

Planned features for ParagraphOptimizer:

- Smart title detection and formatting
- Enhanced list and numbering processing
- Citation and footnote handling
- Multi-column text merging
- Custom optimization profiles per document type

### Testing

```python
# Run tests for the optimizer
python -m pytest tools/readers/utils/tests/

# Test with PDFParser integration
from tools.readers.readers_pdf import PDFParser
parser = PDFParser()
assert hasattr(parser, 'text_optimizer')
assert parser.parse("test.pdf", optimize_text=False) != parser.parse("test.pdf", optimize_text=True)
```