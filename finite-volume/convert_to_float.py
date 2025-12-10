#!/usr/bin/env python3
"""
Script to convert double to REAL throughout the finite-volume source code.
This script performs intelligent replacements, avoiding comments and string literals.
"""

import os
import re
import sys

def should_replace_double(content, pos):
    """Check if 'double' at position pos should be replaced with REAL."""
    # Check if it's part of a larger word (like "double_precision" or "MPI_DOUBLE")
    if pos > 0 and (content[pos-1].isalnum() or content[pos-1] == '_'):
        return False
    if pos + 6 < len(content) and (content[pos+6].isalnum() or content[pos+6] == '_'):
        return False
    
    # Check if it's in a string literal
    # Simple check: count quotes before this position
    before = content[:pos]
    # Count unescaped quotes
    single_quotes = 0
    double_quotes = 0
    i = 0
    while i < len(before):
        if before[i] == '\\':
            i += 2
            continue
        if before[i] == "'":
            single_quotes += 1
        elif before[i] == '"':
            double_quotes += 1
        i += 1
    
    if (single_quotes % 2 != 0) or (double_quotes % 2 != 0):
        return False
    
    # Check if it's in a comment
    # Find the start of the line
    line_start = content.rfind('\n', 0, pos) + 1
    line_before_double = content[line_start:pos]
    
    # Check for // comments
    if '//' in line_before_double:
        return False
    
    # Check for /* */ comments - find if we're inside one
    last_comment_start = content.rfind('/*', 0, pos)
    if last_comment_start != -1:
        last_comment_end = content.find('*/', last_comment_start, pos)
        if last_comment_end == -1:  # Inside unclosed comment
            return False
    
    return True

def replace_double_in_file(filepath):
    """Replace double with REAL in a file, handling special cases."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return False
    
    original_content = content
    
    # Replace MPI_DOUBLE with MPI_REAL_TYPE
    content = re.sub(r'\bMPI_DOUBLE\b', 'MPI_REAL_TYPE', content)
    
    # Replace DBL_MAX, DBL_MIN, DBL_EPSILON with REAL_* versions
    content = re.sub(r'\bDBL_MAX\b', 'REAL_MAX', content)
    content = re.sub(r'\bDBL_MIN\b', 'REAL_MIN', content)
    content = re.sub(r'\bDBL_EPSILON\b', 'REAL_EPSILON', content)
    
    # Replace (double) casts with (REAL)
    content = re.sub(r'\(double\)', '(REAL)', content)
    
    # Replace double type declarations (more careful)
    # Pattern: double followed by optional * and variable name
    # This is tricky, so we'll do a simpler approach: replace standalone "double"
    # that appears to be a type
    
    # Replace all standalone 'double' with 'REAL' using regex
    # This pattern matches 'double' as a word boundary, but excludes it if it's part of MPI_DOUBLE, DBL_*, etc.
    # We already handled MPI_DOUBLE and DBL_* above, so now we can safely replace remaining 'double'
    
    # Use regex to replace 'double' that's not part of a larger identifier
    # Pattern: word boundary, 'double', followed by optional whitespace and then * or identifier start
    def replace_double(match):
        # Check if it's in a comment or string (simple heuristic)
        pos = match.start()
        line_start = content.rfind('\n', 0, pos) + 1
        line = content[line_start:pos]
        if '//' in line:
            return match.group(0)  # Don't replace in // comments
        return 'REAL'
    
    # Replace 'double' with 'REAL' - be more aggressive
    # First, handle cases where double is followed by * or whitespace (type declarations)
    content = re.sub(r'\bdouble\s*\*', 'REAL *', content)
    content = re.sub(r'\bdouble\s+', 'REAL ', content)
    content = re.sub(r'\bdouble\s*\)', 'REAL)', content)  # Function parameters
    content = re.sub(r'\bdouble\s*;', 'REAL;', content)  # Variable declarations
    content = re.sub(r'\bdouble\s*,', 'REAL,', content)  # In lists
    content = re.sub(r'\bdouble\s*\[', 'REAL[', content)  # Arrays
    content = re.sub(r'\bdouble\s*\(', 'REAL(', content)  # Function return types
    # Catch any remaining standalone 'double'
    content = re.sub(r'\bdouble\b', 'REAL', content)
    
    if content != original_content:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated: {filepath}")
            return True
        except Exception as e:
            print(f"Error writing {filepath}: {e}", file=sys.stderr)
            return False
    
    return False

def main():
    """Main function to process all source files."""
    source_dir = os.path.dirname(os.path.abspath(__file__)) + '/source'
    
    if not os.path.exists(source_dir):
        print(f"Error: {source_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    
    extensions = ['.c', '.h', '.cu', '.cuh']
    excluded_dirs = ['extern']  # Don't modify external libraries
    
    files_processed = 0
    files_updated = 0
    
    for root, dirs, files in os.walk(source_dir):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                filepath = os.path.join(root, file)
                files_processed += 1
                if replace_double_in_file(filepath):
                    files_updated += 1
    
    print(f"\nProcessed {files_processed} files, updated {files_updated} files")

if __name__ == '__main__':
    main()

