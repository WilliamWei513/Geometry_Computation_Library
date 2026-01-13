import sys

def cleanup_script(input_file, func_ranges, main_code_range):
    """
    Clean up a Python script by keeping only specified functions and main code.
    
    Parameters:
        input_file: Path to the Python file to clean
        func_ranges: Dictionary mapping function names to (start_line, end_line) tuples (1-indexed)
        main_code_range: Tuple (start_line, end_line) for main code to keep (1-indexed)
    
    Example:
        cleanup_script(
            'Archive/Algorithm/TinySitePacker.py',
            {
                'coerce_to_curve': (14, 28),
                'graft_tree': (263, 283),
                'evaluate_curve': (1815, 1994),
                'bin_pack_segments': (5695, 6021)
            },
            (7096, 7110)
        )
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Start with imports (assume first 9 lines are imports, adjust if needed)
    result = lines[:9]

    # Add functions in order
    for name, (start, end) in sorted(func_ranges.items(), key=lambda x: x[1][0]):
        result.extend(lines[start-1:end])

    # Add main code
    main_start, main_end = main_code_range
    result.extend(lines[main_start-1:main_end])

    # Write back
    with open(input_file, 'w', encoding='utf-8') as f:
        f.writelines(result)

    print(f"Cleaned file: kept {len(result)} lines (was {len(lines)} lines)")
    print(f"Kept functions: {', '.join(sorted(func_ranges.keys()))}")

if __name__ == '__main__':
    # Example usage for TinySitePacker.py
    cleanup_script(
        'Archive/Algorithm/TinySitePacker.py',
        {
            'coerce_to_curve': (14, 28),
            'graft_tree': (263, 283),
            'evaluate_curve': (1815, 1994),
            'bin_pack_segments': (5695, 6021)
        },
        (7096, 7110)
    )

