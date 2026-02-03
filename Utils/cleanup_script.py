import sys
import os

def cleanup_script(input_file, func_ranges, main_code_range, import_lines=9):
    """
    Clean up a Python script by keeping only specified functions and main code.
    
    Parameters:
        input_file: Path to the Python file to clean
        func_ranges: Dictionary mapping function names to (start_line, end_line) tuples (1-indexed)
        main_code_range: Tuple (start_line, end_line) for main code to keep (1-indexed)
        import_lines: Number of lines to keep from the beginning (default 9)
    """
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return False
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    result = lines[:import_lines]

    for name, (start, end) in sorted(func_ranges.items(), key=lambda x: x[1][0]):
        if start < 1 or end > len(lines) or start > end:
            print(f"Warning: Invalid range for function '{name}': ({start}, {end})")
            continue
        result.extend(lines[start-1:end])

    main_start, main_end = main_code_range
    if main_start < 1 or main_end > len(lines) or main_start > main_end:
        print(f"Warning: Invalid main code range: ({main_start}, {main_end})")
        return False
    
    result.extend(lines[main_start-1:main_end])

    with open(input_file, 'w', encoding='utf-8') as f:
        f.writelines(result)

    print(f"\nCleaned file: kept {len(result)} lines (was {len(lines)} lines)")
    print(f"Kept functions: {', '.join(sorted(func_ranges.keys()))}")
    return True

def get_input(prompt, default=None, input_type=str):
    """Get user input with optional default value."""
    if default is not None:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "
    
    user_input = input(prompt).strip()
    if not user_input and default is not None:
        return default
    if not user_input:
        return None
    try:
        return input_type(user_input)
    except ValueError:
        print(f"Invalid input. Expected {input_type.__name__}.")
        return None

def get_line_range(prompt):
    """Get a line range (start, end) from user."""
    while True:
        range_input = get_input(prompt)
        if not range_input:
            return None
        try:
            if ',' in range_input:
                parts = range_input.split(',')
                start = int(parts[0].strip())
                end = int(parts[1].strip())
                return (start, end)
            else:
                print("Please enter range as 'start, end' (e.g., '10, 20')")
        except (ValueError, IndexError):
            print("Invalid format. Please enter as 'start, end' (e.g., '10, 20')")

def interactive_cleanup():
    """Interactive CLI for cleanup script."""
    print("=" * 60)
    print("Python Script Cleanup Tool")
    print("=" * 60)
    
    print()
    
    input_file = None
    while not input_file:
        file_path = get_input("Enter the path to the Python file to clean")
        if file_path and os.path.exists(file_path):
            input_file = file_path
        else:
            print(f"File not found: {file_path}")
    
    print()
    
    import_lines = get_input("How many import lines to keep from the beginning", default=9, input_type=int)
    if import_lines is None:
        import_lines = 9
    
    print()
    print("Enter functions to keep (function name and line range).")
    print("Format: function_name,start_line,end_line")
    print("Press Enter with empty input when done.")
    print()
    
    func_ranges = {}
    while True:
        func_input = get_input("Function (name,start,end)")
        if not func_input:
            break
        parts = func_input.split(',')
        if len(parts) == 3:
            try:
                func_name = parts[0].strip()
                start = int(parts[1].strip())
                end = int(parts[2].strip())
                func_ranges[func_name] = (start, end)
                print(f"Added: {func_name} ({start}, {end})")
            except (ValueError, IndexError):
                print("Invalid format. Use: function_name,start,end")
        else:
            print("Invalid format. Use: function_name,start,end")
    
    print()
    print("Enter main code range to keep.")
    main_code_range = None
    while not main_code_range:
        main_code_range = get_line_range("Main code range (start, end)")
    
    print()
    print("Summary:")
    print(f"  Input file: {input_file}")
    print(f"  Import lines: {import_lines}")
    print(f"  Functions to keep: {len(func_ranges)}")
    for name, (start, end) in func_ranges.items():
        print(f"    - {name}: lines {start}-{end}")
    print(f"  Main code: lines {main_code_range[0]}-{main_code_range[1]}")
    print()
    
    confirm = get_input("Proceed with cleanup? (yes/no)", default="no")
    if confirm and confirm.lower() in ['yes', 'y']:
        success = cleanup_script(input_file, func_ranges, main_code_range, import_lines)
        if success:
            print("\nCleanup completed successfully!")
        return success
    else:
        print("\nCleanup cancelled.")
        return False

if __name__ == '__main__':
    interactive_cleanup()

