import json
import os

def clean_jsonl_file(file_path):
    """Remove conversation turns with empty content from JSONL file."""
    cleaned_lines = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # Filter out conversation turns with empty content
                if 'conversations' in data and isinstance(data['conversations'], list):
                    original_count = len(data['conversations'])
                    data['conversations'] = [
                        turn for turn in data['conversations']
                        if isinstance(turn, dict) and
                           turn.get('content', '').strip() != ''
                    ]
                    removed_count = original_count - len(data['conversations'])
                    if removed_count > 0:
                        print(f"File {os.path.basename(file_path)}, line {line_num}: Removed {removed_count} empty conversation turns")

                cleaned_lines.append(json.dumps(data, ensure_ascii=False))

            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num} in {file_path}: {e}")
                # Keep the line as-is if we can't parse it
                cleaned_lines.append(line)

    # Write cleaned data back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in cleaned_lines:
            f.write(line + '\n')

    return len(cleaned_lines)

def main():
    output_dir = "F:/manas-project/manas-dataset/output"
    jsonl_files = [
        f for f in os.listdir(output_dir)
        if f.endswith('.jsonl')
    ]

    print(f"Found {len(jsonl_files)} JSONL files to process")

    total_lines = 0
    for filename in jsonl_files:
        file_path = os.path.join(output_dir, filename)
        print(f"\nProcessing {filename}...")
        lines_processed = clean_jsonl_file(file_path)
        total_lines += lines_processed
        print(f"Processed {lines_processed} lines in {filename}")

    print(f"\nTotal lines processed across all files: {total_lines}")

if __name__ == "__main__":
    main()