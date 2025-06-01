import json
import sys
import os

def inspect_json_structure(file_path):
    """Loads a JSON file and prints information about its structure."""
    print(f"Inspecting JSON file: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"Top-level data type: {type(data)}")

    if isinstance(data, list):
        print(f"The JSON is a list with {len(data)} elements.")
        if len(data) > 0:
            first_element = data[0]
            print(f"Type of the first element: {type(first_element)}")
            if isinstance(first_element, dict):
                print(f"Keys in the first element (dictionary): {list(first_element.keys())}")
                # Try to print a snippet of text based on common keys
                for key in ['text', 'content', 'article', 'passage', 'document']:
                    if key in first_element and isinstance(first_element[key], str):
                        print(f"Snippet from first_element['{key}']: '{first_element[key][:200]}...'" )
                        break
            elif isinstance(first_element, str):
                 print(f"First element is a string: '{first_element[:200]}...'" )

    elif isinstance(data, dict):
        print(f"The JSON is a dictionary with top-level keys: {list(data.keys())}")
        # If it's a dictionary, you might want to inspect a specific key further
        # For example, if you expect a list of articles under a key like 'articles'
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0:
                print(f"Value for key '{key}' is a list with {len(value)} elements.")
                first_item_in_value = value[0]
                print(f"  Type of the first item in data['{key}']: {type(first_item_in_value)}")
                if isinstance(first_item_in_value, dict):
                    print(f"  Keys in the first item of data['{key}']: {list(first_item_in_value.keys())}")
                    for text_key in ['text', 'content', 'article', 'passage', 'document']:
                        if text_key in first_item_in_value and isinstance(first_item_in_value[text_key], str):
                            print(f"  Snippet from data['{key}'][0]['{text_key}']: '{first_item_in_value[text_key][:200]}...'" )
                            break
                elif isinstance(first_item_in_value, str):
                    print(f"  First item in data['{key}'] is a string: '{first_item_in_value[:200]}...'" )
            elif isinstance(value, str):
                 print(f"Value for key '{key}' is a string: '{value[:200]}...'" )

    else:
        print(f"The JSON data is of an unexpected type: {type(data)}")

if __name__ == "__main__":
    # You can change this path to test different files
    # Path relative to the project root, assuming the script is run from 'proj/src/'
    # or if run from 'proj/', then the path would be 'Dataset/cn/human/zh_unicode/news-zh.json'
    
    # Default path, assuming script is run from project root `proj/`
    default_file_to_inspect = "Dataset/cn/human/zh_unicode/news-zh.json"
    
    # If script is run from `proj/src/` then path should be `../Dataset/...`
    # Check if current working directory is src, if so adjust path
    if os.getcwd().endswith("src"):
        default_file_to_inspect = "../" + default_file_to_inspect

    if len(sys.argv) > 1:
        file_to_inspect = sys.argv[1]
    else:
        print(f"No file path provided as argument. Using default: {default_file_to_inspect}")
        file_to_inspect = default_file_to_inspect

    inspect_json_structure(file_to_inspect)

    # You might want to test other files as well:
    # inspect_json_structure("../Dataset/cn/human/zh_unicode/webnovel.json")
    # inspect_json_structure("../Dataset/cn/generated/zh_qwen2/news-zh.qwen2-72b-base.json") 