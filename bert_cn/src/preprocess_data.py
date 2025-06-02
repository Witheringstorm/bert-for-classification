import os
import json
# import glob # No longer needed as we are not loading English txt data here
from sklearn.model_selection import train_test_split
import pandas as pd
import tqdm

def extract_text_from_human_item(item_dict, file_path_for_warning):
    """Helper function for HUMAN data items.
    Extracts and combines text from an item's 'input' and 'output' fields.
    'input' and 'output' can be strings or dictionaries of string segments.
    """
    combined_text_parts = []
    if not isinstance(item_dict, dict):
        # This case should ideally not be hit if called from the list processing part
        print(f"Warning: Expected a dictionary item for human data, but got {type(item_dict)} in {file_path_for_warning}")
        return None

    # Process 'input' field
    input_val = item_dict.get('input')
    if isinstance(input_val, dict):
        for part in input_val.values():
            if isinstance(part, str):
                combined_text_parts.append(part)
    elif isinstance(input_val, str):
        combined_text_parts.append(input_val)
        
    # Process 'output' field
    output_val = item_dict.get('output')
    if isinstance(output_val, dict):
        for part in output_val.values():
            if isinstance(part, str):
                combined_text_parts.append(part)
    elif isinstance(output_val, str):
        combined_text_parts.append(output_val)
    
    if combined_text_parts:
        return "\n".join(combined_text_parts)
    else:
        print(f"Warning: No text parts extracted from human data item in {file_path_for_warning}: {item_dict if len(str(item_dict)) < 200 else str(item_dict)[:200] + '...'}")
        return None

def load_chinese_json_data(file_path):
    """Loads text data from a Chinese JSON file.
    Handles two main structures:
    1. Human data: A list of dictionaries. Each dictionary is an item processed by extract_text_from_human_item.
    2. Generated data: A single top-level dictionary. This dictionary contains 'input' and 'output' keys,
       which are themselves dictionaries of numbered string segments. Each corresponding pair of
       input/output segments forms a document.
    """
    texts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

            if isinstance(data, list):
                # Case 1: Human data (list of items)
                for item_dict in tqdm.tqdm(data, desc=f"Processing Human data list in {os.path.basename(file_path)}"):
                    if isinstance(item_dict, dict) and 'input' in item_dict and 'output' in item_dict:
                        extracted_text = extract_text_from_human_item(item_dict, file_path)
                        if extracted_text:
                            texts.append(extracted_text)
                    else:
                        # This warning means an item in the list isn't the expected dict structure
                        print(f"Warning: Item in human data list from {file_path} is not a dict with 'input'/'output' keys: {item_dict if len(str(item_dict)) < 200 else str(item_dict)[:200] + '...'}")
            
            elif isinstance(data, dict):
                # Case 2: Generated data (single dictionary containing maps of segments)
                tqdm.tqdm.write(f"Processing Generated data (dict of segments) in {os.path.basename(file_path)}")
                input_segments_map = data.get('input')
                output_segments_map = data.get('output')

                if isinstance(input_segments_map, dict) and isinstance(output_segments_map, dict):
                    # Iterate through keys of input segments (e.g., "0", "1", "2", ...)
                    # Assuming corresponding keys exist in output_segments_map
                    for key in tqdm.tqdm(input_segments_map.keys(), desc=f"  Segments in {os.path.basename(file_path)}"):
                        if key in output_segments_map:
                            input_segment = input_segments_map[key]
                            output_segment = output_segments_map[key]

                            if isinstance(input_segment, str) and isinstance(output_segment, str):
                                texts.append(input_segment + "\n" + output_segment)
                            else:
                                # This would mean a segment itself is not a string, which contradicts `head` output
                                print(f"Warning: Segment for key '{key}' in generated file {file_path} is not a string. Input type: {type(input_segment)}, Output type: {type(output_segment)}")
                        else:
                            print(f"Warning: Missing corresponding output segment for key '{key}' in generated file {file_path}")
                    else:
                    print(f"Warning: Generated file {file_path} does not have dictionary structures for 'input' and 'output' fields. Input type: {type(input_segments_map)}, Output type: {type(output_segments_map)}")
            else:
                print(f"Warning: Top-level JSON structure in {file_path} is neither a list nor a dict: {type(data)}")

    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
    except FileNotFoundError:
        print(f"Error: File not found {file_path}")
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
    return texts

# Removed load_english_txt_data function as it's not needed for Chinese data processing

def main():
    # Path to the raw Chinese dataset, relative to this script in bert_cn/src/
    base_raw_dataset_path = "/public/share/yinxiangrong/qinxiaoyu/kdc2024/S/nlp/proj/Dataset/cn"
    
    # Path to save processed data, relative to this script in bert_cn/src/
    processed_data_output_path = "/public/share/yinxiangrong/qinxiaoyu/kdc2024/S/nlp/proj/bert_cn/data/processed"
    os.makedirs(processed_data_output_path, exist_ok=True)

    # Chinese data paths
    human_cn_news_path = os.path.join(base_raw_dataset_path, "human/zh_unicode/news-zh.json")
    human_cn_webnovel_path = os.path.join(base_raw_dataset_path, "human/zh_unicode/webnovel.json")
    human_cn_wiki_path = os.path.join(base_raw_dataset_path, "human/zh_unicode/wiki-zh.json")

    generated_cn_news_path = os.path.join(base_raw_dataset_path, "generated/zh_qwen2/news-zh.qwen2-72b-base.json")
    generated_cn_webnovel_path = os.path.join(base_raw_dataset_path, "generated/zh_qwen2/webnovel.qwen2-72b-base.json")
    generated_cn_wiki_path = os.path.join(base_raw_dataset_path, "generated/zh_qwen2/wiki-zh.qwen2-72b-base.json")

    # --- Load Chinese Data ---
    print("loading human chinese data")
    human_cn_texts = []
    human_cn_texts.extend(load_chinese_json_data(human_cn_news_path))
    human_cn_texts.extend(load_chinese_json_data(human_cn_webnovel_path))
    human_cn_texts.extend(load_chinese_json_data(human_cn_wiki_path))
    print(f"Loaded {len(human_cn_texts)} human Chinese texts.")

    generated_cn_texts = []
    generated_cn_texts.extend(load_chinese_json_data(generated_cn_news_path))
    generated_cn_texts.extend(load_chinese_json_data(generated_cn_webnovel_path))
    generated_cn_texts.extend(load_chinese_json_data(generated_cn_wiki_path))
    print(f"Loaded {len(generated_cn_texts)} generated Chinese texts.")

    # Create labels for Chinese data
    labels_cn_human = [0] * len(human_cn_texts)
    labels_cn_generated = [1] * len(generated_cn_texts)

    all_texts_cn = human_cn_texts + generated_cn_texts
    all_labels_cn = labels_cn_human + labels_cn_generated

    if all_texts_cn and all_labels_cn:
        # Split Chinese data
        # Ensure there are enough samples in each class for stratification if len(set(all_labels_cn)) > 1
        min_samples_for_stratify_initial = 2 * len(set(all_labels_cn)) # At least 2 samples per class for the initial split
        can_stratify_initial = (len(set(all_labels_cn)) > 1 and
                        all(pd.Series(all_labels_cn).value_counts() >= min_samples_for_stratify_initial // len(set(all_labels_cn))))

        # First split: 80% train, 20% temp (for validation + test)
        train_texts_cn, temp_texts_cn, train_labels_cn, temp_labels_cn = train_test_split(
            all_texts_cn, all_labels_cn, test_size=0.2, random_state=42,
            stratify=all_labels_cn if can_stratify_initial else None
        )

        # Second split: 50% of temp for validation, 50% for test (making it 10% val, 10% test of original)
        # Check stratification for the temporary set
        min_samples_for_stratify_temp = 2 * len(set(temp_labels_cn)) if temp_labels_cn else 0
        can_stratify_temp = False
        if temp_labels_cn and len(set(temp_labels_cn)) > 1:
            temp_label_counts = pd.Series(temp_labels_cn).value_counts()
            if all(count >= min_samples_for_stratify_temp // len(set(temp_labels_cn)) for count in temp_label_counts) and all(count >=1 for count in temp_label_counts): # ensure at least 1 sample per class for the second split if stratifying
                 can_stratify_temp = True
    else:
                print(f"Warning: Cannot stratify temp split. Label counts: {temp_label_counts}. Min samples required per class for strat: {min_samples_for_stratify_temp // len(set(temp_labels_cn)) if len(set(temp_labels_cn)) > 0 else 'N/A'}")
        elif temp_labels_cn and len(set(temp_labels_cn)) == 1:
            print("Warning: Only one class present in the temporary data for val/test split. Stratification disabled for this split.")
        elif not temp_labels_cn:
            print("Warning: temp_labels_cn is empty. Cannot perform validation/test split.")
            # Handle case where temp_labels_cn might be empty or too small
            val_texts_cn, test_texts_cn, val_labels_cn, test_labels_cn = [], [], [], []
        
        if temp_texts_cn: # Proceed only if temp_texts_cn is not empty
            val_texts_cn, test_texts_cn, val_labels_cn, test_labels_cn = train_test_split(
                temp_texts_cn, temp_labels_cn, test_size=0.5, random_state=42, # 0.5 of 0.2 gives 0.1 for test
                stratify=temp_labels_cn if can_stratify_temp else None
        )
            print(f"Chinese data: {len(train_texts_cn)} train, {len(val_texts_cn)} validation, {len(test_texts_cn)} test examples.")
        else: # temp_texts_cn was empty
             val_texts_cn, test_texts_cn, val_labels_cn, test_labels_cn = [], [], [], []
             print(f"Chinese data: {len(train_texts_cn)} train. Validation and test sets are empty due to insufficient data in temp split.")


        # Save processed data to CSV files
        df_cn_train = pd.DataFrame({'text': train_texts_cn, 'label': train_labels_cn})
        df_cn_val = pd.DataFrame({'text': val_texts_cn, 'label': val_labels_cn})
        df_cn_test = pd.DataFrame({'text': test_texts_cn, 'label': test_labels_cn})
        
        train_output_file = os.path.join(processed_data_output_path, "train_cn.csv")
        val_output_file = os.path.join(processed_data_output_path, "val_cn.csv")
        test_output_file = os.path.join(processed_data_output_path, "test_cn.csv")

        if not df_cn_train.empty:
            df_cn_train.to_csv(train_output_file, index=False, encoding='utf-8')
            print(f"Saved Chinese training data to: {train_output_file}")
        else:
            print(f"Chinese training data is empty. Skipping save for {train_output_file}")

        if not df_cn_val.empty:
            df_cn_val.to_csv(val_output_file, index=False, encoding='utf-8')
            print(f"Saved Chinese validation data to: {val_output_file}")
        else:
            print(f"Chinese validation data is empty. Skipping save for {val_output_file}")

        if not df_cn_test.empty:
            df_cn_test.to_csv(test_output_file, index=False, encoding='utf-8')
            print(f"Saved Chinese testing data to: {test_output_file}")
        else:
            print(f"Chinese testing data is empty. Skipping save for {test_output_file}")
    else:
        print("No Chinese data loaded or labels are empty. Skipping Chinese data processing and saving.")

    # Removed all English data processing sections
    # Removed TODOs that were specific to English data or general structure not relevant here

if __name__ == "__main__":
    main() 