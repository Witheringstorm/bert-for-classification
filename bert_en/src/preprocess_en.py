import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import tqdm
import random # Added for sampling

# Configuration for English data preprocessing
BASE_RAW_DATA_PATH = "/public/share/yinxiangrong/qinxiaoyu/kdc2024/S/nlp/proj/Dataset/ghostbuster-data"
PROCESSED_DATA_OUTPUT_PATH = "/public/share/yinxiangrong/qinxiaoyu/kdc2024/S/nlp/proj/bert_en/data/processed" # Relative to bert_en/src/
os.makedirs(PROCESSED_DATA_OUTPUT_PATH, exist_ok=True)

# Define the domains and LLM types to include
DOMAINS = ["wp", "reuter", "essay"] # Add other domains if present and relevant
LLM_TYPES = ["gpt", "claude"] # Add other LLM generator types if present and relevant

def load_english_txt_files(directory_path, desc_prefix="Processing files in"):
    """Loads text data from all .txt files in a directory.
    If no .txt files are found directly, it searches one level deeper in subdirectories.
    """
    texts = []
    if not os.path.isdir(directory_path):
        print(f"Warning: Directory not found {directory_path}")
        return texts
    
    # Try loading .txt files directly from the directory_path
    file_paths = glob.glob(os.path.join(directory_path, "*.txt"))
    
    current_desc_base = os.path.basename(directory_path)

    if not file_paths: # If no .txt files found directly, look into subdirectories
        subdirectories = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
        if subdirectories:
            for subdir_name in subdirectories:
                # We only want to go one level deeper. If subdir_name is 'logprobs' or 'headlines', skip it.
                # This is a simple heuristic; actual .txt files should not be in these utility folders.
                if subdir_name.lower() in ["logprobs", "headlines"]:
                    continue 
                subdir_path = os.path.join(directory_path, subdir_name)
                # Update description for tqdm to show the subdirectory being processed
                desc = f"{desc_prefix} {current_desc_base}/{subdir_name}"
                subdir_file_paths = glob.glob(os.path.join(subdir_path, "*.txt"))
                for txt_file in tqdm.tqdm(subdir_file_paths, desc=desc):
                    try:
                        with open(txt_file, 'r', encoding='utf-8') as f:
                            texts.append(f.read())
                    except Exception as e:
                        print(f"Error loading file {txt_file}: {e}")
            return texts # Return texts found in subdirectories

    # Process direct files (if any) or if no subdirectories with files were found
    desc = f"{desc_prefix} {current_desc_base}"
    for txt_file in tqdm.tqdm(file_paths, desc=desc):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                texts.append(f.read())
        except Exception as e:
            print(f"Error loading file {txt_file}: {e}")
    return texts

def main():
    all_human_texts = []
    all_llm_texts = []
    random.seed(42) # For reproducible sampling

    print("Loading English human-written texts...")
    for domain in DOMAINS:
        human_domain_path = os.path.join(BASE_RAW_DATA_PATH, domain, "human")
        print(f"Loading from: {human_domain_path}")
        domain_human_texts = load_english_txt_files(human_domain_path, desc_prefix=f"Processing {domain}/human in")
        all_human_texts.extend(domain_human_texts)
        print(f"Loaded {len(domain_human_texts)} texts from {domain}/human. Total human texts: {len(all_human_texts)}")

    print("\nLoading English LLM-generated texts...")
    for domain in DOMAINS:
        for llm_type in LLM_TYPES:
            llm_domain_path = os.path.join(BASE_RAW_DATA_PATH, domain, llm_type)
            print(f"Loading from: {llm_domain_path}")
            domain_llm_texts = load_english_txt_files(llm_domain_path)
            all_llm_texts.extend(domain_llm_texts)
            print(f"Loaded {len(domain_llm_texts)} texts from {domain}/{llm_type}. Total LLM texts: {len(all_llm_texts)}")

    print(f"\nTotal human texts loaded: {len(all_human_texts)}")
    print(f"Total LLM texts loaded: {len(all_llm_texts)}")

    if not all_human_texts or not all_llm_texts: # Ensure both lists have data before balancing
        print("One or both text lists are empty. Cannot balance or proceed. Exiting.")
        return

    # Balance the number of samples per class by undersampling the majority class
    num_human = len(all_human_texts)
    num_llm = len(all_llm_texts)

    if num_llm > num_human:
        print(f"Undersampling LLM texts from {num_llm} to {num_human} to match human texts.")
        all_llm_texts = random.sample(all_llm_texts, num_human)
    elif num_human > num_llm:
        print(f"Undersampling Human texts from {num_human} to {num_llm} to match LLM texts.")
        all_human_texts = random.sample(all_human_texts, num_llm)
    else:
        print("Human and LLM text counts are already equal.")

    print(f"Balanced human texts: {len(all_human_texts)}")
    print(f"Balanced LLM texts: {len(all_llm_texts)}")

    # Create labels for the balanced lists
    human_labels = [0] * len(all_human_texts)
    llm_labels = [1] * len(all_llm_texts)

    all_texts = all_human_texts + all_llm_texts
    all_labels = human_labels + llm_labels
    
    if not all_texts or not all_labels or len(all_texts) != len(all_labels):
        print("Data and label mismatch or empty data after balancing. Cannot proceed with splitting.")
        return

    # Split data
    num_unique_labels = len(set(all_labels))
    can_stratify = False
    if num_unique_labels > 1:
        label_counts = pd.Series(all_labels).value_counts()
        if all(count >= 2 for count in label_counts):
            can_stratify = True
        else:
            print("Warning: Not all classes have at least 2 samples for stratification. Stratification will be disabled.")
    elif num_unique_labels == 1:
        print("Warning: Only one class present in the data. Stratification will be disabled.")
    else:
        print("No labels found. Cannot stratify.")

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        all_texts, all_labels, test_size=0.2, random_state=42,
        stratify=all_labels if can_stratify else None
    )

    print(f"\nEnglish data (balanced): {len(train_texts)} train, {len(test_texts)} test examples.")

    # Save to CSV
    df_train = pd.DataFrame({'text': train_texts, 'label': train_labels})
    df_test = pd.DataFrame({'text': test_texts, 'label': test_labels})

    train_output_file = os.path.join(PROCESSED_DATA_OUTPUT_PATH, "train_en.csv")
    test_output_file = os.path.join(PROCESSED_DATA_OUTPUT_PATH, "test_en.csv")

    df_train.to_csv(train_output_file, index=False, encoding='utf-8')
    df_test.to_csv(test_output_file, index=False, encoding='utf-8')

    print(f"Saved English training data to: {train_output_file}")
    print(f"Saved English testing data to: {test_output_file}")

if __name__ == "__main__":
    main() 