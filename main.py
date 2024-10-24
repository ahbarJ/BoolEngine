import pandas as pd
from bitarray import bitarray
import glob
import pickle
import os
from collections import defaultdict

# Step 1: Load Data
def load_data(dir):
    csv_files = glob.glob(dir + '*.csv')
    tweets = pd.DataFrame() 
    for file in csv_files:
        df = pd.read_csv(file)
        tweets = tweets._append(df, ignore_index=True)
    return tweets

# Step 2: Normalize Hashtags
def normalize_hashtags(hashtags):
    if pd.isna(hashtags):
        return []
    return [tag.strip().lower().replace('#', '') for tag in hashtags.split(',')]  # Remove '#' for consistency

# Step 3: Create DTM using bitarray
def create_dtm(df):
    hashtags = []
    for index, row in df.iterrows():
        hashtags += normalize_hashtags(row['hashtags'])
    
    unique_hashtags = sorted(set(hashtags))
    print(f"Unique hashtags identified: {len(unique_hashtags)}")

    dtm = []
    for index, row in df.iterrows():
        if index % 1000 == 0:
            print(f"Processing tweet {index}/{len(df)}")
        
        bit_vector = bitarray(len(unique_hashtags))
        bit_vector.setall(0)

        normalized_tags = normalize_hashtags(row['hashtags'])

        for tag in normalized_tags:
            if tag in unique_hashtags:
                bit_index = unique_hashtags.index(tag)
                bit_vector[bit_index] = 1

        dtm.append(bit_vector)

    return dtm, unique_hashtags

# Step 4: Create Inverted Index
def create_inverted_index(df):
    inverted_index = defaultdict(set)

    for index, row in df.iterrows():
        normalized_tags = normalize_hashtags(row['hashtags'])

        for tag in normalized_tags:
            inverted_index[tag].add(index)

    print(f"Inverted index created with {len(inverted_index)} unique hashtags.")
    return inverted_index

# Step 5: Simple Query Processing
def parse_simple_query(query):
    tokens = query.lower().split()  # Split the query by spaces and normalize to lowercase
    return tokens

def evaluate_simple_query(tokens, inverted_index):
    print(f"Evaluating query tokens: {tokens}")  # Debugging output
    if len(tokens) == 1:
        # Single hashtag query
        tag = tokens[0]
        if tag in inverted_index:
            print(f"Tag '{tag}' found in inverted index. IDs: {inverted_index[tag]}")  # Debugging output
            return inverted_index[tag]
        else:
            print(f"Tag '{tag}' not found in inverted index.")  # Debugging output
            return set()
    
    if len(tokens) == 3:
        tag1, operator, tag2 = tokens
        tag1_set = inverted_index.get(tag1, set())
        tag2_set = inverted_index.get(tag2, set())

        print(f"Tag 1: '{tag1}', IDs: {tag1_set}")  # Debugging output
        print(f"Tag 2: '{tag2}', IDs: {tag2_set}")  # Debugging output

        if operator == 'and':
            return tag1_set.intersection(tag2_set)
        elif operator == 'or':
            return tag1_set.union(tag2_set)
        elif operator == 'not':
            return tag1_set.difference(tag2_set)
    
    return set()  # Return empty set for invalid queries

# Save DTM and unique hashtags to file
def save_dtm(dtm, unique_hashtags, filename='dtm.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump((dtm, unique_hashtags), f)
    print(f"DTM saved to {filename}")

# Load DTM and unique hashtags from file
def load_dtm(filename='dtm.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            dtm, unique_hashtags = pickle.load(f)
        print(f"DTM loaded from {filename} with {len(dtm)} tweets and {len(unique_hashtags)} unique hashtags.")
        return dtm, unique_hashtags
    else:
        return None, None

# Save Inverted Index to a file
def save_inverted_index(inverted_index, filename='inverted_index.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(inverted_index, f)
    print(f"Inverted index saved to {filename}")

# Load Inverted Index from a file
def load_inverted_index(filename='inverted_index.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            inverted_index = pickle.load(f)
        print(f"Inverted index loaded from {filename} with {len(inverted_index)} unique hashtags.")
        return inverted_index
    else:
        return None

def search_single_hashtag(hashtag, inverted_index):
    # Step 1: Print the original hashtag input
    print(f"Original hashtag input: '{hashtag}'")  # Debugging statement
    # Step 2: Check if the hashtag exists in the inverted index without normalization
    print(inverted_index.keys())
    if hashtag in inverted_index.keys():
        print('Found..', hashtag)
        print(f"Hashtag '{hashtag}' found in inverted index.")  # Debugging statement
        result = inverted_index[hashtag]
        print(f"Tweet IDs for hashtag '{hashtag}': {result}")  # Debugging statement
        return result
    else:
        print(f"Hashtag '{hashtag}' NOT found in inverted index.")  # Debugging statement
        return set()  # Return an empty set if not found




# Main Execution
if __name__ == "__main__":
    dtm, unique_hashtags = load_dtm()
    inverted_index = load_inverted_index()

    if dtm is None or unique_hashtags is None or inverted_index is None:
        df = load_data('data/')
        print(f"Loaded {len(df)} tweets with hashtags.")

        dtm, unique_hashtags = create_dtm(df)
        print(f"Document-Term Matrix created with {len(dtm)} tweets and {len(unique_hashtags)} unique hashtags.")
        
        save_dtm(dtm, unique_hashtags)

        inverted_index = create_inverted_index(df)
        save_inverted_index(inverted_index)
    else:
        print("Using existing Document-Term Matrix and Inverted Index.")

    single_tag_query = input('Enter a single hashtag to search:')
    resultSet = search_single_hashtag(single_tag_query, inverted_index)
    print(resultSet)

    # Simple query processing loop
    """ while True:
        user_query = input("Enter a query (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break

        try:
            # Normalize user query hashtags
            normalized_query = user_query.replace('#', '').strip()  # Normalize user input
            tokens = parse_simple_query(normalized_query)
            print(f"Normalized query tokens: {tokens}")  # Debugging output
            result = evaluate_simple_query(tokens, inverted_index)
            print(f"Resulting tweet IDs: {sorted(result)}")
        except Exception as e:
            print(f"Error processing query: {e}")
 """