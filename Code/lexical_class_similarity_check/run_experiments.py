import re
import json
import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

def process_file_to_dict(file_path, label_mapping):
    """
    Processes a file containing tokenized sentences with corresponding argument labels and predicates.
    Converts the data into a structured list of dictionaries, where each dictionary represents a sentence
    with its tokens, predicate status (True/False), and numeric argument labels.

    This function assumes the input file is in a tab-separated format where:
    - The second column contains the token (word),
    - The 11th column indicates whether the token is a predicate ('_' means no, any other value means yes),
    - The 12th column contains the argument label (or '_' for no label).

    The function returns a list of dictionaries, where each dictionary represents a sentence.

    Parameters:
        file_path (str): The path to the file to be processed.

        label_mapping (dict): A dictionary that maps argument labels (e.g., "ARG1", "ARG2", "O") to their corresponding numeric values.
            This mapping is used to convert string labels into numeric labels.

    Output:
        list: A list of dictionaries, where each dictionary represents a sentence. Each dictionary contains:
            - 'text' (list): A list of tokens (words) in the sentence.
            - 'predicate' (list): A list of Boolean values indicating whether each token is a predicate (True or False).
            - 'label' (list): A list of numeric values corresponding to the argument label for each token.
                The numeric labels are based on the `label_mapping.json` provided.


    """

    file_sentences = []
    #for path in file_paths:
    with open(file_path, encoding='utf-8') as infile:
        sentence = []
        sentence_labels = []
        sentence_predicate = []
        sentence_pred_labels = []
        sentence_ent_labels = []

        for line in infile:
            if line != '\n':
                # remove the newline at the end of the line, then split the line based on the tab separator
                line_split = line.strip('\n').split('\t')
                # extract the token from the line in the file
                token = line_split[4]
                region = line_split[1]

                # check whether the token is the predicate
                pred = line_split[5]
                if pred == 'O':
                    predicate = False
                    pred_label = 'O'
                else:
                    predicate = True
                    pred_label = f"[{re.sub(r'^[BI]-', '', pred)}]"

                # check whether label for argument is in label_mapping.json (train_labels), else 'O'
                argument = line_split[6]
                entity = line_split[7]

                # get numeric label
                if argument in label_mapping.keys():
                    numeric_arg = label_mapping[argument]
                else:
                    numeric_arg = label_mapping['O']

                # per sentence
                sentence.append(token)
                sentence_predicate.append(predicate)
                sentence_labels.append(numeric_arg)
                sentence_pred_labels.append(pred_label)
                sentence_ent_labels.append(entity)
            else:
                if sentence and sentence_labels and sentence_predicate and sentence_pred_labels and sentence_ent_labels:
                    sentence_dict = {'text': sentence, 'region': region, 'predicate': sentence_predicate, 'label': sentence_labels,
                                     'pred_label': sentence_pred_labels, 'ent_label': sentence_ent_labels}
                    # get list of dicts
                    file_sentences.append(sentence_dict)
                    sentence_dict = {}
                    sentence = []
                    sentence_labels = []
                    sentence_predicate = []
                    sentence_pred_labels = []
                    sentence_ent_labels = []
                else:
                    sentence_dict = {}
                    sentence = []
                    sentence_labels = []
                    sentence_predicate = []
                    sentence_pred_labels = []
                    sentence_ent_labels = []
        if sentence and sentence_labels and sentence_predicate and sentence_pred_labels and sentence_ent_labels:
            sentence_dict = {
                'text': sentence,
                'region': region,
                'predicate': sentence_predicate,
                'label': sentence_labels,
                'pred_label': sentence_pred_labels,
                'ent_label': sentence_ent_labels
            }
            file_sentences.append(sentence_dict)

    return file_sentences


def extract_index_groups(data, label1, label2):
    groups = []
    current_group = []

    for i, val in enumerate(data):
        if val in (label1, label2):
            if not current_group:
                current_group = [i]
            else:
                # Check if it's contiguous
                if i == current_group[-1] + 1:
                    current_group.append(i)
                else:
                    groups.append(current_group)
                    current_group = [i]
        else:
            if current_group:
                groups.append(current_group)
                current_group = []

    # Append the last group if it exists
    if current_group:
        groups.append(current_group)

    return groups

def get_token_groups(index_groups, tokens):
    return [[tokens[i] for i in group] for group in index_groups]

def find_conllu_files(directory):
    conllu_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".conllu"):
                full_path = os.path.join(root, file)
                conllu_files.append(full_path)
    return conllu_files

def remove_duplicates(list_of_lists):
    # Convert inner lists to tuples (hashable), then use a set to remove duplicates
    seen = set()
    unique_lists = []
    for lst in list_of_lists:
        tuple_lst = tuple(lst)
        if tuple_lst not in seen:
            seen.add(tuple_lst)
            unique_lists.append(lst)  # Keep the original list format
    return unique_lists

with open('../SRL/label_mapping', 'r', encoding='utf-8') as f:
    label_mapping = json.load(f)


file_paths = find_conllu_files('../../SRL_data_with_entities/')

print("Selecting token sequences...")

all_tokens_for_ships = []
all_tokens_for_leaving_agents = []
all_tokens_for_moving_instruments = []

# for each file in the training data
for f in file_paths:
    list_emb = process_file_to_dict(f, label_mapping)
    # for each text region
    for dict in list_emb:
        # if there is a B-Agent in the argument labels and if there is a leaving event in the region
        if 0 in dict['label'] and '[Leaving]' in dict['pred_label']:
            # get the indices of arguments (B-Agent ... I-Agent)
            indices = extract_index_groups(dict['label'], 0, 11)
            tokens = get_token_groups(indices, dict['text'])
            for group in tokens:
                all_tokens_for_leaving_agents.append(group)
        # if there is a B-Instrument in the argument labels and a Leaving or Arriving event in the region
        if 4 in dict['label'] and ('[Leaving]' in dict['pred_label'] or '[Arriving]' in dict['pred_label']  or '[Voyage]' in dict['pred_label'] or '[Transportation]' in dict['pred_label']):
            # get the indices of arguments (B-Instrument.... I-Instrument)
            indices = extract_index_groups(dict['label'], 4, 15)
            tokens = get_token_groups(indices, dict['text'])
            for group in tokens:
                all_tokens_for_moving_instruments.append(group)
        # if there is an entity annotated as ship in the region
        if 'B-SHIP' in dict['ent_label']:
            indices = extract_index_groups(dict['ent_label'], 'B-SHIP', 'I-SHIP')
            tokens = get_token_groups(indices, dict['text'])
            for group in tokens:
                all_tokens_for_ships.append(group)


all_tokens_for_ships = remove_duplicates(all_tokens_for_ships)
all_tokens_for_leaving_agents = remove_duplicates(all_tokens_for_leaving_agents)
all_tokens_for_moving_instruments = remove_duplicates(all_tokens_for_moving_instruments)

print()
print('Selected sequences referring to ships:')
print(all_tokens_for_ships)
print(len(all_tokens_for_ships))
print()
print('Selected sequences referring to leaving agents:')
print(all_tokens_for_leaving_agents)
print(len(all_tokens_for_leaving_agents))
print()
print('Selected sequences referring to moving instruments:')
print(all_tokens_for_moving_instruments)
print(len(all_tokens_for_moving_instruments))

print()
print("Selected token sequences.")
print()


############## Embedding similarity comparison

def get_sequence_embedding(tokens, model, tokenizer):
    """
    Takes a list of tokens (words) and returns a single BERT embedding vector for the sequence.
    Aggregates token embeddings (excluding special tokens) by averaging.
    """
    text = " ".join(tokens)
    inputs = tokenizer(text, return_tensors='pt', return_attention_mask=True)

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state.squeeze(0)  # Shape: [seq_len, hidden_size]
    #attention_mask = inputs['attention_mask'].squeeze(0)  # Shape: [seq_len]

    # Exclude special tokens ([CLS], [SEP]) by checking attention mask and tokenizer
    input_ids = inputs['input_ids'].squeeze(0)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    valid_embeddings = []
    for i, token in enumerate(tokens):
        if token not in ['[CLS]', '[SEP]']:
            valid_embeddings.append(last_hidden_state[i])

    if not valid_embeddings:
        raise ValueError("No valid tokens found for embedding.")

    # Average embeddings of valid tokens to get a single vector
    sequence_embedding = torch.stack(valid_embeddings).mean(dim=0)

    return sequence_embedding

def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two PyTorch vectors.
    """
    return F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()


def get_matrix(model_name, mentions, tokentype):

    # Load pre-trained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    #Get embeddings for all mentions
    print("Computing embeddings...")
    mention_embeddings = [get_sequence_embedding(m, model, tokenizer) for m in mentions]

    #Compute similarity matrix
    n = len(mention_embeddings)
    similarity_matrix = np.zeros((n, n))

    print("Computing similarity matrix...")
    for i in range(n):
        for j in range(n):
            similarity_matrix[i][j] = cosine_similarity(mention_embeddings[i], mention_embeddings[j])

    mention_labels = [' '.join(m) for m in mentions]

    df_matrix = pd.DataFrame(similarity_matrix, index=mention_labels, columns=mention_labels)

    # Save to CSV
    df_matrix.to_csv("sim_matrices/sim_matrix_"+str(tokentype)+"_"+model_name.split('/')[1]+".csv", encoding="utf-8")
    print("Saved similarity matrix")


    return (similarity_matrix)

def get_delta(matrixA, matrixB):
    delta_matrix = matrixA - matrixB

    # Compute average signed delta
    average_signed_delta = np.mean(delta_matrix)

    std_A = np.std(matrixA)  # or np.std(B, ddof=1)

    std_B = np.std(matrixB)  # or np.std(B, ddof=1)

    #print('standard dev A', std_A)
    #print('standard dev B', std_B)

    normalized_delta = average_signed_delta / std_A

    return(normalized_delta)


def main():

    glob_SFT_matrix_mi = get_matrix('globalise/SFT-globertise', all_tokens_for_moving_instruments, 'moving_instruments')
    glob_MFT_matrix_mi = get_matrix('globalise/MFT-globertise', all_tokens_for_moving_instruments, 'moving_instruments')

    mBERT_SFT_matrix_mi= get_matrix('globalise/SFT-mBERT', all_tokens_for_moving_instruments, 'moving_instruments')
    mBERT_MFT_matrix_mi=get_matrix('globalise/MFT-mBERT', all_tokens_for_moving_instruments, 'moving_instruments')

    XLM_SFT_matrix_mi= get_matrix('globalise/SFT-XLM-R', all_tokens_for_moving_instruments, 'moving_instruments')
    XLM_MFT_matrix_mi=get_matrix('globalise/MFT-XLM-R', all_tokens_for_moving_instruments, 'moving_instruments')

    print()
    print('Calculating normalized deltas...')
    print()


    delta_glob = get_delta(glob_SFT_matrix_mi, glob_MFT_matrix_mi)
    delta_bert = get_delta(mBERT_SFT_matrix_mi, mBERT_MFT_matrix_mi)
    delta_xlm = get_delta(XLM_SFT_matrix_mi, XLM_MFT_matrix_mi)

    #Positive average delta: On average, values in A are greater than in B.
    #Negative average delta: On average, values in B are greater than in A.
    #Zero: On average, they are equal.

    print("AVG SIGNED DELTA GLOBERTISE MOVING INSTRUMENTS")
    print(delta_glob)

    print("AVG SIGNED DELTA MBERT MOVING INSTRUMENTS")
    print(delta_bert)

    print("AVG SIGNED DELTA XLM MOVING INSTRUMENTS")
    print(delta_xlm)
    print()

    glob_SFT_matrix_s = get_matrix('globalise/SFT-globertise', all_tokens_for_ships, 'ships')
    glob_MFT_matrix_s = get_matrix('globalise/MFT-globertise', all_tokens_for_ships, 'ships')

    mBERT_SFT_matrix_s= get_matrix('globalise/SFT-mBERT', all_tokens_for_ships, 'ships')
    mBERT_MFT_matrix_s=get_matrix('globalise/MFT-mBERT', all_tokens_for_ships, 'ships')

    XLM_SFT_matrix_s= get_matrix('globalise/SFT-XLM-R', all_tokens_for_ships, 'ships')
    XLM_MFT_matrix_s=get_matrix('globalise/MFT-XLM-R', all_tokens_for_ships, 'ships')



    delta_glob = get_delta(glob_SFT_matrix_s, glob_MFT_matrix_s)
    delta_bert = get_delta(mBERT_SFT_matrix_s, mBERT_MFT_matrix_s)
    delta_xlm = get_delta(XLM_SFT_matrix_s, XLM_MFT_matrix_s)

    #Positive average delta: On average, values in A are greater than in B.
    #Negative average delta: On average, values in B are greater than in A.
    #Zero: On average, they are equal.

    print("AVG SIGNED DELTA GLOBERTISE SHIPS")
    print(delta_glob)

    print("AVG SIGNED DELTA MBERT SHIPS")
    print(delta_bert)

    print("AVG SIGNED DELTA XLM SHIPS")
    print(delta_xlm)
    print()


    glob_SFT_matrix_la = get_matrix('globalise/SFT-globertise', all_tokens_for_leaving_agents, 'leaving_agents')
    glob_MFT_matrix_la = get_matrix('globalise/MFT-globertise', all_tokens_for_leaving_agents, 'leaving_agents')

    mBERT_SFT_matrix_la= get_matrix('globalise/SFT-mBERT', all_tokens_for_leaving_agents, 'leaving_agents')
    mBERT_MFT_matrix_la=get_matrix('globalise/MFT-mBERT', all_tokens_for_leaving_agents, 'leaving_agents')

    XLM_SFT_matrix_la= get_matrix('globalise/SFT-XLM-R', all_tokens_for_leaving_agents, 'leaving_agents')
    XLM_MFT_matrix_la=get_matrix('globalise/MFT-XLM-R', all_tokens_for_leaving_agents, 'leaving_agents')



    delta_glob = get_delta(glob_SFT_matrix_la, glob_MFT_matrix_la)
    delta_bert = get_delta(mBERT_SFT_matrix_la, mBERT_MFT_matrix_la)
    delta_xlm = get_delta(XLM_SFT_matrix_la, XLM_MFT_matrix_la)

    #Positive average delta: On average, values in A are greater than in B.
    #Negative average delta: On average, values in B are greater than in A.
    #Zero: On average, they are equal.

    print("AVG SIGNED DELTA GLOBERTISE LEAVING AGENTS")
    print(delta_glob)

    print("AVG SIGNED DELTA MBERT LEAVING AGENTS")
    print(delta_bert)

    print("AVG SIGNED DELTA XLM LEAVING AGENTS")
    print(delta_xlm)


main()