import re
import json

def process_file_to_dict(file_paths, label_mapping):
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
    for path in file_paths:
        with open(path, encoding='utf-8') as infile:
            sentence = []
            sentence_labels = []
            sentence_predicate = []
            sentence_pred_labels = []

            for line in infile:
                if line != '\n':
                    # remove the newline at the end of the line, then split the line based on the tab separator
                    line_split = line.strip('\n').split('\t')
                    # extract the token from the line in the file
                    token = line_split[4]

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
                else:
                    if sentence and sentence_labels and sentence_predicate and sentence_pred_labels:
                        sentence_dict = {'text': sentence, 'predicate': sentence_predicate, 'label': sentence_labels,
                                         'pred_label': sentence_pred_labels}
                        # get list of dicts
                        file_sentences.append(sentence_dict)
                        sentence_dict = {}
                        sentence = []
                        sentence_labels = []
                        sentence_predicate = []
                        sentence_pred_labels = []
                    else:
                        sentence_dict = {}
                        sentence = []
                        sentence_labels = []
                        sentence_predicate = []
                        sentence_pred_labels = []
            if sentence and sentence_labels and sentence_predicate and sentence_pred_labels:
                sentence_dict = {
                    'text': sentence,
                    'predicate': sentence_predicate,
                    'label': sentence_labels,
                    'pred_label': sentence_pred_labels
                }
                file_sentences.append(sentence_dict)

    return file_sentences

with open('label_mapping.json', 'r', encoding='utf-8') as f:
    label_mapping = json.load(f)

dickt = process_file_to_dict(['Data/1_p_80-ner-event-preanno_NL-HaNA_1.04.02_3604_0270-0276 - 1782 -.conllu'], label_mapping)

print(dickt)