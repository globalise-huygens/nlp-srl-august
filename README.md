This is code of the experiments and results presented in 'Comparing Domain-Specific, Multilingual and Multitask Strategies to Overcome Data Scarcity for Semantic Role Labelling in Highly Noisy Early Modern Dutch'.


# Code

Folders
- Code/SRL
Contains code to singletask fine-tune (SRL) with two settings: [PRED] special token information and ET: specific [Event_Type] special token information for the predicate.

- Code/MultiTask
Contains code to multitask fine-tune (SRL and NER) with two settings: [PRED] special token information and ET: specific [Event_Type] special token information for the predicate.

- Code/lexical_class_similarity_test
Contains the .py file to perform our extra similarity test experiment. Directly prints results.

# Results

Contains all results of our main experiments. Also contains py files with functions to reproduce the data distribution, significance tests, and mean ± std across seeds, and an ipynb file that can be run to easily reproduce the results as reported in the paper.

