from fuzzywuzzy import fuzz

from utils.text_preprocessing import remove_stopwords


'''
    Generally in an answer the first few words are common and related to the words in the question asked.
    Thus Fuzz Ratios can be a good feature to consider

    Eg: What is mental illness ?
    Valid-Answer: Mental illness is .... (the word mental illness exists in both question and answer)
    Invalid-Answer: Today weather is.... (as the answer in invalid, hence chances of it having common words is minimal)
    
'''
def fn_fuzz_ratio(row):
    q = row['q']
    a = row['a'][:20] # just picking first few words in the answer
    return fuzz.ratio(q, a)

def fn_fuzz_partial_ratio(row):
    q = row['q']
    a = row['a'][:20] # just picking first few words in the answer
    return fuzz.partial_ratio(q, a)
    
def fn_fuzz_token_sort_ratio(row):
    q = row['q']
    a = row['a'][:20] # just picking first few words in the answer
    return fuzz.token_sort_ratio(q, a)

def fn_fuzz_token_set_ratio(row):
    q = row['q']
    a = row['a'][:20] # just picking first few words in the answer
    return fuzz.token_set_ratio(q, a)

def fn_common_words_count(row):
    ''' Valid pairs have similar words in both question and answer'''
    q = remove_stopwords(row['q'])
    a = remove_stopwords(row['a'])
    return len(list(set(q.split())&set(a.split())))


def jaccard_similarity(row): 
    
    doc1 = row['q']
    doc2 = row['a']
    
    doc1 = remove_stopwords(doc1)
    doc2 = remove_stopwords(doc2)
    
    # List the unique words in a document
    words_doc1 = set(doc1.lower().split()) 
    words_doc2 = set(doc2.lower().split())
    
    # Find the intersection of words list of doc1 & doc2
    intersection = words_doc1.intersection(words_doc2)

    # Find the union of words list of doc1 & doc2
    union = words_doc1.union(words_doc2)
        
    # Calculate Jaccard similarity score 
    # using length of intersection set divided by length of union set
    return round(100* float(len(intersection)) / len(union), 2)
