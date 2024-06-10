import os
import re
import nltk
import math
import json
from collections import defaultdict
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.metrics.distance import edit_distance
from tabulate import tabulate

nltk.download('stopwords')

class RetrievalSystem:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.inverted_index = defaultdict(list)
        self.doc_lengths = {}
        self.doc_vectors = {}
        self.doc_paths = {}  # Dictionary to store document paths
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def index_documents(self):
        doc_id = 0
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith('.html'):
                    doc_path = os.path.join(root, file)
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        terms = self.process_document(content)
                        self.doc_lengths[doc_id] = len(terms)
                        self.doc_paths[doc_id] = doc_path  # Store document path
                        for position, term in enumerate(terms):
                            self.inverted_index[term].append((doc_id, position))
                        doc_id += 1
        self.save_index('inverted_index.json', 'doc_paths.json')

    def process_document(self, content):
        terms = re.findall(r'\w+', content.lower())
        terms = [self.stemmer.stem(term) for term in terms if term not in self.stop_words]
        return terms

    def save_index(self, index_filepath, paths_filepath):
        with open(index_filepath, 'w') as f:
            json.dump(self.inverted_index, f)
        with open(paths_filepath, 'w') as f:
            json.dump(self.doc_paths, f)

    def load_index(self, index_filepath, paths_filepath):
        with open(index_filepath, 'r') as f:
            self.inverted_index = json.load(f)
        with open(paths_filepath, 'r') as f:
            self.doc_paths = json.load(f)

    def search(self, query):
        terms = self.process_document(query)
        if not terms:
            return []

        if 'AND' in query or 'OR' in query:
            return self.handle_boolean_query(terms)
        else:
            return self.retrieve_documents(terms)

    def handle_boolean_query(self, terms):
        if 'AND' in terms:
            terms = [term for term in terms if term != 'and']
            return self.conjunctive_query(terms)
        elif 'OR' in terms:
            terms = [term for term in terms if term != 'or']
            return self.disjunctive_query(terms)
        else:
            return self.retrieve_documents(terms)

    def retrieve_documents(self, terms):
        results = defaultdict(int)
        for term in terms:
            postings = self.inverted_index.get(term, [])
            for doc_id, _ in postings:
                results[doc_id] += 1
        return sorted(results, key=results.get, reverse=True)

    def conjunctive_query(self, terms):
        if not terms:
            return []
        postings_lists = [self.inverted_index.get(term, []) for term in terms]
        postings_lists.sort(key=len)
        result = set([doc_id for doc_id, _ in postings_lists[0]])
        for postings in postings_lists[1:]:
            result.intersection_update([doc_id for doc_id, _ in postings])
        return list(result)

    def disjunctive_query(self, terms):
        result = set()
        for term in terms:
            postings = self.inverted_index.get(term, [])
            result.update([doc_id for doc_id, _ in postings])
        return list(result)

    def compute_document_vectors(self):
        for term, postings in self.inverted_index.items():
            idf = math.log(len(self.doc_lengths) / len(postings))
            for doc_id, _ in postings:
                if doc_id not in self.doc_vectors:
                    self.doc_vectors[doc_id] = defaultdict(float)
                self.doc_vectors[doc_id][term] += idf

    def compute_cosine_similarity(self, query_vector, doc_vector):
        dot_product = sum(query_vector[term] * doc_vector.get(term, 0) for term in query_vector)
        query_norm = math.sqrt(sum(value ** 2 for value in query_vector.values()))
        doc_norm = math.sqrt(sum(value ** 2 for value in doc_vector.values()))
        if query_norm == 0 or doc_norm == 0:
            return 0.0
        return dot_product / (query_norm * doc_norm)

    def search_with_vector_space_model(self, query):
        query_terms = self.process_document(query)
        query_vector = defaultdict(float)
        for term in query_terms:
            query_vector[term] += 1
        self.compute_document_vectors()
        scores = {}
        for doc_id, doc_vector in self.doc_vectors.items():
            scores[doc_id] = self.compute_cosine_similarity(query_vector, doc_vector)
        return sorted(scores, key=scores.get, reverse=True)

    def suggest_corrections(self, query):
        terms = self.process_document(query)
        suggestions = []
        for term in terms:
            if term not in self.inverted_index:
                suggestions.append((term, self.find_closest_term(term)))
        return suggestions

    def find_closest_term(self, term):
        min_distance = float('inf')
        closest_term = term
        for indexed_term in self.inverted_index.keys():
            distance = edit_distance(term, indexed_term)
            if distance < min_distance:
                min_distance = distance
                closest_term = indexed_term
        return closest_term

def main():
    dataset_path = 'D:/1'
    retrieval_system = RetrievalSystem(dataset_path)

    # Index documents and save the index to a file (run in advance)
    print("Indexing documents... This may take a while.")
    retrieval_system.index_documents()
    print("Indexing complete.")

    # Load the inverted index from the file
    retrieval_system.load_index('inverted_index.json', 'doc_paths.json')

    while True:
        query = input("Enter your search query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        suggestions = retrieval_system.suggest_corrections(query)
        if suggestions:
            print("Did you mean:")
            for original, suggestion in suggestions:
                print(f"'{original}' -> '{suggestion}'")
        
        results = retrieval_system.search(query)
        if results:
            print("Search Results:")
            print(tabulate([[res, retrieval_system.doc_paths[str(res)]] for res in results], headers=["Document ID", "Document Path"], tablefmt="grid"))
        else:
            print("No documents found for the search query.")

        vector_results = retrieval_system.search_with_vector_space_model(query)
        if vector_results:
            print("Vector Space Model Search Results:")
            print(tabulate([[res, retrieval_system.doc_paths[str(res)]] for res in vector_results], headers=["Document ID", "Document Path"], tablefmt="grid"))
        else:
            print("No documents found for the search query using Vector Space Model.")

if __name__ == "__main__":
    main()
