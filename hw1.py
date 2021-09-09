import argparse
import math
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List
from typing import Tuple
from typing import Generator


# Generator for all n-grams in text
# n is a (non-negative) int
# text is a list of strings
# Yields n-gram tuples of the form (string, context), where context is a tuple of strings
def get_ngrams(n: int, test: List[str]) -> Generator[Tuple[str, Tuple[str, ...]], None, None]:
    addStart = ["<s>"]
    #print(addStart)
    i = n-1
    while i > 0:
        test = addStart + test
        i -= 1
    #appending end token
    test = test + ["</s>"]
    nonRealTokens = ["<s>"]
    #Creating tokens
    tokens = test
    #Creating nGramTuple of each token
    index = 0
    while index < len(tokens):
        if tokens[index] not in nonRealTokens:
            context = tuple()
            count = index-n+1
            while count < index:
                added_value = tokens[count]
                added_value_tuple = (added_value,)
                context = context + added_value_tuple
                count += 1
            nGramTuple = (tokens[index], tuple(context))
            yield nGramTuple
        index += 1

# Loads and tokenizes a corpus
# corpus_path is a string
# Returns a list of sentences, where each sentence is a list of strings


def load_corpus(corpus_path: str) -> List[List[str]]:
    f = open(corpus_path)
    text = f.read()
    paragraph = text.split("\n")
    sentences = sent_tokenize(str(paragraph))
    sentence_words = []
    for sentence in sentences:
        #word_list = nltk.word_tokenize(sentence)
        word_list = word_tokenize(sentence)
        sentence_words.append(list(word_list))
    return sentence_words


# Builds an n-gram model from a corpus
# n is a (non-negative) int
# corpus_path is a string
# Returns an NGramLM
def create_ngram_lm(n: int, corpus_path: str) -> 'NGramLM':
    sentence_words = load_corpus(corpus_path)
    nGramObj = NGramLM(n)
    for sentences in sentence_words:
        nGramObj.update(sentences)
    return nGramObj


# An n-gram language model
class NGramLM:
    def __init__(self, n: int):
        self.n = n
        self.ngram_counts = {}
        self.context_counts = {}
        self.vocabulary = set()

    # Updates internal counts based on the n-grams in text
    # text is a list of strings
    # No return value
    def update(self, text: List[str]) -> None:
        nGramTuples = get_ngrams(self.n, text)
        for nGramTuple in nGramTuples:
            token = nGramTuple[0]
            context = nGramTuple[1]
            self.vocabulary.add(token)
            if nGramTuple not in self.ngram_counts.keys():
                self.ngram_counts[nGramTuple] = 1
            else:
                self.ngram_counts[nGramTuple] += 1
            if context not in self.context_counts.keys():
                self.context_counts[context] = [token]
            else:
                self.context_counts[context].append(token)

    # Calculates the MLE probability of an n-gram
    # word is a string
    # context is a tuple of strings
    # delta is an float
    # Returns a float
    def get_ngram_prob(self, word: str, context: Tuple[str, ...], delta=.0) -> float:
        #Use delta = some value for smoothing ( Q2)
        if (word,context) in self.ngram_counts.keys():
            count_of_token = self.ngram_counts[(word,context)]
        else:
            count_of_token = 0.0
        if context in self.context_counts.keys():
            count_of_context = len(self.context_counts[context])
        else:
            return 1/len(self.vocabulary)
        result = count_of_token / count_of_context
        return result

    # Calculates the log probability of a sentence
    # sent is a list of strings
    # delta is a float
    # Returns a float
    def get_sent_log_prob(self, sent: List[str], delta=.0) -> float:
        n_grams_sent = get_ngrams(self.n, sent)
        prob = 0
        for n_gram in n_grams_sent:
            word = n_gram[0]
            context = n_gram[1]
            prob = self.get_ngram_prob(word, context, delta)
            if prob == 0:
                continue;
            current_ngram_prob = math.log(prob, 2)
            prob = prob + current_ngram_prob
        return prob
        

    # Calculates the perplexity of a language model on a test corpus
    # corpus is a list of lists of strings
    # Returns a float
    def get_perplexity(self, corpus: List[List[str]]) -> float:
        pass

    # Samples a word from the probability distribution for a given context
    # context is a tuple of strings
    # delta is an float
    # Returns a string
    def generate_random_word(self, context: Tuple[str, ...], delta=.0) -> str:
        r = random.random()
        probs = {}
        tokens = self.context_counts[context]
        for token in tokens:
            probs[token] = self.get_ngram_prob(context, token)

        sum_p = 0
        for token in sorted(probs):
            sum_p += probs[token]
            if sum_p > r:
                return token

    # Generates a random sentence
    # max_length is an int
    # delta is a float
    # Returns a string
    def generate_random_text(self, max_length: int, delta=.0) -> str:
        n = self.n
        context_queue = (n - 1) * ['<s>']
        result = []
        for _ in range(max_length):
            obj = self.random_token(tuple(context_queue))
            result.append(obj)
            if n > 1:
                context_queue.pop(0)
                if context_queue == '</s>':
                    break
                if obj == '.':
                    context_queue = (n - 1) * ['<s>']
                else:
                    context_queue.append(obj)
        return ' '.join(result)


def main(corpus_path: str, delta: float, seed: int):
    trigram_lm = create_ngram_lm(3, corpus_path)
    s1 = 'God has given it to me, let him who touches it beware!'
    s2 = 'Where is the prince, my Dauphin?'

    print(trigram_lm.get_sent_log_prob(word_tokenize(s1)))
    print(trigram_lm.get_sent_log_prob(word_tokenize(s2)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CS6320 HW1")
    parser.add_argument('corpus_path', nargs="?", type=str,
                        default='warpeace.txt', help='Path to corpus file')
    parser.add_argument('delta', nargs="?", type=float,
                        default=.0, help='Delta value used for smoothing')
    parser.add_argument('seed', nargs="?", type=int, default=82761904,
                        help='Random seed used for text generation')
    args = parser.parse_args()
    random.seed(args.seed)
    main(args.corpus_path, args.delta, args.seed)
