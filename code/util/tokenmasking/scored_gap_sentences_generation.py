from .abstract_gap_sentences_generation import AbstractGapSentencesGeneration
from ..heap import Heap

import abc
import itertools

__all__ = ["ScoredGapSentencesGeneration"]


class ScoredGapSentencesGeneration(AbstractGapSentencesGeneration):

    # Original implementation: https://github.com/google-research/pegasus/blob/29fe4b974676fdd790069923b4e38f9aec01ff08/pegasus/ops/sentence_selection.cc

    def __init__(self, mask_token, gap_sentence_ratio, select_sequentially, antimask_no_context=None):
        super().__init__(mask_token, gap_sentence_ratio, antimask_no_context=antimask_no_context)
        self.select_sequentially = select_sequentially

    @abc.abstractmethod
    def score(self, selected_sentences, unselected_sentences):
        return 0

    def select_sentences_to_mask(self, tokenized_sentences):
        tokenized_sentences = list(tokenized_sentences)
        if self.select_sequentially:
            return self._select_sentences_sequentially(tokenized_sentences)
        else:
            return self._select_sentences_indepentently(tokenized_sentences)

    def _select_sentences_sequentially(self, tokenized_sentences):
        max_length = self.number_of_sentences_to_mask(tokenized_sentences)
        selected_indices = set()
        selected_sentences = []
        while len(selected_sentences) < max_length:
            max_heap = self.rank_by_score(selected_sentences, tokenized_sentences)
            # Find best non-empty sentence, if possible.
            best_score = max_heap.heap_items[0].key
            while max_heap and best_score == max_heap.heap_items[0].key:
                best_index = max_heap.pop()
                best_sentence = tokenized_sentences[best_index]
                if best_sentence:
                    break
            selected_indices.add(best_index)
            selected_sentences.append(best_sentence)
            tokenized_sentences[best_index] = []
        return selected_indices

    def _select_sentences_indepentently(self, tokenized_sentences):
        max_heap = self.rank_by_score([], tokenized_sentences)
        return set(itertools.islice(max_heap.consume(), self.number_of_sentences_to_mask(tokenized_sentences)))

    def rank_by_score(self, selected_sentences, tokenized_sentences):
        current_score = None
        max_heap = Heap(key=lambda x: -current_score)
        for i, sentence in enumerate(tokenized_sentences):
            current_score = self.score(selected_sentences + [sentence], tokenized_sentences[:i] + tokenized_sentences[i + 1:])
            max_heap.push(i)
        return max_heap
