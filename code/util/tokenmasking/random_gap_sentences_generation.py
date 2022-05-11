from .abstract_gap_sentences_generation import AbstractGapSentencesGeneration

import random

__all__ = ["RandomGapSentencesGeneration"]


class RandomGapSentencesGeneration(AbstractGapSentencesGeneration):

    def __init__(self, mask_token, gap_sentence_ratio, antimask_no_context=None):
        super().__init__(mask_token, gap_sentence_ratio, antimask_no_context=antimask_no_context)
        self.rng = random.Random()

    def set_seed(self, seed_object=None):
        self.rng.seed(seed_object)

    def get_state(self):
        return self.rng.getstate()

    def set_state(self, state):
        self.rng.setstate(state)

    def select_sentences_to_mask(self, tokenized_sentences):
        max_length = self.number_of_sentences_to_mask(tokenized_sentences)
        indices = list(range(len(tokenized_sentences)))
        self.rng.shuffle(indices)
        return set(indices[:max_length])
