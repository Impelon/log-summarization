from .abstract_masking_algorithm import AbstractMaskingAlgorithm

import abc

__all__ = ["AbstractGapSentencesGeneration"]


class AbstractGapSentencesGeneration(AbstractMaskingAlgorithm):

    def __init__(self, mask_token, gap_sentence_ratio, antimask_no_context=None):
        super().__init__(mask_token, gap_sentence_ratio)
        if antimask_no_context is None:
            antimask_no_context = True
        self.antimask_no_context = antimask_no_context

    @property
    def gap_sentence_ratio(self):
        return self.masking_ratio

    def number_of_sentences_to_mask(self, tokenized_sentences):
        return round(len(tokenized_sentences) * self.gap_sentence_ratio)

    @abc.abstractmethod
    def select_sentences_to_mask(self, tokenized_sentences):
        return set()

    def mask(self, tokenized_sentences):
        selected_sentence_indexes = self.select_sentences_to_mask(tokenized_sentences)
        for index, sentence in enumerate(tokenized_sentences):
            if index in selected_sentence_indexes:
                yield (self.mask_token,)
            else:
                yield sentence

    def antimask(self, tokenized_sentences):
        selected_sentence_indexes = self.select_sentences_to_mask(tokenized_sentences)
        for index, sentence in enumerate(tokenized_sentences):
            if index in selected_sentence_indexes:
                yield sentence
            elif not self.antimask_no_context:
                yield (self.mask_token,) * len(sentence)
