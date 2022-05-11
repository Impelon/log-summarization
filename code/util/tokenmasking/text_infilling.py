from .abstract_masking_algorithm import AbstractMaskingAlgorithm

import numpy as np

__all__ = ["TextInfilling"]

class TokenReplacement:

    def __init__(self, *contents):
        self.contents = contents

    def __iter__(self):
        return iter(self.contents)

    def __len__(self):
        return len(self.contents)

class TextInfilling(AbstractMaskingAlgorithm):

    # Original implementation: https://github.com/pytorch/fairseq/blob/fcca32258c8e8bcc9f9890bf4714fa2f96b6b3e1/fairseq/data/denoising_dataset.py
    # SpanBERT: https://github.com/facebookresearch/SpanBERT/blob/main/pretraining/fairseq/data/masking.py
    # unofficial reimplementation: https://github.com/prajdabre/yanmtt/blob/4d329c3bcb81ca432d5947bb4673897086ee7f32/common_utils.py#L484-L502

    def __init__(self, mask_token, masking_ratio, mean_length):
        super().__init__(mask_token, masking_ratio)
        self.mean_length = mean_length
        self.set_seed(None)

    def set_seed(self, seed_object=None):
        self.rng = np.random.default_rng(seed_object)

    def get_state(self):
        return self.rng.bit_generator.state

    def set_state(self, state):
        self.rng.bit_generator.state = state

    def _is_kept(self, token):
        return token == self.sentence_separator or isinstance(token, TokenReplacement)

    def _is_not_mask(self, token):
        return token == self.sentence_separator or token != self.mask_token

    def _find_free_mask_position(self, tokens, position, mask_length, is_kept=None, callback=None):
        """
        Find a position where enough tokens (an amount equal to mask_length)
        can be masked, starting from an initial position.
        """
        if not is_kept:
            is_kept = self._is_kept
        tokens_found = 0
        initial_position = position
        first_position = initial_position
        while tokens_found < mask_length and position < len(tokens):
            if not is_kept(tokens[position]):
                if not tokens_found:
                    first_position = position
                tokens_found += 1
                if callback:
                    callback(position, tokens_found)
            position += 1
        if tokens_found >= mask_length:
            return first_position
        position = initial_position
        while tokens_found < mask_length and position > 0:
            position -= 1
            if not is_kept(tokens[position]):
                tokens_found += 1
                if callback:
                    callback(position, tokens_found)
        return position

    def _expand_replacements(self, tokens):
        for token in tokens:
            if isinstance(token, TokenReplacement):
                yield from token
            else:
                yield token

    def mask(self, tokenized_sentences):
        tokens = list(self.flatten_sentences(tokenized_sentences))
        masks_to_insert = int(np.ceil((len(tokens) - len(tokenized_sentences)) * self.masking_ratio))
        while masks_to_insert > 0:
            length = min(masks_to_insert, self.rng.poisson(self.mean_length))
            masks_to_insert -= length
            position = self.rng.integers(len(tokens))
            if length == 0:
                if isinstance(tokens[position], TokenReplacement):
                    replacement = TokenReplacement(self.mask_token, *tokens[position])
                else:
                    replacement = TokenReplacement(self.mask_token, tokens[position])
                tokens[position] = replacement
                continue

            def replace_at(index, tokens_found):
                tokens[index] = TokenReplacement()
            position = self._find_free_mask_position(tokens, position, length, callback=replace_at)
            tokens[position] = TokenReplacement(self.mask_token)

        return self.unflatten_sentences(self._expand_replacements(tokens))

    def antimask(self, tokenized_sentences):
        tokens = tuple(self.flatten_sentences(tokenized_sentences))
        antitokens = [token if token == self.sentence_separator else self.mask_token for token in tokens]
        masks_to_insert = int(np.ceil((len(antitokens) - len(tokenized_sentences)) * self.masking_ratio))
        while masks_to_insert > 0:
            length = min(masks_to_insert, self.rng.poisson(self.mean_length))
            masks_to_insert -= length
            position = self.rng.integers(len(antitokens))
            if length == 0:
                continue

            def copy_at(index, tokens_found):
                antitokens[index] = tokens[index]
            self._find_free_mask_position(antitokens, position, length, is_kept=self._is_not_mask, callback=copy_at)

        return self.unflatten_sentences(antitokens)
