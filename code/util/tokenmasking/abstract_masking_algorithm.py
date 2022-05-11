import abc
import itertools

__all__ = ["AbstractMaskingAlgorithm"]


class AbstractMaskingAlgorithm(abc.ABC):

    sentence_separator = None

    def __init__(self, mask_token, masking_ratio):
        self.mask_token = mask_token
        self.masking_ratio = max(0.0, min(1.0, masking_ratio))
        self.join_sentence_tokens = lambda tokens: " ".join(tokens)

    @abc.abstractmethod
    def mask(self, tokenized_sentences):
        return None

    @abc.abstractmethod
    def antimask(self, tokenized_sentences):
        """
        Mask every token except the tokens that mask() would replace.
        """
        return None

    # Deterministic algorithms can simply ignore these methods and
    # use these default-implementations that do nothing.

    def set_seed(self, seed_object=None):
        """
        Initialize internal state of the random number generator from the given object.
        """
        pass

    def get_state(self):
        """
        Return internal state; can be passed to set_state() later.
        """
        return None

    def set_state(self, state):
        """
        Restore internal state from object returned by get_state().
        """
        pass

    @classmethod
    def flatten_sentences(cls, tokenized_sentences):
        separated_sentences = itertools.chain.from_iterable(zip(tokenized_sentences, itertools.repeat((cls.sentence_separator,))))
        flattened_tokens = itertools.chain.from_iterable(separated_sentences)
        return flattened_tokens

    @classmethod
    def unflatten_sentences(cls, flattened_tokens):
        token_iterator = iter(flattened_tokens)
        while True:
            found_separator = False

            def find_separator(x):
                nonlocal found_separator
                found_separator = x == cls.sentence_separator
                return not found_separator

            sentence = tuple(itertools.takewhile(find_separator, token_iterator))
            if found_separator:
                yield sentence
            else:
                break
