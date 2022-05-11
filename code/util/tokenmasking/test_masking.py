from .abstract_masking_algorithm import AbstractMaskingAlgorithm

import abc
import contextlib
import importlib
import unittest
from unittest import mock


def skipUnlessCanImport(module_name, top_level=__name__.rsplit(".", 1)[0]):
    exception = None
    try:
        if top_level:
            module = importlib.import_module("." + module_name, top_level)
        else:
            module = importlib.import_module(module_name)
        globals()[module_name] = module
    except Exception as ex:
        exception = ex
    return unittest.skipIf(exception, str(exception))


def as_nested_list(sequence):
    sequence = list(sequence)
    for i in range(len(sequence)):
        try:
            sequence[i] = list(sequence[i])
        except TypeError:
            pass
    return sequence


class TestAbstractMaskingAlgorithm(unittest.TestCase):

    def test_flatten_unflatten(self):
        test_examples = (
            [[1, 2, 3]],
            [[1, 2, 3], [], [4, 5]],
            [["Hello", "World", "!"], ["Mask", "this!"]],
            [[]],
            [],
        )
        for tokenized_sentences in test_examples:
            with self.subTest():
                tokens = AbstractMaskingAlgorithm.flatten_sentences(tokenized_sentences[:])
                result = tuple(AbstractMaskingAlgorithm.unflatten_sentences(tokens))
                self.assertEqual(len(tokenized_sentences), len(result), msg="reference (first) does not have the same length as result (second)!")
                for i in range(len(tokenized_sentences)):
                    self.assertSequenceEqual(tokenized_sentences[i], result[i], msg="reference (first) does not match result (second)!")


class TestMaskingAlgorithmBase(unittest.TestCase, abc.ABC):

    @classmethod
    @abc.abstractmethod
    def initialize_masker(cls):
        pass

    @classmethod
    def setUpClass(cls):
        cls.masker = cls.initialize_masker()

    @contextlib.contextmanager
    def masking_ratio(self, ratio):
        original, self.masker.masking_ratio = self.masker.masking_ratio, ratio
        try:
            yield
        finally:
            self.masker.masking_ratio = original

    def assert_mask(self, tokenized_sentences, expected_sentences, *args, **kwargs):
        return self._assert_masking_method(self.masker.mask, tokenized_sentences, expected_sentences, *args, **kwargs)

    def assert_antimask(self, tokenized_sentences, expected_sentences, *args, **kwargs):
        return self._assert_masking_method(self.masker.antimask, tokenized_sentences, expected_sentences, *args, **kwargs)

    def _assert_masking_method(self, method, tokenized_sentences, expected_sentences, msg=None):
        if not msg:
            msg = "reference (first) does not match result (second)!"
        result = method(tokenized_sentences[:])
        self.assertEqual(as_nested_list(expected_sentences), as_nested_list(result), msg=msg)

    def test_rng_state(self):
        example_sentences = [["Hello", "world", "!"], ["This", "is", "an", "example", "sentence", "."],
                             ["Is", "this", "another", "?"], ["Yep", ",", "yet", "another", "..."]]
        with self.masking_ratio(0.5):
            state = self.masker.get_state()
            masked_results = []
            for i in range(5):
                masked_results.append(as_nested_list(self.masker.mask(example_sentences)))
            self.masker.set_state(state)
            for masked_sentences in masked_results:
                self.assertEqual(masked_sentences, as_nested_list(self.masker.mask(example_sentences)))

    def test_set_seed(self):
        example_sentences = [["Round", "two", "!"], ["Are", "you", "ready", "world", "?"],
                             ["Probably", "yes", "..."], ["But", "I", "thought", "it", "'d", "be", "nice", "to", "ask", "."]]
        with self.masking_ratio(0.5):
            self.masker.set_seed(0)
            masked_results = []
            for i in range(5):
                masked_results.append(as_nested_list(self.masker.antimask(example_sentences)))
            self.masker.set_seed(0)
            for masked_sentences in masked_results:
                self.assertEqual(masked_sentences, as_nested_list(self.masker.antimask(example_sentences)))


@skipUnlessCanImport("text_infilling")
class TestTextInfilling(TestMaskingAlgorithmBase):

    @classmethod
    def initialize_masker(cls):
        return text_infilling.TextInfilling(-1, 0, 3)

    @contextlib.contextmanager
    def mock_rng(self, **method_specs):
        mocked_rng = mock.Mock()
        for method, return_values in method_specs.items():
            getattr(mocked_rng, method).side_effect = return_values
        original, self.masker.rng = self.masker.rng, mocked_rng
        try:
            yield mocked_rng
        finally:
            self.masker.rng = original

    def _assert_masking_method(self, method, tokenized_sentences, expected_sentences, mask_lengths, mask_positions, **kwargs):
        amount_tokens = sum(1 for _ in self.masker.flatten_sentences(tokenized_sentences)) - len(tokenized_sentences)
        if amount_tokens == 0:
            masking_ratio = 0
        else:
            masks_to_insert = sum(mask_lengths)
            masking_ratio = masks_to_insert / amount_tokens
        with self.mock_rng(poisson=mask_lengths, integers=mask_positions):
            with self.masking_ratio(masking_ratio):
                super()._assert_masking_method(method, tokenized_sentences, expected_sentences, **kwargs)

    def test_mask(self):
        test_examples = {
            "once singular": ([[0, 1, 2, 3, -1, 7, 8, 9]], [[0, 1, 2, 3, -1, 7, -1, 9]], (1,), (6,)),
            "once multiple": ([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], [[0, 1, 2, 3, -1, 7, 8, 9]], (3,), (4,)),
            "twice singular": ([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], [[0, 1, 2, 3, -1, 5, 6, 7, -1, 9]], (1, 1), (4, 8)),
            "twice multiple": ([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], [[0, -1, -1, 7, 8, 9]], (3, 3), (1, 1)),
            "complex": ([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], [[0, 1, 2, 3, -1, 7, -1, 9]], (3, 1), (4, 8)),
            "complex same position": ([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], [[0, -1, -1, -1, 5, 6, 7, 8, 9]], (3, 0, 1), (1, 1, 1)),
            "three sent. complex": ([[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]], [[0, 1, 2], [3, -1], [7, -1, 9]], (3, 1), (5, 10)),
            "three sent. complex with zero": ([[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]], [[0, -1, 1, 2], [3, -1], [7, -1, 9]], (3, 0, 1), (5, 1, 10)),
            "three short sent. endpoints": ([[0, 1], [2, 3], [4]], [[-1, 1], [2, 3], [-1]], (1, 1), (0, 7)),
            "three short sent. middle": ([[0, 1], [2, 3], [4]], [[0, -1], [-1], [4]], (2, 1), (3, 1)),
            "three short sent. complex with zero": ([[0, 1], [2, 3], [4]], [[-1, -1, 1], [-1, 2, -1], [4]], (1, 0, 0, 1), (0, 0, 3, 4)),
            "two short sent. complex with zero": ([[0], [1, 2]], [[0, -1], [-1, 1, -1]], (0, 0, 1), (1, 2, 3)),
            "ten sent. complex": ([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
                                  [[0], [-1], [], [], [-1], [5], [6], [-1], [8], [-1]], (1, 4, 0, 1), (14, 2, 8, 18)),
            "three empty sentences": ([[], [], []], [[], [], []], None, None),
            "empty sentence": ([[]], [[]], None, None),
            "empty": ([], [], None, None),
        }
        for name, example in test_examples.items():
            with self.subTest(name):
                self.assert_mask(*example)

    def test_antimask(self):
        test_examples = {
            "once multiple": ([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], [[-1, -1, -1, -1, 4, 5, 6, -1, -1, -1]], (3,), (4,)),
            "twice singular": ([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], [[-1, -1, -1, -1, 4, -1, -1, -1, 8, -1]], (1, 1), (4, 8)),
            "twice multiple": ([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], [[-1, 1, 2, 3, 4, 5, 6, -1, -1, -1]], (3, 3), (1, 1)),
            "complex": ([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], [[-1, -1, -1, -1, 4, 5, 6, -1, 8, -1]], (3, 1), (4, 8)),
            "complex same position": ([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], [[-1, 1, 2, 3, 4, -1, -1, -1, -1, -1]], (3, 0, 1), (1, 1, 1)),
            "three sent. complex": ([[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]], [[-1, -1, -1], [-1, 4, 5], [6, -1, 8, -1]], (3, 1), (5, 10)),
            "three sent. complex with zero": ([[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]], [[-1, -1, -1], [-1, 4, 5], [6, -1, 8, -1]], (3, 0, 1), (5, 1, 10)),
            "three short sent. endpoints": ([[0, 1], [2, 3], [4]], [[0, -1], [-1, -1], [4]], (1, 1), (0, 7)),
            "three short sent. middle": ([[0, 1], [2, 3], [4]], [[-1, 1], [2, 3], [-1]], (2, 1), (3, 1)),
            "three short sent. complex with zero": ([[0, 1], [2, 3], [4]], [[0, -1], [-1, 3], [-1]], (1, 0, 0, 1), (0, 0, 3, 4)),
            "two short sent. complex with zero": ([[0], [1, 2]], [[-1], [-1, 2]], (0, 0, 1), (1, 2, 3)),
            "ten sent. complex": ([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
                                  [[-1], [1], [2], [3], [4], [-1], [-1], [7], [-1], [9]], (1, 4, 0, 1), (14, 2, 8, 18)),
            "three empty sentences": ([[], [], []], [[], [], []], None, None),
            "empty sentence": ([[]], [[]], None, None),
            "empty": ([], [], None, None),
        }
        for name, example in test_examples.items():
            with self.subTest(name):
                self.assert_antimask(*example)


@skipUnlessCanImport("random_gap_sentences_generation")
class TestRandomGapSentencesGeneration(TestMaskingAlgorithmBase):

    @classmethod
    def initialize_masker(cls):
        return random_gap_sentences_generation.RandomGapSentencesGeneration("*", 0)


class TestScoredGapSentencesGenerationBase(TestMaskingAlgorithmBase):

    @contextlib.contextmanager
    def mock_score(self, *scores):
        mocked_score = mock.Mock()
        mocked_score.side_effect = scores
        original, self.masker.score = self.masker.score, mocked_score
        try:
            yield mocked_score
        finally:
            self.masker.score = original

    def _assert_masking_method(self, method, tokenized_sentences, expected_sentences, *scores, **kwargs):
        with contextlib.ExitStack() as context_stack:
            if scores:
                context_stack.enter_context(self.mock_score(*scores))
            super()._assert_masking_method(method, tokenized_sentences, expected_sentences, **kwargs)


@skipUnlessCanImport("rouge_gap_sentences_generation")
class TestIndepententRougeGapSentencesGeneration(TestScoredGapSentencesGenerationBase):

    @classmethod
    def initialize_masker(cls):
        return rouge_gap_sentences_generation.RougeGapSentencesGeneration("<*>", 0.3, False)

    def test_prescored_mask(self):
        test_examples = {
            "one sentence": (0, [[0, 1, 2, 3, 4]], [[0, 1, 2, 3, 4]], (5,)),
            "one masked sentence": (0.5, [[0, 1, 2, 3, 4]], [[0, 1, 2, 3, 4]], (5,)),
            "few masked sentences": (0.3, [["Hello", "!"], ["I", "am"], ["a", "test."]], [[self.masker.mask_token], ["I", "am"], ["a", "test."]], (0.55, 0.3, 0.1)),
            "many masked sentences": (0.6, [["Hello", "!"], ["I", "am"], ["a", "test."]], [["Hello", "!"], [self.masker.mask_token], [self.masker.mask_token]], (-3, 5, 5)),
            "all masked sentences": (1, [["Hello", "!"], ["I", "am"], ["a", "test."]], [[self.masker.mask_token], [self.masker.mask_token], [self.masker.mask_token]], (-3, -4, -5)),
            "three empty sentences": (0.3, [[], [], []], [[], [self.masker.mask_token], []], (-8, 0, -1)),
            "empty": (1, [], [], (None,)),
        }
        for name, example in test_examples.items():
            with self.subTest(name):
                with self.masking_ratio(example[0]):
                    self.assert_mask(*example[1:-1], *example[-1])

    def test_mask(self):
        test_examples = {
            "one sentence": ([["This", "is", "a", "test", "."]], [["This", "is", "a", "test", "."]]),
            "identical sentences": ([["1", "2"], ["1", "2"], ["1", "2"]], [["<*>"], ["1", "2"], ["1", "2"]]),
            "few masked sentences": ([["1", "2"], ["1", "2", "3"], ["3", "4", "5"]], [["1", "2"], ["<*>"], ["3", "4", "5"]]),
            "more masked sentences": ([["1", "2"], ["1", "2", "3"], ["3", "4", "5"], ["5", "6"], ["1", "9"], ["1"], ["3", "6"], ["2"], ["1", "8"], ["3", "6"]],
                                      [["<*>"], ["<*>"], ["<*>"], ["5", "6"], ["1", "9"], ["1"], ["3", "6"], ["2"], ["1", "8"], ["3", "6"]]),
            "more masked sentences shuffled": ([["1", "2"], ["3", "6"], ["3", "4", "5"], ["1", "8"], ["1", "9"], ["5", "6"], ["1"], ["2"], ["1", "2", "3"], ["3", "6"]],
                                               [["<*>"], ["3", "6"], ["<*>"], ["1", "8"], ["1", "9"], ["5", "6"], ["1"], ["2"], ["<*>"], ["3", "6"]]),
            "other masked sentences": ([["2", "3"], ["1", "4", "5"], ["4", "6"], ["2", "4", "5"], ["7"], ["8"], ["9"], ["9"], ["9"], ["9"]],
                                       [["<*>"], ["<*>"], ["4", "6"], ["<*>"], ["7"], ["8"], ["9"], ["9"], ["9"], ["9"]]),
            "three empty sentences": ([[], [], []], [["<*>"], [], []]),
            "empty sentence": ([[]], [[]]),
            "empty": ([], []),
        }
        for name, example in test_examples.items():
            with self.subTest(name):
                self.assert_mask(*example)

    def test_antimask(self):
        test_examples = {
            "one sentence": ([["This", "is", "a", "test", "."]], []),
            "identical sentences": ([["1", "2"], ["1", "2"], ["1", "2"]], [["1", "2"]]),
            "few masked sentences": ([["1", "2"], ["1", "2", "3"], ["3", "4", "5"]], [["1", "2", "3"]]),
            "more masked sentences": ([["1", "2"], ["1", "2", "3"], ["3", "4", "5"], ["5", "6"], ["1", "9"], ["1"], ["3", "6"], ["2"], ["1", "8"], ["3", "6"]],
                                      [["1", "2"], ["1", "2", "3"], ["3", "4", "5"]]),
            "more masked sentences shuffled": ([["1", "2"], ["3", "6"], ["3", "4", "5"], ["1", "8"], ["1", "9"], ["5", "6"], ["1"], ["2"], ["1", "2", "3"], ["3", "6"]],
                                               [["1", "2"], ["3", "4", "5"], ["1", "2", "3"]]),
            "other masked sentences": ([["2", "3"], ["1", "4", "5"], ["4", "6"], ["2", "4", "5"], ["7"], ["8"], ["9"], ["9"], ["9"], ["9"]],
                                       [["2", "3"], ["1", "4", "5"], ["2", "4", "5"]]),
            "three empty sentences": ([[], [], []], [[]]),
            "empty sentence": ([[]], []),
            "empty": ([], []),
        }
        for name, example in test_examples.items():
            with self.subTest(name):
                self.assert_antimask(*example)


@skipUnlessCanImport("rouge_gap_sentences_generation")
class TestSequentialRougeGapSentencesGeneration(TestScoredGapSentencesGenerationBase):

    @classmethod
    def initialize_masker(cls):
        return rouge_gap_sentences_generation.RougeGapSentencesGeneration("<*>", 0.3, True, antimask_no_context=False)

    def test_prescored_mask(self):
        test_examples = {
            "one sentence": (0, [[0, 1, 2, 3, 4]], [[0, 1, 2, 3, 4]], (5,)),
            "one masked sentence": (0.5, [[0, 1, 2, 3, 4]], [[0, 1, 2, 3, 4]], (5,)),
            "few masked sentences": (0.3, [["Hello", "!"], ["I", "am"], ["a", "test."]], [[self.masker.mask_token], ["I", "am"], ["a", "test."]], (0.55, 0.3, 0.1)),
            "many masked sentences": (0.6, [["Hello", "!"], ["I", "am"], ["a", "test."]], [["Hello", "!"], [self.masker.mask_token], [self.masker.mask_token]], (-3, 5, 5, 0, -9, 8)),
            "all masked sentences": (1, [["Hello", "!"], ["I", "am"], ["a", "test."]], [[self.masker.mask_token], [self.masker.mask_token], [self.masker.mask_token]], (-3, -4, -5, -1, 10, 100, -1, 0, -1)),
            "three empty sentences": (0.3, [[], [], []], [[], [self.masker.mask_token], []], (-8, 0, -1)),
            "empty": (1, [], [], (None,)),
        }
        for name, example in test_examples.items():
            with self.subTest(name):
                with self.masking_ratio(example[0]):
                    self.assert_mask(*example[1:-1], *example[-1])

    def test_mask(self):
        test_examples = {
            "one sentence": ([["This", "is", "a", "test", "."]], [["This", "is", "a", "test", "."]]),
            "identical sentences": ([["1", "2"], ["1", "2"], ["1", "2"]], [["<*>"], ["1", "2"], ["1", "2"]]),
            "few masked sentences": ([["1", "2"], ["1", "2", "3"], ["3", "4", "5"]], [["1", "2"], ["<*>"], ["3", "4", "5"]]),
            "more masked sentences": ([["1", "2"], ["1", "2", "3"], ["3", "4", "5"], ["5", "6"], ["1", "9"], ["1"], ["3", "6"], ["2"], ["1", "8"], ["3", "6"]],
                                      [["<*>"], ["<*>"], ["<*>"], ["5", "6"], ["1", "9"], ["1"], ["3", "6"], ["2"], ["1", "8"], ["3", "6"]]),
            "more masked sentences shuffled": ([["1", "2"], ["3", "6"], ["3", "4", "5"], ["1", "8"], ["1", "9"], ["5", "6"], ["1"], ["2"], ["1", "2", "3"], ["3", "6"]],
                                               [["<*>"], ["<*>"], ["3", "4", "5"], ["1", "8"], ["1", "9"], ["5", "6"], ["1"], ["2"], ["<*>"], ["3", "6"]]),
            "other masked sentences": ([["2", "3"], ["1", "4", "5"], ["4", "6"], ["2", "4", "5"], ["7"], ["8"], ["9"], ["9"], ["9"], ["9"]],
                                       [["2", "3"], ["1", "4", "5"], ["4", "6"], ["<*>"], ["7"], ["8"], ["<*>"], ["<*>"], ["9"], ["9"]]),
            "three empty sentences": ([[], [], []], [[], [], ["<*>"]]),
            "empty sentence": ([[]], [[]]),
            "empty": ([], []),
        }
        for name, example in test_examples.items():
            with self.subTest(name):
                self.assert_mask(*example)

    def test_antimask(self):
        test_examples = {
            "one sentence": ([["This", "is", "a", "test", "."]], [["<*>", "<*>", "<*>", "<*>", "<*>"]]),
            "identical sentences": ([["1", "2"], ["1", "2"], ["1", "2"]], [["1", "2"], ["<*>", "<*>"], ["<*>", "<*>"]]),
            "few masked sentences": ([["1", "2"], ["1", "2", "3"], ["3", "4", "5"]], [["<*>", "<*>"], ["1", "2", "3"], ["<*>", "<*>", "<*>"]]),
            "more masked sentences": ([["1", "2"], ["1", "2", "3"], ["3", "4", "5"], ["5", "6"], ["1", "9"], ["1"], ["3", "6"], ["2"], ["1", "8"], ["3", "6"]],
                                      [["1", "2"], ["1", "2", "3"], ["3", "4", "5"], ["<*>", "<*>"], ["<*>", "<*>"], ["<*>"], ["<*>", "<*>"], ["<*>"], ["<*>", "<*>"], ["<*>", "<*>"]]),
            "more masked sentences shuffled": ([["1", "2"], ["3", "6"], ["3", "4", "5"], ["1", "8"], ["1", "9"], ["5", "6"], ["1"], ["2"], ["1", "2", "3"], ["3", "6"]],
                                               [["1", "2"], ["3", "6"], ["<*>", "<*>", "<*>"], ["<*>", "<*>"], ["<*>", "<*>"], ["<*>", "<*>"], ["<*>"], ["<*>"], ["1", "2", "3"], ["<*>", "<*>"]]),
            "other masked sentences": ([["2", "3"], ["1", "4", "5"], ["4", "6"], ["2", "4", "5"], ["7"], ["8"], ["9"], ["9"], ["9"], ["9"]],
                                       [["<*>", "<*>"], ["<*>", "<*>", "<*>"], ["<*>", "<*>"], ["2", "4", "5"], ["<*>"], ["<*>"],  ["9"], ["9"], ["<*>"], ["<*>"]]),
            "three empty sentences": ([[], [], []], [[], [], []]),
            "empty sentence": ([[]], [[]]),
            "empty": ([], []),
        }
        for name, example in test_examples.items():
            with self.subTest(name):
                self.assert_antimask(*example)


@skipUnlessCanImport("fast_rouge_gap_sentences_generation")
class TestSequentialFastRougeGapSentencesGeneration(TestSequentialRougeGapSentencesGeneration):
    # Should behave the same as the regular version.
    pass


@skipUnlessCanImport("fast_rouge_gap_sentences_generation")
class TestSequentialFastRougeGapSentencesGeneration(TestSequentialRougeGapSentencesGeneration):
    # Should behave the same as the regular version.
    pass


# Prevent abstract base classes from being executed as a TestCase:
del TestMaskingAlgorithmBase
del TestScoredGapSentencesGenerationBase

if __name__ == "__main__":
    unittest.main()
