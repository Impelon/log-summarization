from .scored_gap_sentences_generation import ScoredGapSentencesGeneration

from rouge import rouge_n_summary_level

__all__ = ["FastRougeGapSentencesGeneration"]


class FastRougeGapSentencesGeneration(ScoredGapSentencesGeneration):
    """
    This implementation is faster, but also does not reproduce rouge-scores
    from the original implementation, as it skips stemming.

    Nevertheless, this should not be that impactful,
    as ROUGE is just used as an *approximation* of importance.
    """

    # Original implementation: https://github.com/google-research/pegasus/blob/29fe4b974676fdd790069923b4e38f9aec01ff08/pegasus/ops/sentence_selection.cc
    # See also: https://github.com/google-research/pegasus/blob/29fe4b974676fdd790069923b4e38f9aec01ff08/pegasus/ops/pretrain_parsing_ops.cc
    # See also: https://github.com/google-research/pegasus/blob/29fe4b974676fdd790069923b4e38f9aec01ff08/pegasus/ops/pretrain_parsing_ops_test.py

    def __init__(self, mask_token, gap_sentence_ratio, select_sequentially, antimask_no_context=None, ngram=None):
        super().__init__(mask_token, gap_sentence_ratio, select_sequentially, antimask_no_context=antimask_no_context)
        if ngram == None:
            ngram = 1
        self.ngram = ngram

    def score(self, selected_sentences, unselected_sentences):
        return rouge_n_summary_level(selected_sentences, unselected_sentences, self.ngram).f1_measure
