from .scored_gap_sentences_generation import ScoredGapSentencesGeneration

from rouge_score.rouge_scorer import RougeScorer

__all__ = ["RougeGapSentencesGeneration"]


class RougeGapSentencesGeneration(ScoredGapSentencesGeneration):

    # Original implementation: https://github.com/google-research/pegasus/blob/29fe4b974676fdd790069923b4e38f9aec01ff08/pegasus/ops/sentence_selection.cc
    # See also: https://github.com/google-research/pegasus/blob/29fe4b974676fdd790069923b4e38f9aec01ff08/pegasus/ops/pretrain_parsing_ops.cc
    # See also: https://github.com/google-research/pegasus/blob/29fe4b974676fdd790069923b4e38f9aec01ff08/pegasus/ops/pretrain_parsing_ops_test.py

    def __init__(self, mask_token, gap_sentence_ratio, select_sequentially, antimask_no_context=None, rouge_type=None, scorer_options=None):
        super().__init__(mask_token, gap_sentence_ratio, select_sequentially, antimask_no_context=antimask_no_context)
        if not rouge_type:
            rouge_type = "rouge1"
        if not scorer_options:
            scorer_options = {}
            scorer_options["use_stemmer"] = True
        self.scorer = RougeScorer([rouge_type], **scorer_options)

    @property
    def rouge_type(self):
        return self.scorer.rouge_types[0]

    def score(self, selected_sentences, unselected_sentences):
        selected_text = "\n".join(map(self.join_sentence_tokens, selected_sentences))
        unselected_text = "\n".join(map(self.join_sentence_tokens, unselected_sentences))
        return self.scorer.score(unselected_text, selected_text)[self.rouge_type].fmeasure
