"""
A generation-time StoppingCriteria that ends a sequence early once a
phrase-sized chunk of its generated text has repeated (near-identically)
many times in a row -- the "verbatim repetition loop" failure mode
documented in ../new_paper_reasoning_signatures/truncation_error_examples.md,
which accounts for ~90% of truncated (hit max_new_tokens) completions in
this project's pruning/steering experiments.

Without this, a rollout that starts looping at token 500 still burns the
full max_new_tokens budget (16384) before returning -- this catches it
early and frees that sequence's slot for the next batch, without affecting
sequences that are still making real progress. Stops per-sequence (uses
HF's per-row StoppingCriteria protocol), not the whole batch.

Usage (assumes left-padded batch tokenization, as used throughout this
project's steering scripts):

    from repetition_stopping import RepetitionStoppingCriteria
    from transformers import StoppingCriteriaList

    prompt_len = input_ids.shape[1]
    stopping = StoppingCriteriaList([RepetitionStoppingCriteria(tokenizer, prompt_len)])
    gen = model.generate(..., stopping_criteria=stopping)
"""
import torch
from transformers import StoppingCriteria


class RepetitionStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len, phrase_words=12, consecutive_repeats=10,
                 check_every=20, similarity_threshold=0.75):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.phrase_words = phrase_words
        self.consecutive_repeats = consecutive_repeats
        self.check_every = check_every
        self.similarity_threshold = similarity_threshold
        self.stopped = None  # per-sequence bool list, sized lazily on first call

    @staticmethod
    def _word_set(chunk):
        return set(chunk.lower().split())

    @classmethod
    def _similar(cls, a, b, threshold):
        sa, sb = cls._word_set(a), cls._word_set(b)
        if not sa or not sb:
            return a.strip() == b.strip()
        union = len(sa | sb)
        return (len(sa & sb) / union) >= threshold if union else False

    def _has_repetition(self, text):
        words = text.split()
        needed = self.phrase_words * self.consecutive_repeats
        if len(words) < needed:
            return False
        tail_words = words[-needed:]
        chunks = [" ".join(tail_words[i:i + self.phrase_words])
                  for i in range(0, needed, self.phrase_words)]
        last = chunks[-1]
        return all(self._similar(c, last, self.similarity_threshold) for c in chunks)

    def __call__(self, input_ids, scores, **kwargs):
        batch_size = input_ids.shape[0]
        if self.stopped is None:
            self.stopped = [False] * batch_size
        n_generated = input_ids.shape[1] - self.prompt_len
        if n_generated > 0 and n_generated % self.check_every == 0:
            for i in range(batch_size):
                if self.stopped[i]:
                    continue
                new_tokens = input_ids[i, self.prompt_len:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                if self._has_repetition(text):
                    self.stopped[i] = True
        return torch.tensor(self.stopped, device=input_ids.device, dtype=torch.bool)
