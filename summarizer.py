"""
The summarizer module is used to summarize text using the BART model, and it
would be wrapped into a CLAMS app called 'app-bart-summarizer'.
"""
import torch
from transformers import pipeline
from transformers import BartTokenizer

MAX_LEN = 512  #FIXME: BART model allows maximum 1024 tokens, but this number does not work.
MODEL = "facebook/bart-large-cnn"

class TextSummarizer:
    """
    The TextSummarizer class is used to summarize text using the BART model.
    The logistics of the summarizer is to summarize an entire text by summarizing all
    of its chunks with maximum 512 tokens.
    """
    def __init__(self):
        self.summarizer = None
        self.tokenizer = None
        self.model = MODEL
        self._load()

    def _load(self) -> None:
        device = torch.device("cuda:0" if torch.cuda.is_available()
                              and torch.cuda.mem_get_info()[1] > 4000000000
                              else "cpu")
        self.summarizer = pipeline("summarization", model=self.model, device=device)
        self.tokenizer = BartTokenizer.from_pretrained(self.model)

    def _summarize_chunk(self, small_text: str, max_len=150) -> str:
        min_len = 30 if max_len > 30 else int(max_len/2)
        return self.summarizer(small_text, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]

    def summarize_text(self, text: str) -> str:
        """
        Summarizes the given text by chunking it into smaller texts with maximum 512 tokens.
        """
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > MAX_LEN:
            chunks = [tokens[i:i+MAX_LEN] for i in range(0, len(tokens), MAX_LEN)]
            text_in_chunks = [self.tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]
            chunk_summaries = [self._summarize_chunk(chunk_text) for chunk_text in text_in_chunks]
            summary_text = " ".join(chunk_summaries)
            return self._summarize_chunk(summary_text)
        else:
            return self._summarize_chunk(text)
