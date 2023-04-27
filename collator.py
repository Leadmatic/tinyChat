class TextCollator:
    """
    Collator to process a batch of instruction-response pairs.
    """

    # Source Collator: https://github.com/Reason-Wang/flan-alpaca-lora/blob/main/dataset.py
    def __init__(self, text_tokenizer):
        self.text_tokenizer = text_tokenizer

    def __call__(self, batch):
        instructions = [pair[0] for pair in batch]
        responses = [pair[1] for pair in batch]

        # Tokenize the instructions with padding and truncation
        tokenized_inputs = self.text_tokenizer(
            instructions,
            max_length=40,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Tokenize the responses with padding and truncation
        tokenized_labels = self.text_tokenizer(
            responses,
            max_length=160,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).input_ids

        tokenized_inputs["labels"] = tokenized_labels

        return tokenized_inputs
