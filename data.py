import json
from torch.utils.data import Dataset


PROMPT_FORMATS = {
    "input_present": "{instruction}\n\n {input}\n\n",
    "input_absent": "{instruction}\n\n",
}


class TextDataset(Dataset):
    """
    Dataset for instruction-response sequence tasks.
    """

    # Source Dataloader: https://github.com/Reason-Wang/flan-alpaca-lora/blob/main/dataset.py
    def __init__(self, json_file_path):
        super().__init__()

        # Load the dataset from the JSON file
        with open(json_file_path, "r") as file:
            raw_data = json.load(file)

        # Get the instruction formats
        format_input_present, format_input_absent = (
            PROMPT_FORMATS["input_present"],
            PROMPT_FORMATS["input_absent"],
        )

        # Create the instruction and response lists
        instructions = [
            format_input_present.format_map(record)
            if record.get("input", "") != ""
            else format_input_absent.format_map(record)
            for record in raw_data
        ]
        responses = [record["output"] for record in raw_data]

        self.instructions = instructions
        self.responses = responses

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, index):
        return self.instructions[index], self.responses[index]
