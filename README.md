![Banner](tinyChat.jpeg)

# tinyChat 
tinyChat is an instruction-based large language model (LLM) that harnesses recent breakthroughs in open-source machine learning, such as Databrick's Dolly dataset and Microsoft's LoRA, to deliver NLP capabilities at a fraction of the size of GPT-3.5. tinyChat leverages the Google Flan-T5-Large 770M parameter model as its base combined with adaptor weights trained on the Dolly 15k dataset. This is accomplished using a method of training known as LoRA. tinyChat provides ChatGPT-like capabilities on tasks like summarization, question answering, and sentiment analysis, while remaining open source under the Apache 2.0 license and at 1% the size of GPT-3.5. This project aims to initiate discussions around efficient model architectures and responsible use of generative AI. Its acccessible via HuggingFace model hub and with its code repository on Github, tinyChat is available for experimentation and contribution. 

NOTE: This project is early in development.


### Prerequisites

Ensure you have the following dependencies installed:

- Python 3.6 or later
- [transformers](https://github.com/huggingface/transformers)
- [torch](https://pytorch.org/)
- [pandas](https://pandas.pydata.org/)
- [pyarrow](https://arrow.apache.org/docs/python/install.html)
- [peft](https://github.com/huggingface/peft)
- [sentencepiece](https://pypi.org/project/sentencepiece/)
- [bitsandbytes](https://pypi.org/project/bitsandbytes/)

You can install these dependencies using `pip`:

```
pip install -r requirements.txt
```

### Files in the Repository

1. `train.py`: A Python script for training a PEFT model with LORA using the Hugging Face Transformers library.
2. `convert_data.py`: A Python script for converting data from Parquet format to JSON.
3. `README.md`: This README file with instructions on how to use the scripts.

### Obtain Data
tinyChat is trained on the `databricks_dolly_15k` dataset by Databricks. It can be found [here](https://huggingface.co/datasets/HuggingFaceH4/databricks_dolly_15k). You must obtain the `databricks_data.parquet`.

### Data Conversion

To convert data from Parquet format to JSON, follow these steps:

1. Place your Parquet file (e.g., `databricks_data.parquet`) in the same directory as the `convert_data.py` script.
2. Run the `convert_data.py` script:

```
python convert_data.py
```

3. The script will generate a JSON file named `databricks.json` in the same directory.

### Model Training

To train the PEFT model with LORA, follow these steps:

1. Ensure the training data in JSON format (e.g., `databricks.json`) is in the same directory as the `train.py` script. If you used the `convert_data.py` script, this should already be the case.
2. Run the `train.py` script:

```
python train.py
```

3. The script will train the model and save the trained model in the `output` directory.

### Customizing the Training

You can modify the `train.py` script to use different model configurations or training arguments. The dataclass `Arguments` contains fields for model, data, and training arguments. Update the default values for these fields in the `Arguments` dataclass as needed.

For example, you can change the default model by updating the `model_name_or_path` field, or modify the LORA configuration by updating the `lora_r`, `lora_alpha`, and other LORA-related fields.


### What can tinyChat do?

TinyChat like Flan-T5 can perform well on a variety of NLP tasks such as question answering with tinyChat improving on Flan-T5s creative capabilities and summarization.

#### Question and Answering

In the example below, the question asks which country is mentioned in the input with the name of the country not disclosed. Both models are accurately able to determine the country is Morocco.

Prompt: What country did the event take place? Provide only the name of the COUNTRY.

Input: The event took place in Casablanca.

| Model | Response |
| --- | --- |
| Flan-T5 | Morocco |
| TinyChat | Morocco |

#### Summarization

In the following example, the models are prompted to summarize the Wikipedia entry for the video game Witcher 3. Flan-T5 is not able to provide an accurate summary while TinyChat is able to provide one.

| Model | Response |
| --- | --- |
| Flan-T5 | The Witcher 3: Wild Hunt |
| TinyChat | The Witcher 3: Wild Hunt is an action role-playing game developed and published by CD Projekt Red. The game is the third in the series and takes place in a magical world, where Geralt is the first witcher character. Players battle monsters, build powerful spells and use a series of spells to protect himself and his adopted daughter from the Wild Hunt. The game has set the bar for an open world game series. |

#### Creative Text Generation

Both models were asked to write a poem. In comparison to Flan-T5, TinyChat showed longer prose and creativity in its output.

Prompt: Write a poem

| Model | Response |
| --- | --- |
| Flan-T5 | i love chocolate so i got 2 bags of them |
| TinyChat | This poem is so beautiful - the beauty of the nature, the richness of the trees, the silence of the birds - everything is so peaceful, the way nature flows and the time always reaches such a happy destination. I think it's beautiful in its simplicity and beauty, the beauty of life, the way nature flows. As we look out across the bright fields and trees, it's so simple and yet such a powerful force that changes the whole environment. |

### Run Inference

```python
import transformers
from transformers import PeftModel

model_name = "google/flan-t5-large"
peft_model_id = "Leadmatic/tinyChat"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
base_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)
peft_model = PeftModel.from_pretrained(base_model, peft_model_id)

inputs = tokenizer("""[INSERT INSTRUCTION HERE]""", return_tensors="pt")
outputs = peft_model.generate(**inputs, max_length=300, do_sample=True)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
```

### HuggingFace Model Card

Trained model can be found at [HuggingFace Model Hub](https://huggingface.co/Leadmatic/tinyChat)


### License

This repo and model weights are available under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
