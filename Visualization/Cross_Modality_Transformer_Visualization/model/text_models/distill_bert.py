import torch.nn as nn
from transformers import AutoModel
import transformers


class DistillBert(nn.module):
    def __init__(self)
        super().__init__()
        self.text_model = AutoModel.from_pretrained(text_params['model'])
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(config['arch']['args']['text_params']['model'],
                                                               TOKENIZERS_PARALLELISM=False)

    def forward(self, x):
        if self.tokenizer is not None:
            x = self.tokenizer(x, return_tensors='pt', padding=True,
                                          truncation=True)
        x = {key: val.to(self.device) for key, val in x.items()}
        text_embeddings_all = self.text_model(**x).last_hidden_state
        # print(text_embeddings_all.size()) # batch_size, sequence_length, hidden_size
        text_embeddings = text_embeddings_all[:, 0, :]
        return text_embeddings