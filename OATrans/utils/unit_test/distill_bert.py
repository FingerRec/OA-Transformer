from transformers import AutoModel
import transformers

text = "tree"

tokenizer = transformers.AutoTokenizer.from_pretrained("pretrained/distilbert-base-uncased",
                                                       TOKENIZERS_PARALLELISM=False)

text_data = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
text_data = {key: val.cuda() for key, val in text_data.items()}


text_model = AutoModel.from_pretrained("pretrained/distilbert-base-uncased").cuda()

print(text_model)

text_embeddings_all = text_model(**text_data).last_hidden_state
print(text_embeddings_all.size())
text_embeddings = text_embeddings_all[:, 0, :]
print(text_embeddings)


text_embeddings_2 = text_model.embeddings(text_data['input_ids'])

text_embeddings_2 = text_model.transformer(text_embeddings_2,
                                           attn_mask=attention_mask,
                                           head_mask=head_mask,
                                           output_attentions=output_attentions,
                                           output_hidden_states=output_hidden_states,
                                           return_dict=return_dict,
                                           )

print(text_embeddings - text_embeddings_2)