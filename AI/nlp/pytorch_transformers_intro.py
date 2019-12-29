"""
Tutorial: https://pytorch.org/hub/huggingface_pytorch-transformers/
-
- 

# Usage
export PYTHONPATH=$(pwd)

"""

from pypchutils.generic import create_logger
import torch

logger = create_logger(__name__)

##################
# EXAMPLE
##################
text_1 = "Who was Jim Henson ?"
text_2 = "Jim Henson was a puppeteer"

# FIRST, TOKENIZE THE INPUT
# for BERT: [CLS] at the beginning and [SEP] at the end
tokenizer = torch.hub.load("huggingface/pytorch-transformers", "tokenizer", "bert-base-cased")
indexed_tokens = tokenizer.encode(text_1, text_2, add_special_tokens=True)
logger.info("Succcessfully tokenized the input sentence")

# USING BERTMODEL TO ENCODE THE INPUT SENTENCE IN A SEQUENCE OF LAST LAYER HIDDEN-STATES
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
segments_tensor = torch.tensor([segments_ids])
tokens_tensor = torch.tensor([indexed_tokens])

model = torch.hub.load("huggingface/pytorch-transformers", "model", "bert-base-cased")

logger.info("Encoding the input sentence...")
with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor, token_type_ids=segments_tensor)
logger.info("Succcessfully encoded the input sentence")

# USING MODELWITHLMHEAD TO PREDICT A MASKED TOKEN WITH BERT
# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = 8
indexed_tokens[masked_index] = tokenizer.mask_token_id
tokens_tensor = torch.tensor([indexed_tokens])

masked_lm_model = torch.hub.load("huggingface/pytorch-transformers", "modelWithLMHead", "bert-base-cased")
with torch.no_grad():
    predictions = masked_lm_model(tokens_tensor, token_type_ids=segments_tensor)

# Get the predicted token
predicted_index = torch.argmax(predictions[0][0], dim=1)[masked_index].item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
logger.info(f"Predicted_token is {predicted_token}. True token is:")


########################################
# USING MODELFORSEQUENCECLASSIFICATION TO DO PARAPHRASE CLASSIFICATION WITH BERT
########################################
sequence_classification_model = torch.hub.load(
    "huggingface/pytorch-transformers", "modelForSequenceClassification", "bert-base-cased-finetuned-mrpc"
)
sequence_classification_tokenizer = torch.hub.load(
    "huggingface/pytorch-transformers", "tokenizer", "bert-base-cased-finetuned-mrpc"
)

text_1 = "Jim Henson was a puppeteer"
text_2 = "Who was Jim Henson ?"
indexed_tokens = sequence_classification_tokenizer.encode(text_1, text_2, add_special_tokens=True)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
segments_tensor = torch.tensor([segments_ids])
tokens_tensor = torch.tensor([indexed_tokens])

# Predict the sequence classification logits
with torch.no_grad():
    seq_classif_logits = sequence_classification_model(tokens_tensor, token_type_ids=segments_tensor)

predicted_labels = torch.argmax(seq_classif_logits[0]).item()
assert predicted_labels == 0  # In MRPC dataset this means the two sentences are not paraphrasing each other

# Or get the sequence classification loss (set model to train mode before if used for training)
labels = torch.tensor([1])
seq_classif_loss = sequence_classification_model(tokens_tensor, token_type_ids=segments_tensor, labels=labels)
