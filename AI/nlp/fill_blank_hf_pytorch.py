"""
How to Create a Transformer Architecture Model for Natural Language Processing

# fill_blank_hf.py
# Anaconda 2020.02 (Python 3.7.6)
# PyTorch 1.8.0  HF 4.11.3  Windows 10
"""
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

np.set_printoptions(formatter={"float": "{: 0.4f}".format})


def main():
    print("Begin fill-in-the-blank using Transformer Architecture")

    print("\nLoading (cached) DistilBERT language model into memory...")
    model = AutoModelForMaskedLM.from_pretrained("distilbert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")

    sentence = "The man ran through the {BLANK} door."
    print(f"\nThe source fill-in-the-blank sentence is: {sentence}")

    sentence = f"The man ran through the {tokenizer.mask_token} door."
    print("Converting sentence to token IDs")
    inputs = tokenizer(sentence, return_tensors="pt")
    print(inputs)  # {'input_ids': tensor([[101, 1109, 1299, 1868, 1194, 1103, 103, 1442,  119,  102]]),
    # 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]) }
    print("    ID and its token:")
    for i in range(len(inputs["input_ids"][0])):
        id = inputs["input_ids"][0][i]
        print("%6d %s" % (id, tokenizer.decode(id)))

    print("\nPredicting all possible outputs")
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    with torch.no_grad():
        output = model(ids, mask)
    print(f"output: {output}")

    # Output the predictions
    blank_id = tokenizer.mask_token_id  # ID of {blank} = 103
    blank_id_idx = torch.where(inputs["input_ids"] == blank_id)[1]  # [6]
    all_logits = output.logits  # [1, 10, 28996]
    pred_logits = all_logits[0, blank_id_idx, :]  # [1, 28996]
    top_ids = torch.topk(pred_logits, 5, dim=1).indices[0].tolist()
    print("\nThe top 5 predictions (id - word):")
    for id in top_ids:
        print(f"id: {id} - word: {tokenizer.decode([id])}")

    print("\nConverting raw logit outputs to pseudo probabilities ")
    pred_probs = torch.softmax(pred_logits, dim=1).numpy()
    pred_probs = np.sort(pred_probs[0])[::-1]  # high p to low p
    top_probs = pred_probs[0:5]
    print(top_probs)  # [0.1689  0.0630  0.0447  0.0432  0.0323]
    print("End of demo ")
    return


if __name__ == "__main__":
    main()
