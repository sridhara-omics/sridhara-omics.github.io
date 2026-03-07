---
title: "Understanding ESM model and architecture"
date: '2026-03-06'
layout: post
output:
  html_document:
    df_print: paged
  pdf_document: default
  word_document: default
categories: Genomics
---

Here, we take a ESM model with approx 8M parameters, and show how python modules can be used to understand a bit more about the architecture and related stuff of the model.  

Install libraries  
```python
!pip install -q transformers datasets accelerate
```

Imports  
```python
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import Dataset
```

Detect GPU  
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
```

    Using device: cpu


Load ESM2 model, picking a smaller 8M model  
```python
model_name = "facebook/esm2_t6_8M_UR50D"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

model = model.to(device)
```

Printing the model's architecture  
```python
#This prints the full transformer architecture.
print(model)
```

    EsmForMaskedLM(
      (esm): EsmModel(
        (embeddings): EsmEmbeddings(
          (word_embeddings): Embedding(33, 320, padding_idx=1)
          (dropout): Dropout(p=0.0, inplace=False)
        )
        (encoder): EsmEncoder(
          (layer): ModuleList(
            (0-5): 6 x EsmLayer(
              (attention): EsmAttention(
                (self): EsmSelfAttention(
                  (query): Linear(in_features=320, out_features=320, bias=True)
                  (key): Linear(in_features=320, out_features=320, bias=True)
                  (value): Linear(in_features=320, out_features=320, bias=True)
                  (rotary_embeddings): RotaryEmbedding()
                )
                (output): EsmSelfOutput(
                  (dense): Linear(in_features=320, out_features=320, bias=True)
                  (dropout): Dropout(p=0.0, inplace=False)
                )
                (LayerNorm): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
              )
              (intermediate): EsmIntermediate(
                (dense): Linear(in_features=320, out_features=1280, bias=True)
              )
              (output): EsmOutput(
                (dense): Linear(in_features=1280, out_features=320, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
              )
              (LayerNorm): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
            )
          )
          (emb_layer_norm_after): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
        )
        (contact_head): EsmContactPredictionHead(
          (regression): Linear(in_features=120, out_features=1, bias=True)
          (activation): Sigmoid()
        )
      )
      (lm_head): EsmLMHead(
        (dense): Linear(in_features=320, out_features=320, bias=True)
        (layer_norm): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
        (decoder): Linear(in_features=320, out_features=33, bias=False)
      )
    )


Embedding layer: Amino acids (33 tokens) are converted into 320-dimensional vectors.

Transformer encoder: The sequence passes through 6 transformer layers, each containing self-attention and feed-forward networks that learn relationships between residues.

LM head: The final layer predicts the probability of each amino acid at every position (masked language modeling).

Contact head: An auxiliary module predicts residue–residue contacts, useful for protein structure insights.

Printing the key model parameters  
```python
# This shows key architecture parameters
print(model.config)
```

    EsmConfig {
      "add_cross_attention": false,
      "architectures": [
        "EsmForMaskedLM"
      ],
      "attention_probs_dropout_prob": 0.0,
      "classifier_dropout": null,
      "dtype": "float32",
      "emb_layer_norm_before": false,
      "esmfold_config": null,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.0,
      "hidden_size": 320,
      "initializer_range": 0.02,
      "intermediate_size": 1280,
      "is_decoder": false,
      "is_folding_model": false,
      "layer_norm_eps": 1e-05,
      "mask_token_id": 32,
      "max_position_embeddings": 1026,
      "model_type": "esm",
      "num_attention_heads": 20,
      "num_hidden_layers": 6,
      "pad_token_id": 1,
      "position_embedding_type": "rotary",
      "tie_word_embeddings": true,
      "token_dropout": true,
      "transformers_version": "5.2.0",
      "use_cache": true,
      "vocab_list": null,
      "vocab_size": 33
    }
    


Model size / representation

hidden_size = 320
Each amino-acid token is represented as a 320-dimensional embedding vector.

Transformer depth

num_hidden_layers = 6
The model has 6 transformer blocks, meaning the sequence is processed through 6 attention layers.

Attention structure

num_attention_heads = 20
Each layer has 20 self-attention heads that learn different residue–residue relationships.

Feed-forward network size

intermediate_size = 1280
Inside each transformer layer, the feed-forward network expands 320 → 1280 → 320, giving the model nonlinear capacity.

Sequence length limit

max_position_embeddings = 1026
The model can process protein sequences up to ~1026 residues.

Protein vocabulary

vocab_size = 33
The tokenizer supports 33 tokens (20 amino acids + special tokens like mask, pad, start, etc.).


```python
# Count model parameters
num_params = sum(p.numel() for p in model.parameters())
print("Total parameters:", num_params)
```

    Total parameters: 7512474


Understanding transformer layers  
```python
# Inspect transformer layers
for name, module in model.named_modules():
    if "layer" in name:
        print(name)
```

Understanding in depth of one of the layers  
```python
print(model.esm.encoder.layer[0])
```

    EsmLayer(
      (attention): EsmAttention(
        (self): EsmSelfAttention(
          (query): Linear(in_features=320, out_features=320, bias=True)
          (key): Linear(in_features=320, out_features=320, bias=True)
          (value): Linear(in_features=320, out_features=320, bias=True)
          (rotary_embeddings): RotaryEmbedding()
        )
        (output): EsmSelfOutput(
          (dense): Linear(in_features=320, out_features=320, bias=True)
          (dropout): Dropout(p=0.0, inplace=False)
        )
        (LayerNorm): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
      )
      (intermediate): EsmIntermediate(
        (dense): Linear(in_features=320, out_features=1280, bias=True)
      )
      (output): EsmOutput(
        (dense): Linear(in_features=1280, out_features=320, bias=True)
        (dropout): Dropout(p=0.0, inplace=False)
      )
      (LayerNorm): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
    )


Running inference on a sample sequence  
```python
seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQLR"

inputs = tokenizer(seq, return_tensors="pt")
inputs = {k:v.to(device) for k,v in inputs.items()}

outputs = model(**inputs)

# (batch_size, sequence_length, vocab_size)
print(outputs.logits.shape)
```

    torch.Size([1, 37, 33])



```python

```
