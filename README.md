# GPT-1-From-Scratch
Implementation of 56M GPT-1 from scratch. Since BookCorpus dataset (~1B tokens) is no longer publicly available, I instead use WikiText-103 (103M tokens) dataset to pre-train GPT-1

## GPT-1 parameter values
Model dimension <code>**(d_model)**</code> = 512

Number of Attention Heads <code>**(n_heads)**</code> = 8

Number of Decoders <code>**(num_decoder_layers)**</code> = 8

Maximum sequence length <code>**(max_len)**</code> = 128

Feedforward layer hidden size <code>**(dim_feedforward)**</code> = 2048

Vocabulary size <code>**(vocab_size)**</code> = 30000 for WikiText-103 dataset

Batch size <code>**(batch_size)**</code> = 64
************************************
**TOTAL PARAMETER COUNT** ≈ 56M

## Steps to run
1. Run <code>input_processing.py</code> to generate tokenized Wikitext data and save it in .pt torch tensor format

2. Run <code>main_pretrain.py</code> to pre-train GPT-1. The user can change training settings from this file and model parameters from <code>GPT_Decoder.py</code>

3. Run last cell in <code>test.ipynb</code> to generate random text from pre-trained GPT-1

**Sample generation output:** <em>The earliest known mention of this date was that of 544 , when King Olaf II of Norway was discovered in the reign of King Olaf II of Norway . The earliest recorded mention of this date was from 544 , when King Olaf was assassinated . The date of the birth is unknown , but it is unclear whether Olaf was killed . Olaf 's birth date is unknown , but it is likely that Olaf was killed by the Vikings in 842 , but Olaf 's reign is uncertain . .</em>

This can be improved by increasing model size, but is good enough for 56M parameter model.



