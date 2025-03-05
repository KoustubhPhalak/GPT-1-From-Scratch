import torch
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from custom_dataloader import WikiTextDataset

# Initialize BPE tokenizer
tokenizer = Tokenizer(models.BPE(
    unk_token='[UNK]',
    continuing_subword_prefix='##'        
    ))

# Use byte-level pre-tokenization instead of whitespace
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=False,
    use_regex=False
)

# Configure decoder to handle merging
tokenizer.decoder = decoders.ByteLevel()  # Removes space artifacts

# Configure trainer
trainer = trainers.BpeTrainer(
    vocab_size=30000,
    min_frequency=2,          # Ignore tokens appearing <2 times
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]

)

data = load_dataset('wikitext', 'wikitext-103-raw-v1')

# Add text preprocessing
def preprocess_text(examples):
    return {"text": [t.strip() for t in examples["text"] if t.strip()]}

processed_data = data.map(preprocess_text, batched=True)

# Train and save tokenizer. Comment if already trained
tokenizer.train_from_iterator(processed_data['train']['text'], trainer=trainer)
tokenizer.save('bpe.json')

# Load tokenizer
tokenizer = Tokenizer.from_file('bpe.json')

# Create dataset
train_dataset = WikiTextDataset(processed_data['train'], tokenizer) 
test_dataset = WikiTextDataset(processed_data['test'], tokenizer)   

# Save datasets
torch.save(train_dataset, 'train_dataset.pt')
torch.save(test_dataset, 'test_dataset.pt')
