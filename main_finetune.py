'''Fine-tuning the pre-trained model on the RACE dataset'''

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from GPT_Decoder import *
import os
from tqdm import tqdm

# Define RACE dataset class
class RACEDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.bos_id = tokenizer.token_to_id('[BOS]')
        self.eos_id = tokenizer.token_to_id('[EOS]')
        self.sep_id = tokenizer.token_to_id('[SEP]')
        
        # Preprocess all data upfront
        self.processed_data = []
        for item in data:
            # Format: [BOS] Article [SEP] Question [SEP] Options [SEP] Answer [EOS]
            article = item['article']
            question = item['question']
            options = ' '.join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(item['options'])])
            # Convert letter answer (A,B,C,D) to index (0,1,2,3)
            answer_idx = ord(item['answer']) - ord('A')
            answer = f"({item['answer']}) {item['options'][answer_idx]}"
            
            # Combine all parts
            text = f"{article} [SEP] {question} [SEP] {options} [SEP] {answer}"
            encoding = tokenizer.encode(text).ids
            seq = [self.bos_id] + encoding + [self.eos_id]
            seq = seq[:max_len]  # Truncate if too long
            
            self.processed_data.append(torch.LongTensor(seq))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq = self.processed_data[idx]
        
        # Create decoder input/output pairs
        decoder_input = seq[:-1]
        decoder_target = seq[1:]
        
        return {
            'decoder_input': decoder_input,
            'decoder_target': decoder_target,
        }

def collate_fn(batch):
    pad_token = 1  # [PAD] token
    batched = {
        'decoder_input': [],
        'decoder_target': [],
        'decoder_padding_mask': []
    }
    
    for item in batch:
        # Pad decoder input and target
        dec_pad = max_len - len(item['decoder_input'])
        padded_dec_input = F.pad(item['decoder_input'], (0, dec_pad), value=pad_token)
        padded_dec_target = F.pad(item['decoder_target'], (0, dec_pad), value=pad_token)
        
        # Create boolean masks for padding (False=real token, True=padding)
        if dec_pad <= 0:
            dec_mask = torch.zeros_like(item['decoder_input'], dtype=torch.bool)
        else:
            dec_mask = torch.cat([
                torch.zeros_like(item['decoder_input'], dtype=torch.bool),
                torch.ones(dec_pad, dtype=torch.bool)
            ])
        
        batched['decoder_input'].append(padded_dec_input)
        batched['decoder_target'].append(padded_dec_target)
        batched['decoder_padding_mask'].append(dec_mask)
    
    return {k: torch.stack(v) for k, v in batched.items()}

# Load RACE dataset
print("Loading RACE dataset...")
race_data = load_dataset('race', 'all')

# Load tokenizer
print("Loading tokenizer...")
tokenizer = Tokenizer.from_file('bpe.json')

# Create datasets
print("Creating datasets...")
train_dataset = RACEDataset(race_data['train'], tokenizer)
val_dataset = RACEDataset(race_data['validation'], tokenizer)
test_dataset = RACEDataset(race_data['test'], tokenizer)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

# Load pre-trained model
print("Loading pre-trained model...")
gpt_1 = GPT_1(vocab_size, d_model, nhead, num_decoder_layers, dropout, max_len).to(device)
gpt_1.load_state_dict(torch.load('model/seq_128_d_512_n_8/gpt_1_10.pt'))

# Define optimizer and loss function
opt = torch.optim.Adam(gpt_1.parameters(),
                      betas=(0.9, 0.98),
                      eps=1e-9,
                      lr=1e-5  # Lower learning rate for fine-tuning
                      )
loss_fn = nn.CrossEntropyLoss()

# Training parameters
num_epochs = 5
save_step = 1
train_batch_end = len(train_loader)  # Use all batches
val_batch_end = len(val_loader)

# Training loop
# print("Starting fine-tuning...")
# for epoch in range(num_epochs):
#     tr_loss = 0.
#     val_loss = 0.
    
#     # Training
#     gpt_1.train()
#     progress_bar = tqdm(enumerate(train_loader), total=train_batch_end, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
#     for b_num, batch in progress_bar:
#         if b_num == train_batch_end:
#             break
            
#         opt.zero_grad()
#         decoder_input = batch['decoder_input'].to(device)
#         decoder_target = batch['decoder_target'].to(device)
#         decoder_mask = batch['decoder_padding_mask'].to(device)
        
#         output = gpt_1(decoder_input, decoder_mask)
#         loss = loss_fn(output.view(-1, vocab_size), decoder_target.view(-1))
#         loss.backward()
#         opt.step()
        
#         tr_loss += loss.item()
#         progress_bar.set_postfix({'loss': loss.item()})
    
#     # Validation
#     gpt_1.eval()
#     with torch.no_grad():
#         progress_bar = tqdm(enumerate(val_loader), total=val_batch_end, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
#         for b_num, batch in progress_bar:
#             if b_num == val_batch_end:
#                 break
                
#             decoder_input = batch['decoder_input'].to(device)
#             decoder_target = batch['decoder_target'].to(device)
#             decoder_mask = batch['decoder_padding_mask'].to(device)
            
#             output = gpt_1(decoder_input, decoder_mask)
#             loss = loss_fn(output.view(-1, vocab_size), decoder_target.view(-1))
#             val_loss += loss.item()
#             progress_bar.set_postfix({'loss': loss.item()})
    
#     print(f"Epoch: {epoch+1}, Train Loss: {tr_loss/train_batch_end:.4f}, Val Loss: {val_loss/val_batch_end:.4f}")
    
#     # Save checkpoint
#     if (epoch + 1) % save_step == 0:
#         save_dir = f'model/race_finetuned/'
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         torch.save(gpt_1.state_dict(), f"{save_dir}gpt_1_epoch_{epoch+1}.pt")

# print("Fine-tuning completed!")

# Load the best model if already trained
gpt_1.load_state_dict(torch.load('model/race_finetuned/gpt_1_epoch_5.pt'))

# Test evaluation
print("Evaluating on test set...")
test_loss = 0.
gpt_1.eval()
with torch.no_grad():
    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing")
    for b_num, batch in progress_bar:
        decoder_input = batch['decoder_input'].to(device)
        decoder_target = batch['decoder_target'].to(device)
        decoder_mask = batch['decoder_padding_mask'].to(device)
        
        output = gpt_1(decoder_input, decoder_mask)
        loss = loss_fn(output.view(-1, vocab_size), decoder_target.view(-1))
        test_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

print(f"Test Loss: {test_loss/len(test_loader):.4f}")

def test_question_answering(model, tokenizer, example, max_tokens=50):
    """Test the model on a single RACE example"""
    model.eval()
    
    # Format input like during training
    article = example['article']
    question = example['question']
    options = ' '.join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(example['options'])])
    
    # First tokenize question and options to know their length
    question_options = f" [SEP] {question} [SEP] {options} [SEP]"
    question_options_tokens = tokenizer.encode(question_options).ids
    
    # Calculate how much space we have for the article
    # Subtract 2 for [BOS] and potential [EOS], length of question_options, and space for generation
    max_article_tokens = max_len - 2 - len(question_options_tokens) - max_tokens
    
    # Ensure we have at least some space for the article
    if max_article_tokens < 0:
        # If no space for article, we need to truncate question_options
        max_article_tokens = 0
        question_options_tokens = question_options_tokens[:(max_len - max_tokens - 2)]
    
    # Tokenize article and truncate if needed
    article_tokens = tokenizer.encode(article).ids
    if len(article_tokens) > max_article_tokens:
        # Take the last max_article_tokens tokens as they might be more relevant
        article_tokens = article_tokens[-max_article_tokens:]
    
    # Create input text without answer
    bos_id = tokenizer.token_to_id('[BOS]')
    input_ids = [bos_id] + article_tokens + question_options_tokens
    input_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(device)  # Add batch dimension
    
    # Generate answer
    with torch.no_grad():
        generated = []
        current_input = input_tensor
        
        for _ in range(max_tokens):
            # If sequence gets too long, slide the window
            if current_input.size(1) >= max_len:
                current_input = current_input[:, -max_len+1:]
            
            # Create attention mask
            attn_mask = torch.zeros((1, current_input.size(1)), dtype=torch.bool).to(device)
            
            # Get model prediction
            output = model(current_input, attn_mask)
            next_token_logits = output[0, -1, :]
            next_token = torch.argmax(next_token_logits).unsqueeze(0)
            
            generated.append(next_token.item())
            current_input = torch.cat([current_input, next_token.unsqueeze(0)], dim=1)
            
            # Stop if we generate EOS token
            if next_token.item() == tokenizer.token_to_id('[EOS]'):
                break
    
    # Decode generated answer
    generated_text = tokenizer.decode(generated)
    correct_answer = f"({example['answer']}) {example['options'][ord(example['answer']) - ord('A')]}"
    
    # For display, show if article was truncated
    article_display = "..." + article[-200:] if len(article_tokens) > max_article_tokens else article[:200]
    print(f"\nArticle: {article_display}...")
    print(f"\nQuestion: {question}")
    print(f"\nOptions: {options}")
    print(f"\nModel's Answer: {generated_text}")
    print(f"Correct Answer: {correct_answer}")
    
    return generated_text, correct_answer

# Test on some examples
print("\nTesting model on example questions...")
test_examples = race_data['test'].select(range(5))  # Test on first 5 examples

for i, example in enumerate(test_examples):
    print(f"\n=== Example {i+1} ===")
    generated, correct = test_question_answering(gpt_1, tokenizer, example)

