'''Pre-Train GPT-1 model (with 56M parameters) on WikiText-103M dataset.'''

from torch.utils.data import DataLoader
import os
from tokenizers import Tokenizer, pre_tokenizers
from GPT_Decoder import *
from custom_dataloader import WikiTextDataset, collate_fn
from torch.optim.lr_scheduler import _LRScheduler

class GPT1LRScheduler(_LRScheduler):
    def __init__(self, optimizer, max_lr=2.5e-4, warmup_steps=2000, total_steps=1000000, last_epoch=-1):
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1  # Convert to 1-based indexing
        
        # Linear warmup phase
        if step < self.warmup_steps:
            return [self.max_lr * (step / self.warmup_steps) for _ in self.base_lrs]
        
        # Cosine annealing phase
        # Calculate where we are in the cosine cycle
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        # Ensure progress is capped at 1.0
        progress = min(progress, 1.0)  # Clamp progress at 100%
        # Cosine decay from max_lr to 0
        return [self.max_lr * 0.5 * (1 + math.cos(math.pi * progress)) for _ in self.base_lrs]

    def _get_closed_form_lr(self):
        return self.get_lr()

# Define key parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_step = 1
start_epoch = 0
end_epoch = 10
train_batch_end = 15000
test_batch_end = 10

# Load train and test datasets
train_dataset = torch.load('train_dataset.pt')
test_dataset = torch.load('test_dataset.pt')
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
print('Train batches:', len(train_loader), 'Test batches:', len(test_loader))

# Usage example
tokenizer = Tokenizer.from_file('bpe.json')

cnt = 0
for batch in train_loader:
    if cnt == 0: # Random example batch to print
        print(f"Batch input shape: {batch['decoder_input'].shape}")
        print(f"Sample decoder input text: {tokenizer.decode(batch['decoder_input'][0].tolist())}")
        print(f"Sample decoder target text: {tokenizer.decode(batch['decoder_target'][0].tolist())}")
        break
    cnt += 1

# Define model instance, optimizer and loss function
gpt_1 = GPT_1(vocab_size, d_model, nhead, num_decoder_layers, dropout, max_len).to(device)
opt = torch.optim.Adam(gpt_1.parameters(),
                          betas=(0.9, 0.98),
                          eps=1e-9,
                          lr=0.0
                          )
scheduler = GPT1LRScheduler(opt, max_lr=2.5e-4, warmup_steps=2000, total_steps=train_batch_end*end_epoch)
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(start_epoch, end_epoch):
    tr_loss = 0.
    te_loss = 0.
    gpt_1.train()
    for b_num, batch in enumerate(train_loader):
        if b_num == train_batch_end:
            break
        opt.zero_grad()
        decoder_input = batch['decoder_input'].to(device)
        decoder_target = batch['decoder_target'].to(device)
        decoder_mask = batch['decoder_padding_mask'].to(device)
        output = gpt_1(decoder_input, decoder_mask)
        loss = loss_fn(output.view(-1, vocab_size), decoder_target.view(-1))
        loss.backward()
        opt.step()
        scheduler.step()

        tr_loss += loss.item()
        print(f"Train Batch {b_num+1}/{train_batch_end} Loss: {loss.item()}", end='\r')
    
    gpt_1.eval()
    for b_num, batch in enumerate(test_loader):
        if b_num == test_batch_end:
            break
        decoder_input = batch['decoder_input'].to(device)
        decoder_target = batch['decoder_target'].to(device)
        decoder_mask = batch['decoder_padding_mask'].to(device)
        output = gpt_1(decoder_input, decoder_mask)
        loss = loss_fn(output.view(-1, vocab_size), decoder_target.view(-1))
        te_loss += loss.item()
        print(f"Test Batch {b_num+1}/{test_batch_end} Loss: {loss.item()}", end='\r')

    if (epoch+1) % save_step == 0: # Perform checkpointing
        if not os.path.exists(f'model/seq_{max_len}_d_{d_model}_n_{num_decoder_layers}/'):
            os.makedirs(f'model/seq_{max_len}_d_{d_model}_n_{num_decoder_layers}/')
        torch.save(gpt_1.state_dict(), f"model/seq_{max_len}_d_{d_model}_n_{num_decoder_layers}/gpt_1_{epoch+1}.pt")

    print(f"Epoch: {epoch+1}, Train Loss: {tr_loss/train_batch_end}, Test Loss: {te_loss/test_batch_end}")

