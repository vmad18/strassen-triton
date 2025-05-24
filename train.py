import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from model.model import GPT, replace_linear_with_strassen

lyrics = """
We're no strangers to love
You know the rules and so do I
A full commitment's what I'm thinking of
You wouldn't get this from any other guy

I just wanna tell you how I'm feeling
Gotta make you understand

Never gonna give you up
Never gonna let you down
Never gonna run around and desert you
Never gonna make you cry
Never gonna say goodbye
Never gonna tell a lie and hurt you

We've known each other for so long
Your heart's been aching but you're too shy to say it
Inside we both know what's been going on
We know the game and we're gonna play it

And if you ask me how I'm feeling
Don't tell me you're too blind to see

Never gonna give you up
Never gonna let you down
Never gonna run around and desert you
Never gonna make you cry
Never gonna say goodbye
Never gonna tell a lie and hurt you

Never gonna give you up
Never gonna let you down
Never gonna run around and desert you
Never gonna make you cry
Never gonna say goodbye
Never gonna tell a lie and hurt you

Ooh, give you up
Ooh, give you up
Ooh
Never gonna give, never gonna give, give you up
Ooh
Never gonna give, never gonna give, give you up

We've known each other for so long
Your heart's been aching but you're too shy to say it
Inside we both know what's been going on
We know the game and we're gonna play it

I just wanna tell you how I'm feeling
Gotta make you understand

Never gonna give you up
Never gonna let you down
Never gonna run around and desert you
Never gonna make you cry
Never gonna say goodbye
Never gonna tell a lie and hurt you
"""

original_lines = [line for line in lyrics.strip().split('\n') if line]
numbered_lines = [f"{i} {line}" for i, line in enumerate(original_lines)]

print("--- Sample Numbered Lines ---")
print(numbered_lines[0])
print(numbered_lines[1])
print(numbered_lines[6])
print("-" * 30)


raw_text = "\n".join(numbered_lines)
chars = sorted(list(set(raw_text)))
chars.append('<EOS>')
chars.append('<PAD>')

vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
EOS_TOKEN_ID = char_to_idx['<EOS>']
PAD_TOKEN_ID = char_to_idx['<PAD>']

encode = lambda s: [char_to_idx[c] for c in s]
decode = lambda l: ''.join([idx_to_char[i] for i in l if i != PAD_TOKEN_ID])

print(f"New Vocabulary size: {vocab_size}")
print(f"EOS token ID: {EOS_TOKEN_ID}")
print(f"PAD token ID: {PAD_TOKEN_ID}")


class LyricsLineDataset(Dataset):
    def __init__(self, lines, tokenizer, eos_token_id):
        self.lines = []
        for line in lines:
            tokenized_line = tokenizer(line) + [eos_token_id]
            self.lines.append(torch.tensor(tokenized_line, dtype=torch.long))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        return line[:-1], line[1:]

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=PAD_TOKEN_ID)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=PAD_TOKEN_ID)
    return inputs_padded, targets_padded

lyrics_dataset = LyricsLineDataset(numbered_lines, encode, EOS_TOKEN_ID)
dataloader = DataLoader(lyrics_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

DIM = 512
HEADS = 8
NUM_LAYERS = 4
DEVICE = "cuda"
EPOCHS = 500
LEARNING_RATE = 5e-4
BATCH_SIZE = 8

print(f"Using device: {DEVICE}")

model = GPT(
    dim=DIM,
    heads=HEADS,
    max_tokens=2048,
    num_layers=NUM_LAYERS,
    vocab_size=vocab_size,
    gate=False,
    PAD_TOKEN_ID = PAD_TOKEN_ID,
    device=DEVICE
)
model = torch.compile(model) 

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID) 

dataloader = DataLoader(lyrics_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    num_batches = 0
    for input_batch, target_batch in dataloader:
        input_batch = input_batch.to(DEVICE)
        target_batch = target_batch.to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_batch)
        B, T, C = logits.shape
        loss = criterion(logits.view(B * T, C), target_batch.view(B * T))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    if (epoch) % 25 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}, Batched: {num_batches}")




print("\n--- Generating Text (No Strassen) ---")
model.eval()

for i in range(100):
    start_chars = f"{i} " 
    input_ids = encode(start_chars)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    
    generated_ids = list(input_ids)
    max_gen_len = 100
    temperature = 1.0 
    
    with torch.no_grad():
        for _ in range(max_gen_len):
            logits = model(input_tensor) 
            last_logits = logits[:, -1, :] 
            probs = F.softmax(last_logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            if next_token_id == EOS_TOKEN_ID:
                break
                
            generated_ids.append(next_token_id)
            input_tensor = torch.tensor([generated_ids], dtype=torch.long, device=DEVICE)
    
    print("Generated: ", ''.join([idx_to_char[i] for i in generated_ids if i not in (PAD_TOKEN_ID, EOS_TOKEN_ID)]))
    print("-" * 30)


print()
print()
print()

        

def profile_generation(model_to_profile, num_generations, max_len, temp, title):
    """Helper function to run and profile generation, including best/worst times."""
    print(f"\n--- {title} ---")
    
    # --- WARM-UP ---
    # (Warm-up loop remains the same)
    print("Running warm-up (a few generations)...")
    try:
        model_to_profile.eval()
    except AttributeError: # Catch if .eval() doesn't exist (e.g., on some compiled objects)
        print("  (Skipping model_to_profile.eval() in warm-up due to AttributeError)")
        pass 
    
    for i in range(min(5, num_generations)): 
        start_chars = f"{i} " 
        input_ids = encode(start_chars)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
        generated_ids = list(input_ids)
        with torch.no_grad():
            for _ in range(max_len):
                logits = model_to_profile(input_tensor) 
                last_logits = logits[:, -1, :] 
                probs = F.softmax(last_logits / temp, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
                if next_token_id == EOS_TOKEN_ID: break
                generated_ids.append(next_token_id)
                input_tensor = torch.tensor([generated_ids], dtype=torch.long, device=DEVICE)

    # --- PROFILING ---
    print(f"Running profiling ({num_generations} generations)...")
    total_tokens_generated = 0
    individual_generation_times_ms = [] # Store individual times here
    
    try:
        model_to_profile.eval()
    except AttributeError:
        print("  (Skipping model_to_profile.eval() in profiling due to AttributeError)")
        pass
        
    # Create CUDA events once if using CUDA for per-iteration timing
    if DEVICE == "cuda":
        iter_start_event = torch.cuda.Event(enable_timing=True)
        iter_end_event = torch.cuda.Event(enable_timing=True)

    # --- The Actual Loop to Time (per iteration) ---
    for i in range(num_generations):
        if DEVICE == "cuda":
            torch.cuda.synchronize() # Ensure previous CUDA work is done
            iter_start_event.record()
        else:
            iter_start_time = time.perf_counter()

        # --- Single Generation ---
        start_chars = f"{i} " 
        input_ids = encode(start_chars)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
        generated_ids = list(input_ids)
        with torch.no_grad():
            for _ in range(max_len):
                logits = model_to_profile(input_tensor) 
                last_logits = logits[:, -1, :] 
                probs = F.softmax(last_logits / temp, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
                if next_token_id == EOS_TOKEN_ID: break
                generated_ids.append(next_token_id)
                input_tensor = torch.tensor([generated_ids], dtype=torch.long, device=DEVICE)
        
        tokens_this_generation = len(generated_ids) - len(input_ids)
        total_tokens_generated += tokens_this_generation
        
        # --- End Iteration Timing ---
        if DEVICE == "cuda":
            iter_end_event.record()
            torch.cuda.synchronize() # Wait for this iteration's GPU work
            iter_elapsed_time_ms = iter_start_event.elapsed_time(iter_end_event)
        else:
            iter_end_time = time.perf_counter()
            iter_elapsed_time_ms = (iter_end_time - iter_start_time) * 1000.0
        
        individual_generation_times_ms.append(iter_elapsed_time_ms)

    # --- Report Results ---
    if not individual_generation_times_ms:
        print("No generations were timed. Skipping results.")
        return 0.0

    total_elapsed_time_ms = sum(individual_generation_times_ms)
    total_elapsed_time_s = total_elapsed_time_ms / 1000.0
    
    avg_time_ms = total_elapsed_time_ms / num_generations
    best_time_ms = min(individual_generation_times_ms)
    worst_time_ms = max(individual_generation_times_ms)

    print(f"Total time for {num_generations} generations: {total_elapsed_time_s:.4f} seconds")
    print(f"Average time per generation: {avg_time_ms:.4f} ms")
    print(f"Best time per generation:  {best_time_ms:.4f} ms")
    print(f"Worst time per generation: {worst_time_ms:.4f} ms")
    
    if total_tokens_generated > 0 and total_elapsed_time_s > 0:
      print(f"Total tokens generated: {total_tokens_generated}")
      print(f"Average tokens per second: {total_tokens_generated / total_elapsed_time_s:.2f}")
    elif total_tokens_generated > 0 :
      print(f"Total tokens generated: {total_tokens_generated} (total time was near zero)")


    print("-" * 30)
    return total_elapsed_time_s

# --- Main Profiling Script ---

# Ensure model is in eval mode initially and NOT compiled
model.eval() 

# 1. Profile Standard MatMul
time_standard = profile_generation(model, 
                                 num_generations=100, 
                                 max_len=100, 
                                 temp=1.0, 
                                 title="Profiling Text (Matmult)")

print("\n\n--- Preparing Strassen Model ---\n")
# 2. Replace layers and COMPILE the model
replace_linear_with_strassen(model) # Assumes this modifies model in-place
print("Compiling model with torch.compile (this might take a moment)...")
model.eval()
model = torch.compile(model)
print("Compilation finished.")


# 3. Profile Strassen MatMul (after warm-up for compile)
time_strassen = profile_generation(model, 
                                 num_generations=100, 
                                 max_len=100, 
                                 temp=1.0, 
                                 title="Profiling Text (Strassen + Padding)")

# 4. Final Comparison
print("\n--- Final Results ---")
print(f"Standard MatMul Time: {time_standard:.4f} seconds")
print(f"Strassen + Padding Time: {time_strassen:.4f} seconds")





print()
print()
print()


print("\n--- Generating Text (Strassen) ---")
for i in range(100):
    start_chars = f"{i} " 
    input_ids = encode(start_chars)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    
    generated_ids = list(input_ids)
    max_gen_len = 100
    temperature = 1.0 
    
    with torch.no_grad():
        for _ in range(max_gen_len):
            logits = model(input_tensor) 
            last_logits = logits[:, -1, :] 
            probs = F.softmax(last_logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            if next_token_id == EOS_TOKEN_ID:
                break
                
            generated_ids.append(next_token_id)
            input_tensor = torch.tensor([generated_ids], dtype=torch.long, device=DEVICE)
    
    print("Generated: ", ''.join([idx_to_char[i] for i in generated_ids if i not in (PAD_TOKEN_ID, EOS_TOKEN_ID)]))
    print("-" * 30)

