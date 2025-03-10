# train.py
import torch
from tqdm import tqdm
import time

def train(model, train_loader, optimizer, device, scheduler=None, gradient_accumulation_steps=4):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    # Progress bar
    pbar = tqdm(total=len(train_loader), desc="Training")
    
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / gradient_accumulation_steps  # Scale the loss
        total_loss += loss.item() * gradient_accumulation_steps  # Track original loss
        
        
        loss.backward()
        
        # Update weights only after accumulating gradients
        if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if scheduler:
                scheduler.step()
                
            optimizer.zero_grad()
        
        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"})
        pbar.update(1)
        
        # Show progress every 20 batches
        if (i + 1) % 20 == 0:
            current_loss = total_loss / (i + 1)
            elapsed_time = time.time() - start_time
            remaining = elapsed_time / (i + 1) * (len(train_loader) - i - 1)
            pbar.set_description(
                f"Training (loss: {current_loss:.4f}, elapsed: {elapsed_time:.2f}s, remaining: {remaining:.2f}s)"
            )
    
    pbar.close()
    
    # Calculate average loss
    avg_loss = total_loss / len(train_loader)
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    return avg_loss