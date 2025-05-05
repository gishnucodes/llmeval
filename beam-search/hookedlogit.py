import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc # Garbage collector

# --- Hook Function and Setup ---
raw_logits_output = []

def capture_logits_hook(module, input_args, output):
    """
    Hook function to capture the output of the lm_head layer.
    The output might be a tensor or a tuple containing the tensor.
    We are interested in the tensor containing logits.
    """
    if isinstance(output, torch.Tensor):
        logits = output
    elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
        # Common case for models returning more than just logits (e.g., past_key_values)
        # We assume the first element is the logits tensor. Check model docs if unsure.
        logits = output[0]
    else:
        # Cannot determine logits tensor, skip capture for this call
        print(f"Warning: Hook captured unexpected output type: {type(output)}")
        return

    # We only want the logits for the *last* token prediction in the sequence
    # Shape is usually (batch_beam_size, sequence_length, vocab_size)
    last_token_logits = logits[:, -1, :].clone().detach().cpu()
    raw_logits_output.append(last_token_logits)

# --- Example Usage ---

# 2. Load Model and Tokenizer (same as before)
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# 3. Prepare Input (same as before)
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# 4. Register the Forward Hook
# For GPT2, the final linear layer is model.lm_head
hook_handle = model.lm_head.register_forward_hook(capture_logits_hook)
print("Registered forward hook on lm_head.")

# Reset the storage list before generation
raw_logits_output.clear()

# 5. Perform Generation (No LogitsProcessor needed for extraction here)
max_new_tokens = 5
num_beams = 4
k_top = 10

print(f"\nGenerating text (max_new_tokens={max_new_tokens}, num_beams={num_beams})...")
output_sequences = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=max_new_tokens,
    num_beams=num_beams,
    early_stopping=True,
    # No logits_processor needed if only using the hook for extraction
)

# 6. Remove the Hook
hook_handle.remove()
print("Removed forward hook.")

# --- Analyze and Display Top K Raw Logits from Hook ---
print("\n--- Top K Raw Logits Analysis (from Forward Hook) ---")
print(f"Analyzing Top {k_top} raw logits captured by the hook.")

# Note: The hook captures logits *during* each forward pass.
# The number of captured tensors might slightly differ from the number of *new tokens*
# generated, especially concerning the initial prompt processing.
# We typically expect len(raw_logits_output) >= max_new_tokens.
# We'll analyze the captures relevant to the generated tokens (usually starting from the second capture).

# The first capture is often from processing the initial prompt.
# Subsequent captures correspond to generation steps.
logits_for_generation_steps = raw_logits_output[1:] if len(raw_logits_output) > 0 else []

if not logits_for_generation_steps:
     print("No logits captured during generation steps via hook.")
else:
    batch_beam_size = logits_for_generation_steps[0].shape[0]
    is_beam_search = num_beams > 1 and batch_beam_size > 1
    print(f"Raw logits tensor shape (hooked): {logits_for_generation_steps[0].shape}") # Should be (batch*beams, vocab_size)
    print(f"Batch x Beam size: {batch_beam_size}")
    print("-" * 20)

    # Analyze steps corresponding to new tokens generated
    num_steps_to_analyze = min(len(logits_for_generation_steps), max_new_tokens)

    for step in range(num_steps_to_analyze):
        step_logits = logits_for_generation_steps[step]
        print(f"\n--- Step {step + 1} (Hooked Raw Logits) ---")
        top_k_logits, top_k_indices = torch.topk(step_logits, k=k_top, dim=-1)

        for i in range(batch_beam_size):
            item_label = f"Beam {i}" if is_beam_search else f"Batch Item {i}"
            print(f"  {item_label}:")

            item_top_k_logits = top_k_logits[i]
            item_top_k_indices = top_k_indices[i]
            item_top_k_tokens = tokenizer.convert_ids_to_tokens(item_top_k_indices)

            for rank in range(k_top):
                token_id = item_top_k_indices[rank].item()
                logit_value = item_top_k_logits[rank].item() # These are the raw logits
                token_str = item_top_k_tokens[rank]
                print(f"    Rank {rank+1:>2}: Logit={logit_value: <10.4f} | Token ID={token_id: <6} | Token='{token_str}'")

# --- Final Output ---
print("\n" + "="*30)
print(f"Generated sequence IDs: {output_sequences}")
print(f"Decoded text: {tokenizer.batch_decode(output_sequences, skip_special_tokens=True)}")
print("="*30)

# Clean up memory
del model
del tokenizer
del inputs
del raw_logits_output # If using Method 2
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()