import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# --- Configuration ---
MODEL_NAME = "t5-small"  # Or another seq2seq model like "google/bart-large-cnn", "Helsinki-NLP/opus-mt-en-fr"
INPUT_TEXT = "translate English to French: Hello, how are you?"
# INPUT_TEXT = "summarize: Studies have shown that owning a dog is good for you. Dogs encourage people to get outside and exercise, and they provide companionship, which can reduce stress and anxiety."

NUM_BEAMS = 4
NUM_RETURN_SEQUENCES = 4 # Should be <= NUM_BEAMS
MAX_NEW_TOKENS = 50

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
model.eval() # Set model to evaluation mode

# --- Prepare Input ---
# Some models require specific prefixes for tasks (like T5)
input_ids = tokenizer(INPUT_TEXT, return_tensors="pt").input_ids.to(device)

# --- 1. Generate Sequences using Beam Search ---
print(f"\nGenerating {NUM_RETURN_SEQUENCES} sequences using beam search (num_beams={NUM_BEAMS})...")

# Ensure we generate enough sequences to return
if NUM_RETURN_SEQUENCES > NUM_BEAMS:
    print(f"Warning: num_return_sequences ({NUM_RETURN_SEQUENCES}) > num_beams ({NUM_BEAMS}). Setting num_return_sequences=num_beams.")
    NUM_RETURN_SEQUENCES = NUM_BEAMS

with torch.no_grad():
    generated_outputs = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=NUM_BEAMS,
        num_return_sequences=NUM_RETURN_SEQUENCES,
        early_stopping=True, # Stop when num_beams sequences are finished
        # Important: We don't need output_scores here, as we re-score manually
        # output_scores=True, # This would return logits at each step *during* generation
        # return_dict_in_generate=True # Useful if using output_scores
    )

# generated_outputs are the token IDs for the generated sequences
# Shape: (num_return_sequences, sequence_length)

# Decode generated sequences
decoded_sequences = tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)

print("\nGenerated Sequences:")
for i, seq in enumerate(decoded_sequences):
    print(f"{i+1}: {seq}")

# --- 2. Re-score Generated Sequences to get Raw Logits ---
print("\nCalculating Raw Logits for each token in the generated sequences...")

all_sequence_logits = []

# Get the decoder start token ID (e.g., <pad> for T5, <s> for BART)
try:
    decoder_start_token_id = model.config.decoder_start_token_id
    if decoder_start_token_id is None:
      # Fallback for models like BART which might use bos_token_id
      decoder_start_token_id = model.config.bos_token_id
      if decoder_start_token_id is None:
        # Fallback for models like MBart, check tokenizer
        decoder_start_token_id = tokenizer.bos_token_id
        print(f"Using tokenizer.bos_token_id as decoder_start_token_id: {decoder_start_token_id}")

    if decoder_start_token_id is None:
      raise ValueError("Could not determine decoder_start_token_id.")
    print(f"Using decoder_start_token_id: {decoder_start_token_id}")

except AttributeError:
     # If the model config doesn't explicitly have it, try tokenizer's BOS token
    try:
      decoder_start_token_id = tokenizer.bos_token_id
      if decoder_start_token_id is None:
         raise ValueError("Could not determine decoder_start_token_id from model config or tokenizer bos_token_id.")
      print(f"Using tokenizer.bos_token_id as decoder_start_token_id: {decoder_start_token_id}")
    except AttributeError:
        raise ValueError("Cannot find decoder_start_token_id in model config or tokenizer.")


for i, sequence_ids in enumerate(generated_outputs):
    print(f"\n--- Sequence {i+1} ---")
    print(f"Decoded: {decoded_sequences[i]}")
    print(f"Token IDs: {sequence_ids.tolist()}")

    # Prepare decoder input: shift sequence right and add start token
    # sequence_ids shape: (sequence_length,)
    # Example: if sequence_ids is [pad_id, token1, token2, eos_id]
    # decoder_input_ids should be [pad_id, pad_id, token1, token2] (for T5 style)
    # Note: The generate output usually already includes the start token if needed by the model architecture internally,
    # but for a manual forward pass, we construct decoder_input_ids explicitly.
    # Let's verify the typical output format. T5 includes the start <pad> token.
    # sequence_ids = [dec_start_tok, token1, token2, ..., eos_tok]

    # Create decoder_input_ids: [dec_start_tok, token1, token2, ...]
    # We remove the last token (often EOS) to align with logits
    decoder_input_ids = sequence_ids[:-1].unsqueeze(0) # Add batch dimension -> shape (1, seq_len - 1)

     # Ensure decoder_input_ids starts correctly if generate didn't include it (less common for Seq2Seq)
    # If sequence_ids[0] is not the decoder_start_token_id, prepend it
    # (However, generate output for T5/BART usually starts correctly)
    # Example check (adapt if needed for your specific model):
    # if sequence_ids[0] != decoder_start_token_id:
    #     print(f"Warning: Sequence doesn't start with decoder_start_token_id ({decoder_start_token_id}). Found {sequence_ids[0]} instead.")
        # Force start token (use cautiously, might indicate misunderstanding of generate output)
        # decoder_input_ids = torch.cat([
        #     torch.tensor([[decoder_start_token_id]], device=device),
        #     sequence_ids[:-1].unsqueeze(0)
        # ], dim=1)

    # Labels for cross-entropy calculation (if needed, not required for logits)
    # labels = sequence_ids[1:].unsqueeze(0) # shape (1, seq_len - 1)

    # Perform forward pass to get logits
    with torch.no_grad():
        outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        # logits shape: (batch_size, sequence_length, vocab_size)
        # Here batch_size=1, sequence_length = len(decoder_input_ids[0])
        logits = outputs.logits # Shape: (1, seq_len - 1, vocab_size)

    # Extract the logits for the *actual* tokens generated
    # The logit for the token at position `t` (sequence_ids[t])
    # is found in the logits output at step `t-1` (logits[0, t-1, :])
    # We need sequence_ids starting from the *second* token (index 1)
    # because the first token is predicted based on decoder_start_token_id
    target_token_ids = sequence_ids[1:] # Shape: (seq_len - 1)

    # Use gather to efficiently select the logits corresponding to the target tokens
    # logit shape: (1, seq_len - 1, vocab_size)
    # target_token_ids shape: (seq_len - 1) -> need (1, seq_len - 1, 1) for gather
    target_token_ids_for_gather = target_token_ids.unsqueeze(0).unsqueeze(-1) # Shape: (1, seq_len - 1, 1)
    # Select the specific logit value for each chosen token
    token_logits = torch.gather(logits, 2, target_token_ids_for_gather).squeeze(-1).squeeze(0) # Shape: (seq_len - 1)

    # Store results for this sequence
    sequence_logit_data = []
    # Iterate from the first *predicted* token
    for t in range(target_token_ids.size(0)): # Loop over seq_len - 1 steps
        token_id = target_token_ids[t].item()
        token_logit = token_logits[t].item()
        token_text = tokenizer.decode(token_id)

        # Stop if we hit EOS or PAD (adjust based on model/tokenizer)
        if token_id == tokenizer.eos_token_id or token_id == tokenizer.pad_token_id:
             print(f"  Step {t}: Token='{token_text}' (ID: {token_id}), Logit: {token_logit:.4f} -> Stopping here (EOS/PAD)")
             # Optionally break if you don't want scores post-EOS
             # break # Uncomment to stop logging after EOS/PAD
        else:
             print(f"  Step {t}: Token='{token_text}' (ID: {token_id}), Logit: {token_logit:.4f}")

        sequence_logit_data.append({
            "step": t,
            "token_id": token_id,
            "token_text": token_text,
            "raw_logit": token_logit
        })

    all_sequence_logits.append({
        "decoded_sequence": decoded_sequences[i],
        "token_ids": sequence_ids.tolist(),
        "logit_scores": sequence_logit_data
    })

print("\n--- Summary ---")
for i, result in enumerate(all_sequence_logits):
    print(f"\nSequence {i+1}: {result['decoded_sequence']}")
    total_logit_sum = sum(item['raw_logit'] for item in result['logit_scores'] if item['token_id'] not in [tokenizer.eos_token_id, tokenizer.pad_token_id]) # Example: sum logits excluding eos/pad
    avg_logit = total_logit_sum / len([item for item in result['logit_scores'] if item['token_id'] not in [tokenizer.eos_token_id, tokenizer.pad_token_id]]) if len(result['logit_scores']) > 0 else 0
    print(f"  Logits per token:")
    for item in result['logit_scores']:
         print(f"    '{item['token_text']}' ({item['token_id']}): {item['raw_logit']:.4f}")
    print(f"  Sum of Raw Logits (excluding EOS/PAD): {total_logit_sum:.4f}")
    # print(f"  Average Raw Logit (excluding EOS/PAD): {avg_logit:.4f}") # Average might not be super meaningful