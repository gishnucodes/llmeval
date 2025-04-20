# %%
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from collections import deque
import psutil
import os
# Assuming file_writer_factory.py contains FileWriterFactory
from file_writer_factory import FileWriterFactory

# %% Constants and Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEVICE = "mps" # Use MPS for Apple Silicon GPU
TORCH_DTYPE = torch.bfloat16
INPUT_FILE = 'feed.txt'
OUTPUT_FILE = "output.txt"
MODEL_CONTEXT_WINDOW = 128000 # Max sequence length for the model
SAFETY_THRESHOLD = 2000 # Buffer for generated tokens and safety
MAX_INPUT_TOKENS = MODEL_CONTEXT_WINDOW - SAFETY_THRESHOLD # Max tokens to feed the model
GC_COLLECTION_INTERVAL = 50 # How often to run garbage collection (in turns)
MPS_EMPTY_CACHE_INTERVAL = 100 # How often to empty MPS cache (in turns)

# %% Load Model and Tokenizer
print(f"Loading model: {MODEL_NAME}...")
# Note: 'attn_implementation="eager"' might be less memory efficient than "flash_attention_2"
# if available and compatible with MPS in your transformers version.
# Consider trying 'attn_implementation="sdpa"' (scaled_dot_product_attention) if eager is slow/memory hungry
# or omitting it to let transformers choose the default.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # attn_implementation="eager", # Revisit this if performance issues persist
    torch_dtype=TORCH_DTYPE,
).to(device=DEVICE)
model.eval() # Set model to evaluation mode

print(f"Loading tokenizer: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Add pad token if missing (common practice for some models/tasks)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

print("Model and tokenizer loaded.")

# %% Input Data Handling - Generator
def iter_words_from_file(filename):
    """Yields words one by one from a file."""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                words = line.strip().split()
                for word in words:
                    if word: # Skip empty strings resulting from multiple spaces
                        yield word
    except FileNotFoundError:
        print(f"Error: Input file '{filename}' not found.")
        yield from () # Yield an empty sequence

# %% Set up File Writer and Get User Input
factory = FileWriterFactory(OUTPUT_FILE)
writer = factory.get_file_writer()

promptID = input("Enter Prompt ID: ")
try:
    MultiPV = int(input("Enter MultiPV (number of top predictions to show): "))
    if MultiPV <= 0:
        raise ValueError("MultiPV must be positive.")
except ValueError as e:
    print(f"Invalid input: {e}. Using MultiPV = 1.")
    MultiPV = 1

EngineID = "DeepSeek R1 1.5B"
Depth = 1 # Keeping depth fixed as per original code

# Write header info
writer.write(f"""[PromptID "{promptID}"]\n""")
writer.write(f"""[EngineID "{EngineID}"]\n""")
writer.write(f"""[MultiPV "{MultiPV}"]\n""")
writer.write(f"""[DepthRange "1:1"]\n\n""")
writer.write(f"""1-0\n""") # Initial state notation?

# %% Main Processing Loop

# Initialize context with a starting sequence if desired, or empty
# Using the initial "Complete successive parts..." prompt from the original code
initial_prompt = "Complete successive parts of a sentence given one word at a time:"
input_ids = tokenizer.encode(initial_prompt, add_special_tokens=False) # Start with token IDs

# --- Or start empty if the prompt shouldn't be part of the context ---
# input_ids = []

print(f"Starting processing. Initial context length: {len(input_ids)} tokens.")
print(f"Max input tokens allowed: {MAX_INPUT_TOKENS}")

turnCount = 1
Stem = '' # Accumulates the *model's* predicted top tokens
PrevEval = 'n.a.'

# Use the generator to process words one by one
word_generator = iter_words_from_file(INPUT_FILE)

total_words_processed = 0

for word in word_generator:
    # --- Prepare Input for Model ---

    # 1. Tokenize the *current* word to be added *after* this prediction cycle
    # We need its tokens to add to input_ids for the *next* iteration.
    # Add a space token before the word if the context isn't empty
    prefix_space = " " if input_ids else ""
    word_tokens = tokenizer.encode(prefix_space + word, add_special_tokens=False)

    # 2. Create the input tensor for *this* iteration (using context *before* adding the current word)
    # Ensure input_ids don't exceed the max length
    if len(input_ids) > MAX_INPUT_TOKENS:
        # Truncate from the left (oldest tokens)
        truncation_amount = len(input_ids) - MAX_INPUT_TOKENS
        input_ids = input_ids[truncation_amount:]
        # print(f"Input truncated to {len(input_ids)} tokens.") # Optional debug message

    # Convert to tensor for the model - adding batch dimension [1, seq_len]
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)

    # --- Run Model Inference ---
    logits = None # Initialize variables to ensure they exist for deletion
    prob = None
    topEval = None
    top_id = None
    raw_outputs = None

    try:
        with torch.no_grad(): # Crucial for inference efficiency and memory
            # Get raw model outputs (logits)
            raw_outputs = model(input_tensor) # Shape: [batch, seq_len, vocab_size]

            # We only need the logits for the *last* token's prediction
            last_token_logits = raw_outputs.logits[0, -1, :] # Shape: [vocab_size]

            # Calculate probabilities (optional, if needed elsewhere, but logits are often enough)
            # prob = torch.softmax(last_token_logits, dim=-1)

            # Find the top K predictions based on logits (more numerically stable than probs)
            # Using logits directly for Eval score as in the original code
            topEval_logits, top_id = torch.topk(last_token_logits, MultiPV)

        # --- Extract Results ---
        # Decode the top predicted token
        modelToken = tokenizer.decode(top_id[0].item()).strip() # Top prediction
        modelEval = round(topEval_logits[0].item() * 100) # Score of top prediction (using logit)

        # Update Stem (sequence of model's top predictions)
        Stem += (' ' + modelToken if Stem else modelToken)

        # Get NextEval (score of the second-best prediction)
        NextEval = 'n.a.'
        if MultiPV > 1 and len(topEval_logits) > 1:
            NextEval = round(topEval_logits[1].item() * 100)
        elif turnCount == total_words_processed: # Check if it's the very last word? This condition might need review.
             # The original code checked against len(targetVec). Need a way to know if this is the last word.
             # This is tricky with a generator. We might need to read the file once to get the count
             # or accept 'n.a.' might be wrong on the penultimate word.
             # For now, let's assume 'n.a.' if MultiPV=1 or only one prediction available.
             pass # NextEval remains 'n.a.'

        # --- Write Output ---
        writer.write(f"""[PID "{promptID}"]\n""")
        writer.write(f"""[EID "{EngineID}"]\n""")
        writer.write(f"""[Turn "{turnCount}-w"]\n""") # Assuming '-w' is fixed part of format
        writer.write(f"""[TextToken "{word}"]\n""") # The actual word processed in this turn
        writer.write(f"""[ModelToken "{modelToken}"]\n""") # Model's top prediction
        writer.write(f"""[Eval "{modelEval}"]\n""")
        writer.write(f"""[PrevEval "{PrevEval}"]\n""")
        writer.write(f"""[NextEval "{NextEval}"]\n""")
        writer.write(f"""[Depth "{Depth}"]\n""")
        writer.write(f"""[STEM "{Stem}"]\n""")
        writer.write(f"""[NumLegalMoves "{MultiPV}"]\n""") # Assuming this is meant to be MultiPV
        writer.write(f"""---------------"\n""")
        writer.write(f"""{Depth}\n""") # Fixed depth?
        writer.write(f"""---------------"\n""")

        # Write top K moves and their logit scores
        for i, t_id in enumerate(top_id):
            t_str = tokenizer.decode(t_id.item()).strip()
            t_score = topEval_logits[i].item() # Logit score
            writer.write(f"{t_str}: {t_score}\n")

        writer.write(f"===========================================================================================================\n")
        writer.write(f"""\n\n""")
        writer.write(f"""[Comments]\n""") # Placeholder
        writer.write(f"""[EndMove]\n\n""")
        writer.write(f"""\n\n""")

        # --- Update State for Next Iteration ---
        PrevEval = modelEval # Save current eval for the next turn
        # Add the *actual word's* tokens to the context for the next prediction
        input_ids.extend(word_tokens)
        turnCount += 1
        total_words_processed += 1


    finally:
        # --- Cleanup ---
        # Explicitly delete tensors to help free GPU memory
        del input_tensor
        del raw_outputs
        # del prob # Delete if calculated
        del topEval_logits
        del top_id
        del last_token_logits
        # If any other tensors were created, delete them here

        # Periodically run garbage collection and empty MPS cache
        if turnCount % GC_COLLECTION_INTERVAL == 0:
            gc.collect()
            # print(f"Turn {turnCount}: Ran gc.collect()") # Optional debug

        if turnCount % MPS_EMPTY_CACHE_INTERVAL == 0:
             if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
                # print(f"Turn {turnCount}: Ran torch.mps.empty_cache()") # Optional debug
             else:
                 # Fallback or warning if MPS cache emptying isn't available
                 # print("torch.mps.empty_cache not available in this PyTorch version.") # Optional
                 pass


    if turnCount % 100 == 1: # Print status every 100 turns (using modulo 1 for immediate print on turn 1)
        print(f"Processed Turn {turnCount-1}. Current context length: {len(input_ids)} tokens.")
        process = psutil.Process(os.getpid())
        print(f"RAM Used: {process.memory_info().rss / (1024 * 1024):.2f} MB")
        # if DEVICE == 'mps':
        #     # MPS memory reporting is not straightforward like CUDA.
        #     # Check Activity Monitor (GPU History) on macOS.
        #     pass


# %% Cleanup after loop
writer.close()
print(f"\nProcessing complete. Processed {total_words_processed} words.")
print(f"Output written to {OUTPUT_FILE}")