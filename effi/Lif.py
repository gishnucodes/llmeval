#%%
import torch
import gc
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

#%% Constants and Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEVICE = "cpu"
TORCH_DTYPE = torch.float32
INPUT_FILE = '/kaggle/input/vanity-poem-txt/vanity.txt'
OUTPUT_FILE = "vanity_lif_buffered.lif" # Changed output filename
MODEL_CONTEXT_WINDOW = 128_000
SAFETY_THRESHOLD = 2_000
MAX_INPUT_TOKENS = MODEL_CONTEXT_WINDOW - SAFETY_THRESHOLD

# --- New constant for buffered writing ---
# Write to disk after accumulating output for this many turns.
# Adjust this value to assess performance impact. Larger values mean less frequent writes.
WRITE_BUFFER_TURNS = 50
PRINT_TURN_COUNT=1

#%% Load and Quantize Model & Tokenizer
print("Step 1: Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=TORCH_DTYPE
).to(DEVICE)
print(f"  Model loaded to {DEVICE}.")

print("Step 2: Applying dynamic quantization for faster CPU inference...")
model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
model.eval()
print("  Quantization complete. Model is ready for inference.\n")

print("Step 3: Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    print("  Tokenizer missing pad token; setting pad_token = eos_token")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
print("  Tokenizer loaded and configured.\n")

#%% User Inputs
print("Step 4: Prompting user for inputs...")
promptID = input("  Enter Prompt ID: ")
MultiPV = int(input("  Enter MultiPV (top predictions to show): "))
LegalNumberOfMove = int(input("  Enter Max Number of moves: "))
BeamSearchValue=int(input("  Enter Beam search value: "))
SequenceNumberToReturn = int(input("  Enter Sequence Number to return ( < beam search number) : "))
EngineID = f"DeepSeek R1 1.5B ({DEVICE.upper()})"
Depth = 1
print("  User inputs captured.\n")

#%% Pre-tokenize entire relevant input sequence
print("Step 5: Pre-tokenizing input sequence...")
initial_prompt = "Complete successive parts of a sentence given one word at a time:"
initial_prompt_ids = tokenizer.encode(initial_prompt, add_special_tokens=False)

print(f"  Reading words from {INPUT_FILE}...")
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    words_from_file = f.read().split()
print(f"  Found {len(words_from_file)} words.")

all_tokens = list(initial_prompt_ids)
word_end_indices = [len(initial_prompt_ids)]
processed_words = []

print("  Tokenizing words and building full sequence...")
for word in words_from_file:
    word_tokens = tokenizer.encode(" " + word, add_special_tokens=False)
    all_tokens.extend(word_tokens)
    word_end_indices.append(len(all_tokens))
    processed_words.append(word)

full_token_tensor = torch.tensor(all_tokens, dtype=torch.long).unsqueeze(0)
print(f"  Pre-tokenized {len(processed_words)} words into a sequence of {len(all_tokens)} tokens.\n")

num_words_to_process = min(len(processed_words), LegalNumberOfMove)
if num_words_to_process < len(processed_words):
     print(f"  Will process the first {num_words_to_process} words due to LegalNumberOfMove limit.\n")
elif num_words_to_process == 0:
     print("  Warning: No words to process based on input file or limits.\n")

#%% Build file header
print("Step 8: Preparing output file header...")
header_lines = [
    f'[PromptID "{promptID}"]\n',
    f'[EngineID "{EngineID}"]\n',
    f'[MultiPV "{MultiPV}"]\n',
    f'[DepthRange "1:1"]\n\n',
    "1-0\n"
]
print("  Header prepared.\n")

#%% Main Generation Loop (Using Slicing and Buffered Writing)
print(f"Step 9: Entering main generation loop (Buffering writes every {WRITE_BUFFER_TURNS} turns)...\n")
PrevEval = "n.a."
start_time = time.time()
output_buffer = [] # Initialize the buffer for storing lines before writing

if num_words_to_process > 0:
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as writer:
        print("  Writing header to output file...")
        writer.write(''.join(header_lines))
        print("  Header written. Starting word-by-word prediction.\n")

        for turnCount in range(1, num_words_to_process + 1):
            current_word = processed_words[turnCount-1]
            # --- (Optional: Reduced print frequency for performance) ---
            if turnCount % PRINT_TURN_COUNT == 0 or turnCount == 1: # Print less often
                 print(f"Turn {turnCount}: Predicting after word '{current_word}'")

            # 9.1-9.3: Determine slice indices and create input tensor
            slice_end_index = word_end_indices[turnCount-1]
            slice_start_index = max(0, slice_end_index - MAX_INPUT_TOKENS)
            input_tensor = full_token_tensor[:, slice_start_index:slice_end_index]
            current_input_len = input_tensor.shape[1]
            input_tensor_dev = input_tensor.to(DEVICE)

            # 9.4 Generate next token(s)
            start_time_gen = time.time()
            if turnCount % PRINT_TURN_COUNT == 0 or turnCount == 1:
                print(f"  9.4: Running model.generate() with {current_input_len} input tokens...")
            with torch.no_grad():
                 outputs = model.generate(
                     input_tensor_dev,
                     max_new_tokens=1,
                     num_beams=MultiPV,
                     num_return_sequences=MultiPV,
                     output_scores=True,
                     return_dict_in_generate=True,
                     pad_token_id=tokenizer.pad_token_id
                 )
            end_time_gen = time.time()
            print(f"Time to generate the slice {input_tensor.shape} is : {(end_time_gen-start_time_gen)} seconds")
            next_ids = outputs.sequences[:, -1].tolist()
            score_matrix = outputs.scores[-1]
            next_scores = [score_matrix[i, tid].item() for i, tid in enumerate(next_ids)]
            if turnCount % PRINT_TURN_COUNT == 0 or turnCount == 1:
                print(f"    Received top-{MultiPV} token IDs: {next_ids}")

            # 9.5 Derive primary metrics
            modelToken = tokenizer.decode(next_ids[0]).strip()
            modelEval  = round(next_scores[0] * 100)
            NextEval   = (round(next_scores[1] * 100))
            if turnCount % PRINT_TURN_COUNT == 0 or turnCount == 1:
                print(f"  9.5: Selected top token '{modelToken}' with eval {modelEval}. NextEval: {NextEval}")

            # 9.6 Build lines for this turn
            if turnCount % PRINT_TURN_COUNT == 0 or turnCount == 1:
                print("  9.6: Building output lines for this turn...")
            current_stem = " ".join(processed_words[:turnCount])
            lines = [
                f'[Turn "{turnCount}-w"]\n',
                f'[TextToken "{current_word}:"]\n',
                f'[ModelToken "{modelToken}:"]\n',
                f'[Eval "{modelEval}"]\n',
                f'[PrevEval "{PrevEval}"]\n',
                f'[NextEval "{NextEval}"]\n',
                f'[Depth "{Depth}"]\n',
                f'[STEM "{current_stem}"]\n',
                f'[NumLegalMoves "{LegalNumberOfMove}"]\n',
                "---------------\n"
            ]
            for tid, sc in zip(next_ids, next_scores):
                token_str = tokenizer.decode(tid).strip()
                lines.append(f"{token_str}: {sc:.4f}\n")
            lines.append("===========================================================================================================\n\n")

            # 9.7 Accumulate lines in buffer instead of writing immediately
            output_buffer.extend(lines)
            if turnCount % PRINT_TURN_COUNT == 0 or turnCount == 1:
                print("    Lines added to buffer.")

            # Check if the buffer should be flushed (written to disk)
            if turnCount % WRITE_BUFFER_TURNS == 0:
                print(f"  Turn {turnCount}: Flushing buffer ({len(output_buffer)} lines) to {OUTPUT_FILE}...")
                write_start_time = time.time()
                writer.write(''.join(output_buffer))
                output_buffer.clear() # Clear the buffer after writing
                write_duration = time.time() - write_start_time
                print(f"    Flush complete in {write_duration:.4f} seconds.")


            # 9.8 Update state
            PrevEval = modelEval

            # 9.9 Status update (can be less frequent if desired)
            if turnCount % 10 == 0:
                 elapsed = time.time() - start_time
                 rate = turnCount / elapsed if elapsed > 0 else 0
                 # Estimate buffer memory usage (very rough estimate)
                 buffer_size_bytes = sum(len(s.encode('utf-8')) for s in output_buffer)
                 print(f"\n  Status @ Turn {turnCount}: Rate={rate:.2f} w/s. Current buffer: {len(output_buffer)} lines (~{buffer_size_bytes / 1024:.1f} KB)\n")


        # --- End of loop ---

        # 9.11 Final Flush: Write any remaining lines in the buffer after the loop finishes
        if output_buffer:
            print(f"\n  Flushing remaining {len(output_buffer)} lines from buffer to {OUTPUT_FILE}...")
            write_start_time = time.time()
            writer.write(''.join(output_buffer))
            output_buffer.clear()
            write_duration = time.time() - write_start_time
            print(f"    Final flush complete in {write_duration:.4f} seconds.")
        else:
             print("\n  No remaining lines in buffer to flush.")

        print("\n  Finished processing requested words.\n")

else:
    print("Skipping main generation loop as there are no words to process.")


#%% Final Stats
print("Step 10: Reporting final statistics...")
total_time = time.time() - start_time
avg_rate = (num_words_to_process / total_time) if total_time > 0 and num_words_to_process > 0 else 0
print(f"  Total turns processed: {num_words_to_process}")
print(f"  Total time: {total_time:.2f} seconds")
print(f"  Average speed: {avg_rate:.2f} words/second")
print(f"  Output written to {OUTPUT_FILE}")

# Optional: Clean up memory
del model
del tokenizer
del full_token_tensor
# del outputs # 'outputs' is defined inside the loop, so it's cleaned up automatically mostly
gc.collect()
if DEVICE == 'cuda': # Though DEVICE is "cpu" here, good practice if adaptable
     torch.cuda.empty_cache()
print("\nScript finished.")