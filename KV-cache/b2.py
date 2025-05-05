import numpy as np
# No longer need sympy

# Changed imports: Removed openvino, added AutoModelForCausalLM
# import openvino as ov
# from optimum.intel import OVModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import gc
import time
import os
import io
import argparse
import sys
import torch # Keep torch

# Buffer Configuration remains the same
BUFFER_MEMORY_LIMIT_MB = 200
BUFFER_MEMORY_LIMIT_BYTES = BUFFER_MEMORY_LIMIT_MB * 1024 * 1024
print(f"Using output buffer with memory limit: {BUFFER_MEMORY_LIMIT_MB} MB ({BUFFER_MEMORY_LIMIT_BYTES} bytes)")

# main function signature remains the same
def main(INPUT_FILE, OUTPUT_FILE, PROMPT_ID, PROMPT_TOPIC, MULTI_PV, NUM_ITEMS, ALPHA_MODE, NUM_BEAMS, MODEL_NAME): # Added MODEL_NAME argument

    # --- Device Selection (Standard PyTorch) ---
    if torch.cuda.is_available():
        DEVICE = "cuda"
    # elif torch.backends.mps.is_available(): # Uncomment if running on Apple Silicon Mac
    #     DEVICE = "mps"
    else:
        DEVICE = "cpu"
    print(f"Using PyTorch device: {DEVICE}")

    # --- Model Identifier ---
    # ORIGINAL_MODEL_NAME is now passed as MODEL_NAME argument
    # Default set in argparse (e.g., "Qwen/Qwen1.5-1.8B")
    DEPTH_RANGE = 1 # Seems constant

    # Input file handling remains the same
    if not os.path.exists(INPUT_FILE):
        print(f"Warning: Input file '{INPUT_FILE}' not found. Creating a dummy file.")
        os.makedirs(os.path.dirname(INPUT_FILE), exist_ok=True)
        with open(INPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("The quick brown fox jumps over the lazy dog")

    # --- Load Hugging Face Model & Tokenizer ---
    print(f"Step 1 & 2: Loading standard Hugging Face model '{MODEL_NAME}' and tokenizer...")
    try:
        # Load tokenizer - trust_remote_code might be needed for some models like Qwen
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        print(f"  Tokenizer loaded successfully from '{MODEL_NAME}'.")

        # Load model
        # Consider adding options like torch_dtype=torch.float16 for GPU,
        # or load_in_8bit=True/load_in_4bit=True (requires bitsandbytes and accelerate)
        # for larger models if memory is an issue.
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            # torch_dtype=torch.float16, # Example: Uncomment for faster GPU inference
            # load_in_8bit=True,      # Example: Uncomment for 8-bit quantization
            # device_map='auto'       # Example: Use Accelerate for multi-GPU or large models
        )
        print(f"  Model base loaded successfully from '{MODEL_NAME}'.")

        # --- Move model to device ---
        # Note: If using device_map='auto', this step is handled by Accelerate
        if not hasattr(model, 'hf_device_map'): # Check if device_map was used
             model.to(DEVICE)
             print(f"  Model moved to device: {DEVICE}")
        else:
             print(f"  Model placement handled by device_map: {model.hf_device_map}")
             # In device_map scenarios, model.device might point to 'cpu' or a specific GPU index
             # We'll rely on generate handling tensor placement correctly or use input_ids.to(model.device) later if needed.


    except Exception as e:
        print(f"ERROR: Failed to load standard Hugging Face model or tokenizer from '{MODEL_NAME}'")
        print(f"Error details: {e}")
        print("Please ensure the model identifier is correct, dependencies (like sentencepiece, protobuf, bitsandbytes, accelerate) are installed, and you have enough memory/disk space.")
        exit(1)

    # Set pad token if missing (common practice)
    if tokenizer.pad_token is None:
        print("  Tokenizer missing pad token; setting pad_token = eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        # Update model config if it exists and doesn't match
        if hasattr(model, 'config') and model.config.pad_token_id != tokenizer.eos_token_id:
             model.config.pad_token_id = tokenizer.eos_token_id
             print(f"  Model config pad_token_id updated to {tokenizer.eos_token_id}")
    print("  Model and Tokenizer ready for inference.\n")

    # Hook mechanism is not needed as we use output_scores=True

    # User Inputs & Engine ID update
    promptID = PROMPT_ID
    MultiPV = MULTI_PV
    LegalNumberOfMove = NUM_ITEMS
    # Update EngineID to reflect HF model and PyTorch device
    engine_device_info = next(iter(model.hf_device_map.values())) if hasattr(model, 'hf_device_map') else DEVICE # Get device from device_map if used
    EngineID = f"{MODEL_NAME} HF ({engine_device_info})"
    Depth = 1

    # Pre-processing input sequence (remains largely the same)
    print("Step 5: Pre-processing input sequence...")
    initial_prompt = "Complete successive parts of a sentence given one word at a time for the topic : " + PROMPT_TOPIC + ": "
    initial_prompt_ids = tokenizer.encode(initial_prompt, add_special_tokens=False)

    print(f"  Reading words from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            words_from_file = f.read().split()
        print(f"  Found {len(words_from_file)} words.")
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found. Exiting.")
        exit(1)
    except Exception as e:
        print(f"Error reading file '{INPUT_FILE}': {e}")
        exit(1)

    tokenized_words = []
    total_token_count = len(initial_prompt_ids)
    processed_words = []

    print("  Tokenizing words...")
    for word in words_from_file:
        word_to_encode = " " + word if not word.startswith(("'",".")) else word
        word_tokens = tokenizer.encode(word_to_encode, add_special_tokens=False)
        if not word_tokens:
            print(f"  Warning: Word '{word}' (encoded as '{word_to_encode}') tokenized to empty sequence, skipping.")
            continue
        tokenized_words.append(word_tokens)
        processed_words.append(word)
        total_token_count += len(word_tokens)

    print(f"  Pre-processed {len(processed_words)} words.")
    print(f"  Total estimated tokens (prompt + words): {total_token_count}\n")

    model_context_window = getattr(model.config, "max_position_embeddings", None)
    # Some models use different names, e.g., n_positions
    if model_context_window is None:
        model_context_window = getattr(model.config, "n_positions", None)

    if model_context_window:
        print(f"  Model context window: {model_context_window}")
        if total_token_count > model_context_window:
            print(f"WARNING: Estimated total token count ({total_token_count}) exceeds model context window ({model_context_window}). This might lead to unexpected results or errors.")
    else:
        print("Warning: Could not determine model context window from config.")

    num_words_to_process = min(len(processed_words), LegalNumberOfMove)
    if num_words_to_process < len(processed_words):
        print(f"  Will process the first {num_words_to_process} words out of {len(processed_words)} due to LegalNumberOfMove limit.\n")
    elif num_words_to_process == 0:
        print("  Warning: No words to process based on input file or limits. Exiting.\n")
        exit(0)
    else:
        print(f"  Will process all {num_words_to_process} words.\n")


    # Write File Header (remains the same, uses updated EngineID)
    print("Step 8: Preparing and writing output file header...")
    header_lines = [
        f'[PromptID "{promptID}"]\n',
        f'[EngineID "{EngineID}"]\n',
        f'[MultiPV "{MultiPV}"]\n',
        f'[DepthRange "{DEPTH_RANGE}:{DEPTH_RANGE}"]\n\n',
        "1-0\n\n"
    ]
    try:
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as writer:
            writer.write(''.join(header_lines))
        print(f"  Header written to {OUTPUT_FILE}.\n")
    except IOError as e:
        print(f"Error writing header to {OUTPUT_FILE}: {e}")
        exit(1)

    # Main Generation Loop with Beam Search
    print(f"Step 9: Entering main generation loop (using beam search with {NUM_BEAMS} beams)...\n")
    PrevEval = "n.a."
    start_time = time.time()
    output_buffer = []
    current_buffer_size_bytes = 0

    # Explicitly set inference mode
    model.eval()

    # Determine the primary device for placing input tensors if device_map is used
    # If not using device_map, model.device should be correct.
    # If using device_map, inputs usually go to the device of the first layer/embedding.
    input_device = model.device if not hasattr(model, 'hf_device_map') else \
                   (model.hf_device_map.get('transformer.wte', None) or \
                    model.hf_device_map.get('model.embed_tokens', None) or \
                    next(iter(model.hf_device_map.values()))) # Fallback logic

    print(f"  Input tensors will be placed on: {input_device}")


    # Loop through words
    for turnCount in range(1, num_words_to_process + 1):
        current_word_index = turnCount - 1
        current_word = processed_words[current_word_index]

        words_in_stem = processed_words[:turnCount]
        current_stem = initial_prompt + " ".join(words_in_stem)
        current_stem_ids = tokenizer.encode(current_stem, add_special_tokens=False)

        # Prepare input tensor and move to appropriate device
        input_ids = torch.tensor([current_stem_ids], dtype=torch.long).to(input_device)

        if model_context_window and input_ids.shape[1] >= model_context_window:
             print(f"WARNING: Input length {input_ids.shape[1]} for turn {turnCount} meets or exceeds context window {model_context_window}. Truncation might occur internally, or results may degrade.")
             # Consider implementing sliding window or other handling if needed

        start_time_gen = time.time()
        try:
            # --- Perform Inference ---
            with torch.no_grad(): # Disable gradient calculation for inference
                outputs = model.generate(
                    input_ids=input_ids,
                    num_beams=NUM_BEAMS,
                    max_new_tokens=1, # Generate exactly one new token
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True, # Get logits
                    pad_token_id=tokenizer.pad_token_id # Ensure pad token is set
                )
        except Exception as e:
            print(f"\nERROR during model.generate at turn {turnCount}: {e}")
            print(f"  Input shape: {input_ids.shape}")
            print(f"  Input device: {input_ids.device}")
            # Add more debug info if using CUDA (e.g., memory usage)
            if DEVICE == 'cuda':
                print(f"  CUDA Memory allocated: {torch.cuda.memory_allocated(DEVICE) / 1e9:.2f} GB")
                print(f"  CUDA Memory reserved:  {torch.cuda.memory_reserved(DEVICE) / 1e9:.2f} GB")
            print("Aborting generation loop due to error.")
            break # Exit the loop

        gen_duration = time.time() - start_time_gen

        # --- Extract top predictions from scores ---
        if not outputs.scores:
             print(f"Warning: `outputs.scores` is empty for turn {turnCount}. Cannot extract logits. Skipping turn output.")
             PrevEval = "n.a."
             continue

        # outputs.scores[0] contains logits for the first generated token. Shape: (batch_size, vocab_size)
        last_token_scores = outputs.scores[0] # Logits for the generated token
        scores_for_beam_1 = last_token_scores[0] # Shape: (vocab_size,) - Assuming batch_size=1

        # Get the ID of the token chosen by beam search (last token in sequence)
        # Move to CPU before converting to scalar
        generated_token_id = outputs.sequences[0][-1].cpu().item()
        model_token = tokenizer.decode(generated_token_id, skip_special_tokens=True).strip()

        # Get top K logits and indices. Move scores to CPU before topk if they aren't already.
        # (topk works on CUDA, but subsequent list conversion is easier from CPU)
        scores_for_beam_1_cpu = scores_for_beam_1.cpu()
        top_k_logits_values, top_k_logits_indices = torch.topk(scores_for_beam_1_cpu, k=MultiPV, dim=-1)

        top_k_logits_values = top_k_logits_values.tolist()
        top_k_logits_indices = top_k_logits_indices.tolist()
        top_k_tokens = [tokenizer.decode(tid, skip_special_tokens=True).strip() for tid in top_k_logits_indices]

        # --- Derive metrics (using CPU tensor data) ---
        if top_k_logits_values:
            try:
                # Get the logit score of the *actually generated* token from the CPU tensor
                generated_token_logit = scores_for_beam_1_cpu[generated_token_id].item()
                model_eval = round(float(generated_token_logit)) # Scaling logic removed, adjust if needed
                model_eval_str = f"{generated_token_logit:.4f}"
            except IndexError:
                 print(f"Warning: Generated token ID {generated_token_id} out of bounds for scores tensor with shape {scores_for_beam_1_cpu.shape}")
                 model_eval = "error"
                 model_eval_str = "error"

            if MultiPV > 1 and len(top_k_logits_values) > 1:
                next_eval_logit = top_k_logits_values[1]
                next_eval = round(float(next_eval_logit)) # Scaling logic removed, adjust if needed
                next_eval_str = f"{next_eval_logit:.4f}"
            else:
                next_eval = "n.a."
                next_eval_str = "n.a."
        else:
            model_eval = "n.a."
            next_eval = "n.a."
            model_eval_str = "n.a."
            next_eval_str = "n.a."

        # --- Build output lines (remains the same, uses updated metrics/EngineID) ---
        if turnCount == 1:
            PrevEval = "n.a."

        lines = [
            f'[PID "{promptID}"]\n',
            f'[EID "{EngineID}"]\n',
            f'[Turn "{turnCount}-w"]\n',
            f'[TextToken "{current_word}:"]\n',
            f'[ModelToken "{model_token}:"]\n',
            f'[Eval "{model_eval}"]\n',
            f'[PrevEval "{PrevEval}"]\n',
            f'[NextEval "{next_eval}"]\n',
            f'[Depth "{Depth}"]\n',
            f'[STEM "{current_stem}"]\n',
            f'[NumLegalMoves "{LegalNumberOfMove}"]\n',
            "---------------\n",
            f"{DEPTH_RANGE}\n",
            "---------------\n"
        ]
        for token_str, logit_val in zip(top_k_tokens, top_k_logits_values):
            lines.append(f"{token_str}: {logit_val:.4f}\n")
        lines.append("===========================================================================================================\n\n")
        lines.append(f"[Comments]\n")
        lines.append(f"Generation time: {gen_duration:.4f}s | Eval Logit: {model_eval_str} | Next Best Logit: {next_eval_str}\n")
        lines.append(f"[EndMove]\n\n")

        # --- Buffer handling (remains the same) ---
        turn_output_string = ''.join(lines)
        try:
            turn_output_bytes = len(turn_output_string.encode('utf-8'))
        except Exception as enc_e:
            print(f"Warning: Could not encode output string to UTF-8 for size estimation: {enc_e}")
            turn_output_bytes = len(turn_output_string) * 2

        output_buffer.append(turn_output_string)
        current_buffer_size_bytes += turn_output_bytes

        if current_buffer_size_bytes >= BUFFER_MEMORY_LIMIT_BYTES:
            print(f"--- Flushing buffer ({current_buffer_size_bytes / (1024*1024):.2f} MB exceeds limit)... ---")
            try:
                with open(OUTPUT_FILE, 'a', encoding='utf-8') as writer:
                    writer.write("".join(output_buffer))
                output_buffer = []
                current_buffer_size_bytes = 0
                print(f"--- Buffer flushed to {OUTPUT_FILE}. ---")
            except IOError as e:
                print(f"Error writing buffer to {OUTPUT_FILE}: {e}")
                print("Attempting to continue, but output file might be incomplete.")

        PrevEval = model_eval # Update PrevEval for the next turn

        # --- Progress Status Update (remains the same) ---
        status_interval = 10
        if num_words_to_process >= 100: status_interval = 50
        elif num_words_to_process >= 20: status_interval = 10

        if turnCount % status_interval == 0 or turnCount == num_words_to_process:
            elapsed = time.time() - start_time
            rate = turnCount / elapsed if elapsed > 0 else 0
            current_total_tokens = len(initial_prompt_ids) + sum(len(tk) for tk in tokenized_words[:turnCount])
            token_rate = current_total_tokens / elapsed if elapsed > 0 else 0
            # Add memory usage for GPU
            mem_info = ""
            if DEVICE == 'cuda':
                mem_alloc = torch.cuda.memory_allocated(DEVICE) / 1e9
                mem_reserv = torch.cuda.memory_reserved(DEVICE) / 1e9
                mem_info = f" | Mem: {mem_alloc:.2f}/{mem_reserv:.2f} GB"

            print(f"  Status: Processed {turnCount}/{num_words_to_process} words "
                  f"({rate:.2f} w/s | {token_rate:.2f} tok/s) "
                  f"({elapsed:.2f}s total). "
                  f"Buffer: {current_buffer_size_bytes / (1024*1024):.2f} MB{mem_info}")
            if turnCount != num_words_to_process: print()


    # --- End of Loop ---

    # Final buffer flush (remains the same)
    if output_buffer:
        print(f"--- Flushing remaining buffer ({current_buffer_size_bytes / (1024*1024):.2f} MB)... ---")
        try:
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as writer:
                writer.write("".join(output_buffer))
            print(f"--- Final buffer contents flushed to {OUTPUT_FILE}. ---")
        except IOError as e:
            print(f"Error writing final buffer to {OUTPUT_FILE}: {e}")

    print("\n  Finished processing requested words.\n")

    # Final Stats (remains the same)
    print("Step 10: Reporting final statistics...")
    total_time = time.time() - start_time
    if num_words_to_process > 0 and total_time > 0:
        avg_rate_words = num_words_to_process / total_time
        final_total_tokens = len(initial_prompt_ids) + sum(len(tk) for tk in tokenized_words[:num_words_to_process])
        avg_rate_tokens = final_total_tokens / total_time
    else:
        avg_rate_words = 0
        final_total_tokens = len(initial_prompt_ids)
        avg_rate_tokens = 0

    print(f"  Total turns processed: {num_words_to_process}")
    print(f"  Total tokens processed (including prompt): {final_total_tokens}")
    print(f"  Total time: {total_time:.2f} seconds")
    if num_words_to_process > 0:
        print(f"  Average speed: {avg_rate_words:.2f} words/second")
        print(f"  Average speed: {avg_rate_tokens:.2f} tokens/second")
    print(f"  Output written to {OUTPUT_FILE}")

    # Clean up
    print("\nCleaning up resources...")
    del model
    del tokenizer
    del tokenized_words
    if 'input_ids' in locals(): del input_ids
    if 'outputs' in locals(): del outputs
    if 'output_buffer' in locals(): del output_buffer
    # Explicitly clear PyTorch CUDA cache if GPU was used
    if DEVICE == 'cuda':
         torch.cuda.empty_cache()
         print("  Cleared CUDA cache.")
    gc.collect()

    print("\nScript finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LifGenerator using standard Hugging Face Transformers models with beam search decoding", # Updated description
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Arguments largely the same, but default model name changed
    parser.add_argument("--input-file", "-i", type=str, required=True,
                        help="The path to the input text file containing words.")
    parser.add_argument("--output-file", "-o", type=str,
                        help="Name and path of the output .lif file. Defaults to '[input-file-stem].lif'.")
    parser.add_argument("--prompt-id", "-pid", type=str,
                        help="Identifier for the prompt/run. Defaults to the input filename.")
    parser.add_argument("--prompt-topic", "-pt", type=str, default="general text",
                        help="Topic given to the LLM before the stem words.")
    parser.add_argument("--multi-pv", "-mpv", type=int, default=10,
                        help="Number of top token predictions (and their logits) to record at each turn.")
    parser.add_argument("--num-items", "-n", type=int, default=100,
                        help="Maximum number of words from the input file to process.")
    parser.add_argument("--num-beams", "-nb", type=int, default=4,
                        help="Number of beams for beam search decoding.")
    # Changed model-name default and help text
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen1.5-1.8B", # Changed default model
                        help="Hugging Face Hub model ID or local path for the standard transformer model (e.g., 'gpt2', 'Qwen/Qwen1.5-1.8B').")

    args = parser.parse_args()

    print("Welcome to the LifGenerator Hugging Face script!") # Updated welcome message
    print(f"Using model: {args.model_name}")
    print(f"Beam search decoding with {args.num_beams} beams.")
    print(f"Input file path: {args.input_file}")

    INPUT_FILE = args.input_file
    INPUT_FILE_STEM = os.path.splitext(os.path.basename(INPUT_FILE))[0]
    OUTPUT_FILE = args.output_file if args.output_file else (INPUT_FILE_STEM + ".lif")
    print(f"Output file path: {OUTPUT_FILE}")

    PROMPT_ID = args.prompt_id if args.prompt_id else os.path.basename(INPUT_FILE)
    PROMPT_TOPIC = args.prompt_topic
    MULTI_PV = args.multi_pv
    NUM_ITEMS = args.num_items
    ALPHA_MODE = 0 # Not used currently
    NUM_BEAMS = args.num_beams
    MODEL_NAME = args.model_name # Get model name from args

    # Call main function with the model name
    main(INPUT_FILE, OUTPUT_FILE, PROMPT_ID, PROMPT_TOPIC, MULTI_PV, NUM_ITEMS, ALPHA_MODE, NUM_BEAMS, MODEL_NAME)