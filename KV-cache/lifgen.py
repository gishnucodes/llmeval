#%%
# Ensure necessary libraries are installed:
# pip install optimum[openvino] openvino-dev transformers torch>=2.1.0 accelerate huggingface_hub sentencepiece protobuf<=3.20.3 numpy

import numpy as np
import openvino as ov # Use openvino directly for Core if needed, but Optimum abstracts much of this
from optimum.intel import OVModelForCausalLM # <<< Use Optimum Intel for HF integration
from transformers import AutoTokenizer, AutoConfig
import gc
import time
import os
import io # Needed for potential in-memory string handling if preferred
import argparse
import sys

# --- NEW: Buffer Configuration ---
# Define the memory limit for the write buffer in Megabytes
BUFFER_MEMORY_LIMIT_MB = 200 # MB - Adjust as needed based on available RAM
BUFFER_MEMORY_LIMIT_BYTES = BUFFER_MEMORY_LIMIT_MB * 1024 * 1024
print(f"Using output buffer with memory limit: {BUFFER_MEMORY_LIMIT_MB} MB ({BUFFER_MEMORY_LIMIT_BYTES} bytes)")
# --- End Buffer Configuration ---
OPENVINO_MODEL_DIR = "openvino_model_int8/"


# Login logic (commented out, assumed not needed for runtime if model/tokenizer are local)
# ... (rest of the login logic remains the same if needed) ...

#%% Constants and Configuration
def main(INPUT_FILE, OUTPUT_FILE, PROMPT_ID, PROMPT_TOPIC, MULTI_PV,NUM_ITEMS, ALPHA_MODE):
    # try:
    #     config = AutoConfig.from_pretrained(OPENVINO_MODEL_DIR)
    #     ORIGINAL_MODEL_NAME =
    #     print(f"Inferred original model name: {ORIGINAL_MODEL_NAME}")
    # except Exception:
    #     ORIGINAL_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" # Fallback
    #     print(f"Could not infer original model name, using fallback: {ORIGINAL_MODEL_NAME}")
    #
    DEVICE = "AUTO"
    print(f"Attempting to use OpenVINO device: {DEVICE}")

    ORIGINAL_MODEL_NAME = "OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int8-ov"
    DEPTH_RANGE=1

    if not os.path.exists(INPUT_FILE):
        print(f"Warning: Input file '{INPUT_FILE}' not found. Creating a dummy file.")
        with open(INPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("The quick brown fox jumps over the lazy dog")

    #%% Load OpenVINO Model & Tokenizer
    print("Step 1 & 2: Loading OpenVINO model and tokenizer...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_NAME, trust_remote_code=True)
        print("  Tokenizer loaded successfully from OpenVINO model directory.")
        model = OVModelForCausalLM.from_pretrained(
            ORIGINAL_MODEL_NAME,
            device=DEVICE,
            trust_remote_code=True,
        )
        print(f"  OpenVINO Model loaded and compiled for device {model.device}.")
    except Exception as e:
        print(f"ERROR: Failed to load OpenVINO model or tokenizer from {OPENVINO_MODEL_DIR}")
        print(f"Ensure the directory exists and contains openvino_model.xml/bin and tokenizer files.")
        print(f"Error details: {e}")
        exit()

    if tokenizer.pad_token is None:
        print("  Tokenizer missing pad token; setting pad_token = eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        if hasattr(model, 'config'):
            model.config.pad_token_id = tokenizer.eos_token_id
    print("  Model and Tokenizer ready for inference.\n")


    #%% User Inputs
    promptID = PROMPT_ID
    MultiPV = MULTI_PV
    LegalNumberOfMove = NUM_ITEMS
    EngineID = f"{ORIGINAL_MODEL_NAME} OpenVINO ({model.device})"
    Depth = 1

    #%% Pre-process input sequence (Tokenize all relevant parts)
    print("Step 5: Pre-processing input sequence...")
    initial_prompt = "Complete successive parts of a sentence given one word at a time for the topic : " + PROMPT_TOPIC  + ": " # adding prompt topic
    initial_prompt_ids = tokenizer.encode(initial_prompt, add_special_tokens=False)

    print(f"  Reading words from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            words_from_file = f.read().split()
        print(f"  Found {len(words_from_file)} words.")
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found. Exiting.")
        exit()

    tokenized_words = []
    total_token_count = len(initial_prompt_ids)
    processed_words = []

    print("  Tokenizing words...")
    for word in words_from_file:
        word_tokens = tokenizer.encode(" " + word, add_special_tokens=False)
        if not word_tokens:
            print(f"  Warning: Word '{word}' tokenized to empty sequence, skipping.")
            continue
        tokenized_words.append(word_tokens)
        processed_words.append(word)
        total_token_count += len(word_tokens)

    print(f"  Pre-processed {len(processed_words)} words.")
    print(f"  Total estimated tokens (prompt + words): {total_token_count}\n")

    model_context_window = getattr(model.config, "max_position_embeddings", None)
    if model_context_window:
        print(f"  Model context window: {model_context_window}")
        if total_token_count > model_context_window:
            print(f"WARNING: Estimated total token count ({total_token_count}) exceeds model context window ({model_context_window}).")
            print("KV caching might fail or produce unexpected results if the limit is strictly enforced by the model.")
    else:
        print("Warning: Could not determine model context window from config.")

    num_words_to_process = min(len(processed_words), LegalNumberOfMove)
    if num_words_to_process < len(processed_words):
        print(f"  Will process the first {num_words_to_process} words due to LegalNumberOfMove limit.\n")
    elif num_words_to_process == 0:
        print("  Warning: No words to process based on input file or limits.\n")


    #%% Build and Write File Header Separately
    print("Step 8: Preparing and writing output file header...")
    header_lines = [
        f'[PromptID "{promptID}"]\n',
        f'[EngineID "{EngineID}"]\n',
        f'[MultiPV "{MultiPV}"]\n',
        f'[DepthRange "1:1"]\n\n',
        "1-0\n\n"
    ]
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as writer:
            writer.write(''.join(header_lines))
        print(f"  Header written to {OUTPUT_FILE}.\n")
    except IOError as e:
        print(f"Error writing header to {OUTPUT_FILE}: {e}")
        exit()

    #%% Main Generation Loop (Using KV Caching with OpenVINO/Optimum and Buffered Writes)
    print("Step 9: Entering main generation loop (using OpenVINO KV Caching and buffered writes)...\n")
    PrevEval = "n.a."
    start_time = time.time()
    past_key_values = None

    # --- NEW: Initialize Output Buffer ---
    output_buffer = [] # List to store chunks of output strings
    current_buffer_size_bytes = 0
    # --- End Buffer Initialization ---

    if num_words_to_process > 0:
        # Process the initial prompt first to build the initial cache
        print("  9.0: Processing initial prompt to build cache...")
        initial_input_ids = np.array([initial_prompt_ids], dtype=np.int64)
        start_time_gen = time.time()
        outputs = model(
            input_ids=initial_input_ids,
            past_key_values=None,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        gen_duration = time.time() - start_time_gen
        past_key_values = outputs.past_key_values
        print(f"      Initial prompt processing took: {gen_duration:.4f} seconds")

        # Now loop through the words, using the cache
        for turnCount in range(1, num_words_to_process + 1):
            current_word = processed_words[turnCount-1]
            current_word_tokens = tokenized_words[turnCount-1]
            input_ids = np.array([current_word_tokens], dtype=np.int64)

            start_time_gen = time.time()
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
            )
            gen_duration = time.time() - start_time_gen

            past_key_values = outputs.past_key_values
            logits_for_step = outputs.logits[:, -1, :]
            logits_for_prediction = logits_for_step[0]

            try:
                import torch
            except ImportError:
                print("Error: PyTorch is required for logit processing even with OpenVINO via Optimum.")
                exit()

            logits_for_prediction_cpu = logits_for_prediction.cpu()
            top_k_logits_values, top_k_logits_indices = torch.topk(
                logits_for_prediction_cpu, k=MultiPV, dim=-1
            )
            top_k_logits_values = top_k_logits_values.tolist()
            top_k_logits_indices = top_k_logits_indices.tolist()
            top_k_tokens = [tokenizer.decode(tid) for tid in top_k_logits_indices]

            # Derive metrics
            modelToken = top_k_tokens[0].strip()
            modelEval = f"{top_k_logits_values[0]:.4f}"
            modelEval = round(float(modelEval)*100)
            NextEval = (f"{top_k_logits_values[1]:.4f}" if MultiPV > 1 else "n.a.")
            NextEval = round(float(NextEval)*100) if MultiPV > 1 and isinstance(top_k_logits_values[1], float) else "n.a."

            # Build lines for this turn
            current_stem = initial_prompt + " " + " ".join(processed_words[:turnCount])
            lines = [
                f'[PID "{promptID}"]\n',
                f'[EID "{ORIGINAL_MODEL_NAME}"]\n',
                f'[Turn "{turnCount}-w"]\n',
                f'[TextToken "{current_word}:"]\n',
                f'[ModelToken "{modelToken}:"]\n',
                f'[Eval "{modelEval}"]\n',
                f'[PrevEval "{PrevEval}"]\n',
                f'[NextEval "{NextEval}"]\n',
                f'[Depth "{Depth}"]\n',
                f'[STEM "{current_stem}"]\n',
                f'[NumLegalMoves "{LegalNumberOfMove}"]\n',
                "---------------\n",
                f"{DEPTH_RANGE}\n",
                "---------------\n"
            ]
            for token_str, logit_val in zip(top_k_tokens, top_k_logits_values):
                lines.append(f"{token_str.strip()}: {logit_val:.4f}\n")
            lines.append("===========================================================================================================\n\n")
            lines.append(f"[Comments]\n")
            lines.append(f"[EndMove]\n\n")

            # --- NEW: Add to buffer and check size ---
            turn_output_string = ''.join(lines)
            # Estimate size in bytes (UTF-8 encoding)
            turn_output_bytes = len(turn_output_string.encode('utf-8'))

            output_buffer.append(turn_output_string)
            current_buffer_size_bytes += turn_output_bytes

            # Check if buffer exceeds the memory limit
            if current_buffer_size_bytes >= BUFFER_MEMORY_LIMIT_BYTES:
                print(f"--- Flushing buffer ({current_buffer_size_bytes / (1024*1024):.2f} MB exceeds limit)... ---")
                try:
                    # Open in append mode ('a') to add to the existing file
                    with open(OUTPUT_FILE, 'a', encoding='utf-8') as writer:
                        writer.write("".join(output_buffer)) # Write concatenated buffered strings
                    # Reset buffer
                    output_buffer = []
                    current_buffer_size_bytes = 0
                    print(f"--- Buffer flushed to {OUTPUT_FILE}. ---")
                except IOError as e:
                    print(f"Error writing buffer to {OUTPUT_FILE}: {e}")
                    # Optional: Decide whether to continue or exit if writing fails
                    print("Write to fle failed: Exiting. Re run batch")
                    exit()
            # --- End Buffer Handling ---

            # Update state
            PrevEval = modelEval

            # Status update
            status_interval = min(100, num_words_to_process // 2 if num_words_to_process >= 10 else 10)
            if turnCount % status_interval == 0 or turnCount == num_words_to_process :
                elapsed = time.time() - start_time
                rate = turnCount / elapsed if elapsed > 0 else 0
                current_total_tokens = len(initial_prompt_ids) + sum(len(tk) for tk in tokenized_words[:turnCount])
                token_rate = current_total_tokens / elapsed if elapsed > 0 else 0
                print(f"  Status: Processed {turnCount}/{num_words_to_process} words at {rate:.2f} w/s ({token_rate:.2f} tok/s) ({elapsed:.2f}s total)")
                # Add buffer status to the printout
                print(f"          Current buffer size: {current_buffer_size_bytes / (1024*1024):.2f} MB\n")


        # --- NEW: Final Flush After Loop ---
        if output_buffer: # Check if there's anything left in the buffer
            print(f"--- Flushing remaining buffer ({current_buffer_size_bytes / (1024*1024):.2f} MB)... ---")
            try:
                with open(OUTPUT_FILE, 'a', encoding='utf-8') as writer:
                    writer.write("".join(output_buffer))
                output_buffer = [] # Clear buffer after final write
                current_buffer_size_bytes = 0
                print(f"--- Final buffer contents flushed to {OUTPUT_FILE}. ---")
            except IOError as e:
                print(f"Error writing final buffer to {OUTPUT_FILE}: {e}")
        # --- End Final Flush ---

        print("  Finished processing requested words.\n")

    else:
        print("Skipping main generation loop as there are no words to process.")


    #%% Final Stats
    print("Step 10: Reporting final statistics...")
    total_time = time.time() - start_time
    avg_rate_words = (num_words_to_process / total_time) if total_time > 0 and num_words_to_process > 0 else 0
    final_total_tokens = len(initial_prompt_ids) + sum(len(tk) for tk in tokenized_words[:num_words_to_process])
    avg_rate_tokens = (final_total_tokens / total_time) if total_time > 0 and final_total_tokens > 0 else 0

    print(f"  Total turns processed: {num_words_to_process}")
    print(f"  Total tokens processed (including prompt): {final_total_tokens}")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Average speed: {avg_rate_words:.2f} words/second")
    print(f"  Average speed: {avg_rate_tokens:.2f} tokens/second")
    print(f"  Output written to {OUTPUT_FILE}")

    # Optional: Clean up memory
    print("\nCleaning up resources...")
    del model
    del tokenizer
    del tokenized_words
    if 'initial_input_ids' in locals():
        del initial_input_ids
    if 'input_ids' in locals():
        del input_ids
    if 'outputs' in locals():
        del outputs
    if 'past_key_values' in locals():
        del past_key_values
    if 'output_buffer' in locals():
        del output_buffer # Clean up the buffer list
    gc.collect()

    print("\nScript finished.")

#### Main Call ####

if __name__ == "__main__":

     parser = argparse.ArgumentParser(
         description="LifGenerator for CPU with Hugging face models with greedy decoding",
         epilog="Help Documentation"
     )

     parser.add_argument(
         "-input_file", "-i",
         type=str,
         help="The path to the input file."
     )

     parser.add_argument(
         "-output_file", "-o",
         type=str,
         help="Name and path of output file"
     )

     parser.add_argument(
         "-prompt_id", "-pid",
         type=str,
         help="Overall name of item"
     )

     parser.add_argument(
         "-prompt_topic", "-pt",
         type=str,
         help="Topic given to LLM before stem words"
     )

     parser.add_argument(
         "-multi_pv", "-mpv",
         type=int,
         help="Number of options to consider at each turn"
     )

     parser.add_argument(
         "-num_items", "-nt",
         type=int,
         help="Max # of items to generate"
     )

     parser.add_argument(
         "-alpha_mode", "-a",
         type=int,
         help="0 = all tokens, up thru 4 = alpha chars plus ' only"
     )
     args = parser.parse_args()
     print("Welcome to the LifGenerator CPU script!")
     print("This script generates lif files using a Hugging Face model and greedy decoding.")
     print(f"Input file path: {args.input_file}")
     print(f"Output file path: {args.output_file}")
     INPUT_FILE = args.input_file
     INPUT_FILE_STEM = INPUT_FILE.split('.')[0]
     OUTPUT_FILE = args.output_file if args.output_file else (INPUT_FILE_STEM + ".lif")
     PROMPT_ID = args.prompt_id if args.prompt_id else INPUT_FILE
     PROMPT_TOPIC = args.prompt_topic if args.prompt_topic else INPUT_FILE
     MULTI_PV = args.multi_pv if args.multi_pv else 100
     NUM_ITEMS = args.num_items if args.num_items else 700
     ALPHA_MODE = args.alpha_mode if args.alpha_mode else 0
     main(INPUT_FILE, OUTPUT_FILE, PROMPT_ID, PROMPT_TOPIC, MULTI_PV,NUM_ITEMS, ALPHA_MODE)