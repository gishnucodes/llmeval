# %%
# Ensure necessary libraries are installed:
# pip install optimum[openvino] openvino-dev transformers>=4.36.0 torch>=2.1.0 accelerate huggingface_hub sentencepiece protobuf<=3.20.3 numpy

import numpy as np
import openvino as ov
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, AutoConfig, GenerationConfig
import torch
import gc
import time
import os
import io
import argparse
import sys
import math

# --- NEW: Buffer Configuration ---
BUFFER_MEMORY_LIMIT_MB = 200 # MB
BUFFER_MEMORY_LIMIT_BYTES = BUFFER_MEMORY_LIMIT_MB * 1024 * 1024
print(f"Using output buffer with memory limit: {BUFFER_MEMORY_LIMIT_MB} MB ({BUFFER_MEMORY_LIMIT_BYTES} bytes)")
# --- End Buffer Configuration ---
OPENVINO_MODEL_DIR = "openvino_model_int8/" # Default local model directory

# %% Constants and Configuration
# **** MODIFIED main function signature: removed MULTI_PV ****
def main(INPUT_FILE, OUTPUT_FILE, PROMPT_ID, PROMPT_TOPIC, NUM_ITEMS, ALPHA_MODE, NUM_BEAMS):
    DEVICE = "AUTO"
    print(f"Attempting to use OpenVINO device: {DEVICE}")

    # Use the local directory specified above by default
    ORIGINAL_MODEL_NAME = OPENVINO_MODEL_DIR
    DEPTH_RANGE = 1

    # **** Beam Search Info ****
    if NUM_BEAMS <= 0:
        print("Warning: num_beams must be at least 1. Setting num_beams=1 (greedy search).")
        NUM_BEAMS = 1
    if NUM_BEAMS == 1:
        print("Info: num_beams = 1. Using greedy search via generate().")
    else:
        print(f"Using Beam Search with num_beams: {NUM_BEAMS}")
        print(f"Number of returned sequences will also be: {NUM_BEAMS}")

    if not os.path.exists(INPUT_FILE):
        print(f"Warning: Input file '{INPUT_FILE}' not found. Creating a dummy file.")
        with open(INPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("The quick brown fox jumps over the lazy dog")

    # %% Load OpenVINO Model & Tokenizer
    print("Step 1 & 2: Loading OpenVINO model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_NAME, trust_remote_code=True)
        print("  Tokenizer loaded successfully.")
        model = OVModelForCausalLM.from_pretrained(
            ORIGINAL_MODEL_NAME,
            device=DEVICE,
            trust_remote_code=True,
        )
        print(f"  OpenVINO Model loaded and ready for device {model.device}.")
    except Exception as e:
        print(f"ERROR: Failed to load OpenVINO model or tokenizer from {ORIGINAL_MODEL_NAME}")
        print(f"Ensure the directory/name is correct and contains necessary files (e.g., openvino_model.xml/bin, config.json, tokenizer files).")
        print(f"Error details: {e}")
        exit()

    if tokenizer.pad_token is None:
        print("  Tokenizer missing pad token; setting pad_token = eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(model, 'config'):
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        print("Warning: Could not set pad_token_id on model.config. Generation might fail if padding is needed.")
    print("  Model and Tokenizer ready for inference.\n")

    # %% User Inputs
    promptID = PROMPT_ID
    # **** MultiPV is now directly NUM_BEAMS ****
    LegalNumberOfMove = NUM_ITEMS
    # **** Updated EngineID to reflect num_beams is also the return count ****
    EngineID = f"{ORIGINAL_MODEL_NAME} OpenVINO ({model.device}) BeamSearch(n={NUM_BEAMS}, ret={NUM_BEAMS})"
    Depth = 1

    # %% Pre-process input sequence
    print("Step 5: Pre-processing input sequence...")
    initial_prompt = "Complete successive parts of a sentence given one word at a time for the topic : " + PROMPT_TOPIC  + ": "
    initial_prompt_ids_list = tokenizer.encode(initial_prompt, add_special_tokens=False)

    print(f"  Reading words from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            words_from_file = f.read().split()
        print(f"  Found {len(words_from_file)} words.")
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found. Exiting.")
        exit()

    tokenized_words = []
    total_token_count = len(initial_prompt_ids_list)
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
    else:
        print("Warning: Could not determine model context window from config.")

    num_words_to_process = min(len(processed_words), LegalNumberOfMove)
    if num_words_to_process < len(processed_words):
        print(f"  Will process the first {num_words_to_process} words due to LegalNumberOfMove limit.\n")
    elif num_words_to_process == 0:
        print("  Warning: No words to process based on input file or limits.\n")

    # %% Build and Write File Header Separately
    print("Step 8: Preparing and writing output file header...")
    header_lines = [
        f'[PromptID "{promptID}"]\n',
        f'[EngineID "{EngineID}"]\n',
        # **** [MultiPV] line now uses NUM_BEAMS ****
        f'[MultiPV "{NUM_BEAMS}"]\n',
        f'[DepthRange "1:1"]\n\n',
        "1-0\n"
    ]
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as writer:
            writer.write(''.join(header_lines))
        print(f"  Header written to {OUTPUT_FILE}.\n")
    except IOError as e:
        print(f"Error writing header to {OUTPUT_FILE}: {e}")
        exit()

    # %% Main Generation Loop
    print(f"Step 9: Entering main generation loop (using model.generate with num_beams={NUM_BEAMS}, returning {NUM_BEAMS} sequences)...\n")
    PrevEval = "n.a."
    start_time = time.time()
    output_buffer = []
    current_buffer_size_bytes = 0
    current_sequence_ids = list(initial_prompt_ids_list)

    if num_words_to_process > 0:
        for turnCount in range(1, num_words_to_process + 1):
            current_word = processed_words[turnCount - 1]
            current_word_tokens = tokenized_words[turnCount - 1]
            current_sequence_ids.extend(current_word_tokens)
            input_ids = torch.tensor([current_sequence_ids], dtype=torch.long)

            if model_context_window and input_ids.shape[1] >= model_context_window:
                 print(f"WARNING: Input sequence length ({input_ids.shape[1]}) is reaching or exceeding model context window ({model_context_window}) at turn {turnCount}. Further results may be unreliable.")

            start_time_gen = time.time()
            try:
                outputs = model.generate(
                    input_ids,
                    generation_config=GenerationConfig(
                        max_new_tokens=1,
                        num_beams=NUM_BEAMS,
                        # **** num_return_sequences is now directly NUM_BEAMS ****
                        num_return_sequences=NUM_BEAMS,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        early_stopping=True if NUM_BEAMS > 1 else False,
                        output_scores=True,
                        return_dict_in_generate=True
                    )
                )
            except Exception as e:
                print(f"\nERROR during model.generate at turn {turnCount} with input length {input_ids.shape[1]}:")
                print(f"Input sequence (decoded sample): {tokenizer.decode(input_ids[0, -50:])}")
                print(f"Error details: {e}")
                print("Attempting to continue to the next turn...")
                PrevEval = "error"
                continue

            gen_duration = time.time() - start_time_gen

            generated_sequences = outputs.sequences
            sequence_scores = outputs.sequences_scores # Log-probabilities
            newly_generated_token_ids = generated_sequences[:, input_ids.shape[1]]
            top_k_tokens = [tokenizer.decode(token_id) for token_id in newly_generated_token_ids]
            top_k_scores_list = sequence_scores.tolist() # List of log-probs

            if not top_k_tokens: # Handle case where generate might fail to return sequences
                print(f"Warning: No tokens generated at turn {turnCount}. Skipping output for this turn.")
                PrevEval = "no_output"
                continue

            # Derive metrics from the top result (index 0)
            modelToken = top_k_tokens[0].strip()
            modelEvalLogProb = top_k_scores_list[0]
            modelEval = f"{modelEvalLogProb:.4f}"

            # Get the score for the second-best beam if available
            # **** Condition now checks NUM_BEAMS ****
            if NUM_BEAMS > 1 and len(top_k_scores_list) > 1:
                NextEvalLogProb = top_k_scores_list[1]
                NextEval = f"{NextEvalLogProb:.4f}"
            else:
                NextEval = "n.a."

            # Build lines for this turn
            current_stem_text = tokenizer.decode(current_sequence_ids)
            lines = [
                f'[PID "{promptID}"]\n',
                f'[EID "{EngineID}"]\n',
                f'[Turn "{turnCount}-w"]\n',
                f'[TextToken "{current_word}:"]\n',
                f'[ModelToken "{modelToken}:"]\n',
                f'[Eval "{modelEval}"]\n', # Log-probability of top sequence
                f'[PrevEval "{PrevEval}"]\n',
                f'[NextEval "{NextEval}"]\n', # Log-probability of 2nd sequence
                f'[Depth "{Depth}"]\n',
                f'[STEM "{current_stem_text}"]\n',
                f'[NumLegalMoves "{LegalNumberOfMove}"]\n', # This might be less relevant now? Keep for format consistency.
                "---------------\n",
                f"{DEPTH_RANGE}\n",
                "---------------\n"
            ]
            # Add all returned tokens and their scores (up to NUM_BEAMS)
            for token_str, score_val in zip(top_k_tokens, top_k_scores_list):
                lines.append(f"{token_str.strip()}: {score_val:.4f}\n") # Use log-prob scores
            lines.append("===========================================================================================================\n\n")
            lines.append(f"[Comments]\n")
            lines.append(f"[EndMove]\n\n")

            # --- Add to buffer and check size ---
            turn_output_string = ''.join(lines)
            turn_output_bytes = len(turn_output_string.encode('utf-8'))
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
                    print("Write to file failed: Exiting. Re-run batch")
                    exit()
            # --- End Buffer Handling ---

            PrevEval = modelEval

            # Status update
            status_interval = min(100, num_words_to_process // 2 if num_words_to_process >= 10 else 10)
            if turnCount % status_interval == 0 or turnCount == num_words_to_process :
                elapsed = time.time() - start_time
                rate = turnCount / elapsed if elapsed > 0 else 0
                print(f"  Status: Processed {turnCount}/{num_words_to_process} words at {rate:.2f} w/s. Last step: {gen_duration:.4f}s. ({elapsed:.2f}s total)")
                print(f"          Current buffer size: {current_buffer_size_bytes / (1024*1024):.2f} MB\n")

        # --- Final Flush After Loop ---
        if output_buffer:
            print(f"--- Flushing remaining buffer ({current_buffer_size_bytes / (1024*1024):.2f} MB)... ---")
            try:
                with open(OUTPUT_FILE, 'a', encoding='utf-8') as writer:
                    writer.write("".join(output_buffer))
                output_buffer = []
                current_buffer_size_bytes = 0
                print(f"--- Final buffer contents flushed to {OUTPUT_FILE}. ---")
            except IOError as e:
                print(f"Error writing final buffer to {OUTPUT_FILE}: {e}")
        # --- End Final Flush ---
        print("  Finished processing requested words.\n")
    else:
        print("Skipping main generation loop as there are no words to process.")

    # %% Final Stats
    print("Step 10: Reporting final statistics...")
    total_time = time.time() - start_time
    avg_rate_words = (num_words_to_process / total_time) if total_time > 0 and num_words_to_process > 0 else 0
    print(f"  Total turns processed: {num_words_to_process}")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Average speed: {avg_rate_words:.2f} words/second")
    print(f"  Output written to {OUTPUT_FILE}")

    # Optional: Clean up memory
    print("\nCleaning up resources...")
    del model
    del tokenizer
    del tokenized_words
    del current_sequence_ids
    if 'input_ids' in locals(): del input_ids
    if 'outputs' in locals(): del outputs
    if 'output_buffer' in locals(): del output_buffer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\nScript finished.")

# %% Main Call ####
if __name__ == "__main__":
     parser = argparse.ArgumentParser(
         description="LifGenerator using OpenVINO/Optimum with Beam Search",
         epilog="Generates .lif files predicting next tokens turn-by-turn. The number of beams specified is used for both search and the number of returned sequences."
     )

     # Existing arguments... (Removed -multi_pv)
     parser.add_argument( "-input_file", "-i", type=str, required=True, help="The path to the input file.")
     parser.add_argument( "-output_file", "-o", type=str, help="Name and path of output file (defaults to input_stem.lif)")
     parser.add_argument( "-prompt_id", "-pid", type=str, help="Overall name of item (defaults to input filename stem)")
     parser.add_argument( "-prompt_topic", "-pt", type=str, default="general text", help="Topic given to LLM before stem words")
     parser.add_argument( "-num_items", "-nt", type=int, default=700, help="Max # of items (words) to process from input")
     parser.add_argument( "-alpha_mode", "-a", type=int, default=0, help="0 = all tokens, etc. (Note: Script doesn't currently use this arg)")

     # Beam Search Argument (No MultiPV anymore)
     parser.add_argument(
         "-num_beams", "-nb",
         type=int,
         default=1, # Default to 1 (greedy)
         help="Number of beams for beam search. This also determines the number of sequences returned. Set > 1 for beam search."
     )

     args = parser.parse_args()
     print("Welcome to the LifGenerator OpenVINO script with Beam Search!")
     print(f"Input file path: {args.input_file}")
     print(f"Number of beams (and returned sequences): {args.num_beams}") # Updated print

     INPUT_FILE = args.input_file
     INPUT_FILE_STEM = os.path.splitext(os.path.basename(INPUT_FILE))[0]

     OUTPUT_FILE = args.output_file if args.output_file else (INPUT_FILE_STEM + ".lif")
     PROMPT_ID = args.prompt_id if args.prompt_id else INPUT_FILE_STEM
     PROMPT_TOPIC = args.prompt_topic
     # **** No MULTI_PV assignment needed ****
     NUM_ITEMS = args.num_items
     ALPHA_MODE = args.alpha_mode
     NUM_BEAMS = args.num_beams # Get value from args

     # **** No validation needed between MultiPV and NumBeams ****

     print(f"Effective settings: PromptID='{PROMPT_ID}', Topic='{PROMPT_TOPIC}', NumItems={NUM_ITEMS}, NumBeams={NUM_BEAMS}")

     # **** MODIFIED main call: removed MULTI_PV ****
     main(INPUT_FILE, OUTPUT_FILE, PROMPT_ID, PROMPT_TOPIC, NUM_ITEMS, ALPHA_MODE, NUM_BEAMS)