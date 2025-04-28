# %%
import torch
import gc
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import argparse
import sys

def main(INPUT_FILE,OUTPUT_FILE):
# %% Constants and Configuration
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    MODEL_NAME = input(f"Enter hugging face model name or press enter to default to [{MODEL_NAME}]: ") or MODEL_NAME
    DEVICE = "cpu"
    TORCH_DTYPE = torch.float32
    DEPTH_RANGE = 1
    # Ensure INPUT_FILE path is correct for your environment
    # INPUT_FILE = 'feed.txt'  # Assuming it's in the same directory or provide full path
    # Create the input file if it doesn't exist for testing
    if not os.path.exists(INPUT_FILE):
        print(f"Warning: Input file '{INPUT_FILE}' not found. Creating a dummy file.")
        with open(INPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("The quick brown fox jumps over the lazy dog")

    # OUTPUT_FILE = "output.lif"  # Changed output filename
    MODEL_CONTEXT_WINDOW = 128_000  # Example context window, adjust if needed for the actual model
    SAFETY_THRESHOLD = 2_000
    MAX_INPUT_TOKENS = MODEL_CONTEXT_WINDOW - SAFETY_THRESHOLD  # Max tokens per model *input slice*

    # %% Load and Quantize Model & Tokenizer
    print("Step 1: Loading model...")
    # Add trust_remote_code=True if necessary for the specific model architecture
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=TORCH_DTYPE,
        trust_remote_code=True  # Often needed for Qwen-based models
    ).to(DEVICE)
    print(f"  Model loaded to {DEVICE}.")

    print("Step 2: Applying dynamic quantization for faster CPU inference...")
    # Note: Quantization might slightly affect raw logit values compared to fp32/fp16
    model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    model.eval()
    print("  Quantization complete. Model is ready for inference.\n")

    print("Step 3: Loading tokenizer...")
    # Add trust_remote_code=True if necessary for the specific model architecture
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        print("  Tokenizer missing pad token; setting pad_token = eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        # Important: Ensure model config also reflects this if needed by generation args
        if hasattr(model, 'config'):
            model.config.pad_token_id = tokenizer.eos_token_id
    print("  Tokenizer loaded and configured.\n")

    # %% User Inputs
    print("Step 4: Prompting user for inputs...")
    # Use default values for easier testing
    promptID = input("  Enter Prompt ID [Default: VanityTestGreedy]: ") or "VanityTestGreedy"
    MultiPV_str = input("  Enter MultiPV (top logits to show) [Default: 5]: ") or "5"
    MultiPV = int(MultiPV_str)  # Now only controls how many top logits to display
    LegalNumberOfMove_str = input("  Enter Max Number of moves [Default: 10]: ") or "10"
    LegalNumberOfMove = int(LegalNumberOfMove_str)
    EngineID = f"DeepSeek R1 1.5B Qwen-Distil Greedy ({DEVICE.upper()})"  # Updated EngineID
    Depth = 1
    print("  User inputs captured.\n")

    # %% Pre-tokenize entire relevant input sequence
    print("Step 5: Pre-tokenizing input sequence...")
    initial_prompt = "Complete successive parts of a sentence given one word at a time:"
    initial_prompt_ids = tokenizer.encode(initial_prompt, add_special_tokens=False)

    # Pre-load words from file
    print(f"  Reading words from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            words_from_file = f.read().split()
        print(f"  Found {len(words_from_file)} words.")
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found. Exiting.")
        exit()

    all_tokens = list(initial_prompt_ids)
    word_end_indices = [len(initial_prompt_ids)]  # Index *after* the last token of each word (or initial prompt)
    processed_words = []  # Store the actual words processed

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

    # %% Build file header
    print("Step 8: Preparing output file header...")
    header_lines = [
        f'[PromptID "{promptID}"]\n',
        f'[EngineID "{EngineID}"]\n',
        f'[MultiPV "{MultiPV}"]\n',  # MultiPV now just refers to displayed logits
        f'[DepthRange "1:1"]\n\n',
        "1-0\n"
    ]
    print("  Header prepared.\n")

    # %% Main Generation Loop (Using Slicing & Greedy Decoding)
    print("Step 9: Entering main generation loop (using pre-tokenized slicing and greedy decoding)...\n")
    PrevEval = "n.a."
    start_time = time.time()

    if num_words_to_process > 0:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as writer:
            print("  Writing header to output file...")
            writer.write(''.join(header_lines))
            print("  Header written. Starting word-by-word prediction.\n")

            for turnCount in range(1, num_words_to_process + 1):
                current_word = processed_words[turnCount - 1]
                print(f"Turn {turnCount}: Predicting after word '{current_word}'")

                slice_end_index = word_end_indices[turnCount - 1]
                slice_start_index = max(0, slice_end_index - MAX_INPUT_TOKENS)
                print(f"  9.1/9.2: Context slice indices: [{slice_start_index}:{slice_end_index}]")

                input_tensor = full_token_tensor[:, slice_start_index:slice_end_index]
                current_input_len = input_tensor.shape[1]
                print(f"  9.3: Sliced input tensor shape: {input_tensor.shape}")

                input_tensor_dev = input_tensor.to(DEVICE)

                start_time_gen = time.time()
                # 9.4 Generate next token using GREEDY DECODING
                print(f"  9.4: Running model.generate() with {current_input_len} input tokens (Greedy Decoding)...")
                with torch.no_grad():
                    outputs = model.generate(
                        input_tensor_dev,
                        max_new_tokens=1,
                        min_new_tokens=1,  # Explicitly require 1 new token
                        output_scores=True,  # Get logits
                        return_dict_in_generate=True,  # Get dict output
                        do_sample=False,  # Disable sampling -> Use Greedy Decoding
                        pad_token_id=tokenizer.pad_token_id
                        # Removed num_beams and num_return_sequences
                    )
                end_time_gen = time.time()
                gen_duration = end_time_gen - start_time_gen
                print(f"    Model generation took: {gen_duration:.4f} seconds")

                # ----- UPDATED LOGIC for TopK Logits (Greedy Path) -----
                # outputs.scores is a tuple of length max_new_tokens (1)
                # Each element is a tensor of shape [batch_size, vocab_size] (batch_size is 1 here)
                logits_for_step = outputs.scores[0]  # Logits for the single generated token step. Shape: [1, vocab_size]

                # Get the logits from the single batch item (greedy path)
                logits_for_greedy_path = logits_for_step[0]  # Shape: [vocab_size]

                # Get the top K (MultiPV) logits and their corresponding token IDs
                # Note: The highest logit corresponds to the token chosen by greedy decoding
                top_k_logits_values, top_k_logits_indices = torch.topk(
                    logits_for_greedy_path, k=MultiPV, dim=-1
                )

                # Convert results to lists
                top_k_logits_values = top_k_logits_values.tolist()
                top_k_logits_indices = top_k_logits_indices.tolist()

                # Decode the top K tokens based on logits
                top_k_tokens = [tokenizer.decode(tid) for tid in top_k_logits_indices]

                print(f"    Top {MultiPV} Logits from greedy path (Token | Logit Value):")
                for i in range(MultiPV):
                    token_str_cleaned = top_k_tokens[i].strip()
                    print(f"     - '{token_str_cleaned}': {top_k_logits_values[i]:.4f} (ID: {top_k_logits_indices[i]})")

                # The token actually generated by greedy decoding
                greedy_selected_token_id = outputs.sequences[0, -1].item()  # Last token in the sequence
                greedy_selected_token_str = tokenizer.decode(greedy_selected_token_id).strip()
                # This will always match top_k_tokens[0] because do_sample=False
                # print(f"    (Greedy search selected token: '{greedy_selected_token_str}' ID: {greedy_selected_token_id})") # Optional confirmation
                # ----- END of UPDATED LOGIC -----

                # 9.5 Derive primary metrics USING THE TOP LOGITS
                # modelToken is the token with the highest logit (chosen by greedy)
                modelToken = top_k_tokens[0].strip()  # Equivalent to greedy_selected_token_str
                # modelEval is the highest logit value
                modelEval = f"{top_k_logits_values[0]:.4f}"
                modelEval = round(float(modelEval) * 100)
                # NextEval is the second highest logit value
                NextEval = (f"{top_k_logits_values[1]:.4f}" if MultiPV > 1 else "n.a.")
                NextEval = round(float(NextEval) * 100) if MultiPV > 1 else "n.a."
                print(
                    f"  9.5: Top token (greedy choice): '{modelToken}' (Evalution: {modelEval})|Logit value : {top_k_logits_values[0]:.4f}| Next best Eval: {NextEval} | Logit ")

                # 9.6 Build lines for this turn
                print("  9.6: Building output lines for this turn...")
                current_stem = " ".join(processed_words[:turnCount])
                lines = [
                    f'[PID "{promptID}"]\n',
                    f'[EID "{MODEL_NAME}"]\n',
                    f'[Turn "{turnCount}-w"]\n',
                    f'[TextToken "{current_word}:"]\n',
                    f'[ModelToken "{modelToken}:"]\n',  # The model's greedy prediction
                    f'[Eval "{modelEval}"]\n',  # The highest raw logit value
                    f'[PrevEval "{PrevEval}"]\n',
                    f'[NextEval "{NextEval}"]\n',  # The second highest raw logit value
                    f'[Depth "{Depth}"]\n',
                    f'[STEM "{current_stem}"]\n',
                    f'[NumLegalMoves "{LegalNumberOfMove}"]\n',
                    "---------------\n",
                    f"{DEPTH_RANGE}\n",
                    "---------------\n"
                ]
                # Append the list of top K tokens and their raw logits
                for token_str, logit_val in zip(top_k_tokens, top_k_logits_values):
                    lines.append(f"{token_str.strip()}: {logit_val:.4f}\n")

                lines.append(
                    "===========================================================================================================\n\n")
                lines.append(f"[Comments]\n")
                lines.append(f"[EndMove]\n\n")

                print("    Lines built.")

                # 9.7 Write to file
                print("  9.7: Writing lines to output file...")
                writer.write(''.join(lines))
                print("    Write complete.\n")

                # 9.8 Update state
                PrevEval = modelEval

                # 9.9 Status update
                status_interval = min(100, num_words_to_process // 2 if num_words_to_process >= 10 else 10)
                if turnCount % status_interval == 0 or turnCount == num_words_to_process:
                    elapsed = time.time() - start_time
                    rate = turnCount / elapsed if elapsed > 0 else 0
                    print(
                        f"  Status: Processed {turnCount}/{num_words_to_process} words at {rate:.2f} w/s ({elapsed:.2f}s total)\n")

            print("  Finished processing requested words.\n")

    else:
        print("Skipping main generation loop as there are no words to process.")

    # %% Final Stats
    print("Step 10: Reporting final statistics...")
    total_time = time.time() - start_time
    avg_rate = (num_words_to_process / total_time) if total_time > 0 and num_words_to_process > 0 else 0
    print(f"  Total turns processed: {num_words_to_process}")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Average speed: {avg_rate:.2f} words/second")
    print(f"  Output written to {OUTPUT_FILE}")

    # Optional: Clean up memory
    print("\nCleaning up resources...")
    del model
    del tokenizer
    del full_token_tensor
    if 'outputs' in locals():
        del outputs
    if 'input_tensor' in locals():
        del input_tensor
    if 'input_tensor_dev' in locals():
        del input_tensor_dev
    gc.collect()
    if DEVICE == 'cuda':
        print("Emptying CUDA cache...")
        torch.cuda.empty_cache()
    print("\nScript finished.")

### RUN MAIN ####

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

    args = parser.parse_args()
    print("Welcome to the LifGenerator CPU script!")
    print("This script generates lif files using a Hugging Face model and greedy decoding.")
    print(f"Input file path: {args.input_file}")
    print(f"Output file path: {args.output_file}")
    main(args.input_file, args.output_file)