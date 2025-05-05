# %% File "lifgen2.py" by Gishnu Madhu with editing and feature input by KWR
# Usage:
# python lifgen2.py -i <source-file> -o <output.lif> -pid <identifier> -pt <initial prompt> -mpv <# of scored items>
# optional: -nt <# of tokens to search for text word> -bw <width of beam search, 1 for greedy>, -nw <# of words>
#           -a <mode in 0...4 of treating tokens as in-bounds or matching>
#           -model <model to use>
# Example:
# python lifgen2.py -i YaoJokic.txt -o YaoJokicm50.lif -pid YaoTestByDeepSeek -pt "Compare Yao Ming and Nikola Jokic in NBA basketball" -mpv 50
import math

import torch
import gc
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import argparse
import sys
import re
from huggingface_hub import login
import os

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
    "-num_words", "-nw",
    type=int,
    help="Cap on # of text words to iterate"
)

parser.add_argument(
    "-num_tokens", "-nt",
    type=int,
    help="# of tokens to search for text word match"
)

parser.add_argument(
    "-beam_width", "-bw",
    type=int,
    help="Width of beam search, 0 or 1 for greedy"
)

parser.add_argument(
    "-alpha_mode", "-a",
    type=int,
    help="0 = all tokens, up thru 4 = alpha chars plus ' only"
)

parser.add_argument(
    "-model", "-model",
    type=str,
    help="DS for DeepSeek, DSVINO for DeepSeek by OpenVino"
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
NUM_WORDS = args.num_words if args.num_words else 1000
NUM_TOKENS = args.num_tokens if args.num_tokens else 100000
BEAM_WIDTH = args.beam_width if args.beam_width else 1
ALPHA_MODE = args.alpha_mode if args.alpha_mode else 0
MODEL_TAG = args.model if args.model else "DSVINO"
# main(INPUT_FILE, OUTPUT_FILE, PROMPT_ID, PROMPT_TOPIC, MULTI_PV, NUM_WORDS, NUM_TOKENS, BEAM_WIDTH, ALPHA_MODE, MODEL_TAG)

"""
Match if arg occurs in st surrounded by ends or non-alpha chars.

Intent is e.g. for "Karp" to match "Karp, R" but not "Karpov".
Whether "Karp" matches "Karp-Lipton" depends on whether hyphen is part of name.
Works even if arg itself has non-alpha characters.
Used for player and event names AND to identify tokens in command streams.
Uses C++ "isalpha" for local definition of names.
Prefer to override it to count underscore as a non-delimiting char.
Hyphen is always part of tokens but can be used to delimit place and person names,
so "Khanty" and "Khanty-Mansiysk" can both match "Khanty-Mansiysk" and
"Vachier" can match "Vachier-Lagrave".

With LLM tokens, this allows arg="abc" to match st=" abc" but not vice-versa.
However, if called with arg.strip() then vice-versa is fine.
If the token is @-@ then it will match "--" but NOT match a hyphenated word.
"""


def borderedMatch(arg, st, hyphenDelimits=False, underscoreDelimits=False):
    fromPos = st.find(arg)
    while fromPos != -1:
        leftOK = (fromPos == 0)
        if (fromPos > 0):
            c = st[fromPos - 1]
            if c == '-':
                leftOK = hyphenDelimits
            elif c == '_':
                leftOK = underscoreDelimits
            else:
                leftOK = (not c.isalnum())

        rightEdge = fromPos + len(arg)
        rightOK = (rightEdge == len(st))
        if (not rightOK):
            d = st[rightEdge]
            if d == '-':
                rightOK = hyphenDelimits
            elif d == '_':
                rightOK = underscoreDelimits
            else:
                rightOK = (not d.isalnum())

        if rightOK and leftOK:
            return True
        else:  # try to find another match
            fromPos = st.find(arg, fromPos + 1)

    return False

hf_token = input("Enter your Huggingface token")
# Or better:
# hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")

if hf_token:
    print("Logging in to Hugging Face Hub...")
    login(token=hf_token)
else:
    print("HF Token not found. Gated model download might fail.")


def main(INPUT_FILE, OUTPUT_FILE, PROMPT_ID, PROMPT_TOPIC, MULTI_PV, NUM_WORDS, NUM_TOKENS, BEAM_WIDTH, ALPHA_MODE,
         MODEL_TAG):
    # %% Constants and Configuration
    # MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    # MODEL_NAME = "google/gemma-3-4b-it"
    MODEL_NAME = "Qwen/Qwen3-1.7B"
    # MODEL_NAME = input(f"Enter hugging face model name or press enter to default to [{MODEL_NAME}]: ") or MODEL_NAME
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

    # print("Step 2: Applying dynamic quantization for faster CPU inference...")
    # Note: Quantization might slightly affect raw logit values compared to fp32/fp16
    # model = torch.quantization.quantize_dynamic(
    #     model,
    #     {torch.nn.Linear},
    #     dtype=torch.qint8
    # )
    hook_handle = model.lm_head.register_forward_hook(capture_logits_hook)

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
    promptID = PROMPT_ID  # input("  Enter Prompt ID [Default: VanityTestGreedy]: ") or "VanityTestGreedy"
    # MultiPV_str = input("  Enter MultiPV (top logits to show) [Default: 5]: ") or "5"
    MultiPV = MULTI_PV  # int(MultiPV_str)  # Now only controls how many top logits to display
    # LegalNumberOfMove_str = input("  Enter Max Number of moves [Default: 10]: ") or "10"
    LegalNumberOfMove = NUM_WORDS  # int(LegalNumberOfMove_str)
    EngineID = f"DeepSeek R1 1.5B Qwen-Distil Greedy ({DEVICE.upper()})"  # Updated EngineID
    # EngineID = f"Gemma-3-4b-it ({DEVICE.upper()})" # Indicate CPU in EngineID
    Depth = 1
    print("  User inputs captured.\n")

    # %% Pre-tokenize entire relevant input sequence
    print("Step 5: Pre-tokenizing input sequence...")
    initial_prompt = "Complete successive parts of a sentence given one word at a time:"
    initial_prompt_ids = tokenizer.encode(initial_prompt, add_special_tokens=False)

    print(f"  Reading words from {INPUT_FILE}...")
    lines = []
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            # words_from_file = f.read().split()
            lines = f.readlines()
            words_from_file = "".join(line.replace('\n', '') for line in lines)
            wordList = re.split(r'([a-zA-Z]+|\d+)', words_from_file)
            wordList = [x for x in wordList if x != ' ' and x != '']
            # print("The words are:\n", words_from_file)

        print(f"  Found {len(wordList)} words.")

    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found. Exiting.")
        exit()

    all_tokens = list(initial_prompt_ids)
    word_end_indices = [len(initial_prompt_ids)]  # Index *after* the last token of each word (or initial prompt)
    processed_words = []  # Store the actual words processed

    print("  Tokenizing words and building full sequence...")
    for word in wordList:
        word_tokens = tokenizer.encode(" " + word, add_special_tokens=False)
        all_tokens.extend(word_tokens)
        word_end_indices.append(len(all_tokens))
        processed_words.append(word)

    full_token_tensor = torch.tensor(all_tokens, dtype=torch.long).unsqueeze(0)
    print(f"  Pre-tokenized {len(processed_words)} words into a sequence of {len(all_tokens)} tokens.\n")

    num_words_to_process = min(len(processed_words), LegalNumberOfMove)
    if num_words_to_process < len(processed_words):
        print(f"  Will process the first {num_words_to_process} words due to NUM_WORDS limit.\n")
    elif num_words_to_process == 0:
        print("  Warning: No words to process based on input file or limits.\n")

    # %% Build file header
    print("Step 8: Preparing output file header...")
    header_lines = [
                       f'[PromptID "{promptID}"]\n',
                       f'[EngineID "{EngineID}"]\n',
                       f'[MultiPV "{MultiPV}"]\n',
                       f'[DepthRange "1:1"]\n\n',
                   ] + lines + [f'\n\n']
    print("  Header prepared.\n")

    # %% Main Generation Loop (Using Slicing & Greedy Decoding)
    print("Step 9: Entering main generation loop (using pre-tokenized slicing and greedy decoding)...\n")
    PrevEval = "n.a."
    start_time = time.time()
    current_time = start_time

    if num_words_to_process > 0:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as writer:
            print("  Writing header to output file...")
            writer.write(''.join(header_lines))
            print("  Header written. Starting word-by-word prediction.\n")

            for turnCount in range(1, num_words_to_process + 1):
                current_word = processed_words[turnCount - 1].strip()
                # print(f"Turn {turnCount}: Predicting after word '{current_word}'")

                slice_end_index = word_end_indices[turnCount - 1]
                slice_start_index = max(0, slice_end_index - MAX_INPUT_TOKENS)
                # print(f"  9.1/9.2: Context slice indices: [{slice_start_index}:{slice_end_index}]")

                input_tensor = full_token_tensor[:, slice_start_index:slice_end_index]
                current_input_len = input_tensor.shape[1]
                # print(f"  9.3: Sliced input tensor shape: {input_tensor.shape}")

                input_tensor_dev = input_tensor.to(DEVICE)

                start_time_gen = time.time()
                # 9.4 Generate next token using GREEDY DECODING
                # print(f"  9.4: Running model.generate() with {current_input_len} input tokens (Greedy Decoding)...")
                with torch.no_grad():
                    outputs = model.generate(
                        input_tensor_dev,
                        max_new_tokens=1,
                        min_new_tokens=1,  # Explicitly require 1 new token
                        output_scores=True,  # Get logits
                        return_dict_in_generate=True,  # Get dict output
                        do_sample=True,  # Disable sampling -> Use Greedy Decoding
                        pad_token_id=tokenizer.pad_token_id,
                        # num_beams=BEAM_WIDTH,
                        # num_return_sequences=3,
                        # Removed num_beams and num_return_sequences
                        top_k=50,
                        top_p=0.95,
                        num_return_sequences=3
                    )
                end_time_gen = time.time()
                gen_duration = end_time_gen - start_time_gen
                # print(f"    Model generation took: {gen_duration:.4f} seconds")

                # ----- UPDATED LOGIC for TopK Logits (Greedy Path) -----
                # outputs.scores is a tuple of length max_new_tokens (1)
                # Each element is a tensor of shape [batch_size, vocab_size] (batch_size is 1 here)
                logits_for_step = outputs.scores[
                    0]  # Logits for the single generated token step. Shape: [1, vocab_size]

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
                """
                print(f"Top {MultiPV} Logits from greedy path (Token | Logit Value):")
                for i in range(MultiPV):
                    token_str_cleaned = top_k_tokens[i].strip()
                    print(f"     - '{token_str_cleaned}': {top_k_logits_values[i]:.4f} (ID: {top_k_logits_indices[i]})")
                """
                # The token actually generated by greedy decoding
                greedy_selected_token_id = outputs.sequences[0, -1].item()  # Last token in the sequence
                greedy_selected_token_str = tokenizer.decode(greedy_selected_token_id).strip()
                # This will always match top_k_tokens[0] because do_sample=False
                # print(f"    (Greedy search selected token: '{greedy_selected_token_str}' ID: {greedy_selected_token_id})") # Optional confirmation
                # ----- END of UPDATED LOGIC -----

                print("Turn ", turnCount, " now matching text word ", current_word, " ...", end='', sep='')

                topNUMTvals, topNUMTindices = torch.topk(logits_for_greedy_path, k=NUM_TOKENS, dim=-1)
                topNUMTvalList = topNUMTvals.tolist()
                topNUMTindList = topNUMTindices.tolist()
                topNUMTtokens = [repr(tokenizer.decode(tind)) for tind in
                                 topNUMTindList]  # OK that these keep '...' or "..."
                matchingTextToken = "@@"

                textTokenIndex = 0
                textTokenValue = 0
                for tok in topNUMTtokens:
                    # if tok.find(current_word) != -1:
                    if borderedMatch(current_word, tok, True, True):
                        matchingTextToken = f"@{tok.replace('@', '(at)')[1:-1]}@"
                        textTokenValue = round(topNUMTvalList[textTokenIndex], 4)
                        print("found at index", textTokenIndex, "in token", matchingTextToken, "with score ",
                              textTokenValue)
                        break
                    textTokenIndex += 1

                if textTokenIndex >= NUM_TOKENS:
                    textTokenValue = round(topNUMTvalList[-1], 4)
                    print("not found, using bottom score", textTokenValue)

                # Derive metrics
                modelToken = repr(top_k_tokens[0])  # Equivalent to greedy_selected_token_str
                modelToken = modelToken.replace('@', '(at)')
                modelToken = f"@{modelToken[1:-1]}@"
                # modelEval is the highest logit value
                modelEval = f"{top_k_logits_values[0]:.4f}"
                modelEval = round(float(modelEval) * 100)
                NextEval = (f"{top_k_logits_values[1]:.4f}" if MultiPV > 1 else "n.a.")
                if MultiPV > 1 and isinstance(top_k_logits_values[1], float):
                    float_next_eval = float(NextEval) * 100
                    if math.isinf(float_next_eval) and float_next_eval < 0:
                        NextEval = "-inf"  # Or some other indicator for negative infinity
                    else:
                        NextEval = round(float_next_eval)
                else:
                    NextEval = "n.a."

                # print(
                # f"  9.5: Top token (greedy choice): '{modelToken}' (Evalution: {modelEval})|Logit value : {top_k_logits_values[0]:.4f}| Next best Eval: {NextEval} | Logit ")

                # Build lines for this turn
                current_stem = initial_prompt + " " + " ".join(processed_words[:turnCount])
                lines = [
                    f'[PID "{promptID}"]\n',
                    f'[EID "{MODEL_NAME}"]\n',
                    f'[Turn "{turnCount}-w"]\n',
                    f'[TextToken "@{current_word}@"]\n',
                    f'[ModelToken "{modelToken}"]\n',  # The model's greedy prediction
                    f'[TextTokenIndex "{textTokenIndex}"]\n'
                    f'[TextTokenValue "{textTokenValue}"]\n'
                    f'[Eval "{modelEval}"]\n',  # The highest raw logit value
                    f'[PrevEval "{PrevEval}"]\n',
                    f'[NextEval "{NextEval}"]\n',  # The second highest raw logit value
                    f'[Depth "{Depth}"]\n',
                    f'[STEM "{current_stem}"]\n',
                    f'[NumLegalMoves "{MultiPV}"]\n',
                    "---------------\n",
                    f"{DEPTH_RANGE}\n",
                    "---------------\n"
                ]
                for token_str, logit_val in zip(top_k_tokens, top_k_logits_values):
                    rep = repr(token_str).replace('@', '(at)')  # has ' ' or " " around it
                    rep = f"@{rep[1:-1]}@"  # now has @ ... @ around it
                    lines.append(f"{rep} {logit_val:.4f}\n")

                lines.append(
                    "===========================================================================================================\n")
                lines.append(f"[Comments]\n")
                lines.append(f"[EndMove]\n\n")

                # print("    Lines built.")

                # 9.7 Write to file
                # print("  9.7: Writing lines to output file...")
                writer.write(''.join(lines))
                # print("    Write complete.\n")

                # 9.8 Update state
                PrevEval = modelEval

                # 9.9 Status update
                status_interval = min(100, num_words_to_process // 2 if num_words_to_process >= 10 else 10)
                if turnCount % status_interval == 0 or turnCount == num_words_to_process:
                    last_time = current_time
                    current_time = time.time()
                    elapsed = current_time - start_time
                    elapsedLast = current_time - last_time
                    rate = (turnCount - 1) / elapsed if elapsed > 0 else 0
                    rateLast = 100.0 / elapsedLast if elapsedLast > 0 else 0
                    print()
                    print(f"Processed Turn {turnCount - 1}. Rate: {rate:.2f} words/sec., last 100 rate: {rateLast:.2f}")

            print("  Finished processing requested words.\n")  # ends body of for-loop

    else:
        print("Skipping main generation loop as there are no words to process.")

    hook_handle.remove()
    print("Removed forward hook.")
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

main(INPUT_FILE, OUTPUT_FILE, PROMPT_ID, PROMPT_TOPIC, MULTI_PV, NUM_WORDS, NUM_TOKENS, BEAM_WIDTH, ALPHA_MODE,
     MODEL_TAG)