# %% File "essay_generator.py" based on lifgen2.py by Gishnu Madhu, KWR
# Refactored to generate a full essay based on a topic/summary in a single call.
# Usage:
# python essay_generator.py -i <source-file-for-word-count> -o <output.lif> -pid <identifier> -pt <essay_topic_or_summary>
# optional: -model <model_name> -temp <temperature> -top_p <top_p> -atpw <avg_tokens_per_word>
# Example:
# python essay_generator.py -i YaoJokic.txt -o YaoJokic_GeneratedEssay.lif -pid YaoEssayGen -pt "Write an essay comparing and contrasting the NBA careers and playing styles of Yao Ming and Nikola Jokic."

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

# --- Argument Parsing ---
parser = argparse.ArgumentParser(
    description="Essay Generator using Hugging Face models",
    epilog="Generates an essay based on a topic, matching word count of an input file."
)
parser.add_argument(
    "-input_file", "-i",
    type=str,
    required=True,
    help="Path to the reference input file (used for target word count)."
)
parser.add_argument(
    "-output_file", "-o",
    type=str,
    required=True,
    help="Name and path of the output .lif file."
)
parser.add_argument(
    "-prompt_id", "-pid",
    type=str,
    required=True,
    help="Overall identifier for the generation task."
)
parser.add_argument(
    "-prompt_topic", "-pt",
    type=str,
    required=True,
    help="The topic or summary prompt for the LLM to write the essay about."
)
parser.add_argument(
    "-model", "--model_name",
    type=str,
    default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", # Default model
    help="Hugging Face model name to use (e.g., 'google/gemma-3-4b-it', 'Qwen/Qwen3-1.7B')."
)
parser.add_argument(
    "-temp", "--temperature",
    type=float,
    default=0.7,
    help="Sampling temperature for generation."
)
parser.add_argument(
    "-top_p", "--top_p",
    type=float,
    default=0.9,
    help="Nucleus sampling probability."
)
parser.add_argument(
    "-atpw", "--avg_tokens_per_word",
    type=float,
    default=1.3, # Common heuristic
    help="Estimated average tokens per word for length calculation."
)

args = parser.parse_args()

print("Welcome to the Essay Generator script!")
print(f"Reference input file (for word count): {args.input_file}")
print(f"Output LIF file path: {args.output_file}")
print(f"Prompt ID: {args.prompt_id}")
print(f"Essay Topic/Summary: {args.prompt_topic}")
print(f"Model: {args.model_name}")
print(f"Temperature: {args.temperature}")
print(f"Top_p: {args.top_p}")
print(f"Avg Tokens/Word: {args.avg_tokens_per_word}")

# --- Hugging Face Login ---
hf_token = input("Enter your Huggingface token (or press Enter if not needed/already logged in): ")
# Or better:
# hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")

if hf_token:
    print("Logging in to Hugging Face Hub...")
    try:
        login(token=hf_token)
        print("Login successful.")
    except Exception as e:
        print(f"HF Login failed: {e}. Gated model download might fail.")
else:
    print("HF Token not provided. Assuming public model or cached credentials.")


def main(input_file_path, output_lif_path, prompt_id, essay_prompt, model_name, temperature, top_p, avg_tokens_per_word):
    # %% Constants and Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Prefer GPU if available
    TORCH_DTYPE = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float32 # Use bfloat16 on compatible GPUs

    # %% Load Model & Tokenizer
    print(f"\nStep 1: Loading model '{model_name}'...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=TORCH_DTYPE,
            trust_remote_code=True # Often needed for custom architectures
        ).to(DEVICE)
        model.eval() # Set to evaluation mode
        print(f"  Model loaded to {DEVICE} with dtype {TORCH_DTYPE}.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print("\nStep 2: Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            print("  Tokenizer missing pad token; setting pad_token = eos_token")
            tokenizer.pad_token = tokenizer.eos_token
            if hasattr(model, 'config') and model.config.pad_token_id is None:
                 model.config.pad_token_id = tokenizer.eos_token_id
        print("  Tokenizer loaded.")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)

    # %% Get Target Word Count from Input File
    print(f"\nStep 3: Reading reference file '{input_file_path}' to get word count...")
    original_word_count = 0
    original_content_lines = []
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            original_content_lines = f.readlines() # Keep original content for header if needed
            full_text = "".join(original_content_lines)
            words_from_file = re.findall(r'\b\w+\b', full_text) # Simple word count
            original_word_count = len(words_from_file)
        if original_word_count == 0:
             print(f"Warning: No words found in '{input_file_path}'. Cannot set target length.")
             # Decide on a fallback? Exit? Or use a default? Let's use a default for now.
             print("Using a default target of 500 words.")
             original_word_count = 500 # Fallback
        else:
             print(f"  Found {original_word_count} words in reference file.")
    except FileNotFoundError:
        print(f"Error: Input file '{input_file_path}' not found. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)

    # %% Estimate Target Token Count
    max_tokens_to_generate = int(original_word_count * avg_tokens_per_word)
    print(f"  Estimated target tokens for generation: {max_tokens_to_generate}")

    # %% Prepare Generation Prompt
    # Combine the user's topic/summary with instructions
    full_generation_prompt = f"Please write a comprehensive essay based on the following topic or summary. Aim for approximately {original_word_count} words.\n\nTopic/Summary:\n{essay_prompt}\n\nEssay:"

    print("\nStep 4: Preparing for generation...")
    print(f"  Full prompt for LLM (first 200 chars): {full_generation_prompt[:200]}...")

    input_ids = tokenizer.encode(full_generation_prompt, return_tensors="pt").to(DEVICE)
    prompt_token_length = input_ids.shape[1]
    print(f"  Prompt tokenized into {prompt_token_length} tokens.")

    # %% Single Generation Call
    print("\nStep 5: Generating essay...")
    start_time = time.time()
    generated_text = ""
    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_tokens_to_generate,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id # Ensure generation stops at EOS
            )

        generation_time = time.time() - start_time
        print(f"  Generation completed in {generation_time:.2f} seconds.")

        # Decode the output, skipping the prompt tokens and special tokens
        generated_ids = outputs[0, prompt_token_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        generated_word_count = len(re.findall(r'\b\w+\b', generated_text))
        print(f"  Generated essay word count: {generated_word_count} (Target was approx {original_word_count})")

    except Exception as e:
        print(f"Error during generation: {e}")
        generated_text = f"[Generation Failed: {e}]" # Put error in output
        generation_time = time.time() - start_time


    # %% Write Output LIF File
    print(f"\nStep 6: Writing output to '{output_lif_path}'...")
    try:
        with open(output_lif_path, 'w', encoding='utf-8') as writer:
            # Write Header (adapting from original format)
            writer.write(f'[PromptID "{prompt_id}"]\n')
            engine_id = f"EssayGen_{model_name.split('/')[-1]}_{DEVICE.upper()}"
            writer.write(f'[EngineID "{engine_id}"]\n')
            # These fields from lifgen2 don't directly apply here, use placeholders or omit
            writer.write('[MultiPV "N/A - Full Essay Generation"]\n')
            writer.write('[DepthRange "N/A"]\n')
            writer.write(f'[OriginalWordCount "{original_word_count}"]\n') # Add info
            writer.write(f'[GeneratedWordCount "{generated_word_count}"]\n') # Add info
            writer.write(f'[Temperature "{temperature}"]\n') # Add info
            writer.write(f'[TopP "{top_p}"]\n') # Add info
            writer.write(f'[GenerationTimeSec "{generation_time:.2f}"]\n') # Add info

            # Optionally include original text header lines if desired
            # writer.write("\n--- Original Text Header (for context) ---\n")
            # writer.writelines(original_content_lines) # If you want to include the original text

            writer.write("\n--- Generated Essay ---\n\n")
            writer.write(generated_text)
            writer.write("\n\n--- End Generated Essay ---")

        print("  Output file written successfully.")

    except Exception as e:
        print(f"Error writing output file: {e}")

    # %% Final Stats & Cleanup
    print("\nStep 7: Reporting final statistics...")
    total_script_time = time.time() - (start_time - generation_time) # Approximate total time
    print(f"  Total script execution time: {total_script_time:.2f} seconds")
    print(f"  Generated essay word count: {generated_word_count}")

    print("\nCleaning up resources...")
    del model
    del tokenizer
    del input_ids
    if 'outputs' in locals():
        del outputs
    gc.collect()
    if DEVICE == 'cuda':
        print("  Emptying CUDA cache...")
        torch.cuda.empty_cache()
    print("\nScript finished.")


# --- Run Main ---
if __name__ == "__main__":
    main(
        args.input_file,
        args.output_file,
        args.prompt_id,
        args.prompt_topic,
        args.model_name,
        args.temperature,
        args.top_p,
        args.avg_tokens_per_word
    )