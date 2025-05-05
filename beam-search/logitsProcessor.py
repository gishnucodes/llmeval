import torch
from transformers import LogitsProcessor, LogitsProcessorList
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

# Suppress specific warnings if needed (e.g., during format checks)
warnings.filterwarnings("ignore", message="overflow encountered in exp")

# 1. Define the Custom LogitsProcessor with Verification
class LogitExtractor(LogitsProcessor):
    """
    Stores raw logits and includes a one-time check to verify their format.
    """
    def __init__(self):
        super().__init__()
        self.logits_history = []
        self._checked_format = False # Flag to print check only once per instance usage

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Stores logits and performs a one-time format check.
        """
        # --- Format Check ---
        # Perform checks only once on the first valid scores tensor received
        if not self._checked_format and scores.numel() > 0 and scores.ndim >= 2:
            print("\n--- Verifying Logit Format (Inside LogitsProcessor) ---")
            # Use only the first item in the batch/beam for checking efficiency
            test_scores = scores[0:1].float() # Use float for stability in checks
            is_log_probs = False
            is_raw_logits = False

            try:
                # Check if scores behave like log-probabilities (exp() should sum to ~1)
                with torch.no_grad(): # Avoid tracking gradients during checks
                    sum_exp_scores = torch.exp(test_scores).sum(dim=-1)
                # Use a tolerance for floating point errors
                if torch.allclose(sum_exp_scores, torch.ones_like(sum_exp_scores), rtol=1e-3, atol=1e-3):
                    print(f"[Check] Scores seem consistent with LOG-PROBABILITIES (sum(exp(scores[0])) is approx 1): {sum_exp_scores.item():.4f}")
                    is_log_probs = True
                else:
                    print(f"[Check] Scores DO NOT seem to be log-probabilities (sum(exp(scores[0])) != 1): {sum_exp_scores.item():.4f}")

                # Check if scores behave like raw logits (softmax() should sum to ~1)
                with torch.no_grad():
                    sum_softmax_scores = torch.softmax(test_scores, dim=-1).sum(dim=-1)
                if torch.allclose(sum_softmax_scores, torch.ones_like(sum_softmax_scores), rtol=1e-3, atol=1e-3):
                    print(f"[Check] Scores seem consistent with RAW LOGITS (sum(softmax(scores[0])) is approx 1): {sum_softmax_scores.item():.4f}")
                    is_raw_logits = True
                else:
                    # This case might happen for unusual model outputs
                    print(f"[Check] Scores DO NOT seem to be raw logits (sum(softmax(scores[0])) != 1): {sum_softmax_scores.item():.4f}")

            except Exception as e:
                print(f"[Check] Error during format check: {e}")

            # --- Conclusion ---
            if is_raw_logits: # Prioritize this conclusion if softmax sums to 1
                 print(">>> Conclusion: Receiving RAW LOGITS as expected.")
            elif is_log_probs and not is_raw_logits:
                print(">>> Conclusion: Receiving LOG-PROBABILITIES, not raw logits.")
            else:
                print(">>> Conclusion: Format is unclear or non-standard based on checks.")
            print("-" * 50)
            self._checked_format = True # Don't check again for this generate call
        # --- End Format Check ---

        # Store a CPU copy
        self.logits_history.append(scores.clone().detach().cpu())
        # Return original scores
        return scores

    def get_logits(self):
        return self.logits_history

    def reset(self):
        """ Resets history and the format check flag. """
        self.logits_history = []
        self._checked_format = False

# --- Example Usage (Similar to before) ---

# 2. Load Model and Tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# 3. Prepare Input
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# 4. Instantiate your Custom Logit Extractor
logit_extractor = LogitExtractor()
logit_extractor.reset() # Ensure reset before use

# 5. Perform Generation using the Logit Extractor
logits_processor_list = LogitsProcessorList([logit_extractor])

max_new_tokens = 5
num_beams = 4
k_top = 50

print(f"\nGenerating text (max_new_tokens={max_new_tokens}, num_beams={num_beams})...")
output_sequences = model.generate(
    input_ids=input_ids,
    # attention_mask=attention_mask,
    max_new_tokens=max_new_tokens,
    num_beams=num_beams,
    early_stopping=True,
    logits_processor=logits_processor_list,
)

# 6. Retrieve the Extracted "Logits" (as received by the processor)
extracted_scores = logit_extractor.get_logits()

# --- Analyze and Display Top K ---
print("\n--- Top K Analysis (Based on scores received by processor) ---")
print(f"Analyzing Top {k_top} scores for each generation step.")

batch_beam_size = extracted_scores[0].shape[0] if extracted_scores else 0
is_beam_search = num_beams > 1 and batch_beam_size > 1

print(f"Scores tensor shape at step 0: {extracted_scores[0].shape if extracted_scores else 'N/A'}")
print(f"Batch x Beam size: {batch_beam_size}")
print("-" * 20)

for step, step_scores in enumerate(extracted_scores):
    print(f"\n--- Step {step + 1} ---")
    top_k_scores, top_k_indices = torch.topk(step_scores, k=k_top, dim=-1)

    for i in range(batch_beam_size):
        item_label = f"Beam {i}" if is_beam_search else f"Batch Item {i}"
        print(f"  {item_label}:")

        item_top_k_scores = top_k_scores[i]
        item_top_k_indices = top_k_indices[i]
        item_top_k_tokens = tokenizer.convert_ids_to_tokens(item_top_k_indices)

        for rank in range(k_top):
            token_id = item_top_k_indices[rank].item()
            score_value = item_top_k_scores[rank].item()
            token_str = item_top_k_tokens[rank]
            print(f"    Rank {rank+1:>2}: Score={score_value: <10.4f} | Token ID={token_id: <6} | Token='{token_str}'")

print("\n" + "="*30)
print(f"Generated sequence IDs: {output_sequences}")
print(f"Decoded text: {tokenizer.batch_decode(output_sequences, skip_special_tokens=True)}")
print("="*30)