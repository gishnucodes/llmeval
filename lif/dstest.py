## File "dstest.py" by KWR after Muhammad Ahmad Waseem
## Illustrates getting scores on successive increments of a text string
## For use at command prompts, i.e., outside Jupyter Notebooks.
## Requires the LLMEval environment (or similar) to be activated.


from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, GenerationConfig, pipeline
import torch
import torch.nn as nn

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
).to(device="cpu")
model.eval()

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

#prompt = "Tomorrow is the place for all good zombies to come to the aid"
prompt = "Complete successive parts of a sentence given one word at a time:"
target = "Now is the time for all good men to come to the aid of their country ."
targetVec = target.split()

for word in targetVec:
    prompt += ' '
    prompt += word
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device) 
    with torch.no_grad():
        raw_outputs = model(**inputs)
    prob = torch.softmax(raw_outputs.logits, dim=-1)

####################################################
# Muhammad's code adds---put after computing raw outputs

    max = torch.max(raw_outputs.logits[0, -1, :])
    exp1 = torch.exp(raw_outputs.logits[0, -1, :]-max)   # deltas
    exp2 = torch.exp(exp1) - 1
    double_sf_probs = exp2 / torch.sum(exp2)
    _, top_id = torch.topk(prob[0, -1,:], 50)

    print("Single,doube softmaxed probs from raw output logits on deltas for prompt: ", prompt)

    for t_id in top_id:
        t_str = tokenizer.decode(t_id.item()).strip()
        #print(f"{t_str}: {raw_outputs.logits[0, -1, :][t_id].item()}; {prob[0, -1,:][t_id].item():.4f}; {double_sf_probs[t_id].item():.4f}")
        print(f"{t_str}: {raw_outputs.logits[0, -1, :][t_id].item()}")
    
    print("Press Enter to continue...")
    input()
    
