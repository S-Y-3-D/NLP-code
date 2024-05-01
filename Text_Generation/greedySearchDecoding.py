from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

#Device 
device = "cuda" if torch.cuda.is_available() else "cpu"

#Loading the model and tokenizer
model_name = "gpt2-xl"


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

#Tokenizing the text
text = "Transformer are the"
input_ids = tokenizer(text=text,return_tensors="pt")['input_ids'].to(device)
iterations = []
choices = 5
steps = 8

for step in range(steps):
  iteration = dict()
  iteration["input"] = tokenizer.decode(input_ids[0]) 
  outputs = model(input_ids)
  next_token_logit = outputs.logits[0,-1,:]
  next_token_probs = torch.softmax(next_token_logit,dim =-1)
  sorted_ids = torch.argsort(next_token_probs,dim=-1,descending=True)

  for choice in range(choices):
    token_id = sorted_ids[choice]
    prob = next_token_probs[token_id].detach().numpy()
    token_choice = f"{tokenizer.decode(token_id)} ({100*prob:.2f}%)"
    iteration[f"Choice {choice+1}"] = token_choice

  input_ids = torch.cat([input_ids,sorted_ids[None, 0, None]],dim=-1)
  iterations.append(iteration)
