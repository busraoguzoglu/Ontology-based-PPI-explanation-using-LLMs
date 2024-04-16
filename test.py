import torch
from transformers import  LlamaTokenizer, set_seed
from peft import  AutoPeftModelForCausalLM
from datasets import load_dataset
from datetime import datetime as dt
import logging


# Reproducibility
seed = 42
set_seed(seed)

def create_prompt_formats_for_test(sample):
    """
    Format various fields of the sample ('instruction', 'context', 'response')
    Then concatenate them using two newline characters
    :param sample: Sample dictionnary
    """
    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    # Instruction Key without protein tags:
    # INSTRUCTION_KEY = "### Instruction: What is the key word that represents the interaction between the proteins " + sample["Gene1"] + " and " + sample["Gene2"] + " in the given sentence?"
    
    # Instruction Key with protein tags:
    INSTRUCTION_KEY = "### Instruction: What is the key word that represents the interaction between the proteins which are tagged with [Protein1] and [Protein2] in the given sentence?"
    INPUT_KEY = "### Input:"
    RESPONSE_KEY = "### Response:"
    

    blurb = f"{INTRO_BLURB}"
    instruction = INSTRUCTION_KEY
    input_context = f"{INPUT_KEY}\n{sample['Sentence']}"   # Sentence, passage
    response = f"{RESPONSE_KEY}\n"
    

    parts = [part for part in [blurb, instruction, input_context, response] if part]

    formatted_prompt = "\n\n".join(parts)

    sample["text"] = formatted_prompt

    return sample




if __name__ == '__main__':

    logging.basicConfig(filename="finetune_results/finetuned-7B-chat-test-5.log", level=logging.INFO)
    logging.info(f"({dt.now().strftime('%d/%m/%Y %H:%M:%S')})| START")
    
    
    test_on_lll=True
    
    # Specify device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## TEST Finetuned Model From Checkpoint ##
    tmp_model_path = "results/llama2/final_checkpoint"
    print("Loading the checkpoint in a Llama model.")
    model = AutoPeftModelForCausalLM.from_pretrained(tmp_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
    tokenizer = LlamaTokenizer.from_pretrained(tmp_model_path, use_fast=False)
    
    ## check the total model parameters
    print(sum(p.numel() for p in model.parameters()))
    
    
    if test_on_lll:
        test_dataset = load_dataset("bengisucam/LLL_INO-tagged", split="test")
    else:
        test_dataset = load_dataset("bengisucam/HPRD50_true_only_tagged", split="test")
        print(test_dataset[:2])
        test_dataset = test_dataset.filter(lambda example: example["isValid"]==True)
        print(test_dataset[:2])
    
    print(len(test_dataset))
    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = test_dataset.map(create_prompt_formats_for_test)  # , batched=True)
    print(len(dataset))
    
    
    
    for i in range(len(dataset)):
        # Specify input
        text = dataset[i]["text"]
        sentence_id = dataset[i]["Unnamed: 0"]
    
        
        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt").to(device)
    
        # Get answer
        # (Adjust max_new_tokens variable as you wish (maximum number of tokens the model can generate to answer the input))  #.to(device)
        outputs = model.generate(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"],
                                 max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
        
        print("EXAMPLE ", i+1)
        # Decode output & print it
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("Sentence Id: ", sentence_id)
        print(response)
        print("##############################################################################")
        logging.info("Sentence Id: %s, Response: %s  .\n\n", sentence_id, response)
        
    # clear memory
    del model

    
    
    
    