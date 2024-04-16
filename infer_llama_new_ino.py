
from transformers import set_seed

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

from datetime import datetime as dt
import logging
import psutil

  
def print_cpu_details():  
  
  print("Total CPU RAM GB: ", psutil.virtual_memory()[0]/1000000000)
  print("Available CPU RAM GB: ", psutil.virtual_memory()[1]/1000000000)
  print("Free CPU RAM GB: ", psutil.virtual_memory()[4]/1000000000)
  
  
  
  
def print_all_parameters(model):
   
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
    
    print(
        f"all params: {all_param:,f}"
    )
  
  
def load_model_to_cpu(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#    tokenizer.add_eos_token = True
    tokenizer.pad_token = tokenizer.eos_token
    

    return model, tokenizer


def load_model_to_gpu(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        #torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto"
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#    tokenizer.add_eos_token = True
    tokenizer.pad_token = tokenizer.eos_token
    

    return model, tokenizer

    

def create_prompt_formats_few_shot(context_df, test_sample):
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
    END_KEY = "### End"

    blurb = f"{INTRO_BLURB}"
    
    test_text = INPUT_KEY + "\n" + test_sample["Sentence"] + "\n" + RESPONSE_KEY + "\n"
    context_text = INSTRUCTION_KEY + "\n"
    for index, row in context_df.iterrows():
        context_text += INPUT_KEY + "\n" + row["Sentence"] + "\n" + RESPONSE_KEY + "\n" + row["Keywords"] + "\n"   
    
    
    parts = [part for part in [blurb, context_text, test_text] if part]

    formatted_prompt = "\n\n".join(parts)

    return formatted_prompt
    
    
def create_prompt_formats_zero_shot(sample):
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
    END_KEY = "### End"

    blurb = f"{INTRO_BLURB}"
    instruction = INSTRUCTION_KEY
    input_context = f"{INPUT_KEY}\n{sample['Sentence']}"
    end = f"{END_KEY}"

    parts = [part for part in [blurb, instruction, input_context, end] if part]

    formatted_prompt = "\n\n".join(parts)

    return formatted_prompt

      
      



def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")



    

if __name__ == '__main__':


    #logging.basicConfig(filename="lll_few_shot_inference_70B_chat_portion3.log", level=logging.INFO)
    logging.basicConfig(filename="lll_zero_shot_inference_7B_chat_v3.log", level=logging.INFO)
    logging.info(f"({dt.now().strftime('%d/%m/%Y %H:%M:%S')})| START")
    
    # Specify device
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    strategy = "zero-shot"   # "few-shot", "zero-shot"
    test_on_LLL = True

    # Reproducibility
    seed = 42
    set_seed(seed)

    print_cpu_details()
    # load model & tokenizer
    model_name = "meta-llama/Llama-2-70b-chat-hf"
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    model_name = "meta-llama/Llama-2-13b-chat-hf"

    model, tokenizer = load_model_to_cpu(model_name)
    
    # print model parameter size
    print_all_parameters(model)

    max_length = get_max_length(model)
    print(f'Max Token Length : {max_length}')
      
    model.config.use_cache = False
    # apply instruction formats to the datasets
    # zero-shot inference
    if strategy == "zero-shot":    
        if test_on_LLL:
            test_data = load_dataset("bengisucam/LLL_INO-tagged", split="test").to_pandas()
        else:
            test_data = load_dataset("bengisucam/HPRD50_true_only", split="test").to_pandas()
        for i in range(len(test_data)): 
            text = create_prompt_formats_zero_shot(test_data.iloc[i])            
            # infer    
            inputs = tokenizer(text, return_tensors="pt")
            gen_output = model.generate(**inputs, max_new_tokens=200)
            response = tokenizer.batch_decode(gen_output)[0]
            sentence_id = test_data.iloc[i]["Unnamed: 0"]
            print("Sentence Id: ", sentence_id)
            print("Response: ", response)
            print("-------------------------------------------------") 
            logging.info("Sentence Id: %s, Response: %s  .\n\n", sentence_id, response)
    # few-shot inference       
    elif strategy == "few-shot":
        train_data = load_dataset("bengisucam/LLL_INO-tagged", split="train").to_pandas()
        test_data = load_dataset("bengisucam/LLL_INO-tagged", split="test").to_pandas()
        
        context = train_data[80:120] # 0:40, 40:80, 80:120
        
        for i in range(len(test_data)): 
            text = create_prompt_formats_few_shot(context, test_data.iloc[i])            
            # infer    
            inputs = tokenizer(text, return_tensors="pt")  #.to(device)
            gen_output = model.generate(**inputs, max_new_tokens=10)
            response = tokenizer.batch_decode(gen_output)[0]
            print(response)
            sentence_id = test_data.iloc[i]["Unnamed: 0"]
            print("Sentence Id: ", sentence_id)
            print("Response: ", response)
            print("-------------------------------------------------") 
            logging.info("Sentence Id: %s, Response: %s  .\n\n", sentence_id, response)
        
    else:
        print("please set strategy as either zero-shot or few-shot")
    
    logging.info(f"({dt.now().strftime('%d/%m/%Y %H:%M:%S')})| END...")



    


      
   