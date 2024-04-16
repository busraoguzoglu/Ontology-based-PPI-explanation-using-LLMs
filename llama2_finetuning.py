
####
# https://blog.ovhcloud.com/fine-tuning-llama-2-models-using-a-single-gpu-qlora-and-ai-notebooks/
####

# Import libraries
import bitsandbytes as bnb
from functools import partial
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed,  BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments, LlamaTokenizer, EarlyStoppingCallback
from datasets import load_dataset
import random
import pandas as pd

# Reproducibility
seed = 42
set_seed(seed)


def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    print("N GPUS: ", n_gpus)
    max_memory = f'{40960}MB'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # dispatch efficiently the model on the available ressources
        max_memory = {i: max_memory for i in range(n_gpus)},
    )
    #tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True) # use it for llama
    
    tokenizer = LlamaTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)  # use it for pmc-llama,  , trust_remote_code=True, use_fast=False
    
    # for pmc llama : 
    print("BEFORE: ")
    print("Tokenizer BOS: ", tokenizer.bos_token)
    print("Tokenizer EOS: ", tokenizer.eos_token)
    print("Tokenizer PAD: ", tokenizer.pad_token)
    print("Tokenizer UNK: ", tokenizer.unk_token)  
    
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token
      model.resize_token_embeddings(len(tokenizer))      
    tokenizer.padding_side = 'right'   
   
    print("AFTER: ")
    print("Tokenizer BOS: ", tokenizer.bos_token)
    print("Tokenizer EOS: ", tokenizer.eos_token)
    print("Tokenizer PAD: ", tokenizer.pad_token)
    print("Tokenizer UNK: ", tokenizer.unk_token)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    

    return model, tokenizer

   
    

def create_prompt_formats(sample):
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
    response = f"{RESPONSE_KEY}\n{sample['Keywords']}"
    end = f"{END_KEY}"

    parts = [part for part in [blurb, instruction, input_context, response, end] if part]

    formatted_prompt = "\n\n".join(parts)

    sample["text"] = formatted_prompt

    return sample


# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
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
    return max_length


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, dataset: str):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """

    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats)  # , batched=True)
    
    print("First example")
    print(dataset[0]["text"])

    # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
       # remove_columns=["instruction", "context", "response", "text", "category"],
    )

    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

    
    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset


def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config

def create_peft_config(modules):
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    """
    config = LoraConfig(   
        r=8,  # dimension of the updated matrices
        lora_alpha=32,  # parameter for scaling
        target_modules=modules,
        lora_dropout=0.0,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )

    
    print(config)

    return config


# SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)
    

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    
    print(
        f"all params: {all_param:,f} || trainable params: {trainable_params:,f} || trainable%: {100 * trainable_params / all_param}"
    )


def train(model, tokenizer, train_dataset, val_dataset, output_dir):
    # Apply preprocessing to the model to prepare it by
    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()

    # 2 - Using the prepare_model_for_kbit_training method from PEFT
    model = prepare_model_for_kbit_training(model)

    # Get lora module names
    modules = find_all_linear_names(model)

    # Create PEFT config for these modules and wrap the model to PEFT
    peft_config = create_peft_config(modules)
    model = get_peft_model(model, peft_config)

    # Print information about the percentage of trainable parameters
    print_trainable_parameters(model)

    # Training parameters
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=TrainingArguments(
            num_train_epochs=4,  #num_train_epochs = max_steps / len(train_dataloader)
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            # max_steps=40,  #this overrides the num_train_epochs
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
            optim="paged_adamw_8bit",
            load_best_model_at_end = True,  # for EarlyStoppingCallback, it is needed
            evaluation_strategy = 'steps',  # for EarlyStoppingCallback, it is needed
            metric_for_best_model='eval_loss',
            save_strategy='steps',
            eval_steps=5,
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks = [EarlyStoppingCallback(early_stopping_patience = 1, early_stopping_threshold = 0.0)]
    )

    model.config.use_cache = False  # re-enable for inference to speed up predictions for similar inputs

    ### SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
    # Verifying the datatypes before training

    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    do_train = True
    do_eval = True

    
    if do_train:
        # Launch training
        print("Training...")
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print(metrics)

        ###
    if do_eval: 
        # Launch evaluation
        print("Evaluating...")
        eval_result = trainer.evaluate()
        print(eval_result)

    # Saving model
    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
     # save tokenizer for easy inference
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)

    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()


if __name__ == '__main__':

    #dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    dataset = load_dataset("bengisucam/LLL_INO-tagged", split="train")
    train_test_dataset = dataset.train_test_split(test_size=0.1)  # train and test (val)
    
    train_dataset = train_test_dataset["train"]
    val_dataset = train_test_dataset["test"]
    
    print("train dataset length : ", len(train_dataset))
    print("val dataset length : ", len(val_dataset))
    

    
    ##########################
    #### EXPLORE DATASET ####
    # Generate random indices
    ##########################
    nb_samples = 3
    random_indices = random.sample(range(len(train_dataset)), nb_samples)
    train_samples = []

    for idx in random_indices:
        sample = train_dataset[idx]

        sample_data = {            
            'context': sample['Sentence'],
            'response': sample['Keywords'],
            'genes': [sample['Gene1'], sample["Gene2"]]
        }
        train_samples.append(sample_data)

    # Create a DataFrame and display it
    train_df = pd.DataFrame(train_samples)
    #print(train_df[:3])

    # Load model from HF with user's token and with bitsandbytes config
    model_name = "meta-llama/Llama-2-13b-chat-hf"
    #model_name = "meta-llama/Llama-2-7b-chat-hf"
    bnb_config = create_bnb_config()
    model, tokenizer = load_model(model_name, bnb_config)
    
    print_trainable_parameters(model)


    ## Preprocess dataset
    max_length = get_max_length(model)
    train_dataset_processed = preprocess_dataset(tokenizer, max_length, seed, train_dataset)
    val_dataset_processed = preprocess_dataset(tokenizer, max_length, seed, val_dataset)

    output_dir = "results/llama2/final_checkpoint"
    train(model, tokenizer, train_dataset_processed, val_dataset_processed, output_dir)
    
   
   

