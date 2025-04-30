# Ontology-based-PPI-explanation-using-LLMs
Ontology-based protein-protein interaction explanation using large language models

Run llama2_finetuning.py
Next, test the finetuned model with test.py
Before running token_wise_metrics.py, get the responses generated with the testing (responses are the ones generated after ###Response:, sometimes model generates random outputs that does not follow the required output. Thus, we post-processed the generated outputs manually since we do not have much test sentences.)
Fill the "Predicted" column in the excel file given as "finetuned_13b_chat.xlsx"  ("13b" indicates the Llama-2-13b-chat model, change the file name according to your finetuned model)


## Models Tested 

- Llama2 13b : Cannot run on MPS due to lack of bitsandbytes support (llama2_finetuning_mps.py)
    To be tested on Google Colab with quantization
- Llama2 7b : To be tested on MPS without quantization 
- Mistral 7b: Testing on MPS without quantization (mistral7b_finetuning_mps.py)
- Llama3: TODO
- Falcon 7b: TODO
