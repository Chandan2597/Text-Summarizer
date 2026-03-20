from TextSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()


    
    def predict(self,text):
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}

        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path)

        print("Dialogue:")
        print(text)

        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs["input_ids"], **gen_kwargs)
        output = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        print("\nModel Summary:")
        print(output)

        return output