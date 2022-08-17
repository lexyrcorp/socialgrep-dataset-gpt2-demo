import datasets
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, Trainer

CHECKPOINT = None
# CHECKPOINT = "trainer\\checkpoint-1000"
DEVICE = "cuda"

tokenizer = AutoTokenizer.from_pretrained("gpt2", bos_token="<|begoftext|>")
tokenizer.pad_token = tokenizer.eos_token

if CHECKPOINT is None:
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)
    model.resize_token_embeddings(len(tokenizer))
    dataset = datasets.load_dataset("SocialGrep/reddit-r-bitcoin-data-for-jun-2022", "comments")

    def tokenize_function(examples):
        result = tokenizer(examples["body"], padding="max_length", truncation=True, add_special_tokens=True)
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = dataset.map(tokenize_function, batched=True)
    training_args = TrainingArguments("trainer", per_device_train_batch_size=4, gradient_accumulation_steps=16,
                                      logging_steps=100)
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset["train"])
    trainer.train()
else:
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT).to(DEVICE)

inputs = tokenizer([tokenizer.bos_token + tokenizer.bos_token for _ in range(16)], return_tensors="pt", padding=True,
                   add_special_tokens=True).to(DEVICE)
outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                         do_sample=True, max_length=128, bos_token_id=tokenizer.bos_token_id)

print("RESULTS:")
print("=====")
for line in [i for i in tokenizer.batch_decode(outputs, skip_special_tokens=True) if len(i) > 0]:
    print(line.lstrip())
    print("-----")
