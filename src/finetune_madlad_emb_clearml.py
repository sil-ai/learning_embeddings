from pathlib import Path

import evaluate
import json
import numpy as np
import os
import pandas as pd
import random
import torch
import requests

from datasets import (
    load_dataset, 
    Dataset
)
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
    T5ForConditionalGeneration, 
    T5Tokenizer,
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
)
from tqdm import tqdm
from clearml import Task 

task = Task.init(project_name='MADLAD fine-tuning eng-kss', task_name='My Experiment')


VOL_MOUNT_PATH = Path("/vol")

BASE_MODEL = "jbochi/madlad400-3b-mt"

# Initialize the S3 client


def download_file_from_s3(bucket_name, object_key):
    public_url = f'https://{bucket_name}.s3.amazonaws.com/{object_key}'
    file_name = object_key.split('/')[-1]

    response = requests.get(public_url)

    # Save the file locally
    with open(file_name, 'wb') as file:
        file.write(response.content)

    print(f'File {object_key} downloaded from bucket {bucket_name} and saved as {file_name}')

download_file_from_s3(bucket_name = 'acts2-model-finetuning', object_key = 'embeddings/aligned_embeddings_dict_MADLAD-kss-kss.json')
download_file_from_s3(bucket_name = 'acts2-model-finetuning', object_key = 'input_data/en-web-kisi.csv')

# image = Image.debian_slim().pip_install(
#     "accelerate",
#     "datasets",
#     "evaluate",
#     "numpy",
#     "pandas",
#     "peft",
#     "sacrebleu",
#     "scikit-learn",
#     "sentencepiece",
#     "tensorboard",
#     "torch",
#     "transformers",
# )

# app = App(
#     name="test_madlad_ft", image=image
# )
# output_vol = Volume.from_name("finetune-volume", create_if_missing=True)

# restart_tracker_dict = modal.Dict.from_name(
#     "finetune-restart-tracker", create_if_missing=True
# )

# def track_restarts(restart_tracker: modal.Dict) -> int:
#     if not restart_tracker.contains("count"):
#         preemption_count = 0
#         print(f"Starting first time. {preemption_count=}")
#         restart_tracker["count"] = preemption_count
#     else:
#         preemption_count = restart_tracker.get("count") + 1
#         print(f"Restarting after pre-emption. {preemption_count=}")
#         restart_tracker["count"] = preemption_count
#     return preemption_count

# @app.function(
#     image=image,
#     gpu="H100:2",
#     # gpu="any",
#     timeout=10800,
#     #secrets=[modal.Secret.from_name("mw-wandb-secret")],
#     mounts=[
#         modal.Mount.from_local_dir(
#             "../fine-tuned_models/data", remote_path="/root/data"
#         ),
#     ],
#     volumes={VOL_MOUNT_PATH: output_vol},
#     _allow_background_volume_commits=True,
# )
def finetune(file_path: str, emb_file_path: str, num_train_epochs: int = 1):
    
    print("starting fine-tuning")

    # restarts = track_restarts(restart_tracker_dict)

    model_name = 'jbochi/madlad400-3b-mt'
    model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model.config.max_length = 160

    source_language = 'English'
    source_iso_code = 'eng'
    source_lang_code = 'en'
    target_language = 'Kisi'
    target_iso_code = 'kss'
    target_lang_code = 'ks'
    learning_rate = .0007

    print("got model and tokenizer")

    lora_config = LoraConfig(
        r=32, # Rank
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        #modules_to_save=["decode_head"],
        )

    lora_model = get_peft_model(model, lora_config)
    lora_model.config.max_length = 80

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    # with open(emb_file_path) as f:
    #     new_emb = json.load(f)

    # new_vocab = [new_emb[i]['word'] for i in range(len(new_emb))]
    # # embeddings = torch.tensor([item['embedding'] for item in new_emb])

    # tokenizer.add_tokens(list(new_vocab))

    # # lora_model.config.tie_word_embeddings = True
    # # lora_model.tie_weights()
    # lora_model.resize_token_embeddings(len(tokenizer))

    # # for i in range(len(new_emb)):
    # #     if i%1000==0:
    # #         print(i)
    # #     try:
    # #         lora_model.encoder.embed_tokens.weight.data[
    # #             tokenizer.get_vocab()[new_emb[i]['word']]] = torch.tensor(new_emb[i]['embedding'])
    # #     except:
    # #         print("unable to add token for " + str(new_emb[i]['word']))
    # # vocab_dict = tokenizer.get_vocab()

    # # # Create a tensor for embeddings
    # # embedding_matrix = torch.zeros((len(vocab_dict), len(embeddings[0])))
    
    # # Efficient look-up dictionary for new embeddings
    # # embedding_dict = {item['word']: torch.tensor(item['embedding']) for item in tqdm(new_emb)}

    # # # Update only the new embeddings
    # # with torch.no_grad():
    # #     for word, embedding in tqdm(embedding_dict.items()):
    # #         token_index = tokenizer.get_vocab().get(word)
    # #         if token_index is not None:
    # #             lora_model.encoder.embed_tokens.weight.data[token_index] = embedding

    # print(new_emb[26601]['word'])
    # print(new_emb[26601]['embedding'][:5])
    # print(tokenizer.get_vocab()[new_emb[26601]['word']])
    # print(lora_model.encoder.embed_tokens.weight.data[tokenizer.get_vocab()[new_emb[26601]['word']]])


    pair_list = []

    df = pd.read_csv(file_path, index_col=0)

    df['del'] = 0
    for index, row in df.iterrows():
        if row['Kisi']=='<range>':
            df.loc[index, 'del'] = 1
            df.loc[index-1, 'del'] = 1
    df = df[df['del']==0]
    df.reset_index(inplace=True, drop=True)
    # def preprocess_text(text):
    #     text = ' '.join(['▁' + word.lower() for word in text.split()])
    #     return text
    
    # df[target_language] = df[target_language].apply(preprocess_text)
    #print(df.head())
    #df = df.loc[15:]
    test_df = df.sample(n=250, random_state=7)
    print(test_df.head())

    
    
    df = df[~df.index.isin(test_df.index)]

    df.reset_index(inplace=True, drop=True)

    for i in range(len(df)):
        pair_list.append(
            {'id': i, 
             'translation': {source_lang_code: df.loc[i, source_language],
                             target_lang_code: df.loc[i, target_language]}})

    random.Random(7).shuffle(pair_list)


    prefix = ''
    #target_prefix = '<pad>'
    target_prefix = ''

    #model.config.forced_bos_token_id = tokenizer.pad_token_id
    #model.generation_config.forced_bos_token_id = tokenizer.pad_token_id

    MAX_LENGTH = 160

    def preprocess_function(examples):
        inputs = [prefix + example[source_lang_code] for example in examples["translation"]]
        targets = [target_prefix + ''.join(['▁' + word for word in example[target_lang_code].lower().split()]) for example in examples["translation"]]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=MAX_LENGTH, truncation=True)
        return model_inputs

    
    def gen():
        for i in range(len(pair_list)):
            yield {"id": pair_list[i]['id'], "translation": pair_list[i]['translation']}
    ds = Dataset.from_generator(gen)
    ds = ds.train_test_split(test_size=0.2)
    
    tokenized_verses = ds.map(preprocess_function, batched=True)
    print(f'{tokenized_verses["train"][:10]=}')
    print(f'{tokenized_verses["test"][:10]=}')
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, #model=checkpoint
                                        model=lora_model)

    metric = evaluate.load("chrf")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip().lower()] for label in labels]

        return preds, labels


    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        print(f'{preds[:10]=}')
        print(f'{labels[:10]=}')
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        print(f'{decoded_preds[:10]=}')
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        print(f'{decoded_labels[:10]=}')

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"chrf": result["score"]}
        print(result)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        print(result)
        return result

    learning_rate = learning_rate
    weight_decay = 0
    train_batch_size = 32
    eval_batch_size = 16


    training_args = Seq2SeqTrainingArguments(
        output_dir=str(VOL_MOUNT_PATH / "madlad_eng_kss_emb_model"),
        evaluation_strategy="steps",
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        label_smoothing_factor=0.2,
        gradient_accumulation_steps=2,
        #gradient_checkpointing=True,
        weight_decay=weight_decay,
        save_total_limit=3,
        max_steps=5000,
        warmup_steps=4000,
        save_steps=1000,
        eval_steps=1000,
        seed=42,
        metric_for_best_model='chrf',
        greater_is_better=True,
        load_best_model_at_end=True,
        log_level='info',
        tf32=False,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        fp16=False,
        #fp16=True,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_verses["train"],
        eval_dataset=tokenized_verses["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    try:
        # resume = restarts > 0
        # if resume:
        #     print("resuming from checkpoint")
        # print("starting training")
        #trainer.train(resume_from_checkpoint=resume)
        trainer.train()
    except KeyboardInterrupt:  # handle possible preemption
        print("received interrupt; saving state and model")
        trainer.save_state()
        trainer.save_model()
        raise

    # Save the trained model and tokenizer to the mounted volume
    model.save_pretrained(str(VOL_MOUNT_PATH / f"madlad400-finetuned-{source_iso_code}-{target_iso_code}-emb"))
    tokenizer.save_pretrained(str(VOL_MOUNT_PATH / "madlad_tokenizer-emb"))
    # output_vol.commit()
    print("✅ done")

# @app.function(volumes={VOL_MOUNT_PATH: output_vol})
# @wsgi_app()
# def monitor():
#     import tensorboard

#     board = tensorboard.program.TensorBoard()
#     board.configure(logdir=f"{VOL_MOUNT_PATH}/logs")
#     (data_provider, deprecated_multiplexer) = board._make_data_provider()
#     wsgi_app = tensorboard.backend.application.TensorBoardWSGIApp(
#         board.flags,
#         board.plugin_loaders,
#         data_provider,
#         board.assets_zip_provider,
#         deprecated_multiplexer,
#     )  # Note: prior to April 2024, "app" was called "stub"
#     return wsgi_app

if __name__ == "__main__":
    file_path = "en-web-kisi.csv" 
    emb_file_path = "aligned_embeddings_dict_MADLAD-kss-kss.json"

    finetune(file_path, emb_file_path)
