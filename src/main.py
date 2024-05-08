import modal
from typing import List

app = modal.App("full-embeddings-training")

image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch",
    "wandb",
    "pandas",
    "torch",
)


@app.function(
    image=image,
    gpu="any",
    timeout=3600,
    secrets=[modal.Secret.from_name("mw-wandb-secret")],
)
def train_embeddings(
    train_text,
    val_text=None,
    batch_size=64,
    weight_decay=1e-6,
    dropout_rate=0.5,
    lr=0.001,
    num_epochs=30,
    filename="",
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import wandb

    wandb.init(
        project="learn-embeddings-ngram",
        config={
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "dropout_rate": dropout_rate,
            "lr": lr,
            "num_epochs": num_epochs,
            "filename": filename,
        },
    )
    CONTEXT_SIZE = 2
    EMBEDDING_DIM = 256

    train_text_words = train_text.lower().split()
    train_words = set(train_text_words)
    train_words.add("<UNK>")
    vocab = {word: i for i, word in enumerate(train_words)}

    def make_ngrams(words):
        ngrams = [
            ([words[i - j - 1] for j in range(CONTEXT_SIZE)], words[i])
            for i in range(CONTEXT_SIZE, len(words))
        ]
        return ngrams

    train_ngrams = make_ngrams(train_text_words)
    val_ngrams = []
    if val_text:
        val_words = val_text.lower().split()
        val_ngrams = make_ngrams(
            [word if word in vocab else "<UNK>" for word in val_words]
        )

    # Convert n-grams to indices and create DataLoader
    def ngrams_to_tensor(ngrams):
        contexts, targets = zip(*ngrams)
        context_idxs = torch.tensor(
            [[vocab[w] for w in context] for context in contexts], dtype=torch.long
        )
        target_idxs = torch.tensor(
            [vocab[target] for target in targets], dtype=torch.long
        )
        return TensorDataset(context_idxs, target_idxs)

    train_data = ngrams_to_tensor(train_ngrams)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    val_data = ngrams_to_tensor(val_ngrams)
    val_loader = DataLoader(val_data, batch_size=batch_size) if val_text else None

    class NGramLanguageModeler(nn.Module):
        def __init__(
            self, vocab_size, embedding_dim, context_size, dropout_rate=dropout_rate
        ):
            super(NGramLanguageModeler, self).__init__()
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.dropout = nn.Dropout(dropout_rate)
            self.linear1 = nn.Linear(context_size * embedding_dim, 128)
            self.linear2 = nn.Linear(128, vocab_size)

        def forward(self, inputs):
            embeds = self.embeddings(inputs)
            embeds = embeds.view(-1, CONTEXT_SIZE * EMBEDDING_DIM)
            embeds = self.dropout(embeds)
            out = F.relu(self.linear1(embeds))
            out = self.linear2(out)
            log_probs = F.log_softmax(out, dim=1)
            return log_probs

    model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE).to("cuda")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_function = nn.NLLLoss()
    best_val_loss = float("inf")
    patience = 0

    for epoch in range(num_epochs):
        total_train_loss = 0
        for context_idxs, target_idxs in train_loader:
            context_idxs, target_idxs = context_idxs.to("cuda"), target_idxs.to("cuda")
            model.zero_grad()
            log_probs = model(context_idxs)
            loss = loss_function(log_probs, target_idxs)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch}: Train Loss: {avg_train_loss}")

        if val_loader:
            model.eval()  # Set the model to evaluation mode
            total_val_loss = 0
            with torch.no_grad():  # Disable gradient calculation
                for context_idxs, target_idxs in val_loader:
                    context_idxs, target_idxs = (
                        context_idxs.to("cuda"),
                        target_idxs.to("cuda"),
                    )
                    log_probs = model(context_idxs)
                    loss = loss_function(log_probs, target_idxs)
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                patience = 0
            else:
                patience += 1
            if patience > 2:  # Assuming patience threshold of 3
                print("Stopping early due to increasing val loss")
                break
            print(f"Epoch {epoch}: Val Loss: {avg_val_loss}")
            wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss})
            model.train()  # Set the model back to training mode

    # To get the embedding of a particular word, e.g. "beauty"

    embeddings_dict = {
        word: model.embeddings.weight[vocab[word]].detach().cpu().numpy().tolist()
        for word in vocab
    }
    model_weights = model.embeddings.weight.data.cpu()

    return model_weights, vocab, embeddings_dict


@app.function(
    image=image,
    gpu="any",
    timeout=10800,
    secrets=[modal.Secret.from_name("mw-wandb-secret")],
    mounts=[
        modal.Mount.from_local_dir(
            "../data/ngram", remote_path="/data/ngram"
        )
    ],
)
def train_nn(
    train_sentences,
    targets,
    pretrained_vocab,
    pretrained_embedding_weights,
    val_vrefs=None,
    val_sentences=None,
    val_targets=None,
    batch_size=64,
    weight_decay=1e-6,
    dropout_rate=0.5,
    lr=0.001,
    num_epochs=10,
    filename="",
):
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torch.nn.utils.rnn import pad_sequence
    import torch.nn as nn
    from torch.optim import Adam
    import wandb
    import json

    wandb.init(
        project="learn-embeddings-nn",
        config={
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "dropout_rate": dropout_rate,
            "lr": lr,
            "num_epochs": num_epochs,
            "filename": filename,
        },
    )

    class SentenceDataset(Dataset):
        def __init__(self, sentences, targets):
            self.sentences = [self.tokenize(sentence) for sentence in sentences]
            self.targets = [
                torch.tensor(target, dtype=torch.float32) for target in targets
            ]

        def tokenize(self, sentence):
            tokens = sentence.lower().split()  # simple whitespace tokenization
            return torch.tensor(
                [vocab[word] for word in tokens if word in vocab],
                dtype=torch.long,
            )

        def __len__(self):
            return len(self.sentences)

        def __getitem__(self, idx):
            return self.sentences[idx], self.targets[idx]

    def collate_batch(batch):
        sentence_tensors, target_tensors = zip(*batch)
        sentence_tensors = pad_sequence(
            sentence_tensors, batch_first=True, padding_value=0
        )
        target_tensors = torch.stack(target_tensors)
        return sentence_tensors, target_tensors

    class SentenceEncoder(nn.Module):
        def __init__(
            self, vocab, embed_dim, hidden_dim, output_dim=1, dropout_rate=0.5
        ):
            super().__init__()
            self.embedding = nn.Embedding(len(vocab) + 1, embed_dim)
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            self.linear = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, x):
            embedded = self.embedding(x)
            _, (hidden, _) = self.rnn(embedded)
            output = self.linear(hidden[-1])
            output = self.dropout(output)
            return output

    # Prepare datasets and dataloaders
    train_words = set(
        word for sentence in train_sentences for word in sentence.lower().split()
    )
    vocab = {
        word: idx + 1 for idx, word in enumerate(train_words)
    }  # 0 is reserved for padding
    idx_to_word = {idx: word for word, idx in vocab.items()}

    train_dataset = SentenceDataset(train_sentences, targets)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=collate_batch
    )

    if val_sentences:
        print(f'{val_vrefs[:10]}/n{val_sentences[:10]}')
        val_dataset = SentenceDataset(val_sentences, val_targets)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=False
        )

    model = SentenceEncoder(
        vocab, embed_dim=256, hidden_dim=512, output_dim=768, dropout_rate=dropout_rate
    ).to("cuda")

    # Load the pre-trained embedding weights
    for word, old_idx in pretrained_vocab.items():
        if word in vocab:
            new_idx = vocab[word]
            model.embedding.weight.data[new_idx] = pretrained_embedding_weights[old_idx]

    model.embedding = model.embedding.to("cuda")

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Training loop
    patience = 0
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        for sentences, targets in train_loader:
            sentences, targets = sentences.to("cuda"), targets.to("cuda")
            optimizer.zero_grad()
            outputs = model(sentences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if val_sentences:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for sentences, targets in val_loader:
                    sentences, targets = sentences.to("cuda"), targets.to("cuda")
                    outputs = model(sentences)
                    val_loss += criterion(outputs, targets).item()
                val_loss /= len(val_loader)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                else:
                    patience += 1

            print(
                f"Epoch {epoch+1}, Training Loss: {loss.item()}, Validation Loss: {val_loss}"
            )
            wandb.log({"train_loss": loss.item(), "val_loss": val_loss})
            if patience > 1:
                print("Early stopping")
                break
        else:
            print(f"Epoch {epoch+1}, Training Loss: {loss.item()}")
            wandb.log({"train_loss": loss.item()})

    if val_sentences:
        losses = []
        model.eval()
        overall_index = 0  # Initialize a counter to track the index across all batches
        with torch.no_grad():
            for sentences, targets in val_loader:
                sentences, targets = sentences.to("cuda"), targets.to("cuda")
                outputs = model(sentences)
                batch_losses = [
                    (
                        sentence,
                        criterion(outputs[i : i + 1], targets[i : i + 1]).item(),
                        val_vrefs[overall_index + i],  # Use overall_index to access the correct vref
                    )
                    for i, sentence in enumerate(sentences)
                ]
                losses.extend(batch_losses)
                overall_index += len(sentences)  # Update overall_index by the number of sentences in the current batch

        # Function to convert indices to words
        def indices_to_words(sentence):
            return " ".join(
                [
                    idx_to_word.get(idx.item(), "<UNK>")
                    for idx in sentence
                    # if idx.item() in idx_to_word
                ]
            ).strip()

        # Sort sentences by loss
        losses.sort(key=lambda x: x[1], reverse=True)  # Sort by loss value, highest first

        # Extract highest and lowest loss sentences
        highest_loss_sentences = [
            (vref, indices_to_words(sentence), loss)
            for sentence, loss, vref in losses[:10]
        ]
        lowest_loss_sentences = [
            (vref, indices_to_words(sentence), loss)
            for sentence, loss, vref in losses[-10:]
        ]

        print("Highest loss sentences:")
        for vref, sentence, loss in highest_loss_sentences:
            print(f"{vref}: {sentence} with loss {loss}")

        print("Lowest loss sentences:")
        for vref, sentence, loss in lowest_loss_sentences:
            print(f"{vref}: {sentence} with loss {loss}")


    embeddings_dict = {
        word: model.embedding.weight[vocab[word]].detach().cpu().numpy().tolist()
        for word in vocab
    }

    return embeddings_dict


@app.local_entrypoint()
def main(filename, nn_only=False):
    import pandas as pd
    import json
    import torch
    from pathlib import Path

    filename = Path(filename)
    if not nn_only:
        df = pd.read_csv(filename)
        print(df.head())

        df = df.sample(frac=1).reset_index(drop=True)

        # Define the size of the train set
        train_size = int(0.9 * len(df))  # 90% of the data

        # Split the data
        train_df = df[:train_size]
        val_df = df[train_size:]

        # Join texts for training and validation
        train_text = " ".join(train_df["text"].to_list())
        val_text = " ".join(val_df["text"].to_list())
        model_weights, vocab, embeddings_dict = train_embeddings.remote(
            train_text,
            val_text=val_text,
            batch_size=64,
            weight_decay=1e-6,
            dropout_rate=0.3,
            lr=0.003,
            num_epochs=30,
            filename=filename.stem,
        )
        with open(
            f"../data/ngram/vocab/vocabulary_{filename.stem}_001.json", "w"
        ) as vocab_file:
            json.dump(vocab, vocab_file)
        with open(f"embeddings_dict_{filename.stem}_.json", "w") as f:
            json.dump(embeddings_dict, f)
        torch.save(
            model_weights,
            f"../data/ngram/weights/embedding_weights__{filename.stem}_001.pt",
        )
    else:
        df = pd.read_csv(filename)
        df = df.sample(frac=1).reset_index(drop=True)
        train_size = int(0.9 * len(df))  # 90% of the data

        with open(
            f"../data/ngram/vocab/vocabulary_{filename.stem}_001.json", "r"
        ) as vocab_file:
            vocab = json.load(vocab_file)
        model_weights = torch.load(
            f"../data/ngram/weights/embedding_weights__{filename.stem}_001.pt"
        )

    em_df = pd.read_parquet("../fixtures/BSB_embeddings.parquet")
    df = df.merge(em_df[["vref", "embeddings"]], on="vref", how="inner")
    # Define the size of the train set
    train_size = int(0.9 * len(df))  # 90% of the data
    print(
        f"Training on {train_size} samples, validating on {len(df) - train_size} samples"
    )
    print(f"First 5 samples: {df.head()}")

    # Split the data

    sentences = df["text"].tolist()
    targets = df["embeddings"].tolist()
    train_sentences = sentences[:train_size]
    # val_sentences = sentences[train_size:]
    val_sentences = sentences
    train_targets = targets[:train_size]
    # val_targets = targets[train_size:]
    val_targets = targets
    # val_vrefs = df["vref"].tolist()[train_size:]
    val_vrefs = df["vref"].tolist()

    config = {
        "batch_size": 64,
        "weight_decay": 1e-9,
        "dropout_rate": 0.1,
        "lr": 0.001,
        "num_epochs": 30,
        "filename": filename.stem,
    }

    embeddings_dict = train_nn.remote(
        train_sentences,
        train_targets,
        vocab,
        model_weights,
        val_vrefs=val_vrefs,
        val_sentences=val_sentences,
        val_targets=val_targets,
        **config,
    )

    with open(f"../output/nn_embeddings_dict_{filename.stem}.json", "w") as f:
        json.dump(embeddings_dict, f)
