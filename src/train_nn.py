import modal
import json

nn_app = modal.App("train-embeddings-nn")

image = modal.Image.debian_slim(python_version="3.10").pip_install("torch", "wandb")


@nn_app.function(
    image=image,
    gpu="any",
    timeout=10800,
    secrets=[modal.Secret.from_name("mw-wandb-secret")],
    mounts=[modal.Mount.from_local_dir("data/ngram_model", remote_path="/data/ngram_model")]
    )
def train_nn(
    train_sentences,
    targets,
    val_sentences=None,
    val_targets=None,
    batch_size=64,
    weight_decay=1e-6,
    dropout_rate=0.5,
    lr=0.001,
    num_epochs=10,
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
        },
    )

    class SentenceDataset(Dataset):
        def __init__(self, sentences, targets):
            self.sentences = [self.tokenize(sentence) for sentence in sentences]
            self.targets = [torch.tensor(target, dtype=torch.float32) for target in targets]

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
        def __init__(self, vocab, embed_dim, hidden_dim, output_dim=1, dropout_rate=0.5):
            super().__init__()
            self.embedding = nn.Embedding(len(vocab)+1, embed_dim)
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
    vocab = {word: idx+1 for idx, word in enumerate(train_words)}  # 0 is reserved for padding
    idx_to_word = {idx: word for word, idx in vocab.items()}

    train_dataset = SentenceDataset(train_sentences, targets)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=collate_batch
    )

    if val_sentences:
        val_dataset = SentenceDataset(val_sentences, val_targets)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, collate_fn=collate_batch
        )
    with open('/data/ngram_model/vocab/vocabulary_001.json', 'r') as vocab_file:
        pretrained_vocab = json.load(vocab_file)
    
    pretrained_embedding_weights = torch.load('/data/ngram_model/weights/embedding_weights_001.pt')
    
    model = SentenceEncoder(
        vocab, embed_dim=256, hidden_dim=512, output_dim=768, dropout_rate=dropout_rate
    ).to('cuda')
    
    # Load the pre-trained embedding weights
    for word, old_idx in pretrained_vocab.items():
        if word in vocab:
            new_idx = vocab[word]
            model.embedding.weight.data[new_idx] = pretrained_embedding_weights[old_idx]
    
    model.embedding = model.embedding.to('cuda')

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Training loop
    patience = 0
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        for sentences, targets in train_loader:
            sentences, targets = sentences.to('cuda'), targets.to('cuda')
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
                    sentences, targets = sentences.to('cuda'), targets.to('cuda')
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
        with torch.no_grad():   
            for sentences, targets in val_loader:
                sentences, targets = sentences.to('cuda'), targets.to('cuda')
                outputs = model(sentences)
                losses.extend([(sentence, criterion(outputs[i:i+1], targets[i:i+1]).item()) for i, sentence in enumerate(sentences)])

        # Function to convert indices to words
        def indices_to_words(sentence):
            return ' '.join([idx_to_word.get(idx.item(), '<UNK>') for idx in sentence if idx.item() in idx_to_word]).strip()


        # Sort sentences by loss
        losses.sort(key=lambda x: x[1], reverse=True)  # Sort by loss value, highest first

        # Extract highest and lowest loss sentences
        highest_loss_sentences = [indices_to_words(sentence) for sentence, _ in losses[:10]]
        lowest_loss_sentences = [indices_to_words(sentence) for sentence, _ in losses[-10:]]

        print("Highest loss sentences:")
        for sentence in highest_loss_sentences:
            print(sentence)

        print("Lowest loss sentences:")
        for sentence in lowest_loss_sentences:
            print(sentence)

    embeddings_dict = {word: model.embedding.weight[vocab[word]].detach().cpu().numpy().tolist() for word in vocab}

    return embeddings_dict


@nn_app.local_entrypoint()
def main():
    import pandas as pd
    import wandb

    df = pd.read_parquet('BSB_embeddings.parquet')
    df = df.sample(frac=1).reset_index(drop=True)

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
    val_sentences = sentences[train_size:]
    train_targets = targets[:train_size]
    val_targets = targets[train_size:]
    
    config = {
        'batch_size': 64,
        'weight_decay': 1e-9,
        'dropout_rate': 0.1,
        'lr': 0.001,
        'num_epochs': 30,
    }
    
    embeddings_dict = train_nn.remote(
        train_sentences,
        train_targets,
        val_sentences=val_sentences,
        val_targets=val_targets,
        **config,
    )

    with open("nn_embeddings_dict.json", "w") as f:
        json.dump(embeddings_dict, f)