import modal

emb_app = modal.App("train-embeddings")

image = modal.Image.debian_slim(python_version="3.10").pip_install("torch")


@emb_app.function(
    image=image,
    gpu="any",
    timeout=3600,
)
def train_embeddings(
    train_text,
    val_text=None,
    batch_size=64,
    weight_decay=1e-6,
    dropout_rate=0.5,
    lr=0.001,
    num_epochs=30,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

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

        print(f"Epoch {epoch}: Train Loss: {total_train_loss / len(train_loader)}")

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
            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                patience = 0
            else:
                patience += 1
            if patience > 2:  # Assuming patience threshold of 3
                print("Stopping early due to increasing val loss")
                break
            print(f"Epoch {epoch}: Val Loss: {total_val_loss / len(val_loader)}")
            model.train()  # Set the model back to training mode

    # To get the embedding of a particular word, e.g. "beauty"

    embeddings_dict = {
        word: model.embeddings.weight[vocab[word]].detach().cpu().numpy().tolist()
        for word in vocab
    }
    model_weights = model.embeddings.weight.data.cpu()

    return model_weights, vocab, embeddings_dict


@emb_app.local_entrypoint()
def main():
    import pandas as pd
    import json
    import torch

    df = pd.read_csv("BSB_texts.csv")
    # df = df.iloc[23213:]  # NT only
    # Shuffle the DataFrame
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
        dropout_rate=0.7,
        lr=0.003,
        num_epochs=30,
    )
    with open("data/ngram_model/vocab/vocabulary_001.json", "w") as vocab_file:
        json.dump(vocab, vocab_file)
    with open("embeddings_dict.json", "w") as f:
        json.dump(embeddings_dict, f)
    torch.save(model_weights, "data/ngram_model/weights/embedding_weights_001.pt")
