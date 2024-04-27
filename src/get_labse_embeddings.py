import math
import os
from typing import Literal, Optional, List, Union

import modal
from pydantic import BaseModel


volume = modal.NetworkFileSystem.from_name("pytorch-model-vol", create_if_missing=True)
CACHE_PATH = "/root/model_cache"

stub = modal.Stub(
    "get-labse-embeddings",
    image=modal.Image.debian_slim()
    .pip_install(
        "pandas~=1.5.0", "torch~=2.1.0", "transformers~=4.34.0", "tqdm~=4.66.0"
    )
    .copy_mount(
        modal.Mount.from_local_file(
            local_path="fixtures/vref.txt", remote_path="/root/vref.txt"
        )
    )
)

class GetRevision:
    def __init__(self, revision_verses):
        self.revision = revision_verses
        self.vref = open("/root/vref.txt").read().splitlines()

    def check_vref(self):
        return len(self.revision) == len(self.vref)

    def get_revision(self):
        # check that draft and reference are the same length
        import pandas as pd

        if not self.check_matching_length():
            raise ValueError(
                f"draft and reference differ by {abs(len(self.reference)- len(self.revision))}"
            )
        # check that both draft and reference are the same length as vref
        elif not self.check_vref():
            raise ValueError("draft length doesn't match vref")
        else:
            # merge the two revisions together
            revision = pd.DataFrame(
                {"revision": self.revision},
                index=self.vref,
            )
            revision.index.name = "vref"

            return revision


@stub.function(
    timeout=7200,
    secrets=[modal.Secret.from_dict({"TRANSFORMERS_CACHE": CACHE_PATH})],
    network_file_systems={CACHE_PATH: volume},
)
def get_labse_model(cache_path=CACHE_PATH):
    from transformers import BertTokenizerFast, BertModel

    try:
        print("Trying to load model from cache...")
        semsim_model = BertModel.from_pretrained(
            "setu4993/LaBSE", cache_dir=cache_path
        ).eval()
    except OSError as e:
        print(e)
        print("Downloading model instead of using cache...")
        semsim_model = BertModel.from_pretrained(
            "setu4993/LaBSE", cache_dir=cache_path, force_download=True
        ).eval()
    print("Semantic model initialized...")

    try:
        semsim_tokenizer = BertTokenizerFast.from_pretrained(
            "setu4993/LaBSE", cache_dir=cache_path
        )
    except OSError as e:
        print(e)
        print("Downloading tokenizer instead of using cache...")
        semsim_tokenizer = BertTokenizerFast.from_pretrained(
            "setu4993/LaBSE", cache_dir=cache_path, force_download=True
        )
    print("Tokenizer initialized...")

    return semsim_model, semsim_tokenizer


@stub.function(timeout=600, retries=3, cpu=8)
def get_embeddings(
    rev_sents_output: List[str],
    semsim_model=None,
    semsim_tokenizer=None,
):
    import torch

    if semsim_model is None or semsim_tokenizer is None:
        semsim_model, semsim_tokenizer = get_labse_model.remote()
    rev_sents_input = semsim_tokenizer(
        rev_sents_output, return_tensors="pt", padding=True, truncation=True
    )
    with torch.no_grad():
        rev_sents_output = semsim_model(**rev_sents_input)

    rev_sents_embedding = rev_sents_output.pooler_output

    print(f"{rev_sents_embedding=}")

    return rev_sents_embedding


@stub.function()
def merge(revision_id, revision_verses):
    mr = GetRevision(revision_id, revision_verses)
    return mr.get_revision()


@stub.function(timeout=7200, network_file_systems={"/root/cache": volume}, cpu=8)
def assess(texts: List[dict]):
    from tqdm import tqdm
    import pandas as pd
    """
    texts = [
        {'vref': 'GEN 1:1', 'text': "In the beginning God created the heavens and the earth."},
        {'vref': 'GEN 1:2', 'text': "Now the earth was formless and void, and darkness was over the surface of the deep. And the Spirit of God was hovering over the surface of the waters."},
    ]
    """
    df = pd.DataFrame(texts)
    print(df.head())
    batch_size = 256
    rev_sents = df["text"].to_list()
    vrefs = df['vref'].to_list()
    rev_sents_batched = [
        rev_sents[i : i + batch_size] for i in range(0, len(rev_sents), batch_size)
    ]
    semsim_model, semsim_tokenizer = get_labse_model.remote()
    sim_scores = tqdm(
        get_embeddings.map(
            rev_sents_batched,
            kwargs={"semsim_model": semsim_model, "semsim_tokenizer": semsim_tokenizer},
        )
    )
    embeddings = [item for sublist in sim_scores for item in sublist]

    results = [
        {
            "vref": vrefs[j],
            "embeddings": embeddings[j].tolist(),
        }
        for j in range(len(vrefs))
    ]

    print(results[:20])

    return {"results": results}
