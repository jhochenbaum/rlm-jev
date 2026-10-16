"""E5c: modern open retrieval baselines on the identical E5 corpus + needle placement (no API calls).

  qwen3-emb   : Qwen3-Embedding (query prompt), cosine over all sentences
  qwen3-emb+rr: top-100 from qwen3-emb reranked by Qwen3-Reranker (cross-encoder), the standard retrieve-then-rerank
Usage: uv run python experiments/e5c_modern_baselines.py [0.6b|4b|8b]   or   [embedding_model] [reranker_model]
"""

import json
import sys
import time
from pathlib import Path

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


import numpy as np

sys.path.insert(0, ".")
import torch

from jevlab.needles import load_corpus, load_needles, place_needles

# Presets: `e5c_modern_baselines.py 8b` == Qwen3-Embedding-8B + Qwen3-Reranker-8B (CrossEncoder ports by tomaarsen).
PRESETS = {
    t: (f"Qwen/Qwen3-Embedding-{t.upper()}", f"tomaarsen/Qwen3-Reranker-{t.upper()}-seq-cls")
    for t in ("0.6b", "4b", "8b")
}
_arg = sys.argv[1] if len(sys.argv) > 1 else "0.6b"
if _arg.lower() in PRESETS:
    EMB, RR = PRESETS[_arg.lower()]
else:
    EMB = _arg
    RR = sys.argv[2] if len(sys.argv) > 2 else "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"
TOPK = 100
DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# the same corpus and needle placement as e5_locate.py
needles = load_needles()
sents, idx = place_needles(load_corpus(750_000), needles)
queries = [n["query"] for n in needles]
kinds = np.array([n["kind"] for n in needles])

from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm.auto import tqdm

log(f"embedding={EMB} reranker={RR} device={DEV} | corpus {len(sents):,} sentences, {len(needles)} queries")

# Embedding stage is checkpointed, so a reranker failure never throws away the (slow) corpus embedding.
CKPT = Path("results") / f"e5c_ckpt_{EMB.split('/')[-1]}.npz"
if CKPT.exists():
    z = np.load(CKPT)
    emb_rank, TOP = z["emb_rank"], z["top"]
    print(f"loaded embedding checkpoint {CKPT}")
else:
    log("loading embedding model (first run downloads it)")
    m = SentenceTransformer(EMB, device=DEV, model_kwargs={"torch_dtype": torch.float16})
    log(f"embedding {len(sents):,} sentences (progress bar below; batches of 64)")
    E = m.encode(sents, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    Q = m.encode(queries, prompt_name="query", normalize_embeddings=True)
    S = Q @ E.T
    emb_rank = np.array([int((S[i] > S[i, idx[i]]).sum()) + 1 for i in range(len(needles))])
    TOP = np.stack([np.argsort(-S[i])[:TOPK] for i in range(len(needles))])
    np.savez(CKPT, emb_rank=emb_rank, top=TOP)
    log(f"embedding done; checkpoint saved to {CKPT}  (R@5 = {(emb_rank <= 5).mean():.2f})")
    del m

task = "Given a question, find the sentence that answers it"
prefix = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
    'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
log("loading reranker (first run downloads it)")
ce = CrossEncoder(RR, device=DEV, model_kwargs={"torch_dtype": torch.float16})
# The 4B/8B ports ship a chat template in tokenizer_config.json (the 0.6B port does not), and recent
# sentence-transformers then routes (query, document) pairs through a chat template. We already build the full
# Qwen3-Reranker prompt ourselves (prefix/suffix below), so use a template that just joins the two parts in order.
ce.processor.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
rr_rank = []
log(f"reranking top-{TOPK} for each of {len(queries)} queries")
for i, q in enumerate(tqdm(queries, desc="rerank", unit="query")):
    top = TOP[i]
    if idx[i] not in set(top.tolist()):
        rr_rank.append(10**6)
        continue
    pairs = [(f"{prefix}<Instruct>: {task}\n<Query>: {q}\n", f"<Document>: {sents[j]}{suffix}") for j in top]
    sc = ce.predict(pairs, batch_size=16, show_progress_bar=False)
    tgt = float(sc[top.tolist().index(idx[i])])
    rr_rank.append(int((sc > tgt).sum()) + 1)
rr_rank = np.array(rr_rank)


def table(rk):
    return {
        "R@1": float((rk == 1).mean()),
        "R@5": float((rk <= 5).mean()),
        "R@20": float((rk <= 20).mean()),
        "MRR": float((1 / rk).mean()),
        "R@5_by_kind": {k: float((rk[kinds == k] <= 5).mean()) for k in sorted(set(kinds))},
    }


out = {
    "embedding_model": EMB,
    "reranker": RR,
    "rerank_top_k": TOPK,
    "device": DEV,
    "emb": table(emb_rank),
    "emb+rerank": table(rr_rank),
    "emb_ranks": emb_rank.tolist(),
    "rr_ranks": rr_rank.tolist(),
}
name = EMB.split("/")[-1] + ("" if RR.endswith("0.6B-seq-cls") else "_" + RR.split("/")[-1])
Path("results").joinpath(f"e5c_{name}.json").write_text(json.dumps(out, indent=1))
log(f"done -> results/e5c_{name}.json")
print(json.dumps({k: out[k] for k in ("embedding_model", "emb", "emb+rerank")}, indent=1))
