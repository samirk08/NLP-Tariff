"""
N  →  N+1 mapping with a guaranteed-unique key

If an HS code exists we keep it.
Otherwise we synthesise
    {year}_{row_number}_{slugged_description}

so every row keeps its own identity.
"""

import os, pickle, numpy as np, pandas as pd, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from rapidfuzz import fuzz


# ---------------------------------------------------------------
# 0)  Model / tokenizer
# ---------------------------------------------------------------
def setup_model_and_tokenizer():
    token       = "REDACTED_HF_TOKEN"
    model_name  = "sentence-transformers/all-mpnet-base-v2"
    tok         = AutoTokenizer.from_pretrained(model_name, token=token)
    mdl         = AutoModel.from_pretrained(model_name, token=token)
    dev         = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(dev).eval()
    print("Using device:", dev)
    return tok, mdl, dev


# ---------------------------------------------------------------
# 1)  Similarity helpers
# ---------------------------------------------------------------
def norm_lev(a, b):
    return fuzz.ratio(a.strip().lower(), b.strip().lower()) / 100.0


def cos_sim(e1, e2):
    return F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()


def jaccard(t1, t2):
    return len(t1 & t2) / len(t1 | t2) if t1 and t2 else 0.0


# ---------------------------------------------------------------
# 2)  One-row matcher
# ---------------------------------------------------------------
def match_one(desc_N1, hs_N1, emb_N1,
              descs_N,  hs_Ns,   embs_N, token_cache,
              alpha=.7, beta=.2, gamma=.1, top_k=500):
    emb_N1 = emb_N1.unsqueeze(0)
    scores = F.cosine_similarity(emb_N1, embs_N, dim=1).cpu().numpy()
    top    = np.argsort(scores)[-top_k:][::-1]          # best→worst

    best, best_i, best_c, best_j, best_l = -1, -1, 0, 0, 0
    tokens_N1 = set(desc_N1.lower().split())
    for i in top:
        jacc = jaccard(tokens_N1, token_cache[i])
        lev  = norm_lev(desc_N1, descs_N[i])
        comb = alpha * scores[i] + beta * jacc + gamma * lev
        if comb > best:
            best, best_i, best_c, best_j, best_l = comb, i, scores[i], jacc, lev

    return {
        "HS_N1":        hs_N1,
        "Description_N1": desc_N1,
        "Mapped_HS":      hs_Ns[best_i],
        "Mapped_Description": descs_N[best_i],
        "Cosine_Similarity":   best_c,
        "Jaccard_Similarity":  best_j,
        "Levenshtein_Similarity": best_l,
        "Combined_Similarity": best
    }


# ---------------------------------------------------------------
# 3)  Parallel wrapper
# ---------------------------------------------------------------
def map_year_pair(df_N1, df_N, tok, mdl, dev,
                  alpha=.7, beta=.2, gamma=.1, top_k=500):
    # ----- lists -----
    desc_N1 = df_N1['brief_description'].tolist()
    key_N1  = df_N1['uniq_key'].tolist()
    desc_N  = df_N['brief_description'].tolist()
    key_N   = df_N['uniq_key'].tolist()

    # ----- embeddings -----
    yN1, yN = df_N1['year'].iat[0], df_N['year'].iat[0]
    emb_N1  = torch.as_tensor(
        pickle.load(open(f"embeddings/embeddings_{yN1}.pkl", "rb")),
        device=dev, dtype=torch.float32
    )
    emb_N   = torch.as_tensor(
        pickle.load(open(f"embeddings/embeddings_{yN}.pkl",  "rb")),
        device=dev, dtype=torch.float32
    )

    token_cache = [set(d.lower().split()) for d in desc_N]

    # ----- parallel -----
    out, mw = [], min(32, os.cpu_count()+4)
    with ThreadPoolExecutor(max_workers=mw) as ex:
        futs = [
            ex.submit(match_one, d1, k1, e1,
                      desc_N, key_N, emb_N, token_cache,
                      alpha, beta, gamma, top_k)
            for d1, k1, e1 in zip(desc_N1, key_N1, emb_N1)
        ]
        for f in tqdm(as_completed(futs), total=len(futs),
                      desc=f"{yN}→{yN1}"):
            out.append(f.result())
    return pd.DataFrame(out)


# ---------------------------------------------------------------
# 4)  Main driver
# ---------------------------------------------------------------
def slug(s, n=40):
    """simple slug of first n chars"""
    return ''.join(c for c in s.lower() if c.isalnum() or c == ' ')[:n].strip().replace(' ', '_')

def main():
    tok, mdl, dev = setup_model_and_tokenizer()

    df_all = pd.read_csv(
        "./all_tariffs.csv",
        dtype={'hs': str}
    ).dropna(subset=['brief_description'])

    # --- build unique key ---
    def make_key(row):
        if pd.notna(row.hs) and row.hs.strip():
            return row.hs.strip()
        return f"{row.year}_{row.name}_{slug(row.brief_description)}"

    df_all['uniq_key'] = df_all.apply(make_key, axis=1)

    years = sorted(df_all['year'].unique())            # ascending
    year_pairs = list(zip(years[::-1][1:], years[::-1][:-1]))   # 2022→2023, …

    for yN, yN1 in tqdm(year_pairs, desc="Year pairs"):
        output_filename = f"mapped_{yN}_{yN1}.csv"
        if os.path.exists(output_filename):
            print(f"{output_filename} exists, skipping {yN}→{yN1}")
            continue

        df_N  = df_all[df_all.year == yN ].copy()
        df_N1 = df_all[df_all.year == yN1].copy()
        if df_N.empty or df_N1.empty:
            print(f"skip {yN}→{yN1}")
            continue

    # ----------  swap the arguments here  ----------
        mapped = map_year_pair( df_N, df_N1,      # N → N+1
                                tok, mdl, dev,
                                alpha=.7, beta=.2, gamma=.1, top_k=500)
        
        mapped.to_csv(output_filename, index=False)
        print(f"saved {output_filename}")


if __name__ == "__main__":
    main()
