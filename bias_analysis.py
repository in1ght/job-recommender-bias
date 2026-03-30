import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from textstat import flesch_reading_ease

from dataset import user_prompt_basic
from models import compute_ST_batch

LABELS = [
    "Extroversion",
    "Neuroticism",
    "Agreeableness",
    "Conscientiousness",
    "Openness",
]

def recommend_topk_synthetic_users(
        model,
        user_vectors,
        jobs_df,  
        jobs_meta, 
        k: int = 10,
        device: str = "cuda",
        return_similarity: bool = False
    ):
    """
    Recommends top-k jobs for synthetic users

    Args:
        model: torch.nn.Module - trained UserAdapter
        user_vectors: array-like / torch.Tensor / pd.DataFrame - user embeddings (X, 512)
        jobs_df: pd.DataFrame - job embeddings indexed by id (Y, 512)
        jobs_meta: pd.DataFrame - job metadata
        k: int - max number top jobs to recommend to user
        device: str - device to run computations on
        return_similarity: bool - if True, additionally returns full similarity matrix

    Returns:
        dict or tuple of dict and pd.DataFrame:
            results: dict - keys=user labels, values=dict with top-k job IDs, avg salary, mode experience
            if return_similarity=True, sim_df (optional): pd.DataFrame - similarity scores 
    """
    # user vectors -> torch
    if isinstance(user_vectors, pd.DataFrame):
        user_mat_np = user_vectors.values
        user_labels = user_vectors.index.astype(str).tolist()
    else:
        user_mat_np = np.asarray(user_vectors)
        user_labels = [f"user_{i}" for i in range(user_mat_np.shape[0])]

    if user_mat_np.ndim != 2:
        raise ValueError(f"user_vectors must be 2D (X, 512). Got shape {user_mat_np.shape}")
    if user_mat_np.shape[1] != jobs_df.shape[1]:
        raise ValueError(f"user_vectors dim {user_mat_np.shape[1]} must match jobs embeddings dim {jobs_df.shape[1]}")

    user_mat = torch.as_tensor(user_mat_np, dtype=torch.float32, device=device)

    # jobs embeddings (fixed) 
    job_ids = jobs_df.index.to_numpy()
    job_mat = torch.as_tensor(jobs_df.values, dtype=torch.float32, device=device)
    job_mat = F.normalize(job_mat, p=2, dim=1)

    # project users with trained adapter (jobs untouched)
    model = model.to(device).eval()
    with torch.no_grad():
        user_proj = model(user_mat)
        user_proj = F.normalize(user_proj, p=2, dim=1)

        sims = (user_proj @ job_mat.T).detach().cpu().numpy()  # (X, Y)
    # metadata lookup
    if "JobID" not in jobs_meta.columns:
        raise ValueError("jobs_meta must contain a 'JobID' column.")
    meta = jobs_meta.set_index("JobID", drop=False)

    results = {}
    for ui, ulabel in enumerate(user_labels):
        row = sims[ui]
        kk = min(k, row.shape[0])

        top_idx = np.argpartition(-row, kth=kk-1)[:kk]
        top_idx = top_idx[np.argsort(-row[top_idx])]  # sorted by similarity desc
        rec_jobids = job_ids[top_idx].tolist()

        top_meta = meta.loc[meta.index.intersection(rec_jobids)]

        # avg normalized_salary
        if "normalized_salary" in top_meta.columns:
            avg_salary = top_meta["normalized_salary"].dropna().mean()
            avg_salary = float(avg_salary) if pd.notna(avg_salary) else np.nan
        else:
            avg_salary = np.nan

        # mode formatted_experience_level
        if "formatted_experience_level" in top_meta.columns:
            m = top_meta["formatted_experience_level"].dropna().mode()
            mode_exp = m.iloc[0] if len(m) else None
        else:
            mode_exp = None

        results[ulabel] = {
            "recommended_jobids": rec_jobids,
            "avg_normalized_salary_topk": avg_salary,
            "mode_formatted_experience_level_topk": mode_exp,
        }

    if return_similarity:
        sim_df = pd.DataFrame(sims, index=user_labels, columns=job_ids)
        return results, sim_df

    return results


def synth_users_embed_simple(
        users_df: pd.DataFrame, 
        n_users: int, 
        bias_name: str, 
        bias_groups: list, 
        model,
        tokenizer,
        batch_size: int = 64, 
        max_length: int = 512, 
        seed: int = 0, 
        device: str = "cuda"
    ):
    """
    Generates synthetic user embeddings.
    Applyies bias variations to base prompts (simplified)

    Args:
        users_df: pd.DataFrame - user data
        n_users: int - number of users to sample
        bias_name: str - name of bias attribute
        bias_groups: list - possible bias values to apply
        model: transformer model - outputs hidden states
        tokenizer: tokenizer compatible with the model
        batch_size: int - batch size
        max_length: int - max token length for embeddings
        seed: int - random seed for reproducibility
        device: str - device to run computations on

    Returns:
        tuple:
            vecs: np.ndarray - embeddings for all synthetic users
            pd.DataFrame - metadata for synthetic users (includes bias info)
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(users_df), size=min(n_users, len(users_df)), replace=False)
    sampled = users_df.iloc[idx].reset_index(drop=True)

    prompts, meta = [], []
    for i, row in sampled.iterrows():
        base = user_prompt_basic(row)

        # no bias
        prompts.append(base)
        meta.append({"synthetic_id": f"u{i}_none", "bias": "none", "value": "none"})

        # bias groups
        for g in bias_groups:
            prompts.append(f"{base} {bias_name}: {g}.")
            meta.append({"synthetic_id": f"u{i}_{bias_name}_{g}", "bias": bias_name, "value": g})

    vecs = []
    for start in range(0, len(prompts), batch_size):
        vecs.append(compute_ST_batch(prompts[start:start+batch_size], model, tokenizer, device=device, max_length=max_length))
    vecs = np.vstack(vecs).astype(np.float32)

    return vecs, pd.DataFrame(meta)

def bias_eval_table(
        model,
        user_vecs,
        meta: pd.DataFrame,
        jobs_embeddings_df: pd.DataFrame,
        jobs_meta_df: pd.DataFrame,
        k: int = 20,
        device: str = "cuda"
    ):
    """
    Creates a bias evaluation table with link from synthetic users to their top-k recommended jobs

    Args:
        model: torch.nn.Module - trained UserAdapter
        user_vecs: np.ndarray or pd.DataFrame - synthetic user embeddings 
        meta: pd.DataFrame - synthetic user metadata
        jobs_embeddings_df: pd.DataFrame - job embeddings indexed by id
        jobs_meta_df: pd.DataFrame - job metadata
        k: int - max number of top jobs to recommend
        device: str - device to run computations on

    Returns:
        pd.DataFrame:
            Users metadata merged with their recommended job IDs
    """
    recs = recommend_topk_synthetic_users(
        model, user_vecs, jobs_embeddings_df, jobs_meta_df, k=k, device=device
    )

    rows = [
        {
            "synthetic_id": sid,
            "recommended_jobids": recs[f"user_{i}"]["recommended_jobids"],
        } for i, sid in enumerate(meta["synthetic_id"])
    ]

    merged = meta.merge(pd.DataFrame(rows), on="synthetic_id", how="left")
    return merged


def run_bias_suite(
        model,
        model_emb,
        tokenizer,
        users,
        linkedin_semantics: pd.DataFrame,
        postings_linkedin_edited: pd.DataFrame,
        specs: list,
        n_users: int = 10,
        k: int = 20,
        seed: int = 0,
        device: str = "cuda",
    ):
    """
    Executes a suite of bias evaluations: generates synthetic users for each bias specification provided.

    Args:
        model: torch.nn.Module - trained UserAdapter
        model_emb: transformer model (SBERT) - outputs hidden states
        tokenizer: tokenizer compatible with the model_emb
        users: pd.DataFrame - user data
        linkedin_semantics: pd.DataFrame - embeddings (job)
        postings_linkedin_edited: pd.DataFrame - metadata (job)
        specs: list of tuples - each: (bias_name, bias_groups)
        n_users: int - number of synthetic users per bias group
        k: int - top-k recs to evaluate
        seed: int - random seed for reproducibility
        device: str - device to run computations on

    Returns:
        dict:
            Keys as bias names, values as df with synthetic user metadata and recommended jobs
    """
    outputs = {}

    for bias_name, groups in tqdm(specs):

        user_vecs, meta = synth_users_embed_simple(
            users_df=users,
            n_users=n_users,
            bias_name=bias_name,
            bias_groups=groups,
            model=model_emb,
            tokenizer=tokenizer,
            batch_size=64,
            max_length=512,
            seed=seed,
            device=device,
        )

        merged = bias_eval_table(
            model=model,
            user_vecs=user_vecs,
            meta=meta,
            jobs_embeddings_df=linkedin_semantics,
            jobs_meta_df=postings_linkedin_edited,
            k=k,
            device=device,
        )

        outputs[bias_name] = merged

    return outputs


def plot_bias_summary_table(
        summary_table: pd.DataFrame, 
        title: str = None
    ):
    """
    Plots a summary table of bias eval:
    shows average salary and mode experience.

    No reutrns, in-place, shows the plot.

    Args:
        summary_table: pd.DataFrame - with 'bias', 'value', 'avg_salary', and 'mode_experience'
        title: str - optional plot title
    """
    df = summary_table.copy()
    df["label"] = df["bias"].astype(str) + " = " + df["value"].astype(str)

    x = np.arange(len(df))
    y = df["avg_salary"].values

    plt.figure(figsize=(10, 4))
    plt.bar(x, y)
    plt.xticks(x, df["label"], rotation=30, ha="right")
    plt.ylabel("Average normalized_salary (top-K)")
    plt.title(title if title is not None else "Bias audit summary")

    # annotate with mode_experience
    for i, (yi, exp) in enumerate(zip(y, df["mode_experience"].astype(str).values)):
        plt.text(i, yi, exp, ha="center", va="bottom", fontsize=8, rotation=90)

    plt.tight_layout()
    plt.show()



def results_to_table(results: dict, jobs_meta: pd.DataFrame, salary_fmt: str = "{:,.0f}"):
    """
    Aggregates bias evaluation results to a table with average salary and mode experience.
    
    Args:
        results: dict - output from run_bias_suite
        jobs_meta: pd.DataFrame - ()'job_id', 'normalized_salary', 'formatted_experience_level')
        salary_fmt: str - salaries format
        
    Returns:
        pd.DataFrame with columns: ['Bias Category', 'Group', 'Avg. Salary', 'Mode Experience']
    """
    all_rows = []

    for bias_name, df in results.items():
        df = df.copy()

        for group_val, group_df in df.groupby("value"):
            avg_salaries_per_user = []
            mode_experience_per_user = []

            for _, user_row in group_df.iterrows():
                job_ids = user_row["recommended_jobids"]
                jobs_sel = jobs_meta.loc[job_ids]
        
                avg_sal = jobs_sel["normalized_salary"].dropna().mean()
                avg_salaries_per_user.append(avg_sal)

                mode_exp = jobs_sel["formatted_experience_level"].dropna().mode()
                mode_experience_per_user.append(mode_exp.iloc[0])

            group_avg_salary = np.mean(avg_salaries_per_user)

            group_mode_exp_series = pd.Series(mode_experience_per_user).mode().iloc[0]

            all_rows.append({
                "Bias Category": bias_name,
                "Group": group_val,
                "Avg. Salary": salary_fmt.format(group_avg_salary),
                "Mode Experience": group_mode_exp_series
            })

    final_table = pd.DataFrame(all_rows)
    return final_table



@torch.no_grad()
def personality_detection(
        text: str,
        _personality_model,
        _personality_tokenizer,
        device: str = "cuda"
    ):
    """
    Predicts personality traits from a given text using a pre-trained model.

    Args:
        text: str - text to analyze
        _personality_model: torch.nn.Module - trained personality prediction model
        _personality_tokenizer: tokenizer compatible with the model
        device: str - device to run computations on

    Returns:
        dict:
            Personality trait labels to predicted scores mapping
    """
    _personality_model.to(device)

    inputs = _personality_tokenizer(
        text,
        truncation=True,
        padding=True,
        return_tensors="pt",
        max_length=512,
    ).to(device)

    outputs = _personality_model(**inputs)
    scores = outputs.logits.squeeze().detach().cpu().numpy()

    return {LABELS[i]: float(scores[i]) for i in range(len(LABELS))}


def summarize_bias_personality_readability(
        results: dict,
        jobs_meta: pd.DataFrame,
        personality_fn,
        personality_model,
        personailty_tokenizer,
        k: int = 20
    ):
    """
    Summarizes bias effects on recommended w.r.t. readability and personality scores.

    Args:
        results: dict - def run_bias_suite output, each value is a per-profile DataFrame
        jobs_meta: pd.DataFrame - metadata (job)
        personality_fn: callable - personality scores function
        personality_model: callable - personality scores model
        personailty_tokenizer: callable - personality scores tokenizer alsigned with the model
        k: int - number of top jobs to consider

    Returns:
        pd.DataFrame:
            Summary table grouped by bias and value with mean readability and personality trait scores
    """
    rows = []

    for _, df in tqdm(results.items()):  # df is already per-profile dataframe
        for _, r in tqdm(df.iterrows()):
            job_ids = r["recommended_jobids"][:k]
            texts = jobs_meta.loc[job_ids, "description"].dropna().astype(str)

            if texts.empty:
                continue

            read_scores = [flesch_reading_ease(t) for t in texts]

            pers = [personality_fn(t,personality_model,personailty_tokenizer) for t in texts]
            pers_avg = {name: float(np.mean([p[name] for p in pers])) for name in pers[0].keys()}

            rows.append({
                "bias": r["bias"],
                "value": r["value"],
                "readability": float(np.mean(read_scores)),
                **pers_avg,
            })

    df_out = pd.DataFrame(rows)
    summary = df_out.groupby(["bias", "value"], dropna=False).mean(numeric_only=True).reset_index()
    return summary
