import torch
import gc

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from dataset import user_create_prompt, construct_a_prompt_st_linkedin, construct_a_prompt_st


class UserAdapter(nn.Module):
    """
    Neural network module (supervised part of CBF-U)
    [user-adapter] in the Figure 2 of the corresponding paper

    Args:
        dim: int - dimensionality of input and output
    """
    def __init__(self, dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))  # small correction

    def forward(self, user_emb):
        user_emb = F.normalize(user_emb, p=2, dim=1)
        out = user_emb + self.alpha * self.net(user_emb)
        return F.normalize(out, p=2, dim=1)
    


def compute_all_st_users(
        df: pd.DataFrame,
        df_history: pd.DataFrame,
        model,
        tokenizer,
        device: str = "cuda",
        batch_size: int = 64,
        max_length: int = 1024,
        name: str = "window2"
    ):
    """
    Computes SBERT embeddings users for all users and saves to CSV.

    Args:
        df: pd.DataFrame - user data
        df_history: pd.DataFrame - user work history
        model: transformer model (JinaAI in our case)
        tokenizer: tokenizer compatible with the model
        device: str - device to run computations on
        batch_size: int - number of users in batch
        max_length: int - model's max token length
        name: str - prefix for saved CSV file

    Returns:
        np.ndarray:
            Computed embeddings for all users
    """
    df = df.reset_index(drop=True)
    prompts = [user_create_prompt(row, df_history) for _, row in tqdm(df.iterrows())]  # AI
    semantics = np.zeros((len(prompts), model.config.hidden_size), dtype=np.float32)

    for start in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[start:start + batch_size]
        semantics[start:(start+len(batch))] = compute_ST_batch(batch, model, tokenizer, device=device, max_length=max_length)
        if start % 200 == 0 and start > 0:
            gc.collect()
            torch.cuda.empty_cache()  

    pd.DataFrame(semantics).to_csv(f"{name}_users.csv", index=False)
    return semantics


def compute_all_st_jobs_linkedin(
        df: pd.DataFrame,
        model,
        tokenizer,
        device: str = "cuda",
        batch_size: int = 32,
        max_length: int = 2048,
        name: str = "linkedin"
    ):
    """
    Computes SBERT embeddings for a dataset LinkedIn Job Postings (2023 - 2024) and saves to CSV.
    Works with jobs.

    Args:
        df: pd.DataFrame - job data
        model: transformer model (JinaAI in our case)
        tokenizer: tokenizer compatible with the model
        device: str - device to run computations on
        batch_size: int - number of jobs in batch
        max_length: int - model's max token length
        name: str - prefix for CSV file

    Returns:
        pd.DataFrame:
            All jobs indexed by the df index
    """
    df = df.copy()
    prompts = [construct_a_prompt_st_linkedin(row) for _, row in df.iterrows()]
    semantics = np.zeros((len(prompts), model.config.hidden_size), dtype=np.float32)
    for start in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[start:start + batch_size]
        semantics[start:(start+len(batch))] = compute_ST_batch(batch, model, tokenizer, device=device, max_length=max_length)
        if start % 200 == 0 and start > 0:
            gc.collect()
            torch.cuda.empty_cache()  

    semantics_df = pd.DataFrame(semantics, index=df.index)
    semantics_df.to_csv(f"{name}.csv")
    return semantics_df


def compute_all_st_jobs(
        df: pd.DataFrame,
        model,
        tokenizer,
        device: str = "cuda",
        batch_size: int = 32,
        max_length: int = 2048,
        name: str = "window1"
    ):
    """
    Computes SBERT embeddings jobs for a a dataset CareerBuilder and saves to CSV.

    Args:
        df: pd.DataFrame - job data
        model: transformer model (JinaAI in our case)
        tokenizer: tokenizer compatible with the model
        device: str - device to run computations on
        batch_size: int - number of jobs in batch
        max_length: int - model's max token length
        name: str - prefix for saved CSV file

    Returns:
        pd.DataFrame:
            All jobs indexed by the df index
    """
    df = df.reset_index(drop=True)
    prompts = [construct_a_prompt_st(row) for _, row in tqdm(df.iterrows())]
    semantics = np.zeros((len(prompts), model.config.hidden_size), dtype=np.float32)
    for start in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[start:start + batch_size]
        semantics[start:(start+len(batch))] = compute_ST_batch(batch, model, tokenizer, device=device, max_length=max_length)
        if start % 200 == 0 and start > 0:
            gc.collect()
            torch.cuda.empty_cache()  

    pd.DataFrame(semantics).to_csv(f"{name}.csv", index=False)
    return semantics

def mean_pool(
        last_hidden_state: torch.Tensor, 
        attention_mask: torch.Tensor
    ):
    """
    Performs mean pooling on token embeddings.

    Args:
        last_hidden_state: torch.Tensor - model token embeddings (batch_size, seq_len, hidden_dim)
        attention_mask: torch.Tensor - mask with valid tokens (batch_size, seq_len)

    Returns:
        torch.Tensor:
            Mean-pooled embeddings per sequence (batch_size, hidden_dim)
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state*mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.inference_mode()
def compute_ST_batch(
        texts: list[str],
        model,
        tokenizer,
        device: str = "cuda",
        max_length: int = 2048
    ):
    """
    Computes SBERT embeddings for a single batch of texts.

    used as part of compute_all_st_jobs and compute_all_st_jobs_linkedin and compute_all_st_users.

    Args:
        texts: list[str] - input sentences or prompts
        model: transformer model - outputs hidden states
        tokenizer: tokenizer compatible with the model
        device: str - device to run computations on
        max_length: int - maximum token length for truncation

    Returns:
        np.ndarray:
            L2-normalized embeddings of shape (batch_size, hidden_dim) as float32
    """
    encoded = tokenizer(texts,padding=True,truncation=True, max_length=max_length,return_tensors="pt")
    encoded = encoded.to(device)
    out = model(**encoded)
    emb = mean_pool(out.last_hidden_state, encoded["attention_mask"])
    emb = F.normalize(emb, p=2, dim=1)
    return emb.detach().cpu().numpy().astype(np.float32)
