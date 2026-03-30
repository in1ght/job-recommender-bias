import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

def info_nce_user_to_job(
        user_emb: torch.Tensor,
        job_emb: torch.Tensor,
        temperature: float = 0.07
    ):
    """
    Computes InfoNCE loss between user and job for contrastive learning.

    Args:
        user_emb: torch.Tensor - embeddings - user (batch_size, hidden_dim)
        job_emb: torch.Tensor - embeddings - job (batch_size, hidden_dim)
        temperature: float - scaling factor for logits

    Returns:
        torch.Tensor:
            cross-entropy loss for the batch
    """
    logits = (user_emb @ job_emb.T) / temperature
    labels = torch.arange(user_emb.size(0), device=user_emb.device)
    return F.cross_entropy(logits, labels)


def train_model(
        model,
        train_loader,
        val_loader,
        num_epochs: int = 30,
        earl_stop: int = 4,
        lr: float = 0.0005,
        temperature: float = 0.07,
        show_progress: bool = True
    ):
    """
    Trains a UserAdapter using InfoNCE loss

    The main aim -> maximize cosine similarity between items in the interactions dataset.

    Args:
        model: torch.nn.Module - UserAdapter to train
        train_loader: DataLoader - train dataset loader
        val_loader: DataLoader - val dataset loader
        num_epochs: int - max epochs number
        earl_stop: int - early stopping patience
        lr: float - learning rate
        temperature: float - temperature for InfoNCE loss
        show_progress: bool - if True, displays progress bars

    Returns:
        torch.nn.Module:
            Trained model with best validation loss
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    patience = 0
    best_state = None

    for epoch in tqdm(range(num_epochs), disable=not show_progress):
        model.train()
        train_loss = 0.0

        for user_emb, job_emb in tqdm(train_loader, disable=not show_progress):
            user_emb = user_emb.to("cuda")
            job_emb = job_emb.to("cuda")

            optimizer.zero_grad()
            user_proj = model(user_emb)
            job_emb = F.normalize(job_emb, p=2, dim=1)

            loss = info_nce_user_to_job(user_proj, job_emb, temperature)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for user_emb, job_emb in val_loader:
                user_emb = user_emb.to("cuda")
                job_emb = job_emb.to("cuda")
                user_proj = model(user_emb)
                job_emb = F.normalize(job_emb, p=2, dim=1)
                val_loss += info_nce_user_to_job(user_proj, job_emb, temperature).item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= earl_stop:
                print("Early stopping")
                break

    model.load_state_dict(best_state)
    return model


def get_metric(
    model,
    test_dataloader: torch.utils.data.DataLoader,
    temperature: float = 0.07,
    network: bool = True,
    loss_: bool = True
) -> dict:
    """
    Evaluates a user-job recommendation model on a test dataset

    Args:
        model: torch.nn.Module - trained UserAdapter
        test_dataloader: DataLoader with (user_emb, job_emb) pairs
        temperature: float - temperature for cross-entropy loss
        network: bool - if True, sets model to eval and moves to CUDA
        loss_: bool - if True, computes cross-entropy loss

    Returns:
        dict:
            Metrics : loss, mean positive cosine, recall@1, Recall@5, and MRR
    """
    if network:
        model.eval().to("cuda")

    total_loss = 0.0
    total_cos = 0.0
    total_r1 = 0.0
    total_r5 = 0.0
    total_mrr = 0.0
    counter = 0

    with torch.no_grad():
        for user_emb, job_emb in test_dataloader:
            user_emb = user_emb.to("cuda")
            job_emb = job_emb.to("cuda")

            # user-side adaptation only
            u = model(user_emb)
            j = F.normalize(job_emb, p=2, dim=1)

            sims = u @ j.T  # cosine similarity

            # mean positive cosine (diagonal)
            total_cos += sims.diag().mean().item()

            # ranking per user
            ranks = torch.argsort(sims, dim=1, descending=True)
            targets = torch.arange(sims.size(0), device=sims.device).unsqueeze(1)
            match_pos = (ranks == targets).nonzero(as_tuple=False)[:, 1]

            total_r1 += (match_pos < 1).float().mean().item()
            total_r5 += (match_pos < 5).float().mean().item()
            total_mrr += (1.0 / (match_pos.float() + 1.0)).mean().item()

            if loss_:
                logits = sims / temperature
                labels = torch.arange(sims.size(0), device=sims.device)
                total_loss += F.cross_entropy(logits, labels).item()

            counter += 1

    if loss_:
        print("Loss =", total_loss / counter)
    print("Mean positive cosine =", total_cos / counter)
    print("Recall@1 =", total_r1 / counter)
    print("Recall@5 =", total_r5 / counter)
    print("MRR =", total_mrr / counter)

    return {
        "loss": (total_loss / counter) if loss_ else None,
        "mean_pos_cos": total_cos / counter,
        "recall@1": total_r1 / counter,
        "recall@5": total_r5 / counter,
        "mrr": total_mrr / counter,
    }
