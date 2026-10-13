import argparse
import json
import math
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Dict
from tqdm import tqdm
from scipy.stats import spearmanr  # <--- Spearman Correlation 추가


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d_model)
        return x + self.pe[:, :x.size(1)]


class YieldPredictor(nn.Module):
    def __init__(self, vocab_size: int = 4000, d_model: int = 256, nhead: int = 4, num_layers: int = 2, max_len: int = 1024):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            batch_first=True,
            dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )
            
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        padding_mask = (input_ids == 0) # (B, N)
        
        x = self.embedding(input_ids)   # (B, N, d_model)
        x = self.pos_encoder(x)         # (B, N, d_model)
        
        x = self.encoder(x, src_key_padding_mask=padding_mask) # (B, N, d_model)
        
        mask_expanded = (~padding_mask).unsqueeze(-1).float()  # (B, N, 1)
        x_sum = torch.sum(x * mask_expanded, dim=1)            # (B, d_model)
        mask_sum = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        x_pool = x_sum / mask_sum                               # (B, d_model)
        
        yield_score = self.regressor(x_pool)                   # (B, 1)
        return yield_score


def clean_value(val) -> str:
    if val is None:
        return "NULL"
    if isinstance(val, list):
        val = ", ".join([str(v) for v in val if v])
    val_str = str(val).strip()
    return val_str if val_str else "NULL"

def prepare_formatted_data(raw_json_path: str, formatted_save_path: str) -> List[Dict]:
    if os.path.exists(formatted_save_path):
        print(f"Found existing formatted dataset: {formatted_save_path}")
        with open(formatted_save_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"Processing raw JSON file: {raw_json_path}")
    if not os.path.exists(raw_json_path):
        raise FileNotFoundError(f"Cannot find input JSON file at {raw_json_path}")

    with open(raw_json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    formatted_data = []
    for entry in tqdm(raw_data, desc="Formatting Data"):
        canonical_rxn = clean_value(entry.get("canonical_rxn"))
        solvent = clean_value(entry.get("solvent", entry.get("solvents")))
        reagent = clean_value(entry.get("reagent", entry.get("reagents")))
        catalyst = clean_value(entry.get("catalyst", entry.get("catalysts")))
        
        raw_yield = entry.get("yield", entry.get("yields"))
        if raw_yield is None or raw_yield == "":
            yield_val = 0.0
        else:
            try:
                yield_val = float(raw_yield)
                if yield_val > 1.0:
                    yield_val = yield_val / 100.0
            except (ValueError, TypeError):
                yield_val = 0.0

        prompt_text = (
            f"The canonical_rxn is {canonical_rxn}. "
            f"The solvent is {solvent}. "
            f"The reagent is {reagent}. "
            f"The catalyst is {catalyst}."
        )

        formatted_data.append({
            "text": prompt_text,
            "yield": yield_val
        })

    os.makedirs(os.path.dirname(formatted_save_path) if os.path.dirname(formatted_save_path) else ".", exist_ok=True)
    with open(formatted_save_path, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)
        
    print(f"Formatted dataset saved to {formatted_save_path} (Total: {len(formatted_data)} samples)")
    return formatted_data


class USPTOYieldDataset(Dataset):
    def __init__(self, data_list: List[Dict], vocab_map: Dict[str, int], seq_len: int = 512):
        self.data_list = data_list
        self.vocab_map = vocab_map
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        text = item["text"]
        yield_val = item["yield"]

        tokens = [self.vocab_map.get(char, self.vocab_map["[UNK]"]) for char in text]

        if len(tokens) > self.seq_len:
            tokens = tokens[:self.seq_len]
        else:
            tokens = tokens + [self.vocab_map["[PAD]"]] * (self.seq_len - len(tokens))

        input_ids = torch.tensor(tokens, dtype=torch.long)
        target = torch.tensor([yield_val], dtype=torch.float32)

        return input_ids, target

def build_vocab(data_list: List[Dict]) -> Dict[str, int]:
    unique_chars = set()
    for item in data_list:
        unique_chars.update(item["text"])
        
    vocab_map = {"[PAD]": 0, "[UNK]": 1}
    for char in sorted(unique_chars):
        if char not in vocab_map:
            vocab_map[char] = len(vocab_map)
    return vocab_map


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    sse = np.sum((y_true - y_pred) ** 2)
    
    r2 = 1.0 - (sse / sst) if sst > 1e-8 else 0.0
    
    rho, _ = spearmanr(y_true, y_pred)
    spearman_rho = float(rho) if not np.isnan(rho) else 0.0
    
    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "spearman_rho": spearman_rho
    }

def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def train_epoch(model, dataloader, criterion, optimizer, device, epoch: int, total_epochs: int):
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(
        dataloader, 
        desc=f"Train [{epoch:02d}/{total_epochs:02d}]", 
        leave=False, 
        ncols=100
    )
    
    for input_ids, targets in pbar:
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(input_ids)
        loss = criterion(predictions, targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss * input_ids.size(0)

        pbar.set_postfix({"Loss": f"{batch_loss:.5f}"})

    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device, desc: str = "Evaluating") -> Dict[str, float]:
    model.eval()
    
    all_targets = []
    all_predictions = []
    
    pbar = tqdm(dataloader, desc=desc, leave=False, ncols=100)
    
    with torch.no_grad():
        for input_ids, targets in pbar:
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            predictions = model(input_ids)

            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    y_true = np.vstack(all_targets).reshape(-1)
    y_pred = np.vstack(all_predictions).reshape(-1)

    metrics = calculate_metrics(y_true, y_pred)
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Train Yield Predictor with Formatted USPTO Dataset")
    parser.add_argument("--gpu", type=int, default=0, help="Target GPU index")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--seq_len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--raw_data_path", type=str, default="./preprocess/uspto_sample_extracted_nodict.json")
    parser.add_argument("--formatted_save_path", type=str, default="formatted_uspto_data.json")
    parser.add_argument("--save_path", type=str, default="yield_predictor.pt", help="Path to save trained model")
    args = parser.parse_args()

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device = torch.device(f"cuda:{args.gpu}" if args.gpu < num_gpus else "cuda:0")
    else:
        device = torch.device("cpu")

    print(f"[Yield Predictor Training] Started on: {device}")

    formatted_data = prepare_formatted_data(args.raw_data_path, args.formatted_save_path)
    
    vocab_map = build_vocab(formatted_data)
    VOCAB_SIZE = max(4000, len(vocab_map) + 10)
    SEQ_LEN = args.seq_len
    D_MODEL = 256

    print(f"Dataset Size: {len(formatted_data)} | Vocab Size: {len(vocab_map)} | Seq Len: {SEQ_LEN}")
    
    if len(formatted_data) > 0:
        print("\n[Sample Prompt Format Example]:")
        print(formatted_data[0]["text"])
        print("-" * 50)

    full_dataset = USPTOYieldDataset(formatted_data, vocab_map, seq_len=SEQ_LEN)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    print(f"Data Split -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


    model = YieldPredictor(vocab_size=VOCAB_SIZE, d_model=D_MODEL, max_len=SEQ_LEN).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_r2 = -float("inf")
    
    print("\n" + "="*120)
    print(f"{'Epoch':^7} | {'Train MSE':^11} | {'Val MSE':^10} | {'Val RMSE':^10} | {'Val MAE':^10} | {'Val R^2':^10} | {'Val Rho':^10} | {'Elapsed':^9} | {'ETA':^9}")
    print("="*120)
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        val_metrics = evaluate(model, val_loader, criterion, device, desc=f"Val   [{epoch:02d}/{args.epochs:02d}]")

        epoch_elapsed = time.time() - epoch_start_time
        total_elapsed = time.time() - start_time
        avg_epoch_time = total_elapsed / epoch
        remaining_epochs = args.epochs - epoch
        eta_seconds = remaining_epochs * avg_epoch_time

        print(f"[{epoch:02d}/{args.epochs:02d}] | "
              f"{train_loss:11.6f} | "
              f"{val_metrics['mse']:10.6f} | "
              f"{val_metrics['rmse']:10.6f} | "
              f"{val_metrics['mae']:10.6f} | "
              f"{val_metrics['r2']:10.4f} | "
              f"{val_metrics['spearman_rho']:10.4f} | "
              f"{format_time(epoch_elapsed):^9} | "
              f"{format_time(eta_seconds):^9}")

        if val_metrics['r2'] > best_val_r2:
            best_val_r2 = val_metrics['r2']
            torch.save(model.state_dict(), args.save_path)

    print("="*120)
    print(f"Training complete in {format_time(time.time() - start_time)}! Best model saved to: {args.save_path} (Best Val R^2: {best_val_r2:.4f})")

    if os.path.exists(args.save_path):
        model.load_state_dict(torch.load(args.save_path, map_location=device))
        test_metrics = evaluate(model, test_loader, criterion, device, desc="Testing Best Model")
        
        print("\n==================================================")
        print("[Final Test Performance Results]")
        print("==================================================")
        print(f"  • Spearman's Rho (Rank Correlation)        : {test_metrics['spearman_rho']:.4f}")
        print(f"  • R^2 Score (Coefficient of Determination) : {test_metrics['r2']:.4f}")
        print(f"  • MAE  (Mean Absolute Error)              : {test_metrics['mae']:.6f} ({test_metrics['mae']*100:.2f}%p)")
        print(f"  • RMSE (Root Mean Squared Error)          : {test_metrics['rmse']:.6f}")
        print(f"  • MSE  (Mean Squared Error)               : {test_metrics['mse']:.6f}")
        print("==================================================")

if __name__ == "__main__":
    main()
