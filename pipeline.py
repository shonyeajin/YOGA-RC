import argparse
import json
import os
import random
import math  # Positional Encoding용 추가
from typing import List, Set, Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm

# RDKit (Tanimoto 유사도 계산용)
try:
    from rdkit import Chem
    from rdkit import DataStructs
    from rdkit import RDLogger          
    RDLogger.DisableLog('rdApp.*')      
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit is not installed. Tanimoto metric will fall back to dummy scores.")


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
        return x + self.pe[:, :x.size(1)]



class A3PromptChemicalDataset(Dataset):
    def __init__(self, raw_data: list, tokenizer, max_len: int = 384):
        self.data = raw_data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.group_map = [
            ("catalyst1", 1, " , Catalyst is "),
            ("solvent1", 2, " , Solvent1 is "),
            ("solvent2", 3, " , Solvent2 is "),
            ("reagent1", 4, " , Reagent1 is "),
            ("reagent2", 5, " , Reagent2 is "),
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        comps = item["components"]
        meta = item.get("meta", {})  

        all_input_ids = []
        all_group_ids = []
        null_groups = []

        rxn_class = meta.get("rxn_class_name", "Unknown")
        template = meta.get("retro_template", "None")

        prefix_text = f"Reaction is {comps.get('canonical_rxn', '')} , Class is {rxn_class} , Template is {template}"
        
        p_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        all_input_ids.extend(p_ids)
        all_group_ids.extend([0] * len(p_ids))

        for key, g_id, label_text in self.group_map:
            lbl_ids = self.tokenizer.encode(label_text, add_special_tokens=False)
            all_input_ids.extend(lbl_ids)
            all_group_ids.extend([0] * len(lbl_ids))

            val_text = comps.get(key, "NULL")
            if val_text.strip() == "NULL":
                null_groups.append(g_id)
            
            val_ids = self.tokenizer.encode(val_text, add_special_tokens=False)
            all_input_ids.extend(val_ids)
            all_group_ids.extend([g_id] * len(val_ids))

        if len(all_input_ids) > self.max_len:
            all_input_ids = all_input_ids[:self.max_len]
            all_group_ids = all_group_ids[:self.max_len]
        else:
            pad_len = self.max_len - len(all_input_ids)
            all_input_ids.extend([self.tokenizer.pad_token_id] * pad_len)
            all_group_ids.extend([0] * pad_len)

        return {
            "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
            "group_ids": torch.tensor(all_group_ids, dtype=torch.long),
            "null_groups": null_groups,
            "components": comps
        }

def custom_collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    group_ids = torch.stack([item["group_ids"] for item in batch])
    null_groups = [item["null_groups"] for item in batch]
    components = [item["components"] for item in batch]
    return {
        "input_ids": input_ids,
        "group_ids": group_ids,
        "null_groups": null_groups,
        "components": components
    }



class YieldPredictor(nn.Module):
    def __init__(self, vocab_size: int, pad_token_id: int, d_model: int = 256, nhead: int = 4, num_layers: int = 2, max_len: int = 1024):
        super().__init__()
        self.pad_token_id = pad_token_id
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=pad_token_id)
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
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid() 
        )
            
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        padding_mask = (input_ids == self.pad_token_id)
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        
        # 수정된 부분: .float() 대신 .to(x.dtype) 사용
        mask_expanded = (~padding_mask).unsqueeze(-1).to(x.dtype)
        x_sum = torch.sum(x * mask_expanded, dim=1)
        mask_sum = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        x_pool = x_sum / mask_sum
        
        yield_score = self.regressor(x_pool)
        return yield_score



class A3ChemicalLlama(nn.Module):
    def __init__(self, pretrained_llama, pad_token_id: int):
        super().__init__()
        self.llama = pretrained_llama
        self.pad_token_id = pad_token_id

    def generate_a3_causal_mask(self, input_ids: torch.Tensor, group_ids: torch.Tensor, permutation: List[int]) -> torch.Tensor:
        B, N = group_ids.shape
        full_order = [0] + permutation
        priority = torch.zeros_like(group_ids)
        for rank, g_id in enumerate(full_order):
            priority[group_ids == g_id] = rank

        group_mask = (priority.unsqueeze(2) >= priority.unsqueeze(1)).float()
        non_pad_mask = (input_ids != self.pad_token_id).float().unsqueeze(1).unsqueeze(2) 
        combined_mask = group_mask.unsqueeze(1) * non_pad_mask 

        a3_mask = (1.0 - combined_mask) * -10000.0
        return a3_mask

    def forward(self, input_ids: torch.Tensor, group_ids: torch.Tensor, permutation: List[int] = None) -> torch.Tensor:
        if permutation is None:
            permutation = list(range(1, 6))
            random.shuffle(permutation)

        attention_mask = self.generate_a3_causal_mask(input_ids, group_ids, permutation)
        attention_mask = attention_mask.to(device=input_ids.device, dtype=self.llama.dtype)
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return outputs.logits



class A3InferenceManager:
    def __init__(self, generator: A3ChemicalLlama, yield_predictor: YieldPredictor, mask_token_id: int, device: torch.device):
        self.generator = generator
        self.yield_predictor = yield_predictor
        self.mask_token_id = mask_token_id
        self.device = device

    @torch.no_grad()
    def _autoregressive_decode_group(self, input_seq: torch.Tensor, group_ids: torch.Tensor, target_group: int, perm: List[int], temperature: float = 0.0) -> torch.Tensor:
        seq = input_seq.clone()
        B, N = seq.shape
        
        for b in range(B):
            target_indices = torch.where((group_ids[b] == target_group) & (seq[b] == self.mask_token_id))[0]
            if len(target_indices) == 0:
                continue
            
            for idx in target_indices:
                logits = self.generator(seq[b:b+1], group_ids[b:b+1], permutation=perm)
                
                if temperature > 0.0:
                    probs = torch.softmax(logits[0, idx, :] / temperature, dim=-1)
                    pred_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    pred_token = torch.argmax(logits[0, idx, :], dim=-1)
                seq[b, idx] = pred_token

        return seq

    @torch.no_grad()
    def run_inference(self, input_ids: torch.Tensor, group_ids: torch.Tensor, null_groups_list: List[List[int]], temperature: float = 0.0) -> torch.Tensor:
        B, N = input_ids.shape
        current_input_ids = input_ids.clone().to(self.device)
        group_ids = group_ids.to(self.device)

        batch_U = [set(range(1, 6)) - set(null_groups_list[b]) for b in range(B)]
        batch_history = [[] for _ in range(B)]

        for step in range(5):
            if all(len(U) == 0 for U in batch_U):
                break

            candidate_scores = torch.full((B, 6), -1.0, device=self.device)
            temp_virtual_sequences: Dict[int, torch.Tensor] = {}

            for i in range(1, 6):
                active_batches = [b for b in range(B) if i in batch_U[b]]
                if not active_batches:
                    continue

                virtual_input = current_input_ids.clone().to(self.device)
                
                for b in active_batches:
                    perm = batch_history[b] + [i]
                    decoded_b = self._autoregressive_decode_group(
                        virtual_input[b:b+1], group_ids[b:b+1], target_group=i, perm=perm, temperature=temperature
                    )
                    virtual_input[b] = decoded_b[0]
                
                temp_virtual_sequences[i] = virtual_input

            for i, v_seq in temp_virtual_sequences.items():
                scores = self.yield_predictor(v_seq)
                for b in range(B):
                    if i in batch_U[b]:
                        candidate_scores[b, i] = scores[b, 0]

            next_groups = torch.argmax(candidate_scores, dim=-1)
            for b in range(B):
                if len(batch_U[b]) == 0:
                    continue
                chosen_g = next_groups[b].item()
                target_mask = (group_ids[b] == chosen_g)
                current_input_ids[b, target_mask] = temp_virtual_sequences[chosen_g][b, target_mask]
                batch_U[b].remove(chosen_g)
                batch_history[b].append(chosen_g)

        return current_input_ids



def calculate_tanimoto_similarity(pred_smiles: str, target_smiles: str) -> float:
    if not RDKIT_AVAILABLE or target_smiles == "NULL" or pred_smiles == "NULL": return 0.0
    if pred_smiles == target_smiles: return 1.0
    try:
        mol_pred = Chem.MolFromSmiles(pred_smiles)
        mol_target = Chem.MolFromSmiles(target_smiles)
        if mol_pred is None or mol_target is None: return 0.0
        return DataStructs.TanimotoSimilarity(Chem.RDKFingerprint(mol_pred), Chem.RDKFingerprint(mol_target))
    except:
        return 0.0

def check_exact_match(pred_smiles: str, target_smiles: str) -> int:
    if pred_smiles == target_smiles: return 1
    if not RDKIT_AVAILABLE or target_smiles == "NULL" or pred_smiles == "NULL": return 0
    try:
        if Chem.CanonSmiles(pred_smiles) == Chem.CanonSmiles(target_smiles): return 1
    except: pass
    return 0

def evaluate_single_cand(pred_dict: dict, gt_dict: dict) -> dict:
    scores = {"em": {}, "tani": {}}
    overall_em_count, overall_total, overall_tani_sum = 0, 0, 0.0
    
    gt_c = gt_dict.get("catalyst1", "NULL")
    if gt_c != "NULL":
        p_c = pred_dict.get("catalyst1", "")
        em = check_exact_match(p_c, gt_c)
        tani = calculate_tanimoto_similarity(p_c, gt_c)
        scores["em"]["c"] = em; scores["tani"]["c"] = tani
        overall_em_count += em; overall_tani_sum += tani; overall_total += 1
        
    gt_s1 = gt_dict.get("solvent1", "NULL")
    gt_s2 = gt_dict.get("solvent2", "NULL")
    p_s1 = pred_dict.get("solvent1", "")
    p_s2 = pred_dict.get("solvent2", "")
    
    if gt_s1 != "NULL" and gt_s2 != "NULL":
        s11_em = check_exact_match(p_s1, gt_s1); s22_em = check_exact_match(p_s2, gt_s2)
        s12_em = check_exact_match(p_s1, gt_s2); s21_em = check_exact_match(p_s2, gt_s1)
        
        s11_t = calculate_tanimoto_similarity(p_s1, gt_s1); s22_t = calculate_tanimoto_similarity(p_s2, gt_s2)
        s12_t = calculate_tanimoto_similarity(p_s1, gt_s2); s21_t = calculate_tanimoto_similarity(p_s2, gt_s1)
        
        if (s11_t + s22_t) >= (s12_t + s21_t):
            scores["em"]["s1"] = s11_em; scores["tani"]["s1"] = s11_t
            scores["em"]["s2"] = s22_em; scores["tani"]["s2"] = s22_t
        else:
            scores["em"]["s1"] = s21_em; scores["tani"]["s1"] = s21_t
            scores["em"]["s2"] = s12_em; scores["tani"]["s2"] = s12_t
            
        overall_em_count += scores["em"]["s1"] + scores["em"]["s2"]
        overall_tani_sum += scores["tani"]["s1"] + scores["tani"]["s2"]
        overall_total += 2
    elif gt_s1 != "NULL":
        em = check_exact_match(p_s1, gt_s1); tani = calculate_tanimoto_similarity(p_s1, gt_s1)
        scores["em"]["s1"] = em; scores["tani"]["s1"] = tani
        overall_em_count += em; overall_tani_sum += tani; overall_total += 1
        
    gt_r1 = gt_dict.get("reagent1", "NULL")
    gt_r2 = gt_dict.get("reagent2", "NULL")
    p_r1 = pred_dict.get("reagent1", "")
    p_r2 = pred_dict.get("reagent2", "")
    
    if gt_r1 != "NULL" and gt_r2 != "NULL":
        r11_em = check_exact_match(p_r1, gt_r1); r22_em = check_exact_match(p_r2, gt_r2)
        r12_em = check_exact_match(p_r1, gt_r2); r21_em = check_exact_match(p_r2, gt_r1)
        
        r11_t = calculate_tanimoto_similarity(p_r1, gt_r1); r22_t = calculate_tanimoto_similarity(p_r2, gt_r2)
        r12_t = calculate_tanimoto_similarity(p_r1, gt_r2); r21_t = calculate_tanimoto_similarity(p_r2, gt_r1)
        
        if (r11_t + r22_t) >= (r12_t + r21_t):
            scores["em"]["r1"] = r11_em; scores["tani"]["r1"] = r11_t
            scores["em"]["r2"] = r22_em; scores["tani"]["r2"] = r22_t
        else:
            scores["em"]["r1"] = r21_em; scores["tani"]["r1"] = r21_t
            scores["em"]["r2"] = r12_em; scores["tani"]["r2"] = r12_t
            
        overall_em_count += scores["em"]["r1"] + scores["em"]["r2"]
        overall_tani_sum += scores["tani"]["r1"] + scores["tani"]["r2"]
        overall_total += 2
    elif gt_r1 != "NULL":
        em = check_exact_match(p_r1, gt_r1); tani = calculate_tanimoto_similarity(p_r1, gt_r1)
        scores["em"]["r1"] = em; scores["tani"]["r1"] = tani
        overall_em_count += em; overall_tani_sum += tani; overall_total += 1
        
    if overall_total > 0:
        scores["em"]["overall"] = 1 if overall_em_count == overall_total else 0
        scores["tani"]["overall"] = overall_tani_sum / overall_total
        
    return scores

def evaluate_cands_topk(ranked_cands: list, gt_dict: dict, eval_k_list: list) -> dict:
    valid_keys = []
    if gt_dict.get("catalyst1", "NULL") != "NULL": valid_keys.append("c")
    if gt_dict.get("solvent1", "NULL") != "NULL": valid_keys.append("s1")
    if gt_dict.get("solvent2", "NULL") != "NULL": valid_keys.append("s2")
    if gt_dict.get("reagent1", "NULL") != "NULL": valid_keys.append("r1")
    if gt_dict.get("reagent2", "NULL") != "NULL": valid_keys.append("r2")
    valid_keys.append("overall")
    
    results = {k: {"em": {}, "tani": {}} for k in eval_k_list}
    for k in eval_k_list:
        for key in valid_keys:
            results[k]["em"][key] = 0
            results[k]["tani"][key] = 0.0
            
    if not ranked_cands:
        return results

    cand_scores = [evaluate_single_cand(cand["preds_dict"], gt_dict) for cand in ranked_cands]
        
    for k in eval_k_list:
        k_cand_scores = cand_scores[:k]
        if k_cand_scores:
            for key in valid_keys:
                results[k]["em"][key] = max(cs["em"].get(key, 0) for cs in k_cand_scores)
                results[k]["tani"][key] = max(cs["tani"].get(key, 0.0) for cs in k_cand_scores)
                
    return results

def load_json_data(path: str) -> list:
    print(f"Loading data from: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--save_dir", type=str, default="/LocalData1/yjshon/RCPredictor/checkpoints_with_order")
    
    parser.add_argument("--train_path", type=str, default="./preprocess/train_text_data.json")
    parser.add_argument("--val_path", type=str, default="./preprocess/val_text_data.json")
    parser.add_argument("--test_path", type=str, default="./preprocess/test_text_data.json")
    

    parser.add_argument("--yield_model_path", type=str, default="yield_predictor_transformer.pth")
    
    parser.add_argument("--num_samples", type=int, default=20, help="Top-K 평가 시 생성할 후보의 수")
    parser.add_argument("--temperature", type=float, default=1.0, help="다양성 확보를 위한 Temperature")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Pipeline Running on: {device}")

    model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    raw_train = load_json_data(args.train_path)
    raw_val = load_json_data(args.val_path)
    raw_test = load_json_data(args.test_path)

    RATIO = 1

    raw_train = raw_train[:max(1, int(len(raw_train) * RATIO))]
    raw_val = raw_val[:max(1, int(len(raw_val) * RATIO))]
    raw_test = raw_test[:max(1, int(len(raw_test) * RATIO))]

    print(f"\n[Data Subsampling] {RATIO*100:.0f}% 데이터만 축소 적용:")
    print(f" - Train Samples: {len(raw_train)}")
    print(f" - Val Samples:   {len(raw_val)}")
    print(f" - Test Samples:  {len(raw_test)}\n")


    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.mask_token_id is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    MASK_ID = tokenizer.mask_token_id
    PAD_ID = tokenizer.pad_token_id
    V = len(tokenizer)

    train_dataset = A3PromptChemicalDataset(raw_train, tokenizer, max_len=384)
    val_dataset = A3PromptChemicalDataset(raw_val, tokenizer, max_len=384)
    test_dataset = A3PromptChemicalDataset(raw_test, tokenizer, max_len=384)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=4, pin_memory=True)

    raw_llama = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=model_dtype,           
        device_map={"": device},
        attn_implementation="sdpa"   
    )
    raw_llama.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=["embed_tokens", "lm_head"]
    )
    lora_llama = get_peft_model(raw_llama, lora_config)
    generator_model = A3ChemicalLlama(lora_llama, pad_token_id=PAD_ID)


    print("\n[Yield Predictor] 훈련된 모델 가중치를 로드합니다...")
    yield_model = YieldPredictor(
        vocab_size=V, 
        pad_token_id=PAD_ID, 
        d_model=256, 
        nhead=4, 
        num_layers=2, 
        max_len=512
    ).to(device=device, dtype=model_dtype)
    
    if os.path.exists(args.yield_model_path):
        yield_model.load_state_dict(torch.load(args.yield_model_path, map_location=device))
        print(f" -> ✅ 성공적으로 로드되었습니다: {args.yield_model_path}")
    else:
        print(f" -> ❌ [경고] 가중치 파일을 찾을 수 없습니다: {args.yield_model_path}")
        print(" -> 랜덤 가중치로 진행됩니다.")

    for param in yield_model.parameters():
        param.requires_grad = False
    yield_model.eval()
    # ========================================================

    print("\n[학습 시작] LoRA 파인튜닝을 진행합니다...")
    optimizer = torch.optim.AdamW(generator_model.parameters(), lr=args.lr, weight_decay=0.01, fused=True)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        generator_model.train()
        total_train_loss = 0.0
        valid_train_steps = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=True)
        for step, batch in enumerate(train_pbar):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            group_ids = batch["group_ids"].to(device)
            null_groups_list = batch["null_groups"]

            labels = input_ids.clone()
            masked_input_ids = input_ids.clone()

            for b in range(len(input_ids)):
                for g_id in range(1, 6):
                    if g_id not in null_groups_list[b]:
                        target_mask = (group_ids[b] == g_id)
                        masked_input_ids[b, target_mask] = MASK_ID
                    else:
                        labels[b, group_ids[b] == g_id] = -100
                labels[b, group_ids[b] == 0] = -100

            logits = generator_model(masked_input_ids, group_ids)

            shift_logits = logits.view(-1, V).float()
            shift_labels = labels.view(-1)

            if (shift_labels != -100).sum() == 0:
                continue

            loss = loss_fn(shift_logits, shift_labels)

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator_model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            valid_train_steps += 1
            
            train_pbar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "avg_loss": f"{total_train_loss / valid_train_steps:.4f}"
            })

        avg_train_loss = total_train_loss / max(valid_train_steps, 1)

        generator_model.eval()
        total_val_loss = 0.0
        valid_val_steps = 0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False)
        with torch.no_grad():
            for batch in val_pbar:
                input_ids = batch["input_ids"].to(device)
                group_ids = batch["group_ids"].to(device)
                null_groups_list = batch["null_groups"]

                labels = input_ids.clone()
                masked_input_ids = input_ids.clone()

                for b in range(len(input_ids)):
                    for g_id in range(1, 6):
                        if g_id not in null_groups_list[b]:
                            target_mask = (group_ids[b] == g_id)
                            masked_input_ids[b, target_mask] = MASK_ID
                        else:
                            labels[b, group_ids[b] == g_id] = -100
                    labels[b, group_ids[b] == 0] = -100

                logits = generator_model(masked_input_ids, group_ids)
                shift_logits = logits.view(-1, V).float()
                shift_labels = labels.view(-1)

                if (shift_labels != -100).sum() == 0:
                    continue

                loss = loss_fn(shift_logits, shift_labels)
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
                    valid_val_steps += 1
                    
                val_pbar.set_postfix({"val_loss": f"{total_val_loss / max(valid_val_steps, 1):.4f}"})

        avg_val_loss = total_val_loss / max(valid_val_steps, 1)
        print(f"\n Epoch {epoch+1}/{args.epochs} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(args.save_dir, exist_ok=True)
            lora_llama.save_pretrained(args.save_dir)
            tokenizer.save_pretrained(args.save_dir)
            print(f" -> Best Checkpoint Saved at {args.save_dir} (Val Loss: {best_val_loss:.4f})\n")

    print("\n[Test 시작] 학습된 모델로 Top-K 평가를 수행합니다...")
    generator_model.eval()

    inference_manager = A3InferenceManager(
        generator=generator_model,
        yield_predictor=yield_model,
        mask_token_id=MASK_ID,
        device=device
    )

    category_groups = {
        "catalyst": [(1, "catalyst1")],
        "solvent": [(2, "solvent1"), (3, "solvent2")],
        "reagent": [(4, "reagent1"), (5, "reagent2")]
    }
    group_keys = [("catalyst1", 1), ("solvent1", 2), ("solvent2", 3), ("reagent1", 4), ("reagent2", 5)]

    eval_k_list = [1, 3, 5, 10, 15]
    top_k_metrics = {k: {"em": [], "tani": []} for k in eval_k_list}


    global_metrics = {
        k: {
            "em": {"c": [], "s1": [], "s2": [], "r1": [], "r2": [], "overall": []},
            "tani": {"c": [], "s1": [], "s2": [], "r1": [], "r2": [], "overall": []}
        } for k in eval_k_list
    }

    test_pbar = tqdm(test_loader, desc="[Testing Top-K Progress]", leave=True)
    
    with torch.no_grad():
        for b_idx, batch in enumerate(test_pbar):
            input_ids = batch["input_ids"].to(device)
            group_ids = batch["group_ids"].to(device)
            null_groups_list = batch["null_groups"]
            comps_list = batch["components"]
            B = len(input_ids)

            inf_input_ids = input_ids.clone()
            for b in range(B):
                for g_id in range(1, 6):
                    if g_id not in null_groups_list[b]:
                        inf_input_ids[b, group_ids[b] == g_id] = MASK_ID

            all_candidates = [[] for _ in range(B)]

            for _ in range(args.num_samples):
                final_output_ids = inference_manager.run_inference(inf_input_ids, group_ids, null_groups_list, temperature=args.temperature)
                yield_scores = yield_model(final_output_ids).squeeze(-1) 
                
                for b in range(B):
                    preds_dict = {}
                    for key, g_id in group_keys:
                        pred_tokens = final_output_ids[b][group_ids[b] == g_id]
                        pred_val = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip().replace(" ", "")
                        preds_dict[key] = pred_val
                    
                    all_candidates[b].append({
                        "preds_dict": preds_dict,
                        "yield_score": yield_scores[b].item()
                    })

            for b in range(B):
                unique_cands = {}
                for cand in all_candidates[b]:
                    p = cand["preds_dict"]
                    norm_key = (
                        p.get("catalyst1", ""),
                        tuple(sorted([p.get("solvent1", ""), p.get("solvent2", "")])),
                        tuple(sorted([p.get("reagent1", ""), p.get("reagent2", "")]))
                    )
                    if norm_key not in unique_cands or cand["yield_score"] > unique_cands[norm_key]["yield_score"]:
                        unique_cands[norm_key] = cand
                        
                ranked_cands = sorted(unique_cands.values(), key=lambda x: x["yield_score"], reverse=True)
                
                results = evaluate_cands_topk(ranked_cands, comps_list[b], eval_k_list)
                
                for k in eval_k_list:
                    for key in results[k]["em"]:
                        global_metrics[k]["em"][key].append(results[k]["em"][key])
                        global_metrics[k]["tani"][key].append(results[k]["tani"][key])
                
            curr_top1_overall_em_list = global_metrics[1]["em"]["overall"]
            if curr_top1_overall_em_list:
                curr_top1_em = sum(curr_top1_overall_em_list) / len(curr_top1_overall_em_list)
                test_pbar.set_postfix({"Top-1 Overall EM": f"{curr_top1_em*100:.1f}%"})

    print(f"\n=========================================================")
    print(f" [Exact Match Accuracy (Set-level & Slot-level)] ")
    print(f"=========================================================")
    print(f"{'Conditions':<12} | {'Top-1':<8} | {'Top-3':<8} | {'Top-5':<8} | {'Top-10':<8} | {'Top-15':<8}")
    print("-" * 65)
    
    labels_order = [("c", "c"), ("s1", "s1"), ("s2", "s2"), ("r1", "r1"), ("r2", "r2"), ("overall", "Overall")]
    for key, label in labels_order:
        row_str = f"{label:<12} |"
        for k in eval_k_list:
            vals = global_metrics[k]["em"][key]
            mean_val = sum(vals) / len(vals) if vals else 0.0
            row_str += f" {mean_val:.4f}   |"
        print(row_str)

    print(f"\n=========================================================")
    print(f" [Tanimoto Similarity Score (Set-level & Slot-level)] ")
    print(f"=========================================================")
    print(f"{'Conditions':<12} | {'Top-1':<8} | {'Top-3':<8} | {'Top-5':<8} | {'Top-10':<8} | {'Top-15':<8}")
    print("-" * 65)
    for key, label in labels_order:
        row_str = f"{label:<12} |"
        for k in eval_k_list:
            vals = global_metrics[k]["tani"][key]
            mean_val = sum(vals) / len(vals) if vals else 0.0
            row_str += f" {mean_val:.4f}   |"
        print(row_str)

if __name__ == "__main__":
    main()
