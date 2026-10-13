import os
import json
import re
import torch
import argparse
import urllib.request
import urllib.parse
from typing import Dict, Any
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from rdkit import Chem


def tool_functional_group_tagger_simple(reactants_smiles: list, products_smiles: list) -> str:
    fg_smarts = {
        "acyl chloride": "[CX3](=[OX1])[Cl]",
        "carboxylic acid": "[CX3](=O)[OX2H1]",
        "ester": "[#6][CX3](=O)[OX2H0][#6]",
        "amide": "[NX3][CX3](=[OX1])[#6]",
        "aldehyde": "[CX3H1](=O)[#6]",
        "ketone": "[#6][CX3](=O)[#6]",
        "anhydride": "[CX3](=O)[OX2][CX3](=O)",
        "primary amine": "[NX3H2;!$(NC=O)]",
        "secondary amine": "[NX3H1;!$(NC=O)]([#6])[#6]",
        "tertiary amine": "[NX3H0;!$(NC=O)]([#6])([#6])[#6]",
        "nitro": "[N+](=O)[O-]",
        "nitrile": "[NX1]#[CX2]",
        "imine": "[CX3]=[NX2]",
        "isocyanate": "[NX2]=[CX2]=[OX1]",
        "alcohol": "[OX2H;!$(OC=O)]",
        "ether": "[OD2]([#6])[#6;!$(C=O)]",
        "epoxide": "O1CC1",
        "phenol": "[OX2H][c]",
        "thiol": "[SX2H]",
        "thioether": "[#6][SX2H0][#6]",
        "sulfonic acid": "[SX4](=O)(=O)[OX2H]",
        "sulfonamide": "[SX4](=O)(=O)[NX3]",
        "boronic acid": "[BX3]([OX2H])[OX2H]",
        "aryl halide": "[c][F,Cl,Br,I]",
        "alkyl halide": "[CX4][F,Cl,Br,I]",
        "alkene": "[CX3]=[CX3]",
        "alkyne": "[CX2]#[CX2]",
        "terminal alkyne": "[CX2H]#[CX2]"
    }
    
    def extract_fgs(smiles_list):
        found_fgs = set()
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                for fg_name, smarts in fg_smarts.items():
                    patt = Chem.MolFromSmarts(smarts)
                    if mol.HasSubstructMatch(patt):
                        found_fgs.add(fg_name)
        return list(found_fgs)

    r_fgs = extract_fgs(reactants_smiles)
    p_fgs = extract_fgs(products_smiles)
    
    return json.dumps({
        "reactants_fg": r_fgs,
        "products_fg": p_fgs,
        "main_fg": r_fgs
    })

def tool_knowledge_base_search(query: str) -> str:
    search_query = urllib.parse.quote(f"{query} chemical reaction")
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={search_query}&limit=2&fields=title,abstract,year"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            
        papers = data.get("data", [])
        if not papers:
            return json.dumps({
                "knowledge": [{"keyword": query, "evidence": ["No relevant published literature found."]}]
            })
            
        results = []
        for paper in papers:
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            year = paper.get("year", "Unknown")
            
            if abstract:
                short_abstract = abstract[:300] + "..." if len(abstract) > 300 else abstract
                evidence_text = f"[Paper, {year}] {title} - {short_abstract}"
            else:
                evidence_text = f"[Paper, {year}] {title} - (No abstract available)"
                
            results.append({"keyword": query, "evidence": [evidence_text]})
            
        return json.dumps({"knowledge": results})
        
    except urllib.error.URLError as e:
        return json.dumps({
            "knowledge": [{"keyword": query, "evidence": [f"API Search Error: {str(e)}. Fallback to general chemistry knowledge."]}]
        })

def analyze_bond_changes(rxn_smiles: str) -> str:
    try:
        reactants_smiles, products_smiles = rxn_smiles.split('>>')
        reactants = Chem.MolFromSmiles(reactants_smiles)
        products = Chem.MolFromSmiles(products_smiles)

        if not reactants or not products:
            return "Error: Invalid SMILES for RDKit parsing."

        def build_map_dicts(mol):
            map_to_symbol = {}
            map_to_idx = {}
            for atom in mol.GetAtoms():
                map_num = atom.GetAtomMapNum()
                if map_num > 0:
                    map_to_symbol[map_num] = atom.GetSymbol()
                    map_to_idx[map_num] = atom.GetIdx()
            return map_to_symbol, map_to_idx

        r_map2symbol, r_map2idx = build_map_dicts(reactants)
        p_map2symbol, p_map2idx = build_map_dicts(products)

        def get_bond_set(mol, map2idx):
            bonds = set()
            for bond in mol.GetBonds():
                u_map = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomMapNum()
                v_map = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomMapNum()
                if u_map > 0 and v_map > 0:
                    pair = tuple(sorted([u_map, v_map]))
                    bonds.add((pair[0], pair[1], bond.GetBondType()))
            return bonds

        r_bonds = get_bond_set(reactants, r_map2idx)
        p_bonds = get_bond_set(products, p_map2idx)

        r_pairs = {(b[0], b[1]): b[2] for b in r_bonds}
        p_pairs = {(b[0], b[1]): b[2] for b in p_bonds}

        broken_bonds = []
        created_bonds = []

        for (u, v), btype in r_pairs.items():
            sym1, sym2 = r_map2symbol[u], r_map2symbol[v]
            pair_name = "-".join(sorted([sym1, sym2]))
            if (u, v) not in p_pairs:
                broken_bonds.append(pair_name)
            elif r_pairs[(u, v)] > p_pairs[(u, v)]:
                broken_bonds.append(f"{pair_name} (Order decreased)")

        for (u, v), btype in p_pairs.items():
            sym1, sym2 = p_map2symbol[u], p_map2symbol[v]
            pair_name = "-".join(sorted([sym1, sym2]))
            if (u, v) not in r_pairs:
                created_bonds.append(pair_name)
            elif p_pairs[(u, v)] > r_pairs[(u, v)]:
                created_bonds.append(f"{pair_name} (Order increased)")

        broken_str = ", ".join(broken_bonds) if broken_bonds else "None"
        created_str = ", ".join(created_bonds) if created_bonds else "None"

        return f"Created Bonds: {created_str} | Broken Bonds: {broken_str}"
    except Exception as e:
        return f"Error analyzing bonds: {str(e)}"


class GeneralChemistAgent:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        self.system_prompt = """You are a chemical assistant for reaction understanding and condition reasoning.
You receive Reactant and Product SMILES as inputs. You can call THREE tools:

# Tools Definition
1) Functional Group Tagger
- Invocation: <tragger>{"reactants":[...], "products":[...]}</tragger>
- Purpose: extract key functional groups (FGs).

2) Constraint Engine
- Invocation: <engine>{"rxn_smiles":"..."}</engine>
- Purpose: perform atom mapping, bond change diff, and infer by-products.

3) Chemical Knowledge Base Search
- Invocation: <search>{"query":"..."}</search>
- Purpose: retrieve evidence related to reaction conditions via keyword search.

# Interaction Protocol
- You may call tools at any time using the XML tokens above.
- Tool responses are always returned inside <result> ... </result>.
- Show reasoning process inside <think> ... </think>.
- Provide only the structured final answer.

# Task
Given Reactant and Product SMILES:
- Use tools to (i) extract Main Functional Groups, (ii) infer By-products and bond changes, and (iii) retrieve likely Reaction Type candidates.
- You MUST call <tragger>, <engine>, and <search> at least once before making a final decision.
- Do not provide <answer> until you have gathered information from all three tools.
- Reconcile results to choose the most plausible reaction_type.
- Output in JSON schema, wrapped inside <answer> ... </answer>.

# Example
User: Reactant: CC(=O)Cl...
Assistant: <think>I need to find the functional groups first.</think>
<tragger>{"reactants":["CC(=O)Cl"], "products":["CC(=O)N"]}</tragger>
User: Environment Response: <result>{"main_fg": ["acyl chloride"]}</result>
Assistant: <think>Now I will output the final answer.</think>
<answer>
{
    "reaction_type": "Acylation",
    "main_fg": ["acyl chloride"],
    "by_products": ["HCl"],
    "confidence": 0.95,
    "rationale_short": "The reaction involves an acyl chloride converting to an amide.",
    "evidence": ["General chemistry knowledge"]
}
</answer>"""

    def _generate_llm_response(self, messages: list) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.1,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        input_length = inputs["input_ids"].shape[1]
        generated_text = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        return generated_text

    def _execute_tool(self, tool_name: str, tool_args_str: str) -> str:
        try:
            kwargs = json.loads(tool_args_str)

            if tool_name == "tragger":
                reactants = kwargs.get("reactants", [])
                products = kwargs.get("products", [])
                res = tool_functional_group_tagger_simple(reactants, products) 
            elif tool_name == "engine":
                rxn_smiles = kwargs.get("rxn_smiles", "")
                res = analyze_bond_changes(rxn_smiles)
            elif tool_name == "search":
                query = kwargs.get("query", "")
                res = tool_knowledge_base_search(query) 
            else:
                res = f"Error: Unknown tool {tool_name}"
            return f"<result>\n{res}\n</result>"
        except Exception as e:
            return f"<result>\nError parsing tool arguments: {str(e)}\n</result>"

    def run(self, rxn_smiles: str, max_steps: int = 6) -> Dict[str, Any]:
        try:
            reactants, products = rxn_smiles.split(">>")
        except ValueError:
            return {"error": "Invalid canonical_rxn format."}

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Reactant: {reactants}\nProduct: {products}\nCanonical Rxn: {rxn_smiles}"}
        ]
        
        for step in range(max_steps):
            llm_response = self._generate_llm_response(messages)
            messages.append({"role": "assistant", "content": llm_response})

            answer_match = re.search(r"<answer>(.*?)</answer>", llm_response, re.DOTALL)
            if answer_match:
                try:
                    return json.loads(answer_match.group(1).strip())
                except json.JSONDecodeError:
                    return {"error": "JSON parse failed", "raw": llm_response}

            tool_executed = False
            for tag in ["tragger", "engine", "search"]:
                pattern = f"<{tag}>(.*?)</{tag}>"
                match = re.search(pattern, llm_response, re.DOTALL)
                if match:
                    tool_args = match.group(1)
                    tool_result = self._execute_tool(tag, tool_args)
                    messages.append({"role": "user", "content": f"Environment Response:\n{tool_result}"})
                    tool_executed = True
                    break
            
            if not tool_executed:
                messages.append({"role": "user", "content": "Error: You must call a tool using XML tags or provide the final <answer>."})

        return {"error": "Max steps reached."}


def process_and_save_dataset(input_path: str, output_path: str, chemist: GeneralChemistAgent, split_name: str):
    print(f"\n[{split_name.upper()}] Loading data from: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: {input_path} 파일을 찾을 수 없습니다. 건너뜁니다.")
        return


    print(f"[{split_name.upper()}] Processing {len(dataset)} samples. This may take a while...")
    

    for item in tqdm(dataset, desc=f"Processing {split_name}"):
        canonical_rxn = item.get("components", {}).get("canonical_rxn", "")
        if canonical_rxn:

            report = chemist.run(rxn_smiles=canonical_rxn, max_steps=6)

            item["general_chemist_report"] = report
        else:
            item["general_chemist_report"] = {"error": "canonical_rxn missing"}


    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"[{split_name.upper()}] Saved successfully to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run General Chemist Agent on Datasets")
    parser.add_argument("--train_path", type=str, default="./preprocess/train_text_data.json")
    parser.add_argument("--val_path", type=str, default="./preprocess/val_text_data.json")
    parser.add_argument("--test_path", type=str, default="./preprocess/test_text_data.json")
    parser.add_argument("--output_dir", type=str, default="./preprocessed_qwen_general_chem")
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()


    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading Model: {args.model_id}...")


    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16, device_map={"": device})
    

    chemist = GeneralChemistAgent(model, tokenizer, device)


    dataset_configs = [
        ("train", args.train_path, os.path.join(args.output_dir, "train_text_data_with_report.json")),
        ("val", args.val_path, os.path.join(args.output_dir, "val_text_data_with_report.json")),
        ("test", args.test_path, os.path.join(args.output_dir, "test_text_data_with_report.json"))
    ]

    for split_name, in_path, out_path in dataset_configs:
        process_and_save_dataset(in_path, out_path, chemist, split_name)
        
    print("\nAll datasets processed and saved successfully!")
