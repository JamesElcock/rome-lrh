"""Quick residual scaling around the peak (3.25, 3.5, 3.75)."""
import json, os, sys, numpy as np, torch
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from rome.rome_hparams import ROMEHyperParams
from util import nethook
from util.globals import HPARAMS_DIR
import rome.rome_main as rome_main_module

RELATIONS = ["P176", "P1412", "P37", "P27", "P413"]
N_LDA_DIRS = 9
N_TEST_ENTITIES = 5
SEED = 42
SCALE_FACTORS = [2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3]

EXP1_DIR = Path("results/exp1")
EXP2_DIR = Path("results/exp2")
RESULTS_DIR = Path("results/exp4")


def rescale(v, target_norm):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-10: return v
    return v * (target_norm / n)


def apply_edit_and_eval(model, tok, u, v, hparams, prompt_text, target_str):
    layer = hparams.layers[0]
    weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
    w = nethook.get_parameter(model, weight_name)
    u_t = torch.tensor(u, dtype=torch.float32).to(w.device)
    v_t = torch.tensor(v, dtype=torch.float32).to(w.device)
    upd = u_t.unsqueeze(1) @ v_t.unsqueeze(0)
    if upd.shape != w.shape: upd = upd.T
    with torch.no_grad(): w[...] += upd
    target_tok = tok(f" {target_str.strip()}", return_tensors="pt")["input_ids"][0]
    first_target_tok = target_tok[0].item()
    inputs = tok(prompt_text, return_tensors="pt").to(w.device)
    with torch.no_grad(): logits = model(**inputs).logits[0, -1]
    probs = torch.softmax(logits.float(), dim=0)
    pred_tok = logits.argmax().item()
    target_prob = probs[first_target_tok].item()
    with torch.no_grad(): w[...] -= upd
    return {"efficacy": int(pred_tok == first_target_tok), "target_prob": float(target_prob)}


def orthogonalize_lda_against_concept(lda_dirs, concept_dir):
    d = concept_dir / (np.linalg.norm(concept_dir) + 1e-10)
    residuals = []
    for i in range(lda_dirs.shape[1]):
        li = lda_dirs[:, i].copy()
        li -= np.dot(li, d) * d
        if np.linalg.norm(li) > 1e-8: residuals.append(li)
    if not residuals: return np.zeros((len(d), 0))
    R = np.stack(residuals, axis=1)
    Q, _ = np.linalg.qr(R)
    return Q[:, :min(len(residuals), Q.shape[1])]


def main():
    print(f"Residual fine-tuning: scales {SCALE_FACTORS}")
    model = AutoModelForCausalLM.from_pretrained("gpt2-xl").cuda().eval()
    tok = AutoTokenizer.from_pretrained("gpt2-xl"); tok.pad_token = tok.eos_token
    hparams = ROMEHyperParams.from_json(HPARAMS_DIR / "ROME" / "gpt2-xl.json")
    with open(EXP1_DIR / "context_templates.json") as f:
        rome_main_module.CONTEXT_TEMPLATES_CACHE = json.load(f)

    exp2_data = torch.load(EXP2_DIR / "edit_vectors.pt", map_location="cpu")
    exp2_meta, exp2_V, exp2_U = exp2_data["metadata"], exp2_data["v"].numpy(), exp2_data["u"].numpy()
    cd_all = torch.load(EXP1_DIR / "concept_directions_layer17.pt", map_location="cpu")
    lda_dirs = np.load(EXP2_DIR / "lda_directions.npy")[:, :N_LDA_DIRS].copy()
    relation_targets = json.load(open(EXP1_DIR / "relation_targets.json"))

    tasks = []
    for rid in RELATIONS:
        for target in relation_targets[rid][:2]:
            tasks.append((rid, target))

    rng = np.random.RandomState(SEED)
    grand = defaultdict(lambda: {"eff": [], "prob": []})

    for rid, target in tasks:
        key = f"{rid}_{target}"
        idxs = [i for i, m in enumerate(exp2_meta) if m["relation_id"] == rid and m["target_value"] == target]
        if len(idxs) < 3: continue
        V, U = exp2_V[idxs], exp2_U[idxs]
        v_mean = V.mean(axis=0)
        vm_norm = float(np.linalg.norm(v_mean))

        cd_key = f"{rid}_{target}_logistic"
        if cd_key not in cd_all: continue
        concept_dir = cd_all[cd_key].numpy().astype(np.float64)
        lda_orth = orthogonalize_lda_against_concept(lda_dirs, concept_dir)
        v_concept = np.dot(v_mean, concept_dir) * concept_dir
        v_lda = lda_orth @ (lda_orth.T @ v_mean)
        v_residual = v_mean - v_concept - v_lda

        test_idxs = rng.choice(len(idxs), min(N_TEST_ENTITIES, len(idxs)), replace=False)

        for sf in SCALE_FACTORS:
            v_cond = rescale(v_residual, vm_norm * sf)
            cname = f"residual_{sf}x"
            for eidx in test_idxs:
                meta = exp2_meta[idxs[eidx]]
                prompt_text = meta["prompt"].replace("{}", meta["subject"])
                res = apply_edit_and_eval(model, tok, U[eidx], v_cond, hparams, prompt_text, target)
                grand[cname]["eff"].append(res["efficacy"])
                grand[cname]["prob"].append(res["target_prob"])

    print(f"\n{'Condition':20s} {'Eff':>6s} {'Prob':>8s} {'N':>4s}")
    print("-" * 42)
    results = {}
    for sf in SCALE_FACTORS:
        c = f"residual_{sf}x"
        e, p = np.mean(grand[c]["eff"]), np.mean(grand[c]["prob"])
        n = len(grand[c]["eff"])
        print(f"{c:20s} {e:6.3f} {p:8.4f} {n:4d}")
        results[c] = {"efficacy": float(e), "target_prob": float(p), "n": n}

    with open(RESULTS_DIR / "residual_finetune_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/exp4/residual_finetune.json")

if __name__ == "__main__":
    main()