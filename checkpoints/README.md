# Checkpoints (local-only, gitignored)

Pulled from RunPod 216.81.245.7:25207 on 2026-05-12 before pod termination.

| Dir | Size | Description |
|---|---|---|
| `smiles_lm_200m/` | 770 MB | **200M MLM SMILES backbone** trained on 2.8M ChEMBL (12K steps, loss 2.37→0.20). Used as the chemistry-prior init for the v4 ranker. |
| `smiles_ar_lm_200m/` | 770 MB | **200M autoregressive SMILES LM** trained on same 2.8M ChEMBL (12K steps, loss 13.3→0.62). Generative — supports `.sample()` for de novo molecule generation. 91.4% drug-like validity at temperature 1.0 / top-p 0.95. |
| `smiles_ar_lm_rl_ecoli/` | 770 MB | First REINFORCE RL fine-tune attempt — partial success then reward-hack collapse. Top samples in `top_samples.json` include real bis-sulfonamide piperazines (mid-training before collapse). |
| `smiles_ar_lm_rl_ecoli_v2/` | 770 MB | Hardened-validator RL re-attempt — death-spiraled (validator too strict, 0/64 valid every iter). Kept for reference. |
| `abx_ranker_seed42/` | 778 MB | v3 ranker, seed 42, random init. Val ab_acc 83.9%, cyto 84.5%, fm 84.5%. |
| `abx_ranker_seed43/` | 778 MB | v3 ranker, seed 43. Val ab_acc 83.2%. |
| `abx_ranker_seed44/` | 778 MB | v3 ranker, seed 44. Val ab_acc 82.8%. Used for multi-seed ensemble in v3 inference. |
| `abx_ranker_v4_seed42/` | 775 MB | **Final v4 ranker** — FiLM organism conditioning at every layer + per-organism focal pos_weight (A.baumannii 50×, MRSA 50×, S.aureus 8×, E.coli 3.5×) + anti-memorization regularizer + pretrained from smiles_lm_200m. Val ab 77.4% / cyto 83.9% / fm 84.0%. The canonical inference ranker. |
| `abx_diffusion/` | 62 MB | Real DiGress-style discrete graph diffusion, 60K-step (long) run. Stages 1/2/3 ckpts (5.4M params each). Produces ~11 valid SMILES out of 500 samples — undertrained at our scale. |
| `abx_diffusion_v1_18k/` | 62 MB | First diffusion run (18K steps). Produces 0 valid SMILES. Archived for reference. |
| `abx_baselines/` | 8 KB | The 10-baselines comparison CSV + JSON (§18.1–§18.10). |

## Loading any ranker
```python
from train_abx_ranker_v4 import ABXMultiHeadRankerV4
from h200_smiles_lm_pretrain import VOCAB_SIZE
import torch

ckpt = torch.load("checkpoints/abx_ranker_v4_seed42/checkpoint.pt", weights_only=False)
model = ABXMultiHeadRankerV4(VOCAB_SIZE, d_model=1024, n_heads=16, n_layers=16, max_len=128)
model.load_state_dict({k.removeprefix("module."): v for k, v in ckpt["model"].items()}, strict=False)
model.eval()
```

## Loading the generative AR LM
```python
from train_smiles_ar_lm import ARSMILESLM
ckpt = torch.load("checkpoints/smiles_ar_lm_200m/checkpoint.pt", weights_only=False)
lm = ARSMILESLM(d_model=1024, n_heads=16, n_layers=16, max_len=128)
lm.load_state_dict({k.removeprefix("module."): v for k, v in ckpt["model"].items()})
smiles_list = lm.sample(n=100, max_len=128, temperature=1.0, top_p=0.95, device="cuda")
```

## ChEMBL 36 SQLite
The 28 GB `chembl_36.db` SQLite database is NOT included here (too large + reproducible from the public FTP). To recreate:
```bash
mkdir -p rasyn/data/raw/chembl/extracted && cd rasyn/data/raw/chembl
curl -L -o chembl.tar.gz https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_36_sqlite.tar.gz
tar -xzf chembl.tar.gz -C extracted --strip-components=1
```
