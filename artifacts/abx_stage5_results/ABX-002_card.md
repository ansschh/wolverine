# ABX-002 — ABX sealed-case inference card

**Organism:** A.baumannii (Gram-negative, spectrum=pathogen_specific)
**Hidden answer SMILES (injected for closed-mode rank, decontam'd for open-mode):** `C1CN(CCC12C3=CC=CC=C3NC(=O)O2)CCC4=CC=C(C=C4)C(F)(F)F`

> **Honest verdict — `missed`**. Abaucin is a structurally novel spiro-benzoxazinone with no analogue in classical ChEMBL antibacterial space. Ranker top-20 again dominated by canonical antibiotic scaffolds (β-lactams, fluoroquinolones, oxazolidinones). Abaucin injected at random position 37/775; ranker pushed it to 654 — confirming that the model has no learned signal for this chemotype against *A. baumannii*. This is the *exact gap* abaucin's discovery (Liu et al., NCB 2023) was designed to fill: a generative / active-learning loop over a curated 7,500-compound library that the v1 multi-task ranker alone cannot substitute for.

## Closed ranking verdict: **missed**
Library size: 775  |  Hit rank: 654  |  Top-1pct: False

## Top 20 candidates

| Rank | final | antibacterial | cytotox | artifact | channel | SMILES |
|---|---|---|---|---|---|---|
| 1 | 0.316 | 0.945 | 0.871 | 0.645 | B_screened_library | `COC(=O)C1CCCC(NCc2cc(/C=C\c3cn(C)c4ccc(Cl)cc34)[nH]n2)C1` |
| 2 | 0.311 | 1.000 | 1.000 | 0.629 | B_screened_library | `CC(=O)O.CN(C(=O)C[C@@H](N)CCCCN)C1CN=C(Nc2ccccn2)NC1=O` |
| 3 | 0.307 | 1.000 | 1.000 | 0.645 | A_repurposing | `COC1/C=C/OC2(C)Oc3c(C)c(O)c4c(O)c(c(N5CCN(Cc6c(C)cc(C)cc6C)CC5)c(O)c4c3C2=O)NC(=` |
| 4 | 0.305 | 1.000 | 1.000 | 0.648 | A_repurposing | `Cc1c(N2CC3CCC2CC3)c(N)cc2c(=O)c(C(=O)O)cn(C3CC3)c12` |
| 5 | 0.305 | 1.000 | 1.000 | 0.648 | B_screened_library | `CN(C(=O)C[C@@H](N)CCCCN)C1CN=C(Nc2ncccn2)NC1=O.Cl.Cl` |
| 6 | 0.305 | 0.992 | 0.992 | 0.637 | A_repurposing | `CC[C@H]1OC(=O)[C@H](C)C(=O)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@H](N(C)C)[C@H]2O)` |
| 7 | 0.304 | 1.000 | 1.000 | 0.652 | A_repurposing | `NCC1OC(OC2C(N)CC(N)C(OC3CC(N)C(O)C(CO)O3)C2O)C(N)C(O)C1O` |
| 8 | 0.304 | 1.000 | 1.000 | 0.652 | B_screened_library | `CC(C)(O/N=C(/C(=O)N[C@H]1CN2CC(C#N)=C(C(=O)O)N2C1=O)c1csc(N)n1)C(=O)O` |
| 9 | 0.304 | 1.000 | 1.000 | 0.652 | B_screened_library | `C[C@@H](O)[C@H]1C(=O)N2C(C(=O)O)=C(COc3c(I)cc(Cl)c4cccnc34)S[C@H]12` |
| 10 | 0.304 | 1.000 | 1.000 | 0.652 | B_screened_library | `CCCC[C@H](CC(=O)NO)S(=O)(=O)c1ccccc1` |
| 11 | 0.304 | 1.000 | 1.000 | 0.652 | B_screened_library | `C[C@@H](O)[C@H]1C(=O)N2C(C(=O)O)=C(COC(N)=O)S[C@H]12` |
| 12 | 0.304 | 1.000 | 1.000 | 0.652 | C_organism_analog | `N#Cc1ccc(NC(=O)c2cc3n(n2)C(C(F)(F)F)CC(c2cccs2)N3)cc1` |
| 13 | 0.303 | 1.000 | 1.000 | 0.656 | A_repurposing | `CN(Cc1cn(C)c2ccccc12)C(=O)/C=C/c1cnc2c(c1)CCC(=O)N2` |
| 14 | 0.303 | 1.000 | 1.000 | 0.656 | A_repurposing | `COC1/C=C/OC2(C)Oc3c(C)c(O)c4c(O)c(c(C=O)c(O)c4c3C2=O)NC(=O)/C(C)=C\C=C\C(C)C(O)C` |
| 15 | 0.303 | 1.000 | 1.000 | 0.656 | A_repurposing | `COC1CCOC2(C)Oc3c(C)c(O)c4c(O)c(c(N5CCN(Cc6c(C)cc(C)cc6C)CC5)c(O)c4c3C2=O)NC(=O)C` |
| 16 | 0.303 | 1.000 | 1.000 | 0.656 | B_screened_library | `Br.CN(C(=O)C[C@@H](N)CCCN=C(N)N)C1CN=C(Nc2ncccn2)NC1=O` |
| 17 | 0.303 | 1.000 | 1.000 | 0.656 | C_organism_analog | `Cc1ccc(C(=O)CCCC#Cc2ccc(C3=NCCO3)cc2)s1` |
| 18 | 0.303 | 0.977 | 0.973 | 0.625 | A_repurposing | `CCn1cc(C(=O)O)c(=O)c2cc(F)c(N3CCN(C)CC3)cc21` |
| 19 | 0.302 | 1.000 | 1.000 | 0.660 | B_screened_library | `O=C(N[C@H](CO)[C@H](O)c1ccc([N+](=O)[O-])cc1)C(Cl)Cl` |
| 20 | 0.302 | 1.000 | 1.000 | 0.660 | B_screened_library | `CC(=O)O.CN(C(=O)C[C@@H](N)CCCN=C(N)N)C1CN=C(Nc2ccccn2)NC1=O` |