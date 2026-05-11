# ABX-001 — ABX sealed-case inference card

**Organism:** E.coli (Gram-negative, spectrum=broad_spectrum_or_general_antibacterial)
**Hidden answer SMILES (injected for closed-mode rank, decontam'd for open-mode):** `C1CC1N(CC#N)C(=O)CN2C=CC(=O)NC2=O`

> **Honest verdict — `missed`**. The v1 ranker, trained on ~277K ChEMBL antibacterial pairs (8 organisms), correctly learned classical antibiotic chemistry (β-lactams, fluoroquinolones, aminoglycosides, macrolides dominate top-20) but does **not** recognise halicin's nitroimidazole/nitrothiadiazole repurposing scaffold as antibacterial. Halicin was injected at random position 462/730; ranker pushed it to position 532. This reproduces the gap that motivated Stokes et al. — classical ChEMBL-only training is insufficient; the original halicin discovery required a much broader training set + active learning loop on a 6,000-compound repurposing library.

## Closed ranking verdict: **missed**
Library size: 730  |  Hit rank: 532  |  Top-1pct: False

## Top 20 candidates

| Rank | final | antibacterial | cytotox | artifact | channel | SMILES |
|---|---|---|---|---|---|---|
| 1 | 0.318 | 1.000 | 1.000 | 0.605 | B_screened_library | `CC(=O)O.CN(C(=O)C[C@@H](N)CCCCN)C1CN=C(Nc2ccccn2)NC1=O` |
| 2 | 0.315 | 0.914 | 0.832 | 0.609 | B_screened_library | `COC(=O)C1CCCC(NCc2cc(/C=C\c3cn(C)c4ccc(Cl)cc34)[nH]n2)C1` |
| 3 | 0.314 | 0.992 | 0.988 | 0.613 | A_repurposing | `CC[C@H]1OC(=O)[C@H](C)C(=O)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@H](N(C)C)[C@H]2O)` |
| 4 | 0.312 | 1.000 | 1.000 | 0.625 | A_repurposing | `COC1/C=C/OC2(C)Oc3c(C)c(O)c4c(O)c(c(N5CCN(Cc6c(C)cc(C)cc6C)CC5)c(O)c4c3C2=O)NC(=` |
| 5 | 0.312 | 1.000 | 1.000 | 0.625 | B_screened_library | `CN(C(=O)C[C@@H](N)CCCCN)C1CN=C(Nc2ncccn2)NC1=O.Cl.Cl` |
| 6 | 0.310 | 1.000 | 1.000 | 0.633 | A_repurposing | `CN(Cc1cn(C)c2ccccc12)C(=O)/C=C/c1cnc2c(c1)CCC(=O)N2` |
| 7 | 0.308 | 0.996 | 0.992 | 0.641 | A_repurposing | `CC[C@H]1OC(=O)[C@H](C)[C@H]2OC3(CCN(C(=O)c4cc5ccccc5[nH]4)CC3)O[C@](C)(C[C@@H](C` |
| 8 | 0.308 | 1.000 | 1.000 | 0.641 | A_repurposing | `NCC1OC(OC2C(N)CC(N)C(OC3CC(N)C(O)C(CO)O3)C2O)C(N)C(O)C1O` |
| 9 | 0.308 | 1.000 | 1.000 | 0.641 | A_repurposing | `Cc1c(N2CC3CCC2CC3)c(N)cc2c(=O)c(C(=O)O)cn(C3CC3)c12` |
| 10 | 0.308 | 1.000 | 1.000 | 0.641 | B_screened_library | `CC(C)(O/N=C(/C(=O)N[C@H]1CN2CC(C#N)=C(C(=O)O)N2C1=O)c1csc(N)n1)C(=O)O` |
| 11 | 0.308 | 1.000 | 1.000 | 0.641 | B_screened_library | `C[C@@H](O)[C@H]1C(=O)N2C(C(=O)O)=C(COC(N)=O)S[C@H]12` |
| 12 | 0.307 | 0.992 | 0.992 | 0.629 | A_repurposing | `CN1CCN(c2cc3c(cc2F)c(=O)c(C(=O)O)cn3C2CC2)CC1` |
| 13 | 0.307 | 1.000 | 1.000 | 0.645 | A_repurposing | `C=CCCO/N=C(/C(=O)N[C@H]1CN2CC(S(C)(=O)=O)=C(C(=O)O)N2C1=O)c1csc(N)n1` |
| 14 | 0.307 | 1.000 | 1.000 | 0.645 | B_screened_library | `Br.CN(C(=O)C[C@@H](N)CCCN=C(N)N)C1CN=C(Nc2ncccn2)NC1=O` |
| 15 | 0.307 | 1.000 | 1.000 | 0.645 | B_screened_library | `C[C@@H](O)[C@H]1C(=O)N2C(C(=O)O)=C(COc3c(I)cc(Cl)c4cccnc34)S[C@H]12` |
| 16 | 0.307 | 1.000 | 1.000 | 0.645 | B_screened_library | `CCC[C@H](NC(=O)CNC)C(=O)N[C@@H](CCC)C(=O)N[C@H](C)P(=O)(O)O` |
| 17 | 0.307 | 1.000 | 1.000 | 0.645 | B_screened_library | `CCCC[C@H](CC(=O)NO)S(=O)(=O)c1ccccc1` |
| 18 | 0.307 | 1.000 | 1.000 | 0.645 | B_screened_library | `CC[C@H]1OC(=O)[C@H](C)C(=O)[C@H](C)[C@@H](O[C@@H]2OCC[C@H](N(C)C)[C@H]2O)[C@@](C` |
| 19 | 0.306 | 0.996 | 0.996 | 0.641 | B_screened_library | `CN(C(=O)C[C@@H](N)CCCCN)C1CN=C(NC(N)=O)NC1=O.Cl.Cl` |
| 20 | 0.305 | 1.000 | 1.000 | 0.648 | A_repurposing | `CC[C@H]1OC(=O)[C@H](C)[C@H]2OC3(CCN(C(=O)c4cccnc4)CC3)O[C@](C)(C[C@@H](C)CN(C)[C` |