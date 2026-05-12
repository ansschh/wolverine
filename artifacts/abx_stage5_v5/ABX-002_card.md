# ABX-002 — ABX v5 (FiLM ranker + focal + multiplicative)

**Organism:** A.baumannii (Gram-negative, spectrum=pathogen_specific)
**Hidden answer SMILES:** `C1CN(CCC12C3=CC=CC=C3NC(=O)O2)CCC4=CC=C(C=C4)C(F)(F)F`

## Closed ranking verdict: **missed**
Library size: 1153  |  Hit rank: 1048  |  Top-1pct: False

## Top 20 candidates

| Rank | final | base | sel | nov | memo? | ab | cyto | art | channel | SMILES |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.1636 | 0.1258 | 0.248 | 1.00 | n | 0.508 | 0.264 | 0.543 | B_screened_library | `Cc1noc(NS(=O)(=O)c2ccc(N)cc2)c1C` |
| 2 | 0.1618 | 0.1245 | 0.373 | 1.00 | n | 0.334 | 0.156 | 0.477 | A_repurposing | `C(=O)(Nc(cc1)ccc1CC)c2cccs2` |
| 3 | 0.1592 | 0.1224 | 0.292 | 1.00 | n | 0.420 | 0.208 | 0.535 | B_screened_library | `C(c1ccccc1)(=O)Oc2ccc(cc2OCC)\C=N\NC(CNc(cc3)ccc3C)=O` |
| 4 | 0.1570 | 0.1207 | 0.343 | 1.00 | n | 0.352 | 0.168 | 0.504 | B_screened_library | `C(/C(OCC)=O)(=N/c(cc1)ccc1c(cc2)ccc2\N=C(/C(OCC)=O)\C(F)(F)F` |
| 5 | 0.1569 | 0.1207 | 0.287 | 1.00 | n | 0.420 | 0.200 | 0.551 | B_screened_library | `S(=O)(=O)(c(cc1)ccc1[N+](=O)[O-])Oc2ccc(c3c2\C=N\c4ccc(cc4C)` |
| 6 | 0.1566 | 0.1205 | 0.294 | 1.00 | n | 0.410 | 0.198 | 0.543 | A_repurposing | `C(=S)(Oc(cccc1C(=O)Nc(cc2)ccc2Br)c1)N3CCCCC3` |
| 7 | 0.1563 | 0.1202 | 0.299 | 1.00 | n | 0.402 | 0.198 | 0.535 | B_screened_library | `S(=O)(=O)(c1ccc(cc1)\N=C\c2ccccc2O)Nc3nnc(CC)s3` |
| 8 | 0.1561 | 0.1201 | 0.251 | 1.00 | n | 0.479 | 0.239 | 0.566 | B_screened_library | `n1c2c([nH]c1c3ccccc3)ccc(c2)\N=C\c(cccc4[N+](=O)[O-])c4` |
| 9 | 0.1559 | 0.1199 | 0.295 | 1.00 | n | 0.406 | 0.189 | 0.551 | A_repurposing | `C(=O)(c1cccs1)Nc2c(OC)ccc(Cl)c2` |
| 10 | 0.1539 | 0.1184 | 0.246 | 1.00 | n | 0.480 | 0.246 | 0.566 | B_screened_library | `N(c1c(C)cccc1C)C(=O)C(=S)Nc(cc2)ccc2C(OC)=O` |
| 11 | 0.1538 | 0.1183 | 0.228 | 1.00 | n | 0.520 | 0.275 | 0.566 | A_repurposing | `CCc1oc2ccccc2c1C(=O)c1cc(Br)c(O)c(Br)c1` |
| 12 | 0.1534 | 0.1180 | 0.229 | 1.00 | n | 0.516 | 0.273 | 0.566 | B_screened_library | `c(c(O)ccc1Cl)(S2)c1OC2=O` |
| 13 | 0.1527 | 0.1174 | 0.409 | 1.00 | n | 0.287 | 0.118 | 0.475 | A_repurposing | `O=C1N(Cc2ccc(OC(F)(F)F)cc2)C(C)=C(C)n(c13)cc([N+](=O)[O-])n3` |
| 14 | 0.1522 | 0.1170 | 0.348 | 1.00 | n | 0.336 | 0.155 | 0.512 | B_screened_library | `c(cc(I)cc1I)(\C=N\c2cccc(Cl)c2Cl)c1O` |
| 15 | 0.1518 | 0.1168 | 0.358 | 1.00 | n | 0.326 | 0.150 | 0.504 | B_screened_library | `N1(c2ccccc2c3ccccc3)C(=O)c4c(cc(cc4)Oc(cc5)ccc5Oc(cccc6C(O)=` |
| 16 | 0.1515 | 0.1165 | 0.364 | 1.00 | n | 0.320 | 0.150 | 0.496 | A_repurposing | `c1ccc(c(NC(c2ncccc2)=N)nc(c3ncccc3)c4)c4c1` |
| 17 | 0.1503 | 0.1156 | 0.414 | 1.00 | n | 0.279 | 0.127 | 0.457 | B_screened_library | `c(cc(I)cc1I)(\C=N\c2cccc(C)c2)c1O` |
| 18 | 0.1502 | 0.1156 | 0.320 | 1.00 | n | 0.361 | 0.167 | 0.539 | B_screened_library | `[N+](=O)([O-])c(cc1)cc(c1C(=O)Nc(cccn2)c2)Cl` |
| 19 | 0.1502 | 0.1155 | 0.352 | 1.00 | n | 0.328 | 0.147 | 0.516 | B_screened_library | `C(=O)(N\N=C\c(cc1)ccc1OCc(cc2)ccc2C)C(=O)NC3CCCCC3` |
| 20 | 0.1499 | 0.1153 | 0.347 | 1.00 | n | 0.332 | 0.153 | 0.516 | A_repurposing | `c(non1)(c1n2)ncc2c(cc3OC)c(OC)c4c3OCO4` |