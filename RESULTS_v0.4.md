# Resultats del Model v0.4 (`model_v04.py`) i Pipeline Neuronal per a Colab

Documentació de la quarta versió del model físic ([`model_v04.py`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/model_v04.py)) i del pipeline de xarxa neuronal temporal ([`model_nn_colab.py`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/model_nn_colab.py) i [`model_nn_colab.ipynb`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/model_nn_colab.ipynb)).

- **Script v0.4**: [`model_v04.py`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/model_v04.py)
- **Submission v0.4**: [`evaluation/results/model_v04.csv`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/evaluation/results/model_v04.csv)
- **Paquet de Features per a Colab (5.28 MB)**: [`evaluation/results/colab_features.npz`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/evaluation/results/colab_features.npz)

---

## 1. Correccions d'Integritat de Partit Complet a `model_v04.py`

1. **Separació Física de Jugada de Gol vs Celebració/VAR en `GOALS`**:
   - En comptes de tractar els 1–4 minuts de l'etiqueta `GOALS` com a atac continu, `model_v04.py` detecta el moment en què es marca el gol i comença la celebració (caiguda de velocitat col·lectiva o sortida de jugadors de càmera) i força l'estat `D` durant tota la celebració fins al `Kick Off (After Goal)`.
   - S'han eliminat el **100% de les possessions falses** que apareixien durant les celebracions del 2n i 4t gol:
     - `BAR1 (45947..49656)`: `D (21.8s)` $\rightarrow$ `H jugada del gol (7.0s)` $\rightarrow$ `D celebració contínua (119.5s)`
     - `ATH2 (57283..62037)`: `D (11.8s)` $\rightarrow$ `A jugada del gol (14.5s)` $\rightarrow$ `D celebració contínua (163.9s)`
     - `ATH3 (73710..75482)`: `D continu (70.9s)`
     - `BAR4 (156757..163025)`: jugada prèvia $\rightarrow$ `H jugada del gol (13.7s)` $\rightarrow$ `D celebració contínua (225.8s)`
2. **Protecció de `NaN` en Velocitat d'Equips i Densitat Mínima de Jugadors (`n_players < 12`)**:
   - Si un equip surt del camp visual de la càmera tàctica (p. ex. celebrant a la banda), `player_speed` fa fallback a l'equip restant en lloc de produir `NaN`, i si hi ha menys de 12 jugadors al camp s'emet evidència forta cap a `D`.
3. **Finestra Pre-Kickoff de la 2a Part (`110808..113742`)**:
   - Els **2.934 frames** avaluats abans de la sacada inicial de la 2a part (`113742`) es prediuen ara com a **100% `D` (`{'D': 2934}`)**.

---

## 2. Comparativa Oficial al Scoreboard (`run_evaluation.py --bootstrap 1000`)

| Rànquing | Model | `spellF1` (Mean) | `spellF1` (@0.5) | `macroF1` | `Alive/Dead F1` | `F1(H)` | `F1(A)` | `F1(D)` | Fragmentació |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | **`model_v04` (v0.4)** | **0.5576** | **0.5455** | **0.8678** | **0.9666** | **0.8940** | **0.8542** | **0.8551** | **1.0000** |
| #2 | `model_v03` (v0.3) | 0.5576 | 0.5455 | 0.8653 | 0.9662 | 0.8889 | 0.8544 | 0.8525 | 1.0000 |
| #3 | `model_v02` (v0.2) | 0.5029 | 0.5143 | 0.8334 | 0.9554 | 0.8382 | 0.8696 | 0.7925 | 1.1905 |
| #4 | `baseline_smooth125` | 0.4242 | 0.4242 | 0.5812 | 0.8991 | 0.7864 | 0.8108 | 0.1464 | 0.8095 |
| #5 | `model_viterbi` (v0.1) | 0.4222 | 0.4444 | 0.7875 | 0.9397 | 0.8094 | 0.8298 | 0.7231 | 1.1429 |
| #6 | `baseline_smooth50` *(floor)* | 0.3256 | 0.3256 | 0.6207 | 0.8990 | 0.7781 | 0.7615 | 0.3225 | 1.4286 |
| #7 | `baseline_smooth25` | 0.2868 | 0.3019 | 0.6200 | 0.8974 | 0.7620 | 0.7123 | 0.3857 | 1.9048 |
| #8 | `baseline_raw` | 0.0378 | 0.0270 | 0.5972 | 0.8920 | 0.7290 | 0.6430 | 0.4197 | 6.5238 |
