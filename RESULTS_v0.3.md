# Resultats del Model v0.3 (`model_v03.py`)

Documentació de la tercera versió del model d'inferència de possessió, dissenyada sota restriccions físiques estrictes del reglament del futbol (zero *overfitting*).

- **Script**: [`model_v03.py`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/model_v03.py)
- **Submission**: [`evaluation/results/model_v03.csv`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/evaluation/results/model_v03.csv)

---

## 1. Millores Físiques Implementades a v0.3

1. **Gating Físic de Límits del Camp (`In-Bounds Possession Gating`)**:
   - Cap jugador pot tenir possessió efectiva si la pilota és fora dels límits del camp (`is_oob == True`, `score_h = 0`, `score_a = 0`). Elimina falses possessions quan un porter o jugador va a recollir una pilota darrere la línia de fons o de banda.
2. **Filtre de Col·locació de Pilota Morta (`Dead-Ball Placement Filter`)**:
   - Quan els 22 jugadors caminen (`roll_speed <= 1.65 m/s`), la pilota està quieta/lenta (`kalman_bs < 2.5 m/s`) i un únic jugador s'acosta a la pilota mentre el rival és lluny (`> 5.0 m`), es tracta com a **preparació/col·locació d'un servei**, no com a joc viu (es desactiva la resta `score_d -= 4.0` i s'atenua l'emissió de proximitat `0.25x`).
   - El filtre post-Viterbi (`clean_spurious_segments_v3`) fusiona qualsevol illa falsa `D -> H/A -> D` on la velocitat mitjana dels jugadors no superi `1.65 m/s`, eliminant completament les 4 transicions espúries dins l'aturada `15889..16400`.
3. **Viterbi No-Homogeni amb Barrera Física de Represa (`viterbi_decode_restart_gated`)**:
   - Impedeix la transició `D -> H/A` en frames on la pilota no ha estat accelerada (`kalman_bs <= 3.0 m/s` i `ball_s <= 3.0 m/s`) ni el bloc d'equips ha reprès la cursa (`roll_speed <= 2.20 m/s`).
4. **Velocitat d'Avançament Col·lectiu Auto-Calibrada (`adv_vel_h`)**:
   - Calcula de forma no supervisada a cada meitat quina porteria defensa cada equip segons el baricentre mitjà $\bar{X}_H$ vs $\bar{X}_A$, i aporta un prior direccional suau ($\pm 0.4$) durant les oclusions en joc viu (`not b_vis` i `roll_speed > 1.85 m/s`).

---

## 2. Comparativa Oficial al Scoreboard (`run_evaluation.py --bootstrap 1000`)

| Rànquing | Model | `spellF1` (Mean) | `spellF1` (@0.5) | `macroF1` | `Alive/Dead F1` | `F1(H)` | `F1(A)` | `F1(D)` | Fragmentació | Possessions (5m / Partit) | Durada Mitjana (GT=14.19s) | `transF1` (@25f) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | **`model_v03` (v0.3)** | **0.5576** | **0.5455** | **0.8653** | **0.9662** | **0.8889** | **0.8544** | **0.8525** | **1.0000** | **16** (-1) / **269** | **15.44 s** (+1.25s) | **0.2500** |
| #2 | `model_v02` (v0.2) | 0.5029 | 0.5143 | 0.8334 | 0.9554 | 0.8382 | 0.8696 | 0.7925 | 1.1905 | 18 (+1) / 304 | 14.03 s (-0.16s) | 0.2273 |
| #3 | `baseline_smooth125` | 0.4242 | 0.4242 | 0.5812 | 0.8991 | 0.7864 | 0.8108 | 0.1464 | 0.8095 | 16 (-1) / 398 | 18.46 s (+4.27s) | 0.1667 |
| #4 | `model_viterbi` (v0.1) | 0.4222 | 0.4444 | 0.7875 | 0.9397 | 0.8094 | 0.8298 | 0.7231 | 1.1429 | 19 (+2) / 313 | 13.23 s (-0.96s) | 0.1860 |
| #5 | `baseline_smooth50` *(floor)* | 0.3256 | 0.3256 | 0.6207 | 0.8990 | 0.7781 | 0.7615 | 0.3225 | 1.4286 | 26 (+9) / 594 | 10.80 s (-3.39s) | 0.0816 |
| #6 | `baseline_smooth25` | 0.2868 | 0.3019 | 0.6200 | 0.8974 | 0.7620 | 0.7123 | 0.3857 | 1.9048 | 36 (+19) / 746 | 7.58 s (-6.61s) | 0.2034 |
| #7 | `baseline_raw` | 0.0378 | 0.0270 | 0.5972 | 0.8920 | 0.7290 | 0.6430 | 0.4197 | 6.5238 | 131 (+114) / 1.949 | 2.02 s (-12.17s) | 0.1410 |

### Evolució d'iteracions (`v0.1 -> v0.2 -> v0.3`):
- **`spellF1` (mètrica oficial de rànquing)**: `0.3256` (floor) $\rightarrow$ `0.4222` (v0.1) $\rightarrow$ `0.5029` (v0.2) $\rightarrow$ **`0.5576` (v0.3)** (**+23,2 punts** sobre el floor oficial).
- **`macroF1` (desempat frame-exacte)**: `0.6207` (floor) $\rightarrow$ `0.7875` (v0.1) $\rightarrow$ `0.8334` (v0.2) $\rightarrow$ **`0.8653` (v0.3)** (**+24,5 punts** sobre el floor oficial).
- **Fragmentació**: **`1.0000`** (exactament 21 segments predits per als 21 segments reals de la finestra pública).
