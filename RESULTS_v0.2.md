# Resultats del Model v0.2 (`model_v02.py`)

Documentació de la segona versió del model d'inferència de possessió, que incorpora gating d'interacció física (`< 2.2 m`), pontatge d'esdeveniments de control discret (`build_pass_bridges`), filtre cinemàtic d'espai d'estats sobre la pilota (`kalman_smooth_ball`) i pontatge condicional d'aturades segons fora de límits (`oob`).

- **Script**: [`model_v02.py`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/model_v02.py)
- **Submission**: [`evaluation/results/model_v02.csv`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/evaluation/results/model_v02.csv)

---

## 1. Millores Arquitectòniques de v0.2 respecte a v0.1

1. **Gating d'Interacció Física (`min(dh, da) < 2.2 m`)**:
   - Elimina l'emissió contínua de `da - dh` quan la pilota vola lluny dels jugadors (`> 2.2 m`), evitant que passades llargues o canvis d'orientació per sobre de defensors rivals alterin l'equip en possessió o desplacin les fronteres de transició.
2. **Esdeveniments de Control Discret i Pontatge de Passades (`build_pass_bridges`)**:
   - Identifica tocs de control no disputats (`d < 1.8 m`, rival `> 2.2 m`).
   - Connecta intervals de vol/oclusió ($\le 150\text{ frames} = 6.0\text{ s}$): `+2.0` d'evidència logarítmica quan rep el mateix equip (passada completada) i `+1.5` a favor de l'equip passador fins al frame exacte d'intercepció quan hi ha pèrdua.
3. **Suavitzat Cinemàtic 2D (`kalman_smooth_ball`)**:
   - Interpola oclusions curtes ($\le 5.0\text{ s}$) i calcula la velocitat cinemàtica de la pilota (`kalman_bs`), utilitzada per tallar immediatament la fase de preparació dels `SET PIECES` quan la pilota accelera (`kalman_bs > 4.0 m/s`).
4. **Protecció de Joc Viu i Pontatge Condicional a `oob` (`clean_spurious_segments`)**:
   - Inhibeix l'entrada a `D` (`score_d -= 4.0`) mentre hi ha control dins del camp (`< 2.0 m`).
   - En aturades amb sortida real de camp ($\ge 5$ frames `oob`), relaxa el filtre de durada mínima a `40` frames (`1.6 s`) per preservar serveis de banda ràpids; sense `oob`, filtra falses aturades fins a `90` frames (`3.6 s`).

---

## 2. Comparativa Oficial al Scoreboard (`run_evaluation.py --bootstrap 1000`)

| Rànquing | Model | `spellF1` (Mean) | `spellF1` (@0.5) | `macroF1` | `Alive/Dead F1` | `F1(H)` | `F1(A)` | `F1(D)` | Possessions (5m / Partit) | Durada Mitjana (GT=14.19s) | Latència Mitjana | `transF1` (@25f) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | **`model_v02` (v0.2)** | **0.5029** | **0.5143** | **0.8334** | **0.9554** | **0.8382** | **0.8696** | **0.7925** | **18** (+1) / **304** | **14.03 s** (-0.16s) | **-0.40 s** | **0.2273** |
| #2 | `baseline_smooth125` | 0.4242 | 0.4242 | 0.5812 | 0.8991 | 0.7864 | 0.8108 | 0.1464 | 16 (-1) / 398 | 18.46 s (+4.27s) | -7.67 s | 0.1667 |
| #3 | `model_viterbi` (v0.1) | 0.4222 | 0.4444 | 0.7875 | 0.9397 | 0.8094 | 0.8298 | 0.7231 | 19 (+2) / 313 | 13.23 s (-0.96s) | +4.50 s | 0.1860 |
| #4 | `baseline_smooth50` *(floor)* | 0.3256 | 0.3256 | 0.6207 | 0.8990 | 0.7781 | 0.7615 | 0.3225 | 26 (+9) / 594 | 10.80 s (-3.39s) | +0.62 s | 0.0816 |
| #5 | `baseline_smooth25` | 0.2868 | 0.3019 | 0.6200 | 0.8974 | 0.7620 | 0.7123 | 0.3857 | 36 (+19) / 746 | 7.58 s (-6.61s) | -1.21 s | 0.2034 |
| #6 | `baseline_raw` | 0.0378 | 0.0270 | 0.5972 | 0.8920 | 0.7290 | 0.6430 | 0.4197 | 131 (+114) / 1.949 | 2.02 s (-12.17s) | -0.25 s | 0.1410 |

- **Intervals Bootstrap (1.000 mostres, 5%–95%) de `model_v02`**:
  - `spellF1`: `0.3429 .. 0.6629`
  - `macroF1`: `0.6935 .. 0.8871`
- **Quota de possessió (`Home Share`)**: Prediu `55.04%` per al FC Barcelona contra `57.27%` real (error absolut de només `2.23%`, el més baix de tots els models).

---

## 3. Diagnòstic de Transicions de v0.2 (Objectius per a v0.3)

L'anàlisi d'[`evaluation/reports/model_v02_transitions.csv`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/evaluation/reports/model_v02_transitions.csv) mostra 3 fenòmens físics a corregir a `model_v03.py`:
1. **Illes falses `D -> H -> D` dins d'aturades llargues** (`16039..16077`, `16192..16255` i `18089..18320`): quan el porter o llançador s'acosta a col·locar la pilota morta (`min_dist < 2.0 m`) mentre els 22 jugadors caminen (`roll_speed < 1.65 m/s`), la resta incondicional `score_d -= 4.0` força la sortida prematura de `D`.
2. **Canvis `H <-> A` en frames sense contacte**: amb una matriu de transició de Viterbi homogènia, el canvi d'estat pot produir-se mentre la pilota està en l'aire o oclosa, en lloc d'ancorar-se als frames on el nou equip entra en contacte amb la pilota.
3. **Absència de direccionalitat col·lectiva en oclusions llargues**: durant oclusions $> 3\text{ s}$ en joc viu, l'avançament del baricentre dels 22 jugadors cap a la porteria rival aporta evidència direccional auto-calibrada per meitat.
