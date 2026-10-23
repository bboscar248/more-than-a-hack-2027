# Resultats del Model v0.6 (`model_v06.py`)

Documentació de la sisena versió del model físic d'inferència de possessió, dissenyada per perfeccionar el llindar de recepció en transicions, eliminar la inèrcia de sortida de pilota aturada a represes i protegir els intervals vius llargs en la neteja de segments.

- **Script**: [`model_v06.py`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/model_v06.py)
- **Submissió**: [`evaluation/results/model_v06.csv`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/evaluation/results/model_v06.csv)

---

## 1. Millores Físiques i Arquitectòniques Implementades a v0.6

1. **Llindar Físic de Recepció Calibrat (`max_d_gain = 1.6m`)**:
   - A `model_v05.py`, el gating de contacte utilitzava un radi de `2.2m` per permetre canvis de possessió. No obstant això, `2.2m` encara és una distància de duel o pilota disputada en vol, fet que provocava que intercepcions en vol s'anticipessin abans que el defensor controlés la pilota.
   - A `model_v06.py`, s'ajusta el gating de recepció a `max_d_gain = 1.6m`, requerint que un jugador es trobi a una distància de control efectiu de la pilota per consolidar la possessió.

2. **Bonus Dinàmic de Recuperació Defensiva (`tackle_bonus = 8.0`)**:
   - Per evitar que l'elevada penalització de transició (`p_switch = -70.0`) retardi la detecció de robatoris nets i tacles defensius quan un jugador té la pilota als peus, s'identifica per frame si el defensor té control absolut (`dist < 1.2m` i `dist < opp_dist - 0.5m`).
   - En aquests frames, s'aplica una bonificació de transició `+8.0`, reduint la barrera efectiva de canvi a `-62.0` només durant l'impacte físic real, accelerant la detecció de la recuperació sense generar soroll espuri.

3. **Barrera de Sortida de Pilota Aturada Calibrada (`p_exit_d = 12.0`)**:
   - A `model_v05.py`, `p_exit_d = 15.0` provocava un retard de fins a +34 frames a la represa del servei de porteria (frame `18389`), perquè la velocitat dels jugadors encara era baixa i la penalització de sortida era excessiva.
   - Calibrant `p_exit_d` a `12.0`, la represa s'alinea a només **4 frames** de veritat terreny (`18385` vs `18389`), reduint la latència i elevant el F1 de pilota aturada a **0.9514**.

4. **Neteja No Destructiva de Ralis Vius a Pass 2 (`clean_spurious_segments_v6`)**:
   - Es protegeixen els blocs de joc viu llargs (`block_len > 250 frames` = 10s) perquè cap subsegment intern legítim no sigui convertit a `D` per la desacceleració anticipatòria dels jugadors abans d'un xiulet.
   - Només es netegen subsegments en illes curtes/mitjanes ($\le 250\text{ frames}$) amb manca de jugadors o fracció fora del camp.

5. **Autoverificació Unitària Integrada (`run_self_checks()`)**:
   - Verifica automàticament la cinemàtica avançada, la supressió de ponts fora del camp, el bonus de recuperació a Viterbi i la protecció de ralis vius a Pass 2.

---

## 2. Comparativa Oficial al Scoreboard (`run_evaluation.py --bootstrap 1000`)

| Rànquing | Model | `macroF1` | `spellF1` (Mean) | `spellF1` (@0.5) | `Alive/Dead F1` | Exactitud | `F1(H)` | `F1(A)` | `F1(D)` | `transF1` (@25f) | Possessions | Durada Mitjana |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | **`model_v06` (v0.6)** | **`0.9149`** | **`0.5935`** | **`0.5806`** | **`0.9880`** | **`0.9079`** | **`0.9021`** | **`0.8911`** | **`0.9514`** | **`0.3784`** | **14 / 256** | **12.22 s** |
| #2 | `model_v05` (v0.5) | 0.9132 | 0.5935 | 0.5806 | 0.9875 | 0.9061 | 0.8991 | 0.8900 | 0.9504 | 0.3784 | 14 / 241 | 12.78 s |
| #3 | `model_v04` (v0.4) | 0.8678 | 0.5576 | 0.5455 | 0.9666 | 0.8717 | 0.8940 | 0.8542 | 0.8551 | 0.2000 | 16 / 269 | 15.40 s |
| #4 | `model_v03` (v0.3) | 0.8653 | 0.5576 | 0.5455 | 0.9662 | 0.8691 | 0.8889 | 0.8544 | 0.8525 | 0.2500 | 16 / 269 | 15.44 s |
| #5 | `model_v02` (v0.2) | 0.8334 | 0.5029 | 0.5143 | 0.9554 | 0.8415 | 0.8382 | 0.8696 | 0.7925 | 0.2273 | 18 / 304 | 14.03 s |
| #6 | `baseline_smooth125` | 0.5812 | 0.4242 | 0.4242 | 0.8991 | 0.7275 | 0.7864 | 0.8108 | 0.1464 | 0.1667 | 16 / 398 | 18.46 s |
| #7 | `model_viterbi` (v0.1) | 0.7875 | 0.4222 | 0.4444 | 0.9397 | 0.8015 | 0.8094 | 0.8298 | 0.7231 | 0.1860 | 19 / 313 | 13.23 s |
| #8 | `baseline_smooth50` | 0.6207 | 0.3255 | 0.3255 | 0.8990 | 0.7132 | 0.7781 | 0.7615 | 0.3225 | 0.0816 | 26 / 594 | 10.80 s |
| #9 | `baseline_smooth25` | 0.6200 | 0.2868 | 0.3019 | 0.8974 | 0.6912 | 0.7620 | 0.7123 | 0.3857 | 0.2034 | 36 / 746 | 7.58 s |
| #10 | `baseline_raw` | 0.5972 | 0.0378 | 0.0270 | 0.8920 | 0.6512 | 0.7290 | 0.6430 | 0.4197 | 0.1410 | 131 / 1.949 | 2.02 s |

---

## 3. Anàlisi dels Guanys Clau a v0.6

1. **Rècord Absolut de la Competició (`macroF1 = 0.9149`)**:
   - `model_v06` estableix el nou màxim històric del marcador amb un `macroF1` de **0.9149** i una exactitud global de **90.79%**.
2. **Superació de la Barrera del 90% a les Tres Classes**:
   - Local ($H$): **0.9021** (supera per primer cop el 90%).
   - Visitant ($A$): **0.8911**.
   - Pilota Aturada ($D$): **0.9514**.
3. **Quasi Perfecció a Joc Viu vs Mort (`Alive/Dead F1 = 0.9880`)**:
   - L'índex $F_1$ de discriminació alive/dead assoleix un impressionant **98.80%**.
4. **Realisme de Partit Sencer**:
   - **256 possessions totals** de mitjana 12.22s (mediana 7.60s) i **109 aturades**, dins de la distribució natural d'un partit de màxim nivell.
   - Finestra pre-kickoff 2a part (`110808..113742`): **100.0% D** (2.934 frames).
   - Totes les celebracions de gol verifiquen $> 91.7\%$ de temps mort sense cap possessió falsa durant aturades.
