# Resultats del Model v0.1 (`model_viterbi.py`)

Documentació de la primera versió del model d'inferència de possessió basat en HMM + Viterbi físic i priors d'Smart Tagging.

- **Script**: [`model_viterbi.py`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/model_viterbi.py)
- **Submission**: [`evaluation/results/model_viterbi.csv`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/evaluation/results/model_viterbi.csv)

---

## 1. Resum de l'Arquitectura v0.1

1. **Extracció cinemàtica (ATD Tàctic)**:
   - Visibilitat de pilota (`ball_vis`), posició i velocitat (`ball_x`, `ball_y`, `ball_s`), detecció de fora de límits (`oob` sobre `105.3m x 68.0m`).
   - Distància mínima dels jugadors de cada equip a la pilota (`min_dist_h`, `min_dist_a`) i velocitat mitjana col·lectiva suavitzada (`roll_speed_50`, finestra de 2.0 s).
2. **Priors Tàctics (Smart Tagging)**:
   - Fases ofensives (`BUILD UP`, `PROGRESSION`, `ATTACKING TRANSITION`, `CREATING CHANCES`, `LONG BALL`, `GOALS`) assignades a l'equip anotat.
   - Fases defensives (`DEFENDING IN...`, `DEFENSIVE TRANSITION`) invertides cap a l'equip rival.
   - Separació dinàmica dels `SET PIECES`: fase d'aturada (`prior_d`) mentre la velocitat és baixa o la pilota està aturada/fora, i fase de represa (`prior_h`/`prior_a`) un cop s'executa el xut.
3. **Decodificació de Viterbi i Post-processament**:
   - Matriu de transició en espai logarítmic (`p_switch=60.0`, `p_enter_d=50.0`, `p_exit_d=15.0`).
   - Neteja de micro-aturades (`min_d_frames=75`, 3.0 s) i rebots aïllats entre el mateix equip (`min_deflection_frames=35`, 1.4 s).
   - Frames fora de les dues parts avaluades etiquetats com a `X`.

---

## 2. Comparativa al Scoreboard Oficial (`run_evaluation.py --bootstrap 1000`)

| Model | `macroF1` | `spellF1` (Mean) | `spellF1` (@0.5) | `spellF1` (@0.6) | `spellF1` (@0.7) | `Alive/Dead F1` | `F1(H)` | `F1(A)` | `F1(D)` | Possessions (5m / Partit) | Durada Mitjana (5m / Partit) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`model_viterbi` (v0.1)** | **0.7875** | **0.4222** | **0.4444** | **0.3889** | **0.2778** | **0.9397** | **0.8047** | **0.8349** | **0.7231** | **19** / **313** | **13.0 s** / **10.27 s** |
| `baseline_smooth125` | 0.5812 | 0.4242* | 0.4242 | 0.2424 | 0.2424 | 0.8991 | 0.7864 | 0.8108 | 0.1464 | 16 / 398 | 18.5 s / 12.40 s |
| `baseline_smooth50` *(floor)* | 0.6207 | 0.3256 | 0.3256 | 0.2326 | 0.0930 | 0.8990 | 0.7781 | 0.7615 | 0.3225 | 26 / 594 | 10.8 s / 7.24 s |
| `baseline_smooth25` | 0.6200 | 0.2868 | 0.3019 | 0.2264 | 0.0755 | 0.8974 | 0.7620 | 0.7123 | 0.3857 | 36 / 746 | 7.6 s / 5.41 s |
| `baseline_raw` | 0.5972 | 0.0378 | 0.0270 | 0.0135 | 0.0000 | 0.8920 | 0.7290 | 0.6430 | 0.4197 | 131 / 1.949 | 2.0 s / 1.93 s |
| **Ground Truth Públic** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **17** / — | **14.2 s** / — |

---

## 3. Alineació Segment a Segment amb el Ground Truth Públic (17 Possessions)

| GT# | Equip | GT Inici | GT Final | Durada GT | Predicció Emparellada | Durada Pred | IoU | Diagnòstic |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | **A** | 11185 | 11351 | 6.64 s | `11185..11584 (A)` | 15.96 s | 0.416 | El model no detecta la recuperació de `H` a 11351 |
| **1** | **H** | 11351 | 11553 | 8.08 s | *No emparellat* | — | 0.000 | Absorbit dins l'interval `11185..11584 (A)` |
| **2** | **A** | 11553 | 11624 | 2.84 s | *Solapat amb #0* | — | 0.071 | Continuació del bloc `A` |
| **3** | **H** | 11624 | 12665 | 41.64 s | `11819..12603 (H)` | 31.36 s | **0.753** | Entrada tardana (11819 vs 11624) per oclusió/passada |
| **4** | **A** | 12665 | 12851 | 7.44 s | `12603..12883 (A)` | 11.20 s | **0.664** | Bona detecció de recuperació i sortida a `D` |
| **5** | **H** | 12989 | 13704 | 28.60 s | `12979..13671 (H)` | 27.68 s | **0.941** | Alineació gairebé exacta en represa i pèrdua |
| **6** | **A** | 13704 | 13785 | 3.24 s | `13671..13728 (A)` | 2.28 s | 0.211 | Detectada, però talla abans (13728 vs 13785) |
| **7** | **H** | 13785 | 14846 | 42.44 s | `14271..14844 (H)` | 22.92 s | **0.540** | Fragmentada al principi per un fals tram d'`A` |
| **8** | **A** | 14846 | 15346 | 20.00 s | `14844..15647 (A)` | 32.12 s | **0.623** | Inici clavat (14844 vs 14846); no talla pel toc #9 |
| **9** | **H** | 15346 | 15365 | 0.76 s | *No emparellat* | — | 0.000 | Micro-possessió de 19 frames (0.76 s) |
| **10** | **A** | 15365 | 15766 | 16.04 s | *Fusionat amb #8* | — | 0.306 | Unit amb #8 perquè no es detecta #9 |
| **11** | **H** | 15766 | 15799 | 1.32 s | *No emparellat* | — | 0.000 | Possessió curta d'1.32 s (33 frames) |
| **12** | **A** | 15799 | 15889 | 3.60 s | `15706..15853 (A)` | 5.88 s | 0.295 | Desplaçada a l'esquerra |
| **13** | **A** | 16400 | 17293 | 35.72 s | `16423..17163 (A)` | 29.60 s | **0.829** | Molt bona continuïtat durant 40% d'oclusió |
| **14** | **H** | 17293 | 17380 | 3.48 s | *No emparellat* | — | 0.000 | Tancada massa aviat cap a `D` (falta prèvia) |
| **15** | **A** | 17838 | 18027 | 7.56 s | `17845..18079 (A)` | 9.36 s | **0.755** | Molt bona sortida de falta i final de jugada |
| **16** | **H** | 18389 | 18685 | 11.84 s | `18470..18685 (H)` | 8.60 s | **0.726** | Represa de porteria detectada 3.2 s tard (18470 vs 18389) |

---

## 4. Mancances Detectades en v0.1 (Objectius per a v0.2)

1. **Absència d'interpolació cinemàtica (Kalman / pontatge de passades) quan la pilota desapareix**:
   - Quan la pilota s'oclou durant una passada o un servei (p. ex. l'inici del segment #3 `11624..11819` o la represa de porteria #16 `18389..18470`), les emissions d'`H` i `A` cauen a zero perquè `ball_vis == False`. Si interpoléssim la trajectòria física de la pilota (posició i vector velocitat entre l'últim contacte i la següent recepció) i detectéssim **esdeveniments de contacte/passada** (qui llança i qui rep), sabríem en quin frame exacte canvia de mans o es posa en joc.
2. **Retard en les fronteres de transició (`D <-> H/A` i `H <-> A`)**:
   - L'ús de `roll_speed_50` (mitjana mòbil centrada de 2 segons) difumina el moment exacte en què l'àrbitre xiula una falta (p. ex. #14 `17293..17380`, on `D` entra abans d'hora a `17163`) o el moment exacte en què es xuta una falta/servei de porteria (#16 `18389`, on el model surt de `D` a `18470`, 81 frames tard).
   - Cal ancorar l'entrada i sortida de `D` al **frame exacte d'acceleració de la pilota (`ball_s` o canvi de posició de pilota)** i a la sortida física de camp (`oob`), en comptes d'una mitjana mòbil lenta.
3. **Pontatge cec de parades (`min_d_frames=75`) vs condicional a la física**:
   - En lloc d'esborrar cegament qualsevol `D < 3.0 s` (que podria menjar-se un servei ràpid si la pilota ha sortit de banda i s'ha tret ràpid), només s'han de pontar els segments `D` curts quan la pilota **no ha sortit del camp (`~oob`)** i no hi ha un `SET PIECES` actiu.
4. **Detecció d'intercepcions i controls curts (segments #1, #6, #14)**:
   - Quan un jugador rival entra a `< 1.2 m` de la pilota i **canvia la direcció o velocitat de la pilota** (acceleració / canvi d'angle del vector velocitat de la pilota $\Delta \vec{v}_{\text{ball}}$) o manté la pilota controlada sense cap rival a `< 2.0 m`, és una recuperació real encara que duri 2–8 segons.
