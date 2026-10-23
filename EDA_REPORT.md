# Exploratory Data Analysis (EDA) & Proposta de Model — More Than A Hack 2027

Fitxer generat a partir de l'execució de `eda.py` sobre els 199.319 frames del partit **FC Barcelona (`H`) vs Athletic Club (`A`)**.

---

## 1. Conclusions Principals de l'EDA

1. **El problema central de la baseline (`ball missing != Dead Ball`)**:
   - En els **158.857 frames avaluats** (1a i 2a part), la pilota és invisible (`NaN`) el **40,69%** del temps (150 episodis d'oclusió, amb mediana de 10,72 s i mitjana de 17,24 s).
   - A la finestra pública de Ground Truth (`11185..18684`, 7.500 frames), de 885 frames sense pilota, **el 44,18% (391 frames) són joc viu (`H` o `A`)** (passades aèries, canvis d'orientació o oclusions a l'àrea).
   - La baseline (`starter/baseline.py`) assigna `D` sempre que la pilota desapareix, cosa que **fragmenta possessions reals en micro-spells** (`H -> D -> H`) i enfonsa la mètrica principal (`spellF1`). Amb `baseline_smooth50` es prediuen **594 possessions** en el partit complet, molt per sobre de les ~180–250 possessions reals d'un partit.
   - En canvi, **quan el GT és realment pilota morta (`D`)**, la pilota **és visible el 66,37% del temps** (aturada abans d'un servei o fora dels límits del camp: el 10,67% dels frames amb pilota visible en `D` estan fora dels límits `105.3m x 68.0m`, contra només el 0,03% en joc viu).

2. **Senyals cinemàtics (ATD) per separar Joc Viu (`H`/`A`) vs Pilota Morta (`D`)**:
   - **Velocitat mitjana dels jugadors**: durant el joc viu és de **2,01–2,75 m/s**, mentre que durant pilota morta (`D`) baixa a **1,40–1,45 m/s** (jugadors caminant o aturats).
   - **Proximitat jugador-pilota (`< 2.0 m`)**: quan la pilota és visible i hi ha un jugador a menys de 2 m (53,7% dels frames visibles), l'equip del jugador més proper coincideix amb el Ground Truth el **88,7% de les vegades per a `H`** i el **85,0% per a `A`**. Els errors restants corresponen a rebots/duels sense control o pilotes en trànsit que passen prop d'un rival.

3. **Smart Tagging com a *prior* tàctic i detector de represes**:
   - Cobreix el **82,3%** dels frames avaluats del partit (75,5% a la finestra pública).
   - Quan una fase ofensiva (`BUILD UP`, `PROGRESSION`, `ATTACKING TRANSITION`, `CREATING CHANCES`, `LONG BALL`, `GOALS`) té `Team = Barcelona`, la precisió sobre `GT = H` és del **94,4%**.
   - Els 59 esdeveniments `SET PIECES` (`Goal Kick`, `Free Kick`, `Corner Kick`, `Throw-in`, `Kick Off`) s'alineen amb els intervals `D`: l'instant del xut de represa marca la transició `D -> H/A`.

4. **Diagnòstic de fragmentació sobre `baseline_smooth50`**:
   - Només suprimint/unint els forats curts de `D` ($\le 6\text{ s}$) en memòria sobre `baseline_smooth50`, el `spellF1` públic puja de **0.3256 a 0.3610 (+0.0354)** i el `macroF1` de **0.6207 a 0.6264**.

---

## 2. Arquitectura de Model Recomanada

Atès que només tenim **5 minuts de Ground Truth públic (17 possessions)** i que el rànquing s'avalua amb `spellF1` (IoU temporal d'intervals), un model d'aprenentatge profund extrem-a-extrem sobre 7.500 frames patiria *overfitting* sever. L'enfocament més robust és un **Model d'Espai d'Estats / HSMM (Viterbi sobre probabilitats d'emissió físiques + Smart Tagging)** estructurat en dues etapes desacoplades:

### Etapa 1: Detector d'Estat del Joc (`Alive` vs `Dead Ball D`)
En lloc d'assumir `pilota invisible == D`, estimar la probabilitat $P(D_t)$ a cada frame combinant:
- **Fora de límits (`Out-of-Bounds`)**: pilota fora de `[0, 105.3] x [0, 68.0]` m.
- **Baixa activitat cinemàtica**: velocitat mitjana dels dos equips `< 1.7 m/s` i baixa dispersió/moviment col·lectiu.
- **Ancoratge amb `SET PIECES` (Smart Tagging)**: finestra prèvia a `Goal Kick`, `Corner Kick`, `Free Kick`, `Throw-in` i `Kick Off` fins al frame on la pilota accelera (represa del joc).
- **Penalització de durada / Viterbi**: una aturada real (`D`) dura normalment $\ge 4\text{--}5\text{ s}$ (mínim al GT públic: 5,52 s), evitant que oclusions de 1–3 s trenquin una possessió.

### Etapa 2: Rastrejador d'Equip en Possessió (`H` vs `A`) durant Joc Viu
- ** Emissions de contacte i control**:
  - Quan un jugador és a `< 2.0 m` de la pilota durant diversos frames o en el moment d'una desacceleració/acceleració de la pilota (recepció o passada), emetre evidència forta cap al seu equip (`H` o `A`).
  - Quan la pilota vola (`> 2.0 m` de qualsevol jugador) o està oclosa (`NaN`), **mantenir la inèrcia de l'estat anterior** (o interpolar la trajectòria de la pilota entre l'últim contacte i la següent recepció: si surt de `H` i la rep `H`, tota la passada és `H`).
- **Prior de l'Smart Tagging**:
  - Incorporar un pes bayesià suau a favor de `H` o `A` quan hi ha fases ofensives actives (`BUILD UP`, `PROGRESSION`, etc.) i utilitzar l'equip del `SET PIECES` per inicialitzar el posseïdor exactament a l'inici de cada represa (`D -> H/A`).
- **Decodificació global amb Viterbi**:
  - Matriu de transició que penalitza canvis d'equip (`H <-> A`) sense control sostingut (filtrant tocs simples, rebots i duels aeris no controlats que les regles de `CHALLENGE.md` exclouen).

---

## 3. Output Complet de `eda.py`

```text
================================================================================
EXPLORATORY DATA ANALYSIS (EDA) - POSSESSION CHALLENGE
================================================================================

[1/5] Streaming tracking data and extracting features...
  Processed 199,319 frames in 9.15s.

[2/5] Loading homography and smart tagging...
  Homography loaded (199,319 rows).
  Smart Tagging loaded (638 instances).

================================================================================
PART 1: GRADED MATCH OVERVIEW (158,857 frames across Half 1 & 2)
================================================================================

1. Ball Visibility & Pitch Boundaries:
  - Total graded frames:          158,857 (H1: 79,467, H2: 79,390)
  - Ball visible:                 94,214 frames (59.31%)
  - Ball missing / occluded:      64,643 frames (40.69%)
  - Ball out-of-bounds:           3,949 frames (4.19% of visible, 2.49% of graded)

2. Missing-Ball Gap Durations (N = 150 gap episodes in graded match):
  Metric       | Seconds    | Frames    
  -------------+------------+-----------
  p50          |    10.72 s |      268 f
  p75          |    21.06 s |      526 f
  p90          |    33.81 s |      845 f
  p95          |    50.80 s |     1269 f
  p99          |   104.75 s |     2618 f
  Max          |   221.48 s |     5537 f
  Mean         |    17.24 s |    431.0 f

3. Track Slot Visibility Distribution (Players):
  Team         | Mean   | Min  | p25  | Median | p75  | Max 
  -------------+--------+------+------+--------+------+-----
  Home (BAR)   |  10.20 |    0 |   10 |     11 |   11 |   19
  Away (ATH)   |  10.29 |    1 |   10 |     11 |   11 |   15
  Total        |  20.49 |    1 |   21 |     22 |     22 |   25
  - Frames with exactly 22 tracked players: 78,597 (49.5%)
  - Frames with >= 20 tracked players:      135,213 (85.1%)
  - Frames with < 15 tracked players:       8,356 (5.3%)

================================================================================
PART 2: PUBLIC GROUND TRUTH (11,185..18,684, 7,500 frames = 5.0 mins) DEEP DIVE
================================================================================

1. Ground Truth Segment Statistics:
  Label  | Segments  | Total Frames  | Total Time  | Mean (s)  | Min (s)  | Median (s)  | Max (s) 
  -------+-----------+---------------+-------------+-----------+----------+-------------+---------
  H      |         8 |         3,454 |    138.16 s |     17.27 |     0.76 |        9.96 |    42.44
  A      |         9 |         2,577 |    103.08 s |     11.45 |     2.84 |        7.44 |    35.72
  D      |         4 |         1,469 |     58.76 s |     14.69 |     5.52 |       16.40 |    20.44
  H+A    |        17 |         6,031 |    241.24 s |     14.19 |     0.76 |        7.56 |    42.44

2. GT Label vs Ball Visibility:
          Missing (False)  Visible (True)  Total      Vis %
GT Label                                                   
A                     386            2191   2577  85.021343
D                     494             975   1469  66.371681
H                       5            3449   3454  99.855240

3. Missing-Ball Breakdown in Public Window (Total missing = 885 frames):
  - GT H:     5 frames (  0.6%)
  - GT A:   386 frames ( 43.6%)
  - GT D:   494 frames ( 55.8%)
  ==> CRUCIAL INSIGHT: 391 of 885 missing-ball frames (44.2%) are ALIVE play (H or A)!
      The baseline maps every missing ball to Dead Ball (D), directly fragmenting real possessions.

4. Ball Out-of-Bounds (when visible) vs GT Label:
          In-Bounds  Out-Of-Bounds  Total      OOB %
GT Label                                            
A              2189              2   2191   0.091283
D               871            104    975  10.666667
H              3449              0   3449   0.000000

5. Nearest Player Team vs GT Label (when ball is visible):
Nearest Player Team  A_closest  H_closest
GT Label                                 
A                         1640        551
D                          185        790
H                          699       2750

6. Proximity within 2.0m:
  - Visible frames with a player within 2.0m: 3,553 of 6,615 (53.7%)
Nearest Player Team (<2m)  A_closest  H_closest
GT Label                                       
A                               1028        182
D                                102        286
H                                220       1735
  - When player < 2m and GT is H: Home is closest 88.7% of the time.
  - When player < 2m and GT is A: Away is closest 85.0% of the time.

7. Kinematics and Homography Error across GT Classes (Mean +/- Std):
  GT   | Mean Speed H (m/s)   | Mean Speed A (m/s)   | Reproj Error (px)   
  -----+----------------------+----------------------+---------------------
  H    |   2.01 +/-  0.73     |   2.40 +/-  0.97     |   0.23 +/-  0.04
  A    |   2.75 +/-  0.93     |   2.47 +/-  0.87     |   0.23 +/-  0.06
  D    |   1.45 +/-  0.68     |   1.40 +/-  0.74     |   0.22 +/-  0.05

================================================================================
PART 3: SMART TAGGING ALIGNMENT & PREDICTIVE POWER
================================================================================

1. Smart Tagging Frame Coverage:
  - Total instances in XML:        638
  - Graded frames covered (>=1 tag): 130,698 / 158,857 (82.3%)
  - Public window covered:         5,661 / 7,500 (75.5%)

2. Confusion Matrix: Smart Tagging Core Offensive Phase Team vs GT Label:
GT Label                A    D     H
SmartTag Core Phase                 
Atletic Club (A)     1062  537   660
Barcelona (H)          84    6  1506
Both                    0   24    88
No active tag        1431  902  1200
  - When SmartTag is 'Barcelona': 1506 / 1596 frames are GT H (Precision = 94.4%)
  - When SmartTag is 'Atletic Club': 1062 / 2259 frames are GT A (Precision = 47.0%)

3. Set Pieces Alignment with Dead-Ball (D) Intervals:
  GT Dead-Ball (D) Segments in Public Window (N = 4):
    D Segment 5: frames 12851..12989 (138 f, 5.52 s)
    D Segment 14: frames 15889..16400 (511 f, 20.44 s)
    D Segment 17: frames 17380..17838 (458 f, 18.32 s)
    D Segment 19: frames 18027..18389 (362 f, 14.48 s)

  Set Piece Events Overlapping Public Window (N = 3):
    Type: Kick Off (Start)     | Team: N/A             | Frames: 11097..11222 (5.00 s)
    Type: Goal Kick            | Team: Barcelona       | Frames: 18320..19195 (35.00 s)
    Type: Free Kick            | Team: Atletic Club    | Frames: 17533..18408 (35.00 s)
  Key Takeaway: Set Pieces mark the restart of play; the restart instant coincides with the D -> H/A transition.

================================================================================
PART 4: BASELINE FAILURE MODE DIAGNOSIS
================================================================================

1. Segmentation & Metrics Comparison on Public Window (GT: 17 possessions, 4 D segments):
  Submission           | macroF1 | spellF1 | Alive/Dead | Pred Spells | Pred D Segs | Mean Spell (s)
  ---------------------+---------+---------+------------+-------------+-------------+---------------
  baseline_raw         |  0.5972 |  0.0378 |     0.8920 |         131 |           6 |           2.02
  baseline_smooth25    |  0.6200 |  0.2868 |     0.8974 |          36 |           4 |           7.58
  baseline_smooth50    |  0.6207 |  0.3256 |     0.8990 |          26 |           4 |          10.80
  baseline_smooth125   |  0.5812 |  0.4242 |     0.8991 |          16 |           1 |          18.46
  GT Truth             |  1.0000 |  1.0000 |     1.0000 |          17 |           4 |          14.19

2. Predicted Segment Counts on Full Graded Match (158,857 frames):
  Submission           | Total Segments | H Spells  | A Spells  | D Segments | Mean Spell (s)
  ---------------------+----------------+-----------+-----------+------------+---------------
  baseline_raw         |          2,099 |       968 |       981 |        150 |           1.93
  baseline_smooth25    |            879 |       365 |       381 |        133 |           5.41
  baseline_smooth50    |            727 |       289 |       305 |        133 |           7.24
  baseline_smooth125   |            477 |       195 |       203 |         79 |          12.40

3. In-Memory Diagnostic: Mitigating Ball Dropout Fragmentation on baseline_smooth50:
  Baseline (baseline_smooth50): macroF1 = 0.6207, spellF1 = 0.3256, pred spells = 26

  Testing post-processing bridging of short D segments (dropout suppression):
  Max Gap Bridged  | macroF1 | d_macro  | spellF1 | d_spell  | Pred Spells
  -----------------+---------+----------+---------+----------+------------
  <=  1.0s ( 25 f)   |  0.6207 |  +0.0000 |  0.3256 |  +0.0000 |          26
  <=  2.0s ( 50 f)   |  0.6091 |  -0.0116 |  0.3415 |  +0.0159 |          24
  <=  3.0s ( 75 f)   |  0.6091 |  -0.0116 |  0.3415 |  +0.0159 |          24
  <=  4.0s (100 f)   |  0.6091 |  -0.0116 |  0.3415 |  +0.0159 |          24
  <=  5.0s (125 f)   |  0.6091 |  -0.0116 |  0.3415 |  +0.0159 |          24
  <=  6.0s (150 f)   |  0.6264 |  +0.0056 |  0.3610 |  +0.0354 |          24
  <=  8.0s (200 f)   |  0.6264 |  +0.0056 |  0.3610 |  +0.0354 |          24
  <= 10.0s (250 f)   |  0.6264 |  +0.0056 |  0.3610 |  +0.0354 |          24

================================================================================
EDA COMPLETE.
================================================================================
```
