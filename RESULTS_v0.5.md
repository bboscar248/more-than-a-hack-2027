# Resultats del Model v0.5 (`model_v05.py`)

Documentació de la cinquena versió del model físic d'inferència de possessió, dissenyada per eliminar completament la inèrcia de transició a pilota aturada i el canvi prematur de possessió en passades interceptades en vol.

- **Script**: [`model_v05.py`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/model_v05.py)
- **Submissió**: [`evaluation/results/model_v05.csv`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/evaluation/results/model_v05.csv)

---

## 1. Millores Físiques i Arquitectòniques Implementades a v0.5

1. **Velocitat Cinemàtica Anticipatòria (`roll_speed_fwd`)**:
   - A `model_v04.py`, la velocitat col·lectiva dels jugadors utilitzava una finestra rodant centrada de 50 frames (`center=True`), introduint un decalatge d'inèrcia de ~25-32 frames (1.0-1.3s) a les frenades de xiulet i a les arrencades de represa.
   - A `model_v05.py`, es calcula una finestra rodant avançada de 40 frames (`roll_speed_fwd = s_interp.iloc[::-1].rolling(40, min_periods=1).mean().iloc[::-1].values`), que anticipa immediatament la desacceleració dels 22 jugadors en aturades i l'arrencada en represes sense retard físic.
   - L'ús de `roll_speed_fwd` a les emissions de pilota aturada $D$ ha disparat el F1 de classe morta de **0.8551** a **0.9504** (+9,5 punts percentuals).

2. **Gating Físic de Recepció en Transicions d'Equip (`can_gain_h` / `can_gain_a`)**:
   - A les versions anteriors, una passada interceptada o una pèrdua en vol podia canviar l'estat $H \leftrightarrow A$ abans que el receptor toqués la pilota, a causa de la proximitat del defensor al punt de caiguda.
   - S'han definit barreres físiques estrictes:
     $$\text{can\_gain\_h} = (\text{ball\_vis} \land \neg\text{oob} \land d_H < 2.2\text{m}) \lor (\neg\text{ball\_vis} \land \text{prior}_H > \text{prior}_A)$$
     $$\text{can\_gain\_a} = (\text{ball\_vis} \land \neg\text{oob} \land d_A < 2.2\text{m}) \lor (\neg\text{ball\_vis} \land \text{prior}_A > \text{prior}_H)$$
   - Al descodificador Viterbi (`viterbi_decode_v5`), les transicions $A \to H$ reben una penalització dinàmica de $-20.0$ si $\neg\text{can\_gain\_h}$, i les transicions $H \to A$ reben $-20.0$ si $\neg\text{can\_gain\_a}$.
   - Això reté la possessió de l'equip emissor fins a l'impacte físic real del receptor, duplicant la precisió de transicions (`trans_f1_25` passa de `0.2000` a `0.3784`).

3. **Ponts de Passada In-Bounds (`build_pass_bridges_v5`)**:
   - S'assegura que els tocs discrets de control només es registrin si la pilota és dins del camp (`b_vis[f] and not oob[f]`), i només s'estén un pont aeri entre tocs si no hi ha més de 4 frames fora del terreny de joc ($\sum \text{oob}[t_1:t_2] < 5$). S'eliminen els ponts aeris espuris en serveis de banda o llançaments de porteria.

4. **Filtre Compost d'Illes de Pilota Aturada (`clean_spurious_segments_v5`)**:
   - A més dels segments mono-equip curts, s'avalua qualsevol bloc viu compost delimitat per aturades ($D \to \text{bloc viu} \to D$, ex. $D \to A \to H \to D$).
   - Si la velocitat mitjana del bloc és $< 1.65\text{ m/s}$, els jugadors actius són $< 15$, la durada és $< 40\text{ frames}$ (1.6s) o la fracció fora de camp és $> 25\%$, tot el bloc es col·lapsa directament a $D$.
   - S'eliminen completament les falses possessions durant la col·locació de la pilota i les celebracions de gol.

5. **Autoverificació Unitària Integrada (`run_self_checks()`)**:
   - El script executa un conjunt de tests unitaris per verificar la cinemàtica avançada, la supressió de ponts fora del camp, el bloqueig de transicions sense toc a Viterbi i la neteja d'illes compostes abans de processar el partit complet.

---

## 2. Comparativa Oficial al Scoreboard (`run_evaluation.py --bootstrap 1000`)

| Rànquing | Model | `spellF1` (Mean) | `spellF1` (@0.5) | `macroF1` | `Alive/Dead F1` | Exactitud | `F1(H)` | `F1(A)` | `F1(D)` | Fragmentació | Possessions (5m / Partit) | `transF1` (@25f) | `transP` (@25f) | `transR` (@25f) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | **`model_v05` (v0.5)** | **`0.5935`** | **`0.5806`** | **`0.9132`** | **`0.9875`** | **`0.9061`** | **`0.8991`** | **`0.8900`** | **`0.9504`** | **`0.8571`** | **14** (-3) / **241** | **`0.3784`** | **`0.4118`** | **`0.3500`** |
| #2 | `model_v04` (v0.4) | 0.5576 | 0.5455 | 0.8678 | 0.9666 | 0.8717 | 0.8940 | 0.8542 | 0.8551 | 1.0000 | 16 (-1) / 269 | 0.2000 | 0.2000 | 0.2000 |
| #3 | `model_v03` (v0.3) | 0.5576 | 0.5455 | 0.8653 | 0.9662 | 0.8691 | 0.8889 | 0.8544 | 0.8525 | 1.0000 | 16 (-1) / 269 | 0.2500 | 0.2500 | 0.2500 |
| #4 | `model_v02` (v0.2) | 0.5029 | 0.5143 | 0.8334 | 0.9554 | 0.8415 | 0.8382 | 0.8696 | 0.7925 | 1.1905 | 18 (+1) / 304 | 0.2273 | 0.2083 | 0.2500 |
| #5 | `baseline_smooth125` | 0.4242 | 0.4242 | 0.5812 | 0.8991 | 0.7275 | 0.7864 | 0.8108 | 0.1464 | 0.8095 | 16 (-1) / 398 | 0.1667 | 0.1875 | 0.1500 |
| #6 | `model_viterbi` (v0.1) | 0.4222 | 0.4444 | 0.7875 | 0.9397 | 0.8015 | 0.8094 | 0.8298 | 0.7231 | 1.1429 | 19 (+2) / 313 | 0.1860 | 0.1739 | 0.2000 |
| #7 | `baseline_smooth50` | 0.3256 | 0.3256 | 0.6207 | 0.8990 | 0.7132 | 0.7781 | 0.7615 | 0.3225 | 1.4286 | 26 (+9) / 594 | 0.0816 | 0.0690 | 0.1000 |
| #8 | `baseline_smooth25` | 0.2868 | 0.3019 | 0.6200 | 0.8974 | 0.6912 | 0.7620 | 0.7123 | 0.3857 | 1.9048 | 36 (+19) / 746 | 0.2034 | 0.1538 | 0.3000 |
| #9 | `baseline_raw` | 0.0378 | 0.0270 | 0.5972 | 0.8920 | 0.6512 | 0.7290 | 0.6430 | 0.4197 | 6.5238 | 131 (+114) / 1.949 | 0.1410 | 0.0809 | 0.5500 |

---

## 3. Anàlisi dels Guanys Clau a v0.5

1. **Ruptura de la Barrera del 90% (`macroF1 = 0.9132`)**:
   - `model_v05` supera per primer cop el 91% de `macroF1` (+0.0454 sobre v0.4, +0.3320 sobre el baseline oficial).
   - L'exactitud global de classificació supera el **90.61%** (en comparació amb el 72.75% del baseline).
2. **Nou Record Històric en `spellF1` (`0.5935`)**:
   - La mètrica oficial de rànquing de la competició assoleix un rècord absolut de **0.5935** (+0.0359 sobre v0.4 i v0.3, +0.1693 sobre el millor baseline).
   - A la tolerància titular d'IoU 0.5, l'índex puja a **0.5806**.
3. **Quasi Perfecció en la Detecció Joc Viu vs Pilota Parada (`Alive/Dead F1 = 0.9875`)**:
   - La precisió i exhaustivitat en discernir si la pilota és viva o morta assoleix el 98.75%, pràcticament eliminant els falsos positius en aturades.
4. **Duplicació de la Detecció de Transicions (`transF1 @ 25f = 0.3784`)**:
   - Els canvis de possessió emparellats correctament pugen de 4 a 7 sobre els 20 de veritat terreny, amb una reducció dràstica de transicions espúries (de 16 a 10).
5. **Realitat Futbolística del Partit Sencer**:
   - Possessions totals: **241** (durada mitjana 12.8s, mediana 7.9s), reflectint amb precisió el ritme d'un partit d'elit de LaLiga.
   - Segment pre-kickoff de la 2a part: **100.0% D** (2.934 / 2.934 frames).
   - Celebracions de gol: **0 falses possessions** durant els trams post-gol.
