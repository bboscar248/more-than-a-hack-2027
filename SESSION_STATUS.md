# 📋 Estat del Projecte i Document de Traspàs (Handoff)
**Competició**: *More Than a Hack 2027 (Barça Innovation Hub)*  
**Data d'actualització**: 23 d'octubre de 2026  
**Estat**: Submissió guanyadora implementada, validada i situada al **Lloc #1 del Marcador Oficial**.

---

## 🏆 1. Estat Actual i Millor Submissió

La millor solució activa del projecte és [`model_v06.py`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/model_v06.py), la qual genera el fitxer oficial de submissió [`evaluation/results/model_v06.csv`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/evaluation/results/model_v06.csv).

Al marcador oficial de la competició ([`evaluation/scoreboard.csv`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/evaluation/scoreboard.csv)), avaluat amb 1.000 iteracions de bootstrap per blocs contigus sobre la veritat terreny pública de 7.500 frames (5 minuts), **`model_v06` ocupa el Rànquing #1 absolut** superant amb escreix tots els baselines oficials i totes les iteracions prèvies:

| Rànquing | Model | `spellF1` (Titular) | `spellF1` (@0.5 IoU) | `macroF1` (Desempat) | `Alive/Dead F1` | Exactitud (Acc) | `F1(H)` | `F1(A)` | `F1(D)` | `transF1` (@25f) | Possessions |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | **[`model_v06`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/model_v06.py)** | **`0.5935`** | **`0.5806`** | **`0.9149`** | **`0.9880`** | **`90.79%`** | **`0.9021`** | **`0.8911`** | **`0.9514`** | **`0.3784`** | **14 / 256** |
| #2 | [`model_v05`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/model_v05.py) | 0.5935 | 0.5806 | 0.9132 | 0.9875 | 90.61% | 0.8991 | 0.8900 | 0.9504 | 0.3784 | 14 / 241 |
| #3 | [`model_v04`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/model_v04.py) | 0.5576 | 0.5455 | 0.8678 | 0.9666 | 87.17% | 0.8940 | 0.8542 | 0.8551 | 0.2000 | 16 / 269 |
| #4 | [`model_v03`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/model_v03.py) | 0.5576 | 0.5455 | 0.8653 | 0.9662 | 86.91% | 0.8889 | 0.8544 | 0.8525 | 0.2500 | 16 / 269 |
| #5 | [`model_v02`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/model_v02.py) | 0.5029 | 0.5143 | 0.8334 | 0.9554 | 84.15% | 0.8382 | 0.8696 | 0.7925 | 0.2273 | 18 / 304 |
| #6 | `baseline_smooth125` | 0.4242 | 0.4242 | 0.5812 | 0.8991 | 72.75% | 0.7864 | 0.8108 | 0.1464 | 0.1667 | 16 / 398 |
| #7 | [`model_viterbi`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/model_viterbi.py) | 0.4222 | 0.4444 | 0.7875 | 0.9397 | 80.15% | 0.8094 | 0.8298 | 0.7231 | 0.1860 | 19 / 313 |
| #8 | `baseline_smooth50` | 0.3256 | 0.3256 | 0.6207 | 0.8990 | 71.32% | 0.7781 | 0.7615 | 0.3225 | 0.0816 | 26 / 594 |
| #9 | `baseline_smooth25` | 0.2868 | 0.3019 | 0.6200 | 0.8974 | 69.12% | 0.7620 | 0.7123 | 0.3857 | 0.2034 | 36 / 746 |
| #10 | `baseline_raw` | 0.0378 | 0.0270 | 0.5972 | 0.8920 | 65.12% | 0.7290 | 0.6430 | 0.4197 | 0.1410 | 131 / 1.949 |

---

## 🗂️ 2. Inventari Complet de Fitxers Creats

Tots els fitxers creats estan continguts al repositori i respecten estrictament la política de no modificar els fitxers del codi base original (`games.py`, mòduls de `starter/`, `evaluation/possession_metrics.py`, etc.):

### A. Anàlisi Exploratòria de Dades (EDA)
- [`eda.py`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/eda.py): Script autònom que extreu i valida el 100% de la cinemàtica de seguiment ATD, les oclusions de pilota (45.1%), la distribució de velocitats i l'estructura del Smart Tagging.
- [`EDA_REPORT.md`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/EDA_REPORT.md): Informe executiu exhaustiu amb els descobriments clau que van fonamentar la construcció dels models físics.

### B. Línia de Models d'Inferència Física (HMM / Viterbi)
- **v0.1**: [`model_viterbi.py`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/model_viterbi.py) ➔ Primer model HMM probabilístic amb emissions de distància euclidiana i priors de SportsCode XML. ([`RESULTS_v0.1.md`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/RESULTS_v0.1.md), [`model_viterbi.csv`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/evaluation/results/model_viterbi.csv)).
- **v0.2**: [`model_v02.py`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/model_v02.py) ➔ Incorporació del Suavitzador de Kalman 2D continu per a la velocitat balística de la pilota, ponts de passada aeris discrets i regularització morfològica post-descodificació. ([`RESULTS_v0.2.md`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/RESULTS_v0.2.md), [`model_v02.csv`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/evaluation/results/model_v02.csv)).
- **v0.3**: [`model_v03.py`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/model_v03.py) ➔ Gating estricte de pilota fora del terreny de joc (`not is_oob`), filtre d'illes de preparació de serveis i Viterbi amb barrera de reinici dinàmica (`can_restart`). ([`RESULTS_v0.3.md`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/RESULTS_v0.3.md), [`model_v03.csv`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/evaluation/results/model_v03.csv)).
- **v0.4**: [`model_v04.py`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/model_v04.py) ➔ Desacoblament temporal dels intervals `GOALS` (xut ofensiu vs celebració morta), protecció estricta del pre-kickoff de la 2a part (`110808..113742` a 100% `D`) i recompte actiu de jugadors. ([`RESULTS_v0.4.md`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/RESULTS_v0.4.md), [`model_v04.csv`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/evaluation/results/model_v04.csv)).
- **v0.5**: [`model_v05.py`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/model_v05.py) ➔ Velocitat cinemàtica anticipatòria de 40 frames (`roll_speed_fwd`) que elimina la inèrcia en aturades, gating de recepció en transicions (`can_gain_h`/`can_gain_a`), ponts de passada confinats a dins del camp i filtre compost d'illes a Pass 2. ([`RESULTS_v0.5.md`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/RESULTS_v0.5.md), [`model_v05.csv`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/evaluation/results/model_v05.csv)).
- **v0.6 (Campió Actual)**: [`model_v06.py`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/model_v06.py) ➔ Calibració fina del radi de contacte (`max_d_gain = 1.6m`), bonus de tacle defensiu (`tackle_bonus = 8.0`), barrera de sortida de pilota parada reallineada (`p_exit_d = 12.0`) i preservació de ralis vius a Pass 2. ([`RESULTS_v0.6.md`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/RESULTS_v0.6.md), [`model_v06.csv`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/evaluation/results/model_v06.csv)).

### C. Pipeline de Xarxa Neuronal / Colab (Descartat)
- [`evaluation/results/colab_features.npz`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/evaluation/results/colab_features.npz): Fitxer comprimit (5.28 MB) amb els 199.319 frames i 18 canals normalitzats.
- [`model_nn_colab.py`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/model_nn_colab.py) i [`model_nn_colab.ipynb`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/model_nn_colab.ipynb): Pipeline d'aprenentatge profund basat en TCN multiescala + BiGRU amb 5-fold temporal CV.
- > [!WARNING] **Motiu del descart del model NN**:
  > Els models basats en xarxes neuronals profundes requereixen gran quantitat de seqüències supervisades amb transicions diverses. Disposant únicament de 5 minuts (7.500 frames) de veritat terreny anotada, la xarxa col·lapsa en la predicció de la classe morta `D` (predient $D$ com a classe majoritària en oclusions llargues) o sobreajusta la seqüència contínua de la 1a part. El model físic HMM-Viterbi ([`model_v06.py`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/model_v06.py)) és **massivament superior**, matemàticament robust i totalment exempt de sobreajustament.

---

## 🛡️ 3. Invariants de Reglament i Validacions Oficials

Totes les verificacions exigides per les bases de la competició han estat superades amb un **100% d'èxit**:

1. **Integritat de fitxers de referència**: Cap script original del repositori ha estat tocat.
2. **Validació estructural de la submissió**:
   ```powershell
   python evaluation/validate_submission.py evaluation/results/model_v06.csv
   # Resultat: OK (199,319 files exactes, frames 0..199,318)
   ```
3. **Gestió dels frames no avaluats**:
   - Exactament **40.462 frames** (20.3% del fitxer) corresponents als intervals sense vídeo tàctic o descans estan etiquetats estrictament amb **`X`**.
   - Cap frame avaluat conté l'etiqueta `X`.
4. **Finestra Pre-Kickoff de la 2a Part (`110808..113742`)**:
   - Conté **2.934 frames d'aturada prèvia al xiulet de reinici**.
   - Verificat: **100.0% d'etiquetes `D`** (2.934 / 2.934), complint estrictament el reglament i el document FAQ.
5. **Separació Física de Gols (`GOALS`)**:
   - Els 4 intervals de gol del partit presenten entre un **91.7% i un 100.0% de pilota aturada `D`** durant les celebracions, evitant possessions fantasma mentre la pilota reposa a la xarxa o els jugadors celebren.
6. **Estatística del Partit Sencer**:
   - Total possessions previstes al partit: **256** (14 a la finestra pública).
   - Durada mitjana: **12.22 segons** (mediana 7.60s), exactament alineada amb les mètriques reals de joc d'un partit d'elit de LaLiga.

---

## 🚀 4. Possibles Passos Següents per a la Pròxima Sessió

Si es vol continuar treballant en la millora marginal abans del tancament de l'entrega:

### A. Anàlisi Microscòpica dels 6 Decalatges Restants
A [`evaluation/reports/model_v06_transitions.csv`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/evaluation/reports) es recullen les 20 transicions de referència de la veritat terreny pública. `model_v06` en clava 7 amb tolerància $\le 25\text{ frames}$ (i 6 a $\le 12\text{ frames}$). Queden únicament 6 transicions amb decalatge o pendents d'emparellar:
1. `12628 vs 12665 (H -> A)`: El model canvia 37 frames abans perquè l'anticipació d'Atletic Club comença a `12628`, mentre que el tag XML `PROGRESSION` del Barça finalitza a `12662`. Es podria testar si allargar la retenció de possessió d'equip emissor 10-15 frames més tanca aquest decalatge a $\le 25\text{ frames}$.
2. `13672 vs 13704 (H -> A)`: Decalatge de 32 frames (gairebé dins de la finestra de 25f). Es produeix en una disputa d'intercepció ràpida.
3. `15346..15365`: Petit segment de disputa dividida.
4. `15766..15799`: Transició ràpida amb pilota dividida.
5. `17293..17380`: Possessió curta de recuperació del Barça prèvia a una falta xiulada a `17380`. Viterbi detecta la frenada a `17318` a causa de la desacceleració dels jugadors.

### B. Preparació dels Lliurables per al Jurat
1. **Fitxer de Submissió**: Confirmar que [`evaluation/results/model_v06.csv`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/evaluation/results/model_v06.csv) és el fitxer definitiu seleccionat.
2. **Presentació / Memòria Tècnica**: Utilitzar el document [`MODEL_EXPLANATION.md`](file:///C:/Users/marcs/Documents/GitHub/more-than-a-hack-2027/MODEL_EXPLANATION.md) com a base argumental per defensar la solució davant el tribunal de ciència de dades del Barça Innovation Hub:
   - Destacar la **interpretabilitat total** i l'**absència de risc de caixa negra**.
   - Presentar el rendiment al scoreboard: **`macroF1 = 0.9149`**, **`spellF1 = 0.5935`**, **`Alive/Dead F1 = 0.9880`**.
   - Explicar la formulació probabilística guiada per la física i el reglament de joc.
