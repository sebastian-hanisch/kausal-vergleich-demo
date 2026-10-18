# Welches Verfahren, gemessen woran? – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-kausal-vergleich-demo.streamlit.app/)**

Siebzehntes und **letztes Stück** der Linie **Kausale Diagramme und Adjustierung** in der „Konzepte“-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning). Nachfolger von [diagrammfehler-demo](https://github.com/sebastian-hanisch/diagrammfehler-demo) und Zusammenfluss der Linie: [backdoor-demo](https://github.com/sebastian-hanisch/backdoor-demo), [schlechte-kontrollen-demo](https://github.com/sebastian-hanisch/schlechte-kontrollen-demo), [frontdoor-demo](https://github.com/sebastian-hanisch/frontdoor-demo) und [instrumente-demo](https://github.com/sebastian-hanisch/instrumente-demo) haben je eine Strategie gebaut; hier stehen sie nebeneinander vor derselben Wahrheit. (Der Titel ist derselbe wie bei den Zusammenfluss-Stücken der Ranking-Linien, [mcda-vergleich-demo](https://github.com/sebastian-hanisch/mcda-vergleich-demo) und [rating-vergleich-demo](https://github.com/sebastian-hanisch/rating-vergleich-demo); das Thema ist ein anderes.)

## Kernfrage

**Welche Schätzstrategie der Linie, wenn die Wahrheit bekannt ist, und gemessen woran?** Sechs Strategien (naiv, Backdoor mit zwei Mengen, alles anpassen, Frontdoor, 2SLS) in einer Welt mit **verborgener Ursache, Anstoß und Mediator**, gemessen an **Bias, Streuung, Deckung und Breite**, dazu ob die Annahme der Strategie im Diagramm gilt. Es gibt **kein bestes Verfahren, nur eines, dessen Annahme gilt**, und das Maß entscheidet, welches „das beste“ heißt. Und der Vergleich der Schätzungen untereinander verrät nicht, welche gilt.

## Modell

- **Welt (linear, Einheitsrauschen):** Nachfrage Z → Faktor X (0,8) und Servicegrad Y (0,6); Wetter W → Y (0,6); Anstoß I → X (π = 0,5); **verborgene Standortqualität U** → X (a = 0,6), Y (b = 0,6), dazu Bestand M (f), Anstoß I (h) nach Wahl; X → M (c = 1,0) → Y (0,6); X → Y direkt (e) nach Wahl; I → Y (g) nach Wahl; Y → Beschwerden C (0,8). **Wahrheit:** Gesamtwirkung bei Eingriff, c · 0,6 + e (0,6 in der Voreinstellung).
- **Sechs Strategien**, alle nur aus beobachteten Größen (U wird nie gelesen): **naiv** (Y auf X); **Backdoor {Z}** und **{Z, W}**; **alles anpassen** (Z, W, I, M, C); **Frontdoor** (ĉ · d̂ über den Bestand); **2SLS** mit dem Anstoß und den Kovariaten Z, W.
- **Neun Szenarien** als Presets, jeder Regler einzeln einstellbar: Verborgene Ursache mit Anstoß (Voreinstellung), Alles beobachtet, Kein Anstoß, Nichts brauchbar, Schwacher Anstoß, Anstoß wirkt direkt, Anstoß hängt an der Ursache, Teilweise vermittelt, Ursache wirkt auf den Mediator.
- **Gültigkeit am Diagramm:** Backdoor-, Frontdoor- und Instrument-Kriterium (wortgleich aus Stück 3, 8, 9) am Diagramm der Welt mit der verborgenen Ursache als Knoten. **Kriterium erfüllt ⇒ exakter Grenzwert-Bias null**; umgekehrt gilt es in den neun Szenarien, aber nicht ausnahmslos in der Welt (zufällige Auslöschung zweier Verletzungen ist möglich).

## Methodik

- **Exakt statt gewürfelt** (`kvl_strategies.py`): Grenzwert und Streuung je √n jeder Strategie folgen aus der Bevölkerungskovarianz; die Experimente 1, 2 und 5 haben keinen Zufall.
- **Kreuzungsgröße:** asymptotisch RMSE² = Bias² + σ² / n; das verzerrte, ruhige Backdoor schlägt die gültige, streuende Strategie unter n\* = (σ²_gültig − σ²_Backdoor) / Bias². Die Tabelle der Demo rechnet ohne die Verletzungen e, f, g, h (sie vergleicht mit gültigen Strategien).
- **Experimente 3 und 4** laufen über feste SplitMix64-Ströme (300 und 400 Datensätze je Zelle); Deckung zählt ab 93 %.
- **Vortest:** Hausman zwischen Backdoor {Z, W} und 2SLS (Varianzdifferenz, die Regression ist unter der Nullhypothese effizient); die Strategie „2SLS, wenn verworfen und F ≥ 10, sonst Backdoor“.
- **Reue:** asymptotischer RMSE minus der der besten gültigen Strategie je gewähltem Szenario; gewählt wird die kleinste größte Reue (Minimax-Reue).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. Experimente 1, 2 und 5 sind exakt (Rundung); die Läufe der Experimente 3 und 4 sind fest, ihre Zahl steht dabei, Raten sind auf etwa ±3 bis 6 Punkte genau.

| Frage | Befund | Test |
|---|---|---|
| Handwerte | Naiv ohne verborgene Ursache: Bias **0,254** (= 0,6 · 0,8 / 1,89); „alles anpassen“: **−0,600** (der Bestand enthält die ganze Wirkung); Backdoor mit verborgener Ursache: **0,2236** (= 0,36 / 1,61); Frontdoor bei direktem Pfeil: **−0,300**; Kreuzungsgröße **1 047**. | `test_claim_hand_instances` |
| Gültigkeit und Grenzwert | Das Kriterium am Diagramm und der exakte Grenzwert-Bias null stimmen in allen 54 Zellen überein (neun Szenarien, sechs Strategien). Das Kriterium impliziert Bias null; die Rückrichtung ist **kein Gesetz der Welt**: Heben sich der direkte Pfeil e und der Pfeil f von der verborgenen Ursache zum Bestand auf, kann der Bias bei verletztem Frontdoor-Kriterium null sein (etwa a = 0, b = 0,5, c = 1,0, e = 0,2, f = 0,5). Gemessen in den neun Szenarien, kein Befund über die Welt. | `test_claim_the_criterion_at_the_diagram_and_the_exact_bias_agree_in_the_nine_scenarios` |
| Wer ist wo gültig? (neun Szenarien) | **Frontdoor 6, 2SLS 5, Backdoor 1** (nur „Alles beobachtet“), naiv und „alles anpassen“ **0**; in **einem** Szenario („Nichts brauchbar“) ist keine der sechs gültig. Die Zahlen hängen an der Wahl der Szenarien (Frontdoor ist in sechs nach Konstruktion gültig). | `test_claim_who_is_valid_where` |
| Exakte Grenzwerte | Voreinstellung: naiv **+0,373**, Backdoor **+0,224**, alles anpassen **−0,454**, Frontdoor und 2SLS **0**. Anstoß wirkt direkt: 2SLS **+0,600**. Anstoß an der Ursache gekoppelt: 2SLS **+0,248**, hier zufällig gleich dem Backdoor (bei anderen Stärken nicht). Teilweise vermittelt: Frontdoor **−0,300**. Ursache wirkt auf den Mediator: Frontdoor **+0,211**. Ohne Anstoß ist der Grenzwert des 2SLS nicht definiert. | `test_claim_exact_limits_table` |
| Preis der Gültigkeit (Streuung je √n, alles beobachtet) | Backdoor {Z, W} **1,043**, {Z} 1,173, Frontdoor **1,337** (1,3-fach), 2SLS **2,332** (**2,2-fach**). | `test_claim_price_of_validity_is_exact` |
| Das schmalste Intervall | In **keinem** der neun Szenarien gehört das schmalste Intervall (kleinste exakte Streuung) einer gültigen Strategie (in einem davon ist ohnehin keine gültig); das gilt für diese Welt, und der Vorsprung ist teils klein (bei „Alles beobachtet“ naiv 1,018 gegen Backdoor {Z, W} 1,043). In der Simulation in 8 von 8 Szenarien mit gültiger Strategie. | `test_claim_the_narrowest_interval_never_belongs_to_a_valid_strategy`, `test_claim_the_winner_depends_on_the_measure` |
| Kreuzungsgröße (verborgene Ursache a = b = s; Backdoor gegen 2SLS) | **s = 0,1: 69 739; 0,2: 4 697; 0,3: 1 047; 0,4: 389; 0,6: 117.** Eine **Rechnung** aus Bias (0,008 bis 0,224) und Streuung, keine Beobachtung; die Zahlen hängen an den Stärken dieser Welt, die Form (Bias² gegen den Preis der Gültigkeit) nicht. | `test_claim_crossing_sizes_are_a_calculation` |
| Kreuzung in der Simulation (s = 0,3, 300 Läufe) | n = 300: RMSE Backdoor **0,086** gegen 2SLS 0,145 (Backdoor gewinnt, Deckung 84 %); n = 5 000: Backdoor **0,068** gegen 2SLS **0,035**, Deckung des Backdoor 1 %, des 2SLS 96 %. | `test_claim_the_crossing_agrees_with_the_simulation` |
| Unter den gültigen (alle beobachtet, n = 1 000, 300 Läufe) | Alle vier gültigen Strategien: Bias-Betrag unter 0,015 (**zwischen ihnen entscheidet das Rauschen, nicht der Bias**); RMSE **Backdoor {Z, W} 0,034 < Frontdoor 0,041 < 2SLS 0,075**; naiv und „alles anpassen“ Deckung 0 %. | `test_claim_among_the_valid_rmse_follows_the_assumptions_and_bias_is_noise` |
| Ungültig mit engem Intervall | **Jede ungültige Strategie** hat Deckung höchstens 5 % bei Breite unter 0,5; die gültigen mindestens 88 % (meist 94 bis 97 %). Ausnahme: 2SLS **ohne Anstoß** deckt zu 100 % bei Breite **9 bis 11**. | `test_claim_invalid_strategies_miss_with_narrow_intervals_and_the_empty_instrument_covers_with_a_huge_one` |
| Nichts brauchbar | Fünf Strategien verfehlen die Wahrheit mit enger Breite (0,10 bis 0,15): naiv **+0,42**, Backdoor **+0,26**, „alles anpassen“ **−0,29**, Frontdoor **−0,30**; keine Zahl warnt. | `test_claim_nothing_is_valid_in_one_scenario_and_five_strategies_miss_confidently` |
| Deckung ohne Breite | Ohne Anstoß ist das 2SLS „Sieger“ nach Deckung, mit Breite über 60-mal der des Frontdoor. | `test_claim_coverage_without_width_is_worth_nothing` |
| Schwacher Anstoß (π = 0,05, n = 1 000, 300 Läufe) | 2SLS: **RMSE 8,7**, Quartile **−0,06 bis 1,38**, **Median-Fehler 0,73**, Deckung 99 % bei Breite 4,3; mit π = 0,5: RMSE 0,089, Quartile 0,53 bis 0,67, Breite 0,33. **Mittel und RMSE sind hier unbrauchbar** (die Momente fehlen); Median, Quartile und Breite zählen. | `test_claim_weak_instrument_needs_median_and_width` |
| Der Sieger wechselt mit dem Maß | Voreinstellung: **RMSE → Frontdoor, Breite → naiv** (ungültig, 0,121 gegen 0,173), Deckung ≥ 93 %: Frontdoor und 2SLS. Teilweise vermittelt und Ursache wirkt auf den Mediator: **2SLS** nach RMSE; Anstoß wirkt direkt: Frontdoor. | `test_claim_the_winner_depends_on_the_measure` |
| **Vortest** (Backdoor {Z, W} gegen 2SLS, n = 1 000, **400 Läufe**) | Ohne verborgene Ursache verwirft er in **5,2 %** (die Größe); mit verborgener Ursache in **87 %** (Strategie: RMSE 0,107, Deckung 85 % gegen Backdoor 0,228); Ursache wirkt auf den Mediator **97 %** (Deckung 96 %). **Wirkt der Anstoß direkt, verwirft er in 98 %, die Strategie wählt das falsche 2SLS: RMSE 0,605, Deckung 0 %** (Backdoor 0,317). **Hängt der Anstoß an der Ursache, verwirft er nur in 5,0 %:** beide gleich verzerrt (RMSE 0,251). Schwacher Anstoß (F im Mittel 8,4): 2,2 %. | `test_claim_vortest_shows_disagreement_not_validity` |
| Vortest bei n = 200 | Macht bei verborgener Ursache nur **21 %**; ohne verborgene Ursache ist die Vortest-Strategie schlechter als Backdoor allein (RMSE 0,118 gegen 0,076). | `test_claim_vortest_small_sample` |
| Reue über die plausiblen Szenarien (n = 1 000, ohne das Szenario ohne gültige Strategie) | Größte Reue: naiv **0,398**, Backdoor 0,275, „alles anpassen“ **0,568**, **Frontdoor 0,218** (mittlere 0,044; unter 0,05 in 6 von 8), 2SLS **unendlich** (ohne Anstoß). Wählt man nur die Voreinstellung und „Teilweise vermittelt“, gewinnt **2SLS**: Die Wahl hängt an den Szenarien, die man für möglich hält. | `test_claim_regret_is_a_tool_over_the_chosen_scenarios` |

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Linear und Gauß'sch, ein Faktor, ein Ergebnis** | Alle Zahlen gelten für diese Welt; Schätzvarianten (Gewichten, Matching, doppelt robust) und nichtlineare Wirkungen sind nicht gebaut (Stück 6 und 7). | nicht gebaut |
| **Das Diagramm der Welt ist bekannt** | Die Gültigkeit wird am wahren Diagramm gerechnet; ein falsch gezeichnetes Diagramm ändert, welche Strategie man für gültig hält (Stück 16). | nicht gebaut |
| **Die Wahrheit ist bekannt** | Nur im Orakel. In der Praxis lässt sich nicht prüfen, ob eine einzelne Strategie gilt; der Vortest zeigt Uneinigkeit, nicht den Fehler (Experiment 4). | nicht gebaut |
| **Die Szenarien sind meine Wahl** | Reue und „am häufigsten gültig“ hängen an den neun Szenarien (Frontdoor ist in sechs nach Konstruktion gültig); sie sind ein Werkzeug für die Welten, die Sie für möglich halten, kein Befund über Verfahren. | nicht gebaut |
| **Ein Anstoß, ein Mediator, eine verborgene Ursache** | Mehrere Anstöße (Überidentifikation, J-Test), mehrere Mediatoren und mehrere verborgene Ursachen sind nicht gebaut; gelernte Diagramme (Stück 13 bis 15) und Übertragbarkeit (12) sind keine Strategien dieses Vergleichs. | nicht gebaut |
| **Wenn nichts gilt** | Das Szenario „Nichts brauchbar“ zeigt, dass keine der sechs hilft und kein Intervall warnt; was dann bleibt, ist die Sensitivität (Stück 11). | nicht gebaut |

Was **Identität, Konstruktion oder Rechnung** ist, steht in `test_claims.py` als solches: Kriterium ⇒ Bias null (in den neun Szenarien auch umgekehrt), die Kreuzungsgröße n\* (Rechnung aus Bias und Streuung), die Reue und die Zählung „wer ist wo gültig“ (hängen an den neun Szenarien). Die Kreuzungszahlen gelten für die Stärken dieser Welt (Anstoß π = 0,5, Z und W wie in der Linie); die Form nicht.
Das Backdoor-Kriterium, das Frontdoor-Kriterium, 2SLS, der Hausman-Test und die Minimax-Reue sind aus dem Gedächtnis geschrieben und nicht gegen die Quellen geprüft; sie sind belegt durch die Orakel in `tests/test_oracle_strategies.py` (große Simulation, Kriterium gegen exakten Bias in 64 Welten, Streuung gegen 300 Läufe, Größe des Vortests).

## Bezug zu den Nachbarn im Portfolio

- [backdoor-demo](https://github.com/sebastian-hanisch/backdoor-demo) (3), [schlechte-kontrollen-demo](https://github.com/sebastian-hanisch/schlechte-kontrollen-demo) (4), [adjustierungsmenge-demo](https://github.com/sebastian-hanisch/adjustierungsmenge-demo) (5), [frontdoor-demo](https://github.com/sebastian-hanisch/frontdoor-demo) (8), [instrumente-demo](https://github.com/sebastian-hanisch/instrumente-demo) (9): je eine Strategie in eigener Welt; hier alle sechs in **einer** (die Kriterien `bdr_backdoor.py`, `frd_criterion.py`, `ins_criterion.py` und `dsep_graph.py` wortgleich).
- [schaetzen-demo](https://github.com/sebastian-hanisch/schaetzen-demo) (6) und [doppelt-robust-demo](https://github.com/sebastian-hanisch/doppelt-robust-demo) (7): Schätzvarianten bei erfüllter Backdoor-Annahme; hier nicht neu gebaut.
- [identifizierbarkeit-demo](https://github.com/sebastian-hanisch/identifizierbarkeit-demo) (10): wann **keine** Strategie gilt; hier das Szenario „Nichts brauchbar“. [sensitivitaet-demo](https://github.com/sebastian-hanisch/sensitivitaet-demo) (11) und [diagrammfehler-demo](https://github.com/sebastian-hanisch/diagrammfehler-demo) (16): wie stark die Annahme verletzt sein darf; hier **welche** Strategie man wählt und was die Verletzung kostet (Kreuzungsgröße).
- [mcda-vergleich-demo](https://github.com/sebastian-hanisch/mcda-vergleich-demo) und [rating-vergleich-demo](https://github.com/sebastian-hanisch/rating-vergleich-demo): gleicher Titel, gleiche Rolle (Zusammenfluss der Linie), anderes Thema.

## Tests

Hand-Instanzen (0,254; −0,600; 0,2236; −0,300; Kreuzungsgröße 1 047; 2SLS gegen die zweistufige Regression, Frontdoor gegen zwei Regressionen, Regressionen gegen `lstsq`), der exakte Grenzwert jeder Strategie gegen eine Simulation mit 400 000 Zeilen, **Kriterium am Diagramm ⇔ Grenzwert-Bias null über 64 Welten**, exakte Streuung gegen 300 Läufe, Größe des Hausman-Tests, „U wird nie gelesen“ (NaN in Zeile und Spalte von U ändert nichts), Auswertung, Verdikte, jede README-Zahl in `test_claims.py`, AppTest (Presets, Regler, Checkliste, Experimente, Permalink, PDF), Footer und Laufzeit-Importe.

```bash
python -m pytest tests -q
```

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `kausal_generator.py` | gemeinsamer Generator der Linie (SplitMix64, Normalverteilung) |
| `dsep_graph.py`, `bdr_backdoor.py`, `frd_criterion.py`, `ins_criterion.py` | d-Separation und die drei Kriterien (wortgleich aus Stück 2, 3, 8, 9) |
| `kvl_world.py` | Koeffizienten, Wahrheit, Diagramm der Welt, Bevölkerungskovarianz, Daten |
| `kvl_strategies.py` | die sechs Strategien, Grenzwerte, Streuung je √n, Kreuzungsgröße, Hausman, Monte-Carlo |
| `kvl_validity.py` | Gültigkeit der Strategien am Diagramm, Gründe |
| `kvl_evaluation.py` | Einstellungen, Analyse, Checkliste, fünf Experimente, Verdikte |
| `kvl_visualization.py`, `kvl_pdf_export.py` | Figuren, Tabellen, PDF |
| `kvl_constants.py`, `kvl_presets.py` | Regler, Szenarien, Experimente, Presets, Permalink |
| `tests/` | Hand-Instanzen, Orakel, Claims, App |

## Bewusst nicht umgesetzt

- **Schätzvarianten** (Gewichten, Matching, doppelt robust, Lernverfahren) und **nichtlineare Wirkungen:** Stück 6 und 7.
- **Ein Vergleich von Entscheidungsregeln über zufällige Welten:** Seine Rangfolge ist die Gewichtung der Welten (in der Messreihe 47 % ohne gültige Strategie), nicht ein Befund; die Reue über selbst gewählte Szenarien zeigt dieselbe Lehre als Werkzeug.
- **Mehrere Anstöße (J-Test), mehrere Mediatoren, gelernte Diagramme als Strategie:** nur im Grenzen-Absatz genannt.
- **Praxisdaten:** nicht gebaut; alle Zahlen sind erzeugt.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python 3.12.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Kausale Diagramme: von Konfundierung bis Wahl](https://sebastianhanisch.net/konzepte-kausale-diagramme.html).
