# Pairs Trading Strategy: EWC-EWA


## Projet

Stratégie de pairs trading développée dans le cadre de mes études en finance quantitative (4ème année). La stratégie exploite la relation de cointégration entre **EWC** (Canada) et **EWA** (Australie).



## Résultats du Backtest

Période : 1996-2025 (29 ans)  
Capital initial : 100,000$


## PERFORMANCE METRICS

Annual_Return................. 5.42%
Sharpe_Ratio.................. 0.9731
Sortino_Ratio................. 0.9314
Max_Drawdown.................. -9.09%
Calmar_Ratio.................. 0.5960
Win_Rate...................... 44.06%
Profit_Factor................. 1.4021
Total_Return.................. 355.97%








## Installation

```bash
git clone https://github.com/afchtalahmedquant/pairs-trading-ewc-ewa.git
cd pairs-trading-ewc-ewa
pip install -r requirements.txt
```

-

##  Utilisation

```python
from Pair_trading_strategy import PairstradingStrategy

strategy = PairstradingStrategy(
    ticker1="EWC",
    ticker2="EWA",
    start_date="1996-01-01",
    end_date="2025-01-01"
)

strategy.run_full_analysis(
    entry_threshold=1.5,
    exit_threshold=0.5,
    stop_loss=3.0,
    transaction_cost=0.0002
)
```

---

##  Méthodologie

### 1. Test de Cointégration
- **Test Engle-Granger** : p-value = 0.022 
- Les deux ETFs sont cointégrés (95% confiance)

### 2. Hedge Ratio Dynamique
- **Filtre de Kalman** pour un ratio adaptatif
- Beta moyen : 1.72

### 3. Calcul du Spread
- Spread normalisé par prix moyen
- Test ADF : spread stationnaire (99% confiance)
- Half-life : 5.67 jours

### 4. Signaux de Trading

| Condition | Action |
|-----------|--------|
| Z-score < -1.5 | **LONG** spread |
| Z-score > +1.5 | **SHORT** spread |
| Z-score retourne à ±0.5 | **SORTIE** |
| Perte > stop-loss | **STOP** |

---

##  Limites

Ce projet est un backtest simple réalisé dans un cadre académique :

-  Pas de walk-forward analysis
-  Pas de tests out-of-sample
-  Slippage simplifié
-  Pas d'optimisation robuste des paramètres

**Ne pas utiliser en trading réel sans validation approfondie.**

---

##  Prochaines Étapes(Améliorations)

-  Walk-forward analysis
-  Grid search pour optimisation
-  Tests de robustesse (bootstrap)
-  Analyse par régimes de marché

---

## Auteur

**Ahmed Afchtalahmed**  
Étudiant en Finance - 4ème année(ENCGF)

- GitHub: [@afchtalahmedquant](https://github.com/afchtalahmedquant)
- LinkedIn: [Ahmed Afachtal]




