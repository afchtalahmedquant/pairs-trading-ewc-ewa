import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tools.tools import add_constant
from pykalman import KalmanFilter
import yfinance as yf



class PairstradingStrategy: 
    """
    Stratégie de Pair Trading 
    Développée par Afachtal Ahmed - 2025
    """
    
    def __init__(self, ticker1: str, ticker2: str, start_date: str, end_date: str):
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.returns = None
        self.df_trading = None

    def download_data(self):
        """Télécharge les données depuis Yahoo Finance"""
        tickers = [self.ticker1, self.ticker2]
        self.data = yf.download(tickers, self.start_date, self.end_date)
        self.data_close = self.data["Close"].dropna()
        self.returns = np.log(self.data_close / self.data_close.shift(1)).dropna()
        print(f"✓ Données téléchargées : {len(self.data_close)} observations")

    def calculate_half_life(self, series: pd.Series) -> float:
        """
        Calcule le half-life pour déterminer la fenêtre optimale
        """
        
        series_clean = series.dropna()
        series_lag = series_clean.shift(1).dropna()
        series_clean = series_clean[1:]

        model = sm.OLS(series_clean, sm.add_constant(series_lag)).fit()
        
        
        lambda_coef = model.params.iloc[1]  

        if lambda_coef < 0 or lambda_coef >= 1:
            print(" Série non mean-reverting")
            return np.nan
        
        half_life = -np.log(2) / np.log(lambda_coef)
        print(f"✓ Half-life calculé : {half_life:.2f} jours")
        print(f"  λ (AR coefficient) : {lambda_coef:.4f}")
        return half_life
    
    def cointegration_test(self):
        """Test de cointégration Engle-Granger"""
        score, pvalue, _ = coint(
            np.log(self.data_close[self.ticker1]),
            np.log(self.data_close[self.ticker2])
        )

        is_cointegrated = pvalue < 0.05
        
        print(f"\n{'='*60}")
        print("TEST DE COINTÉGRATION ENGLE-GRANGER")
        print(f"{'='*60}")
        print(f"P-value : {pvalue:.6f}")
        
        if pvalue < 0.01:
            print(f" Cointégration FORTE (99% confiance)")
        elif pvalue < 0.05:
            print(f" Cointégration modérée (95% confiance)")
        else:
            print(f" Pas de cointégration détectée")
        print(f"{'='*60}\n")
        
        return pvalue, is_cointegrated
    
    def apply_kalman_filter(self):
        """
        Filtre de Kalman pour hedge ratio dynamique
        """
        x = self.data_close[self.ticker2].values  # Variable indépendante
        y = self.data_close[self.ticker1].values  # Variable dépendante
        
        kf = KalmanFilter(
            transition_matrices=np.array([[1]]),
            observation_matrices=x.reshape(-1, 1, 1),
            transition_covariance=np.array([[1e-4]]),
            observation_covariance=np.array([[1]]),
            initial_state_mean=np.array([0]),  
            initial_state_covariance=np.array([[1]])
        )

        state_means, state_covariances = kf.filter(y)
        beta_kalman = state_means.flatten()
        beta_std = np.sqrt(state_covariances.flatten())

        self.df_trading = pd.DataFrame({
            self.ticker1: y,
            self.ticker2: x,
            "beta_kalman": beta_kalman,
            "beta_std": beta_std
        }, index=self.data_close.index)
        
        print(f" Filtre de Kalman appliqué")
        print(f"  Beta moyen : {beta_kalman.mean():.4f}")
        print(f"  Beta std : {beta_std.mean():.4f}")

    def calculate_spread(self): 
        """
        Calcule le spread normalisé
        
        Normalisation par prix moyen 
        """
        # Prix moyen pour normalisation
        avg_price = (
            self.df_trading[self.ticker1] + 
            self.df_trading["beta_kalman"] * self.df_trading[self.ticker2]
        ) / 2
        
        # Spread brut
        self.df_trading["spread_raw"] = (
            self.df_trading[self.ticker1] - 
            self.df_trading["beta_kalman"] * self.df_trading[self.ticker2]
        )
        
        # Spread normalisé (en %)
        self.df_trading["spread"] = self.df_trading["spread_raw"] / avg_price
        
        # Test de stationnarité
        adf, pval, _, _, _, _ = adfuller(self.df_trading["spread"].dropna())
        
        print(f"\n{'='*60}")
        print(f"TEST DE STATIONNARITÉ (ADF)")
        print(f"{'='*60}")
        print(f"ADF Statistic : {adf:.4f}")
        print(f"P-value : {pval:.6f}")
        
        if pval < 0.01:
            print(f" Spread TRÈS stationnaire (99% confiance)")
        elif pval < 0.05:
            print(f" Spread stationnaire (95% confiance)")
        else:
            print(f" Spread NON stationnaire - ATTENTION!")
        print(f"{'='*60}\n")
        
        # Calcul half-life
        half_life = self.calculate_half_life(self.df_trading["spread"])
        
        if not np.isnan(half_life):
            
            if half_life < 5:
                window = max(20, int(np.ceil(half_life * 5)))  # 5x le half-life
                print(f" Half-life très court ({half_life:.1f}j) (fenêtre augmentée)")
            else:
                window = int(np.ceil(half_life))
            
            
            window = max(10, min(100, window))  # Limiter entre 20 et 100 jours
        else:
            window = 60
            print(f" Utilisation de la fenêtre par défaut : {window}")
        
        print(f" Fenêtre de rolling : {window} jours")
        return window

    def calculate_zscore(self, window: int):
        """Calcule le z-score du spread"""
        self.df_trading["spread_mean"] = self.df_trading["spread"].rolling(window).mean()
        self.df_trading["spread_std"] = self.df_trading["spread"].rolling(window).std()
        self.df_trading["zscore"] = (
            (self.df_trading["spread"] - self.df_trading["spread_mean"]) / 
            self.df_trading["spread_std"]
        )
        
        print(f"\n{'='*60}")
        print("Z-SCORE CALCULÉ")
        print(f"{'='*60}")
        print(f"Z-score max : {self.df_trading['zscore'].max():.4f}")
        print(f"Z-score min : {self.df_trading['zscore'].min():.4f}")
        print(f"Z-score moyen : {self.df_trading['zscore'].mean():.4f}")
        print(f"Z-score écart-type : {self.df_trading['zscore'].std():.4f}")
        print(f"{'='*60}\n")

    def generate_signals(
        self, 
        entry_threshold: float = 2.0, 
        exit_threshold: float = 0.5, 
        stop_loss: float = 3.5
    ):
    
        # Génère les signaux de trading
    
        signals = pd.DataFrame(index=self.df_trading["spread"].index)
        signals["z"] = self.df_trading["zscore"]
        signals["positions"] = 0
        position = 0

        for t in range(len(self.df_trading["zscore"])):
            z = signals["z"].iloc[t]

            if np.isnan(z):
                continue
            
            # Pas de position
            if position == 0:
                if z > entry_threshold:
                    position = -1  # Short spread
                elif z < -entry_threshold:
                    position = 1   # Long spread
            
            # Position ouverte
            elif position != 0:
                if abs(z) < exit_threshold:
                    position = 0
                elif position == 1 and z < -stop_loss:  # Long spread, z diverge vers le bas
                    position = 0
                    print(f"Stop-loss (LONG) : {signals.index[t].date()}, z={z:.2f}")
                
                elif position == -1 and z > stop_loss:  # Short spread, z diverge vers le haut
                    position = 0
                    print(f"Stop-loss (SHORT) : {signals.index[t].date()}, z={z:.2f}")
                
            signals.iloc[t, signals.columns.get_loc("positions")] = position

        self.signals = signals
        
        print(f"\n{'='*60}")
        print("SIGNAUX GÉNÉRÉS")
        print(f"{'='*60}")
        print(f"Nombre de trades : {(self.signals['positions'].diff() != 0).sum()}")
        print(f"Jours en position : {(self.signals['positions'] != 0).sum()}")
        
        long_days = (self.signals['positions'] == 1).sum()
        short_days = (self.signals['positions'] == -1).sum()
        print(f"  * Long spread : {long_days} jours")
        print(f"  * Short spread : {short_days} jours")
        print(f"{'='*60}\n")

    def backtest_strategy(
        self, 
        transaction_cost: float = 0.0001, 
        initial_capital: float = 100000
    ):
       
        # Backtest de la stratégie
        capital_per_leg = initial_capital / 2
        
        positions = self.signals["positions"].shift(1).fillna(0)
    
        pnl_ticker1 = positions * self.returns[self.ticker1] * capital_per_leg
        pnl_ticker2 = -positions * self.df_trading["beta_kalman"] * self.returns[self.ticker2] * capital_per_leg
        
        # Coûts de transaction
        trades = self.signals["positions"].diff().abs()
        cost = trades * transaction_cost * initial_capital
        
        pnl_net = pnl_ticker1 + pnl_ticker2 - cost
        pnl_cumulative = pnl_net.cumsum()
        
        # Returns nets
        capital_series = initial_capital + pnl_cumulative.shift(1).fillna(0)
        returns_net = pnl_net / capital_series.replace(0, np.nan)
        returns_net = returns_net.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Métriques
        metrics = self.calculate_metrics(returns_net, pnl_cumulative, initial_capital)
        
        self.returns_net = returns_net
        self.pnl_cumulative = pnl_cumulative
        self.initial_capital = initial_capital
        
        return metrics

    def calculate_metrics(
        self, 
        returns: pd.Series, 
        pnl_cuml: pd.Series, 
        initial_capital: float = 100000
    ):
        
        # Calcule les métriques de performance
        returns_clean = returns.dropna()
        pnl_clean = pnl_cuml.dropna()

        if len(pnl_clean) == 0:
            print("Pas de données pour calculer les métriques")
            return {}

        total_return = pnl_clean.iloc[-1] / initial_capital
        
        n_trading_days = len(returns_clean)
        n_years = n_trading_days / 252
        
        if n_years > 0:
            annual_return = (1 + total_return) ** (1 / n_years) - 1
        else:
            annual_return = 0

        # Sharpe Ratio
        if returns_clean.std() > 0:
            sharpe = annual_return / (returns_clean.std() * np.sqrt(252))
        else:
            sharpe = 0

        # Sortino Ratio
        downside_returns = returns_clean[returns_clean < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = annual_return / (downside_returns.std() * np.sqrt(252))
        else:
            sortino = 0

        # Max Drawdown
        cum_returns = (1 + returns_clean).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar Ratio
        if max_drawdown != 0:
            calmar = annual_return / abs(max_drawdown)
        else:
            calmar = 0

        # Win Rate
        winning_days = (returns_clean > 0).sum()
        total_days = len(returns_clean[returns_clean != 0])
        win_rate = winning_days / total_days if total_days > 0 else 0

        # Profit Factor
        gross_profit = returns_clean[returns_clean > 0].sum()
        gross_loss = abs(returns_clean[returns_clean < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        metrics = {
            "Annual_Return": annual_return,
            "Sharpe_Ratio": sharpe,
            "Sortino_Ratio": sortino,
            "Max_Drawdown": max_drawdown,
            "Calmar_Ratio": calmar,
            "Win_Rate": win_rate,
            "Profit_Factor": profit_factor,
            "Total_Return": total_return
        }

        print(f"\n{'='*60}")
        print(f"PERFORMANCE METRICS")
        print(f"{'='*60}")

        for key, value in metrics.items():
            if "Rate" in key or "Return" in key or "Drawdown" in key:
                print(f"{key:.<30} {value:.2%}")
            else:
                print(f"{key:.<30} {value:.4f}")
        print(f"{'='*60}\n")

        return metrics
    
    def plot_results(self, entry_threshold: float, exit_threshold: float):
        # Visualisations complètes
        fig = plt.figure(figsize=(16, 12))

        # 1. Prix des actifs
        ax1 = plt.subplot(3, 2, 1)
        self.data_close.plot(ax=ax1)
        ax1.set_title("Prix des actifs", fontsize=12, fontweight="bold")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # 2. Hedge Ratio Dynamique
        ax2 = plt.subplot(3, 2, 2)
        self.df_trading["beta_kalman"].plot(ax=ax2, color="purple")
        ax2.set_title("Hedge Ratio Dynamique (Kalman)", fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)

        # 3. Spread
        ax3 = plt.subplot(3, 2, 3)
        self.df_trading["spread"].plot(ax=ax3, color='orange')
        ax3.axhline(0, color='black', linestyle="--", alpha=0.5)
        ax3.set_title("Spread Normalisé", fontsize=12, fontweight="bold")
        ax3.grid(alpha=0.3)

        # 4. Z-score avec seuils
        ax4 = plt.subplot(3, 2, 4)
        self.df_trading["zscore"].plot(ax=ax4, color="blue", alpha=0.7)
        ax4.axhline(entry_threshold, color="red", linestyle="--", label='Entry')
        ax4.axhline(-entry_threshold, color="red", linestyle="--")
        ax4.axhline(exit_threshold, color="green", linestyle="--", label='Exit')
        ax4.axhline(-exit_threshold, color="green", linestyle="--")
        ax4.axhline(0, color="black", linestyle="-", alpha=0.3)
        ax4.set_title("Z-score du Spread", fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)

        # 5. PnL Cumulatif
        ax5 = plt.subplot(3, 2, 5)
        self.pnl_cumulative.plot(ax=ax5, color='green', linewidth=2)
        ax5.set_title(f"PnL Cumulatif (Capital initial: {self.initial_capital:,.0f}$)", 
                     fontsize=12, fontweight='bold')
        ax5.grid(alpha=0.3)

        # 6. Distribution des returns
        ax6 = plt.subplot(3, 2, 6)
        self.returns_net.hist(bins=50, ax=ax6, color="skyblue", edgecolor="black")
        ax6.axvline(0, color="red", linestyle="--", linewidth=2)
        ax6.set_title("Distribution des Rendements", fontsize=12, fontweight='bold')
        ax6.set_xlabel("Returns")
        ax6.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        plt.show()


    def generate_publication_linkedin(self, entry_threshold, exit_threshold):

        plt.style.use('seaborn-v0_8-white')
        plt.rcParams['font.family']='sans-serif'
        plt.rcParams['font.size']=11
        fig, ax=plt.subplots(figsize=(10,6))

        #PnL cumulative:
        ax.plot(self.pnl_cumulative.index, self.pnl_cumulative, color='#000000', linewidth=2)
        ax.set_title('Cumulative PnL ', weight='bold', pad=20)
        ax.set_ylabel('PnL ($)', fontsize=12)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('linkedin _pnl.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()


        # zscore
        fig, ax=plt.subplots(figsize=(10,6))
        ax.plot(self.df_trading["zscore"].index, self.df_trading["zscore"], color='#4A90E2',linewidth=2, alpha=0.8)
        ax.axhline(entry_threshold, color='red', linestyle='--', label='Entry', alpha=0.7)
        ax.axhline(-entry_threshold, color='red' , linestyle='--', alpha=0.7)
        ax.axhline(exit_threshold, color='green', linestyle='--', label='Exit', alpha=0.7)
        ax.axhline(-exit_threshold, color='green', linestyle='--', label='Exit')
        ax.axhline(0, color='black', linestyle='-', alpha =0.3)
        ax.set_title("Zscore of Spread",fontsize=12, fontweight='bold', pad=20)
        ax.set_ylabel("Zscore", fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right', frameon=False)
        ax.grid(alpha=0.2, axis='y')
        plt.tight_layout()
        plt.savefig('linkedin_zscore.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        # Hedge Ratio (Kalman)

        fig,ax=plt.subplots(figsize=(10,6))
        ax.plot(self.df_trading['beta_kalman'].index, self.df_trading['beta_kalman'], color='#9B59B6', linewidth=2)
        ax.set_title('Hedge Ratio Dynamique ( Beta Kalman)', fontsize=12, fontweight='bold', pad=20)
        ax.set_ylabel('Beta kalman', fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.2, axis='y')
        plt.tight_layout()
        plt.savefig('linkedin_hedgeratio.png',dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        # Distribution des rendements
        fig, ax=plt.subplots(figsize=(10,6))
        ax.hist(self.returns_net, bins=50,color='#34495E' ,edgecolor='black', linewidth=1.5, alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', alpha=0.7)
        ax.set_title( "Distribution of Returns", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Returns', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.2, axis='y')
        plt.tight_layout()
        plt.savefig('linkedin_dist_returns.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f" 4 images généres pour la publication ")




    def run_full_analysis(
        self, 
        entry_threshold: float = 2.0, 
        exit_threshold: float = 0.5,
        stop_loss: float = 3.5,  
        transaction_cost: float = 0.0001,
        initial_capital: float = 100000
    ):
    
    
        print(f"\n{'#'*60}")
        print(f"  PAIR TRADING: {self.ticker1} vs {self.ticker2}")
        print(f"{'#'*60}\n")

        # 1. Téléchargement
        self.download_data()

        # 2. Test de cointégration
        pval, is_cointegrated = self.cointegration_test()
        
        if not is_cointegrated:
            print(" Paire non cointégrée - résultats peu fiables!")

        # 3. Filtre de Kalman
        self.apply_kalman_filter()

        # 4. Spread et half-life
        window = self.calculate_spread()

        # 5. Z-score
        self.calculate_zscore(window)

        # 6. Signaux
        self.generate_signals(entry_threshold, exit_threshold, stop_loss)

        # 7. Backtest
        metrics = self.backtest_strategy(transaction_cost, initial_capital)

        # 8. Visualisations

        self.plot_results(entry_threshold, exit_threshold)
        

        return metrics
    

