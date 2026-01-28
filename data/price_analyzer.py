"""
Stock Price Analyzer - Refactored from the original Streamlit app.

Analyzes historical stock price data to identify key price events.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import logging

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PriceEvent:
    """Represents a significant price event."""
    date: datetime
    event_type: str
    price: float
    prev_price: Optional[float] = None
    pct_change: Optional[float] = None
    volume: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date.isoformat() if isinstance(self.date, datetime) else str(self.date),
            "event_type": self.event_type,
            "price": self.price,
            "prev_price": self.prev_price,
            "pct_change": self.pct_change,
            "volume": self.volume,
            "metadata": self.metadata
        }


@dataclass
class PriceStatistics:
    """Summary statistics for price data."""
    mean_daily_return: float
    std_daily_return: float
    max_daily_gain: float
    max_daily_loss: float
    positive_days: int
    negative_days: int
    total_days: int
    start_price: float
    end_price: float
    total_return: float
    annualized_return: Optional[float] = None
    annualized_volatility: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_daily_return": self.mean_daily_return,
            "std_daily_return": self.std_daily_return,
            "max_daily_gain": self.max_daily_gain,
            "max_daily_loss": self.max_daily_loss,
            "positive_days": self.positive_days,
            "negative_days": self.negative_days,
            "total_days": self.total_days,
            "start_price": self.start_price,
            "end_price": self.end_price,
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "annualized_volatility": self.annualized_volatility
        }


@dataclass
class AnalysisResults:
    """Complete analysis results."""
    ticker: str
    company_name: str
    data_range: Dict[str, Any]
    all_time_high: PriceEvent
    all_time_low: PriceEvent
    top_rallies: List[PriceEvent]
    top_drawdowns: List[PriceEvent]
    anomalies: List[Dict[str, Any]]
    statistics: PriceStatistics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "data_range": self.data_range,
            "all_time_high": self.all_time_high.to_dict(),
            "all_time_low": self.all_time_low.to_dict(),
            "top_rallies": [r.to_dict() for r in self.top_rallies],
            "top_drawdowns": [d.to_dict() for d in self.top_drawdowns],
            "anomalies": self.anomalies,
            "statistics": self.statistics.to_dict()
        }


class StockPriceAnalyzer:
    """Analyzes historical stock price data to identify key price events."""

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        ticker: str = "STOCK",
        company_name: Optional[str] = None
    ):
        self.ticker = ticker.upper()
        self.company_name = company_name or ticker
        self.df = df.copy() if df is not None else pd.DataFrame()
        self.analysis_results: Optional[AnalysisResults] = None

        if not self.df.empty:
            self._prepare_data()

    @classmethod
    def from_yfinance(
        cls,
        ticker: str,
        period: str = "5y",
        interval: str = "1d"
    ) -> "StockPriceAnalyzer":
        """
        Create analyzer from Yahoo Finance data.

        Args:
            ticker: Stock ticker symbol
            period: Data period (e.g., "1y", "5y", "max")
            interval: Data interval (e.g., "1d", "1wk")

        Returns:
            StockPriceAnalyzer instance
        """
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance not installed. Install with: pip install yfinance")

        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        if df.empty:
            raise ValueError(f"No data found for ticker: {ticker}")

        # Reset index to make Date a column
        df = df.reset_index()
        df = df.rename(columns={'index': 'Date'})

        # Get company name
        try:
            info = stock.info
            company_name = info.get('longName') or info.get('shortName') or ticker
        except Exception:
            company_name = ticker

        return cls(df=df, ticker=ticker, company_name=company_name)

    @classmethod
    def from_file(
        cls,
        file_path: str,
        ticker: str = "STOCK",
        company_name: Optional[str] = None
    ) -> "StockPriceAnalyzer":
        """
        Create analyzer from a CSV or Excel file.

        Args:
            file_path: Path to the file
            ticker: Stock ticker symbol
            company_name: Optional company name

        Returns:
            StockPriceAnalyzer instance
        """
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        return cls(df=df, ticker=ticker, company_name=company_name)

    def _prepare_data(self) -> None:
        """Prepare and clean the data."""
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
        elif self.df.index.name == 'Date' or isinstance(self.df.index, pd.DatetimeIndex):
            self.df = self.df.reset_index()
            self.df = self.df.rename(columns={'index': 'Date'})

        self.df = self.df.sort_values('Date').reset_index(drop=True)
        self.df['Daily_Pct_Change'] = self.df['Close'].pct_change() * 100

    def find_all_time_high(self) -> PriceEvent:
        """Find the all-time high price."""
        max_idx = self.df['Close'].idxmax()
        row = self.df.loc[max_idx]

        return PriceEvent(
            date=row['Date'],
            event_type='All-Time High',
            price=row['Close'],
            volume=row.get('Volume'),
            metadata={'index': int(max_idx)}
        )

    def find_all_time_low(self) -> PriceEvent:
        """Find the all-time low price."""
        min_idx = self.df['Close'].idxmin()
        row = self.df.loc[min_idx]

        return PriceEvent(
            date=row['Date'],
            event_type='All-Time Low',
            price=row['Close'],
            volume=row.get('Volume'),
            metadata={'index': int(min_idx)}
        )

    def find_top_rallies(self, n: int = 5) -> List[PriceEvent]:
        """Find the top N single-day rallies."""
        top = self.df.nlargest(n, 'Daily_Pct_Change')
        rallies = []

        for idx, row in top.iterrows():
            prev_close = self.df.loc[idx - 1, 'Close'] if idx > 0 else None
            rallies.append(PriceEvent(
                date=row['Date'],
                event_type='Top Rally',
                price=row['Close'],
                prev_price=prev_close,
                pct_change=row['Daily_Pct_Change'],
                volume=row.get('Volume'),
                metadata={'rank': len(rallies) + 1}
            ))

        return rallies

    def find_top_drawdowns(self, n: int = 5) -> List[PriceEvent]:
        """Find the top N single-day drawdowns."""
        top = self.df.nsmallest(n, 'Daily_Pct_Change')
        drawdowns = []

        for idx, row in top.iterrows():
            prev_close = self.df.loc[idx - 1, 'Close'] if idx > 0 else None
            drawdowns.append(PriceEvent(
                date=row['Date'],
                event_type='Top Drawdown',
                price=row['Close'],
                prev_price=prev_close,
                pct_change=row['Daily_Pct_Change'],
                volume=row.get('Volume'),
                metadata={'rank': len(drawdowns) + 1}
            ))

        return drawdowns

    def detect_data_anomalies(self) -> List[Dict[str, Any]]:
        """Detect potential data anomalies (e.g., stock splits not adjusted)."""
        anomalies = []

        for i in range(1, len(self.df) - 1):
            curr = self.df.loc[i, 'Daily_Pct_Change']
            next_val = self.df.loc[i + 1, 'Daily_Pct_Change']

            if pd.notna(curr) and pd.notna(next_val):
                # Detect sudden drop followed by sudden recovery
                if curr < -20 and next_val > 30:
                    anomalies.append({
                        'date': self.df.loc[i, 'Date'],
                        'drop_pct': curr,
                        'recovery_pct': next_val,
                        'likely_cause': 'Data artifact (possible stock split adjustment error)'
                    })

        return anomalies

    def get_statistics(self) -> PriceStatistics:
        """Calculate summary statistics."""
        daily_returns = self.df['Daily_Pct_Change'].dropna()
        trading_days = len(self.df)

        start_price = self.df['Close'].iloc[0]
        end_price = self.df['Close'].iloc[-1]
        total_return = ((end_price / start_price) - 1) * 100

        # Calculate annualized metrics
        years = trading_days / 252
        annualized_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else None
        annualized_volatility = daily_returns.std() * np.sqrt(252)

        return PriceStatistics(
            mean_daily_return=daily_returns.mean(),
            std_daily_return=daily_returns.std(),
            max_daily_gain=daily_returns.max(),
            max_daily_loss=daily_returns.min(),
            positive_days=int((daily_returns > 0).sum()),
            negative_days=int((daily_returns < 0).sum()),
            total_days=trading_days,
            start_price=start_price,
            end_price=end_price,
            total_return=total_return,
            annualized_return=annualized_return,
            annualized_volatility=annualized_volatility
        )

    def run_full_analysis(self, num_events: int = 5) -> AnalysisResults:
        """Run complete price analysis."""
        if self.df.empty:
            raise ValueError("No data loaded for analysis")

        self.analysis_results = AnalysisResults(
            ticker=self.ticker,
            company_name=self.company_name,
            data_range={
                'start_date': self.df['Date'].min().isoformat(),
                'end_date': self.df['Date'].max().isoformat(),
                'trading_days': len(self.df)
            },
            all_time_high=self.find_all_time_high(),
            all_time_low=self.find_all_time_low(),
            top_rallies=self.find_top_rallies(num_events),
            top_drawdowns=self.find_top_drawdowns(num_events),
            anomalies=self.detect_data_anomalies(),
            statistics=self.get_statistics()
        )

        return self.analysis_results

    def get_key_events(self) -> List[PriceEvent]:
        """Get all key events for news search, excluding anomalies."""
        if not self.analysis_results:
            self.run_full_analysis()

        events = []
        anomaly_dates = [a['date'] for a in self.analysis_results.anomalies]

        # Add ATH and ATL
        events.append(self.analysis_results.all_time_high)
        events.append(self.analysis_results.all_time_low)

        # Add rallies (excluding anomalies)
        for rally in self.analysis_results.top_rallies:
            if rally.date not in anomaly_dates:
                events.append(rally)

        # Add drawdowns (excluding anomalies)
        for dd in self.analysis_results.top_drawdowns:
            if dd.date not in anomaly_dates:
                events.append(dd)

        return events

    def get_price_around_date(
        self,
        target_date: Union[str, datetime],
        days_before: int = 5,
        days_after: int = 5
    ) -> Optional[pd.DataFrame]:
        """
        Get price data around a specific date.

        Args:
            target_date: The target date
            days_before: Number of trading days before
            days_after: Number of trading days after

        Returns:
            DataFrame with price data or None
        """
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)

        # Normalize both to timezone-naive for comparison
        # Handle target_date
        if hasattr(target_date, 'tzinfo') and target_date.tzinfo is not None:
            target_date = target_date.replace(tzinfo=None)

        # Handle df dates - convert to naive if tz-aware
        df_dates = self.df['Date'].copy()
        try:
            if df_dates.dt.tz is not None:
                df_dates = df_dates.dt.tz_convert(None)
        except (AttributeError, TypeError):
            pass  # Already timezone-naive

        # Find the closest date in our data
        date_diffs = abs(df_dates - target_date)
        closest_idx = date_diffs.idxmin()

        start_idx = max(0, closest_idx - days_before)
        end_idx = min(len(self.df) - 1, closest_idx + days_after)

        return self.df.iloc[start_idx:end_idx + 1].copy()

    def calculate_drawdown_from_peak(self) -> pd.DataFrame:
        """Calculate drawdown from rolling peak."""
        df = self.df.copy()
        df['Peak'] = df['Close'].cummax()
        df['Drawdown'] = (df['Close'] - df['Peak']) / df['Peak'] * 100
        return df[['Date', 'Close', 'Peak', 'Drawdown']]

    def calculate_rolling_metrics(
        self,
        window: int = 20
    ) -> pd.DataFrame:
        """Calculate rolling statistics."""
        df = self.df.copy()
        df['Rolling_Mean'] = df['Close'].rolling(window=window).mean()
        df['Rolling_Std'] = df['Daily_Pct_Change'].rolling(window=window).std()
        df['Rolling_Volatility'] = df['Rolling_Std'] * np.sqrt(252)
        df['Relative_Strength'] = (df['Close'] / df['Rolling_Mean'] - 1) * 100
        return df


def fetch_stock_data(
    ticker: str,
    period: str = "5y",
    interval: str = "1d"
) -> Dict[str, Any]:
    """
    Convenience function to fetch and analyze stock data.

    Args:
        ticker: Stock ticker symbol
        period: Data period
        interval: Data interval

    Returns:
        Dictionary with analysis results
    """
    try:
        analyzer = StockPriceAnalyzer.from_yfinance(ticker, period, interval)
        results = analyzer.run_full_analysis()
        return {
            "success": True,
            "results": results.to_dict(),
            "dataframe": analyzer.df
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
