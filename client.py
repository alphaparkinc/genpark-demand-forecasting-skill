"""
demand-forecasting-skill: Client SDK
Forecast product demand using moving averages, trend detection, and seasonality decomposition.
"""
from __future__ import annotations
import math
from typing import Optional


class DemandForecastingClient:
    """
    SDK for product demand forecasting using:
    - Simple Exponential Smoothing (SES) for baseline
    - Linear trend detection via OLS regression
    - Multiplicative seasonality decomposition
    - Safety stock and reorder point calculation
    """

    def forecast(
        self,
        sales_history: list[dict],
        forecast_periods: int = 12,
        lead_time_periods: int = 2,
        safety_stock_factor: float = 1.5,
        alpha: float = 0.3,
    ) -> dict:
        """
        Forecast future demand.

        Args:
            sales_history:       List of {period (str), units_sold (int/float)}.
            forecast_periods:    Periods to forecast ahead.
            lead_time_periods:   Supplier lead time in periods.
            safety_stock_factor: Safety stock multiplier (default 1.5x avg demand).
            alpha:               Exponential smoothing factor (0-1).

        Returns:
            dict with forecast, trend, seasonality_index, reorder_recommendation, summary
        """
        if len(sales_history) < 3:
            raise ValueError("Need at least 3 historical periods.")

        units = [float(d.get("units_sold", 0)) for d in sales_history]
        periods = [str(d.get("period", f"P{i+1}")) for i, d in enumerate(sales_history)]
        n = len(units)

        # Trend detection via linear regression
        slope, intercept = self._linear_regression(units)
        trend_direction = "growing" if slope > 0.02 * (sum(units)/n) else "declining" if slope < -0.02 * (sum(units)/n) else "stable"

        # Seasonality decomposition (multiplicative, cycle = min(12, n//2))
        cycle = min(12, max(2, n // 2))
        seasonality = self._seasonality_index(units, cycle)

        # Exponential smoothing on deseasonalized data
        deseason = [units[i] / max(seasonality[i % cycle], 0.01) for i in range(n)]
        smoothed = self._exp_smooth(deseason, alpha)

        # Generate forecast
        last_smooth = smoothed[-1]
        forecast_out = []
        for i in range(forecast_periods):
            base = last_smooth + slope * (i + 1)
            seasonal_mult = seasonality[(n + i) % cycle]
            predicted = max(0, base * seasonal_mult)
            forecast_out.append({
                "period": f"F+{i+1}",
                "predicted_units": round(predicted, 1),
                "lower_bound": round(max(0, predicted * 0.80), 1),
                "upper_bound": round(predicted * 1.20, 1),
            })

        # Reorder recommendation
        avg_demand = sum(units) / n
        std_demand = math.sqrt(sum((u - avg_demand)**2 for u in units) / n)
        safety_stock = round(safety_stock_factor * std_demand * math.sqrt(lead_time_periods), 1)
        reorder_point = round(avg_demand * lead_time_periods + safety_stock, 1)
        reorder_qty = round(avg_demand * max(forecast_periods // 2, 4), 1)

        summary = (
            f"Trend: {trend_direction} (slope: {slope:+.2f}/period) | "
            f"Avg demand: {avg_demand:.1f}/period | "
            f"Forecast horizon: {forecast_periods} periods | "
            f"Reorder point: {reorder_point} units"
        )

        return {
            "historical_periods": n,
            "avg_historical_demand": round(avg_demand, 1),
            "trend": trend_direction,
            "trend_slope": round(slope, 3),
            "seasonality_index": [round(s, 3) for s in seasonality],
            "forecast": forecast_out,
            "reorder_recommendation": {
                "reorder_point_units": reorder_point,
                "safety_stock_units": safety_stock,
                "suggested_order_qty": reorder_qty,
                "lead_time_periods": lead_time_periods,
                "reorder_when_stock_falls_below": reorder_point,
            },
            "summary": summary,
        }

    @staticmethod
    def _linear_regression(values: list[float]) -> tuple[float, float]:
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        num = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / den if den != 0 else 0
        intercept = y_mean - slope * x_mean
        return slope, intercept

    @staticmethod
    def _seasonality_index(values: list[float], cycle: int) -> list[float]:
        grand_avg = sum(values) / len(values) or 1.0
        indices = []
        for c in range(cycle):
            cycle_vals = [values[i] for i in range(c, len(values), cycle)]
            cycle_avg = sum(cycle_vals) / len(cycle_vals) if cycle_vals else 1.0
            indices.append(cycle_avg / grand_avg)
        # Normalize so average = 1.0
        avg_idx = sum(indices) / len(indices)
        return [idx / avg_idx for idx in indices]

    @staticmethod
    def _exp_smooth(values: list[float], alpha: float) -> list[float]:
        smoothed = [values[0]]
        for v in values[1:]:
            smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
        return smoothed
