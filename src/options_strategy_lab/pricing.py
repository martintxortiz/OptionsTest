from __future__ import annotations

import math

from scipy.stats import norm


def black_scholes_price(
    spot: float,
    strike: float,
    years_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str,
) -> float:
    if years_to_expiry <= 0:
        if option_type == "call":
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)

    sigma = max(volatility, 1e-4)
    root_t = math.sqrt(years_to_expiry)
    d1 = (
        math.log(spot / strike)
        + (risk_free_rate + 0.5 * sigma * sigma) * years_to_expiry
    ) / (sigma * root_t)
    d2 = d1 - sigma * root_t

    if option_type == "call":
        return spot * norm.cdf(d1) - strike * math.exp(-risk_free_rate * years_to_expiry) * norm.cdf(d2)
    return strike * math.exp(-risk_free_rate * years_to_expiry) * norm.cdf(-d2) - spot * norm.cdf(-d1)


def bull_put_credit_spread_value(
    spot: float,
    short_strike: float,
    long_strike: float,
    days_to_expiry: int,
    implied_volatility: float,
    risk_free_rate: float = 0.04,
) -> float:
    years_to_expiry = max(days_to_expiry, 0) / 252.0
    short_put = black_scholes_price(
        spot=spot,
        strike=short_strike,
        years_to_expiry=years_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=implied_volatility + 0.02,
        option_type="put",
    )
    long_put = black_scholes_price(
        spot=spot,
        strike=long_strike,
        years_to_expiry=years_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=implied_volatility + 0.01,
        option_type="put",
    )
    return max(short_put - long_put, 0.0)


def bear_call_credit_spread_value(
    spot: float,
    short_strike: float,
    long_strike: float,
    days_to_expiry: int,
    implied_volatility: float,
    risk_free_rate: float = 0.04,
) -> float:
    years_to_expiry = max(days_to_expiry, 0) / 252.0
    short_call = black_scholes_price(
        spot=spot,
        strike=short_strike,
        years_to_expiry=years_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=implied_volatility + 0.01,
        option_type="call",
    )
    long_call = black_scholes_price(
        spot=spot,
        strike=long_strike,
        years_to_expiry=years_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=implied_volatility,
        option_type="call",
    )
    return max(short_call - long_call, 0.0)
