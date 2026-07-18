"""
Shared request budget so multiple collectors can split one API's daily
quota within a single script run instead of each independently blowing
past the limit (see fundamentals.py + news.py, both of which draw on
Alpha Vantage's 25 requests/day free tier).
"""


class RequestBudget:
    def __init__(self, limit: int):
        self.remaining = limit

    def has(self, n: int = 1) -> bool:
        return self.remaining >= n

    def spend(self, n: int = 1):
        self.remaining -= n
