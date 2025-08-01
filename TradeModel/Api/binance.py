import requests

def get_latest_kline(symbol, interval):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=1"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()[0]
    return None