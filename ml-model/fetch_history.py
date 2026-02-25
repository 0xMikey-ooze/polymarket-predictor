import urllib.request, json, time, datetime, csv, os

os.makedirs('data', exist_ok=True)

def fetch_candles(bar, filename):
    url_base = f'https://www.okx.com/api/v5/market/history-candles?instId=ETH-USDT&bar={bar}&limit=300'
    
    # Target: 2 years ago
    two_years_ago = int((datetime.datetime.now() - datetime.timedelta(days=730)).timestamp() * 1000)
    
    all_candles = []
    after_ts = None  # Start from most recent, paginate backwards
    
    while True:
        url = url_base + (f'&after={after_ts}' if after_ts else '')
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            resp = urllib.request.urlopen(req, timeout=15)
            d = json.loads(resp.read())
            candles = d.get('data', [])
        except Exception as e:
            print(f'Error: {e}, retrying...')
            time.sleep(1)
            continue
            
        if not candles:
            break
            
        all_candles.extend(candles)
        oldest_ts = int(candles[-1][0])
        
        if oldest_ts <= two_years_ago:
            break
            
        after_ts = oldest_ts
        
        if len(all_candles) % 10000 == 0:
            print(f'  {bar}: fetched {len(all_candles)} candles, oldest: {datetime.datetime.fromtimestamp(oldest_ts/1000)}')
        
        time.sleep(0.15)  # rate limit safe
    
    # Sort ascending (oldest first)
    all_candles.sort(key=lambda x: int(x[0]))
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        for c in all_candles:
            dt = datetime.datetime.fromtimestamp(int(c[0])/1000)
            writer.writerow([dt.strftime('%Y-%m-%d %H:%M:%S'), c[1], c[2], c[3], c[4], c[5]])
    
    print(f'{bar}: saved {len(all_candles)} candles to {filename}')
    return len(all_candles)

print('Fetching ETH 5m...')
n5 = fetch_candles('5m', 'data/eth_5m.csv')
print('Fetching ETH 15m...')
n15 = fetch_candles('15m', 'data/eth_15m.csv')
print(f'Done. 5m: {n5} candles, 15m: {n15} candles')
