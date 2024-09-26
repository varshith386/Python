 #Final Code

import requests
import pandas as pd
import datetime

def scrape_crypto_data():
    url = 'https://api.coingecko.com/api/v3/coins/markets'
    params = {
        'vs_currency': 'usd',  # Specify the currency (e.g., usd, eur, etc.)
        'order': 'market_cap_desc',  # Sort by market cap descending
        'per_page': 100,  # Number of results per page
        'page': 1,  # Page number
        'sparkline': False,  # Exclude sparkline data
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print("Failed to fetch data from the CoinGecko API.")
        return None

def process_crypto_data(data):
    if data:
        crypto_data = []
        for item in data:
            name = item['name']
            symbol = item['symbol'].upper()
            price = item['current_price']
            market_cap = item['market_cap']
            timestamp = item['last_updated']
            timestamp = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()
            high_24h = item['high_24h'] 
            low_24h = item['low_24h']     

            crypto_data.append([name, symbol, price, market_cap, timestamp, high_24h, low_24h])
        return crypto_data
    else:
        return None



# Scrape cryptocurrency data from CoinGecko API
crypto_data = scrape_crypto_data()

# Process the data
processed_data = process_crypto_data(crypto_data)

# Convert the processed data into a DataFrame
if processed_data:
    columns = ['Name', 'Symbol', 'Price', 'Market Cap', 'Timestamp','high_24h', 'low_24h']
    crypto_df = pd.DataFrame(processed_data, columns=columns)

    # Save the DataFrame to a CSV file
    crypto_df.to_csv('crypto_data_from_api.csv', index=False)

    print("Dataset saved to crypto_data_from_api.csv")
else:
    print("No data to process.")
