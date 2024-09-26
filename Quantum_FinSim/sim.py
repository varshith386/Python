#Final Code





import numpy as np
import pandas as pd
import hashlib
import datetime

class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        sha = hashlib.sha256()
        sha.update(str(self.index).encode('utf-8') +
                   str(self.timestamp).encode('utf-8') +
                   str(self.data).encode('utf-8') +
                   str(self.previous_hash).encode('utf-8'))
        return sha.hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self):
        # Manually create the first block (genesis block)
        return Block(0, datetime.datetime.now(), "Genesis Block", "0")

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, new_block):
        new_block.previous_hash = self.get_latest_block().hash
        new_block.hash = new_block.calculate_hash()
        self.chain.append(new_block)

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            if current_block.hash != current_block.calculate_hash():
                return False

            if current_block.previous_hash != previous_block.hash:
                return False

        return True

class CryptoSimulator:
    def __init__(self, crypto_data):
        self.crypto_data = crypto_data
        self.portfolio = {}
        self.blockchain = Blockchain()  # Initialize blockchain
        self.balance = 0  # Initial balance
        self.transaction_history = []  # Track transactions

    def buy_crypto(self, crypto_name, amount):
        crypto_info = self.crypto_data[self.crypto_data['Name'].str.lower() == crypto_name.lower()]
        if crypto_info.empty:
            print(f"Sorry, {crypto_name} not found in the dataset.")
            return
        
        current_price = crypto_info['Price'].values[0]
        total_cost = current_price * amount
        
        if total_cost > self.balance:
            print("Insufficient balance to buy this amount of cryptocurrency.")
            return
        
        if crypto_name in self.portfolio:
            self.portfolio[crypto_name] += amount
        else:
            self.portfolio[crypto_name] = amount
        
        self.balance -= total_cost
        
        # Add transaction to blockchain
        self.blockchain.add_block(Block(len(self.transaction_history) + 1, datetime.datetime.now(), f"Bought {amount} units of {crypto_name} at ${current_price} each.", self.blockchain.get_latest_block().hash))
        self.transaction_history.append(f"Bought {amount} units of {crypto_name} at ${current_price} each.")

        print(f"Bought {amount} units of {crypto_name} at ${current_price} each.")

    def sell_crypto(self, crypto_name, amount):
        if crypto_name not in self.portfolio or self.portfolio[crypto_name] < amount:
            print("You don't have enough units of this cryptocurrency to sell.")
            return
        
        crypto_info = self.crypto_data[self.crypto_data['Name'].str.lower() == crypto_name.lower()]
        if crypto_info.empty:
            print(f"Sorry, {crypto_name} not found in the dataset.")
            return
        
        current_price = crypto_info['Price'].values[0]
        total_sale = current_price * amount
        
        self.portfolio[crypto_name] -= amount
        self.balance += total_sale
        
        # Add transaction to blockchain
        self.blockchain.add_block(Block(len(self.transaction_history) + 1, datetime.datetime.now(), f"Sold {amount} units of {crypto_name} at ${current_price} each.", self.blockchain.get_latest_block().hash))
        self.transaction_history.append(f"Sold {amount} units of {crypto_name} at ${current_price} each.")

        print(f"Sold {amount} units of {crypto_name} at ${current_price} each.")

    def display_portfolio(self):
        print("Current Portfolio:")
        for crypto, amount in self.portfolio.items():
            print(f"{crypto}: {amount} units")
        print(f"Balance: ${self.balance}")

    def run(self):
        print("Welcome to the Crypto Trading Simulator.")
        while True:
            print("\nOptions:")
            print("1. Buy Cryptocurrency")
            print("2. Sell Cryptocurrency")
            print("3. Display Portfolio")
            print("4. Exit")
            choice = input("Enter your choice (1-4): ")

            if choice == '1':
                crypto_name = input("Enter the name of the cryptocurrency to buy: ")
                amount = float(input("Enter the amount to buy: "))
                self.buy_crypto(crypto_name, amount)
            elif choice == '2':
                crypto_name = input("Enter the name of the cryptocurrency to sell: ")
                amount = float(input("Enter the amount to sell: "))
                self.sell_crypto(crypto_name, amount)
            elif choice == '3':
                self.display_portfolio()
            elif choice == '4':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please choose a valid option.")

# Load cryptocurrency data obtained from web scraping
crypto_data = pd.read_csv('crypto_data_from_api.csv')

# Initialize simulator with initial balance
simulator = CryptoSimulator(crypto_data)
simulator.balance = 1000000
simulator.run()
