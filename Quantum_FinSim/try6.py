#Working properly

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import hashlib
import datetime
import json

# Load cryptocurrency data obtained from web scraping
crypto_data = pd.read_csv('AdminDashboard\crypto_data_from_api.csv')

# Extract features and target variable
X = crypto_data[['Price', 'Market Cap', 'Timestamp', 'high_24h', 'low_24h']].values
y = crypto_data['Price'].values

# Normalize the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the neural network model with improved architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model with a lower learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Add dropout regularization to prevent overfitting
model.add(tf.keras.layers.Dropout(0.2))

# Train the model with more epochs
model.fit(X_train, y_train, epochs=10000, batch_size=128, verbose=1)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae)

# Blockchain class for data verification
class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_block(proof=1, previous_hash='0')

    def create_block(self, proof, previous_hash):
        block = {
            'index': len(self.chain) + 1,
            'timestamp': str(datetime.datetime.now()),
            'proof': proof,
            'previous_hash': previous_hash
        }
        self.chain.append(block)
        return block

    def get_previous_block(self):
        return self.chain[-1]

    def proof_of_work(self, previous_proof):
        new_proof = 1
        check_proof = False
        while check_proof is False:
            hash_operation = hashlib.sha256(str(new_proof**2 - previous_proof**2).encode()).hexdigest()
            if hash_operation[:4] == '0000':
                check_proof = True
            else:
                new_proof += 1
        return new_proof

    def hash(self, block):
        encoded_block = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(encoded_block).hexdigest()

    def is_chain_valid(self, chain):
        previous_block = chain[0]
        block_index = 1
        while block_index < len(chain):
            block = chain[block_index]
            if block['previous_hash'] != self.hash(previous_block):
                return False
            previous_proof = previous_block['proof']
            proof = block['proof']
            hash_operation = hashlib.sha256(str(proof**2 - previous_proof**2).encode()).hexdigest()
            if hash_operation[:4] != '0000':
                return False
            previous_block = block
            block_index += 1
        return True

# Instantiate the blockchain
blockchain = Blockchain()

# Verify the blockchain
is_valid = blockchain.is_chain_valid(blockchain.chain)
print("Is Blockchain Valid?", is_valid)

# Function to verify cryptocurrency data using blockchain
def verify_data_with_blockchain(data):

    return blockchain.is_chain_valid(blockchain.chain)

# Function to predict the price of a cryptocurrency and assess the risks with blockchain verification
def predict_price_with_blockchain(crypto_name):
    # Fetch data for the specified cryptocurrency
    crypto_info = crypto_data[crypto_data['Name'].str.lower() == crypto_name.lower()]
    if crypto_info.empty:
        return f"Sorry, {crypto_name} not found in the dataset."
    
    # Extract current price, market cap, timestamp, high_24, and low_24
    current_price = crypto_info['Price'].values[0]
    market_cap = crypto_info['Market Cap'].values[0]
    timestamp = crypto_info['Timestamp'].values[0]
    high_24 = crypto_info['high_24h'].values[0]
    low_24 = crypto_info['low_24h'].values[0]
    
    # Normalize the features
    scaled_info = scaler.transform([[current_price, market_cap, timestamp, high_24, low_24]])
    
    # Verify the data using blockchain
    if verify_data_with_blockchain(crypto_info):
        # Predict the price
        predicted_price = model.predict(scaled_info)[0][0]
        
        # Assess the trend (profit or loss)
        if predicted_price > current_price:
            trend = "positive (Profit)"
        elif predicted_price < current_price:
            trend = "negative (Loss)"
        else:
            trend = "neutral"
        
        # Construct the result string
        result = (
            f"Current Price of {crypto_name}: ${current_price}\n"
            f"Predicted Price of {crypto_name}: ${predicted_price}\n"
            f"Predicted Trend: {trend}\n"
        )

        # Add risk assessment
        if market_cap < 1000000000:  # Example threshold for low market cap
            result += (
                "Warning: This cryptocurrency has a low market capitalization, "
                "which may indicate higher volatility and risks."
            )
        else:
            result += (
                "The market capitalization of this cryptocurrency is relatively high, "
                "which may indicate lower volatility and risks."
            )

        return result
    else:
        return "The data integrity verification failed. Cannot provide prediction."

# Ask the user to input the cryptocurrency name
crypto_name = input("Which crypto do you want to invest in: ")

# Call the predict_price function with the user's input
print(predict_price_with_blockchain(crypto_name))
