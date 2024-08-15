# Stock Price Prediction

This project aims to predict stock prices using historical data. We use a dataset containing Tesla's stock prices and various machine learning techniques to build and evaluate a predictive model.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [Model Evaluation](#model-evaluation)
- [License](#license)

## Project Overview

The dataset used in this project is a CSV file containing Tesla's historical stock prices. The data includes:

- **Date**: The date of the stock price.
- **Open**: The opening price of the stock.
- **High**: The highest price of the stock during the day.
- **Low**: The lowest price of the stock during the day.
- **Close**: The closing price of the stock.
- **Adj Close**: The adjusted closing price, accounting for dividends and stock splits.
- **Volume**: The number of shares traded on that day.

The goal of this project is to predict Tesla's future stock prices based on this historical data. The dataset is used to train and evaluate a machine learning model that forecasts stock prices.

## Installation

To run this project, you'll need to have Python installed along with the following packages:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `plotly`

You can install the required packages using `pip`:

    pip install pandas numpy matplotlib seaborn scikit-learn plotly

## Usage
- Clone the Repository:
  
      git clone https://github.com/chandkund/stock-price-prediction.git
      cd stock-price-prediction

- Prepare the Dataset:
  Place your dataset (e.g., tesla.csv) in the project directory or adjust the file path in the code.

 - Run the Code:
  You can run the Jupyter notebook or Python script to execute the code. Make sure to update the path to 
  the dataset if necessary.

## Code Explanation
- Import Relevant Libraries:

      import pandas as pd
      import numpy as np
      import matplotlib.pyplot as plt
      import seaborn as sns
      from sklearn.model_selection import train_test_split
      from sklearn.metrics import mean_squared_error
      from sklearn.preprocessing import MinMaxScaler, StandardScaler
      from sklearn.linear_model import LinearRegression
      import plotly.graph_objs as go
      from plotly.offline import init_notebook_mode, plot, iplot
- Load and Preprocess the Data:
    
      raw_data = pd.read_csv("D:\\Project\\Project_3\\tesla.csv")
      df = raw_data.copy()
      df['Date'] = pd.to_datetime(df["Date"])
- Visualize the Data:
  
       fig, ax = plt.subplots(figsize=(12, 6))
       sns.boxplot(df[['Open', 'High', 'Low', 'Close', 'Adj Close']])
       plt.show()

       fig, ax = plt.subplots(figsize=(12, 6))
       plt.plot(df["Date"], df['Adj Close'], marker="|", linestyle="-", color='dodgerblue', linewidth=2, 
       markersize=8)
      ax.set_title('Adjusted Closing Prices Over Time', fontsize=16, fontweight='bold')
      ax.set_xlabel("Date")
      ax.set_ylabel("Price")
      plt.xticks(rotation=45)
      ax.grid(True, linestyle="--", linewidth=0.7)
      plt.tight_layout()
- Create a Plotly Visualization:

      init_notebook_mode(connected=True)

      layout = go.Layout(
              title="Stock Prices of Tesla",
              xaxis=dict(
                     title="Date",
                     titlefont=dict(
                               family='Courier New, monospace',
                               size=18,
                               color='#7f7f7f'
                              )
                            ),
               yaxis=dict(
                     title="Price",
                     titlefont=dict(
                               family='Courier New, monospace',
                               size=18,
                               color='#7f7f7f'
                           )
                      )
                   )

      tesla_data = go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price')
      plot = go.Figure(data=[tesla_data], layout=layout)
      iplot(plot)
- Normalization and Standardization:
  
      from sklearn.preprocessing import MinMaxScaler, StandardScaler

      cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

      # Min-Max Normalization
      scaled = MinMaxScaler()
      df[cols] = pd.DataFrame(scaled.fit_transform(df[cols]), columns=cols)

      # Standardization
      scaled = StandardScaler()
      df[cols] = pd.DataFrame(scaled.fit_transform(df[cols]), columns=cols)




- Split Data and Train Model:
  
      X = df[['Open', 'High', 'Low', 'Volume']]
      Y = df['Close']
      X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
- Model
  
      model = LinearRegression()
      model.fit(X_train, Y_train)

  
## Model Evaluation
  
  After training the model, evaluate its performance using Mean Squared Error (MSE):

      pred = model.predict(X_test)
      mse = mean_squared_error(Y_test, pred)
      print(f"Mean Squared Error: {mse}")


## License
This project is licensed under the MIT License. See the LICENSE file for details.

Make sure to adjust any file paths and repository URLs as needed. This `README.md` provides a structured overview of the project, including installation instructions, code explanation, and model evaluation.













