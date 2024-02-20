# BinanceTrader

BinanceTrader class that uses DNN for predictions and making fake/real trades.

## Preparation

1. To use BinanceTrader Class, you should download python packages first. You can copy the cell below and run it:
```
python -m pip install numpy pandas schedule pytz tensorflow stock-indicators scikit-learn
```
2. You need to fetch and download the data from Binance. Run the `Downloades` file. After that, run the `Preprocessing` file to train and save the model and scaler (It is optional, you can use the model and scaler I prepared for you. But it is recommended to do it to have fresh data).

3. Open the `BinanceTrader` file and replace the `api_key` and `secret_key` (situated on almost the end of the file) based on your API information which you have already got from Binance (Want more details? Click [here](https://support.coinigy.com/hc/en-us/articles/360001144614-How-do-I-find-my-API-key-on-Binance-com-)). If you want to use Binance Testnet, do not forget to set the `testnet` value to `True`. You should also replace the `model_path` and `scaler_path` with the path of your model and scaler in your local computer or server (Having trouble finding the path? Click [here](https://www.wikihow.com/Find-a-File%27s-Path-on-Windows)). When you are done, run the file and enjoy!

## What does this code do?

It connects to your Binance account (through `api_key` and `secret_key`) and makes trades with leverage based on DNN predictions.

## Hints

- By using our services, you agree that you are aware of the possible risks of trading in the futures market.
- This class uses all your funds in the futures section for trades, please make sure to withdraw or transfer your funds on time.

## Resources

I have used these resources to build my project:

- Data Analysis with Pandas and Python (Video, [Details](https://www.udemy.com/course/data-analysis-with-pandas/))
- Python for Data Science and Machine Learning Bootcamp (Video, [Details](https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/))
- Algorithmic Trading A-Z with Python, Machine Learning & AWS (Video, [Details](https://www.udemy.com/course/algorithmic-trading-with-python-and-machine-learning/))
- Cryptocurrency Algorithmic Trading with Python and Binance (Video, [Details](https://www.udemy.com/course/cryptocurrency-algorithmic-trading-with-python-and-binance/))
- Performance Optimization and Risk Management for Trading (Video, [Details](https://www.udemy.com/course/performance-optimization-and-risk-management-for-trading/))
- Python-for-Finance_Mastering-Data-Driven-Finance-Book-OReilly-2018 (Book, [Available here](https://www.pdfdrive.com/python-for-finance-mastering-data-driven-finance-e196886670.html))
