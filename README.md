# BinanceTrader

BinanceTrader is a Python class that utilizes Deep Neural Networks (DNN) for making predictions and executing demo/real trades on the Binance platform. This project aims to provide a framework for algorithmic trading in the cryptocurrency market.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Risk Disclaimer](#risk-disclaimer)
- [Resources](#resources)
- [Contributing](#contributing)
- [License](#license)

## Features

- Leverages Deep Neural Networks for price predictions
- Supports both demo (testnet) and real trading environments (mainnet)
- Automated trading with customizable parameters
- Real-time data fetching and preprocessing
- Integration with Binance API

## Prerequisites

- Python 3.7+
- Binance account (with API access)
- Basic understanding of cryptocurrency trading and Python programming

## Installation

1. Clone this repository:
```
git clone https://github.com/AlrzA2003/BinanceTrader.git
cd BinanceTrader
```

2. Install the required packages:
```
pip install -r requirements.txt
```

For specific versions to ensure compatibility:
```
pip install -r requirements_specific.txt
```

## Setup

1. Data Preparation:
- Run `Downloades.py` to fetch historical data from Binance.
- Execute `Preprocessing.py` to train and save the model and scaler.

Note: While pre-trained models are provided, it's recommended to train your own for the most up-to-date data.

2. Configuration:
- Open `credentials.txt` and replace `API_KEY` and `SECRET` with your Binance API credentials. [How to get Binance API keys](https://support.coinigy.com/hc/en-us/articles/360001144614-How-do-I-find-my-API-key-on-Binance-com-)
- In `BinanceTrader.py` set `testnet` to `True` if you want to use Binance Testnet for demo trading.
- Update `model_path` and `scaler_path` with the correct paths to your model and scaler files.

## Usage

After completing the setup, run the BinanceTrader:
```
python BinanceTrader.py
```
The script will connect to your Binance account and start trading based on the DNN predictions.

## How It Works

BinanceTrader operates by:

1. Connecting to your Binance account using the provided API keys.
2. Making trades with leverage based on predictions from the Deep Neural Network model.
3. Using all available funds in the futures section for trades.

## Risk Disclaimer

Trading cryptocurrencies, especially with leverage, carries significant financial risk. By using BinanceTrader, you acknowledge and accept these risks. Please note:

- This software is for educational and experimental purposes only.
- Never trade with funds you cannot afford to lose.
- The author is not responsible for any financial losses incurred while using this software.
- Always monitor your trades and be prepared to intervene manually if necessary.

## Resources

This project was developed with the help of the following educational resources:

1. **Data Analysis with Pandas and Python**
   - Platform: Udemy
   - [Course Link](https://www.udemy.com/course/data-analysis-with-pandas/)
   - Description: Comprehensive course on data manipulation and analysis using Pandas.

2. **Python for Data Science and Machine Learning Bootcamp**
   - Platform: Udemy
   - [Course Link](https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/)
   - Description: Extensive course covering various machine learning algorithms and their implementation in Python.

3. **Algorithmic Trading A-Z with Python, Machine Learning & AWS**
   - Platform: Udemy
   - [Course Link](https://www.udemy.com/course/algorithmic-trading-with-python-and-machine-learning/)
   - Description: Comprehensive overview of algorithmic trading, from basic concepts to advanced strategies.

4. **Cryptocurrency Algorithmic Trading with Python and Binance**
   - Platform: Udemy
   - [Course Link](https://www.udemy.com/course/cryptocurrency-algorithmic-trading-with-python-and-binance/)
   - Description: Focused course on cryptocurrency trading using the Binance API.

5. **Performance Optimization and Risk Management for Trading**
   - Platform: Udemy
   - [Course Link](https://www.udemy.com/course/performance-optimization-and-risk-management-for-trading/)
   - Description: Course on optimizing trading strategies and managing risk in trading systems.

6. **Python for Finance: Mastering Data-Driven Finance**
   - Author: Yves Hilpisch
   - Publisher: O'Reilly Media
   - [Book Link](https://www.oreilly.com/library/view/python-for-finance/9781492024323/)
   - Description: Comprehensive book covering various aspects of financial analysis and algorithmic trading using Python.

## Contributing

Contributions to improve BinanceTrader are welcome! Please feel free to submit pull requests or open issues to discuss potential enhancements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
