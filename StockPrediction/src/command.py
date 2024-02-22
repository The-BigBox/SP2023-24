import prediction_function as cmd

for stock in cmd.STOCK_LIST:
    cmd.stock_tuning(stock, [1])

# cmd.stock_tuning("BH", [1])