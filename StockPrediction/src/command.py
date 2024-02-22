import prediction_function as cmd

for stock in cmd.STOCK_LIST:
    cmd.moving_average(stock)

# cmd.moving_average("ADVANC")