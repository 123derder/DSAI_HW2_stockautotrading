# DSAI_HW2_stockautotrading

stockautotrading.py 是股票預測的程式，我是用LSTM來預測股票的漲跌，我的LSTM預測目標是從今天看，五天後股票是會漲還是跌，漲的話，預設的的值會是1，跌的話，預測的值會是0，當我得到一串漲跌預測的list後
ex:[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

我用預測五天後會漲會跌的模型，來決定要買要賣

那要買要賣的預測呢，我是把上面的5天後漲跌的預測list，經過我的策略後，轉換成預測明天要買要賣的list，
策略: 當0轉1的時候，1那個點就要買 (output: 1) 當1轉0的時候，0那個點就要賣 (output: -1 )  其他(output: 0) (買時股票數量-> 1 賣時股票數量-> 0)
經過我得策略轉換後，就可以得到，預測明天要買要賣的結果了
