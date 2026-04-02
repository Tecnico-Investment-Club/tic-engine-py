# Backtesting beggining
DATE_INIT = '2017-01-01'


# Minimum acceptable in sample asset return for further selection
RETURN_MIN = 0.0

# Walk Forward
N_SPLITS = 9*4
TRAIN_BLOCKS_INIT = 2
TEST_BLOCKS = 1
WF_METHOD = 'rolling' # rolling or expanding

# Clustering
N_CLUSTERS = 4

# Treshold to zero assets weights with too small weights
WEIGHT_CUTOFF = 0.00001

# Optimization Universe
crypto = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'BNB-USD', 'SOL-USD', 'TRX-USD', 'DOGE-USD', 'ADA-USD']
commodities = ["USO", "BNO", "UNG", "GLD", "SLV", "PPLT", "PALL", "DBA", "DBC", "GSG"]

