import time
from iqoptionapi.stable_api import IQ_Option
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import MetaTrader5 as mt5
import joblib
iq = IQ_Option(f"email", "senha")
iq.connect()  # connect to iqoptionimport xgboost as xgb
par ='EURUSD'
def checktempo2():
    while True:
        time.sleep(0.2)
        if datetime.now().second == 55:
            break

def getcandles(mt5):
    bars = 5000
    if not mt5.initialize():
	    print("initialize() failed, error code =", mt5.last_error())
	    quit()
    symbol = "EURUSD"
    timeframe = mt5.TIMEFRAME_M1
    candles = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    mt5.shutdown()
    data = pd.DataFrame(candles)
    data = data.rename(columns={'open': 'open', 'close': 'close', 'high': 'max', 'low': 'min', 'tick_volume': 'volume'})
    data = data[['time', 'open', 'close', 'min', 'max', 'volume']]
    data = data.drop(columns=['time'])
    # Remova a coluna de datas (índice)
    data = data.reset_index(drop=True)
    return data


def datamain(iq, par):
    data_frames = []
    dias = 1
    current_time = time.time()  # Obtenha o horário atual
    for _ in range(dias):
        num_velas = 1000  # Defina o número de velas para coletar (máximo de 1000)
        end_time = current_time
        velas = iq.get_candles(par, 60, num_velas, end_time)
        if not velas:
            break  # Se não houver mais velas, interrompa o loop
        data = pd.DataFrame(velas)
        data_frames.append(data)
        last_timestamp = data['from'].iloc[-1]
        current_time = last_timestamp - 1  # Subtrair 1 segundo do carimbo de data/hora mais recente para o próximo dia
        # Aguarde alguns segundos antes de fazer a próxima chamada para evitar a duplicação de registros
        time.sleep(0.2)
    combined_data = pd.concat(data_frames, ignore_index=True)
    combined_data.fillna(method="ffill", inplace=True)
    X = combined_data[["open", "close", "min", "max","volume"]]
    #print(X.info())
    #print(X.describe())
    #duplicates = X.duplicated()
    #print(X[duplicates])
    return X

def processprediction(iq,par):
	par = 'EURUSD'
	df = datamain(iq, par)
	df = pd.DataFrame(df)
	df['Pips'] = df['close'] - df['open']
	df['Maxmin'] = df['max'] / df['min']
	df['volume'] = np.where((df['open'] < df['close']), df['volume'],
	                        np.where(
		                        (df['open'] > df['close']),
		                        -df['volume'],
		                        0))
	df['volume'] = df['volume']
	df['EMA_3'] = df['close'].ewm(span=9, adjust=False).mean()
	df['EMA_5'] = df['close'].ewm(span=21, adjust=False).mean()
	df['EMA_9'] = df['close'].ewm(span=36, adjust=False).mean()
	df['EMA_3'] = df['EMA_3'] - df['close']
	df['EMA_5'] = df['EMA_5'] - df['close']
	df['EMA_9'] = df['EMA_9'] - df['close']
	df['EMA_14'] = df['close'].ewm(span=14, adjust=False).mean()
	df['EMA_21'] = df['close'].ewm(span=21, adjust=False).mean()
	df['EMA_14'] = df['close'] - df['EMA_14']
	df['EMA_21'] = df['close'] - df['EMA_21']
	df['ltamax'] = df['max'].ewm(span=3, adjust=False).mean()
	df['Ltamin'] = df['min'].ewm(span=3, adjust=False).mean()
	df['ltamax'] = df['close'] - df['ltamax']
	df['Ltamin'] = df['close'] - df['Ltamin']
	df['WAD'] = 0
	price_diff = df['close'].diff()
	high_low_range = df['max'] - df['min']
	positive_flow = (price_diff > 0).astype(int)
	negative_flow = (price_diff < 0).astype(int)
	df['WAD'] = df['WAD'].shift(1) + (
		(positive_flow * (df['close'] - df['min']) / high_low_range - negative_flow * (
				df['max'] - df['close']) / high_low_range))
	atr_period = 14
	k_period = 14
	d_period = 3
	df['Lowest_Low'] = df['close'].rolling(window=k_period).min()
	df['Highest_High'] = df['close'].rolling(window=k_period).max()
	tr = np.maximum(df['Highest_High'] - df['Lowest_Low'], np.abs(df['Highest_High'] - df['close'].shift(1)))
	tr = np.maximum(tr, np.abs(df['Lowest_Low'] - df['close'].shift(1)))
	atr = tr.rolling(window=atr_period).mean()
	df['atr'] = atr
	df['Lowest_Low'] = df['close'].rolling(window=k_period).min()
	df['Highest_High'] = df['close'].rolling(window=k_period).max()
	df['%K'] = ((df['close'] - df['Lowest_Low']) / (df['Highest_High'] - df['Lowest_Low'])) * 100
	df['%D'] = df['%K'].rolling(window=d_period).mean()
	df['max'] = df['close'] - df['max']
	df['min'] = df['close'] - df['min']
	df['LTA'] = df['close'].ewm(span=21, adjust=False).mean()
	df['LTB'] = df['close'].ewm(span=50, adjust=False).mean()
	df['LTA'] = (df['close'] - df['LTA']).abs()
	df['LTB'] = (df['close'] - df['LTB']).abs()
	df['typical_price'] = (df['open'] + df['close'] + df['max'] + df['min']) / 4
	df['price_difference'] = df['typical_price'] - df['typical_price'].shift(1)
	df['average_price_difference'] = df['price_difference'].rolling(window=60).mean()
	lower_threshold = 51
	upper_threshold = 49
	df['price_neutral_zone'] = np.where((df['close'] >= lower_threshold) & (df['close'] <= upper_threshold), 0,
	                                    df['close'])
	lag_columns = [3, 5, ]  # Exemplo de números da sequência de Fibonacci para lags
	for lag in lag_columns:
		df[f'average_price_difference_lag_{lag}'] = df['average_price_difference'].shift(lag)
		df[f'LTB_lag_{lag}'] = df['LTB'].shift(lag)
		df[f'LTA_lag_{lag}'] = df['LTA'].shift(lag)
		df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
		df[f'EMA_14_lag_{lag}'] = df['EMA_14'].shift(lag)
		df[f'EMA_21_lag_{lag}'] = df['EMA_21'].shift(lag)
	df = df.dropna()
	return df

def processanalizeotc(par, iq):
    df = datamain(iq, par)
    df = pd.DataFrame(df)
    df['volume'] = np.where((df['open'] < df['close']), df['volume'],
	                        np.where(
		                        (df['open'] > df['close']),
		                        -df['volume'],
		                        0))
    max_value = df['volume'].max()
    min_value = df['volume'].min()
    range_value = max_value - min_value
    df['volume'] = ((df['volume'] - min_value) / range_value) * 2 - 1
    df['diffcsx'] = df['open'] - df['close']
    df['diffcsx'] = df['diffcsx'].shift(1) - df['diffcsx']
    df['diffcs'] = df['close'].shift(1)-df['close']
    df['diffcs2'] = df['max'].shift(1)-df['max']
    df['diffcs3'] = df['close'].shift(1)-df['close']
    df['diffcs4'] = df['open'].shift(1)-df['open']
    df['diffcsmeadi1'] = df['diffcs'].ewm(span=12, adjust=False).mean()
    df['diffcsmeadi2'] = df['diffcs2'].ewm(span=12, adjust=False).mean()
    df['diffcsmeadi3'] = df['diffcs3'].ewm(span=12, adjust=False).mean()
    df['diffcsmeadi4'] = df['diffcs4'].ewm(span=12, adjust=False).mean()
    df['supports'] = np.where((df['open'] == df['min']) | (df['close'] == df['min']), df['close'], 0)
    df['resistances'] = np.where((df['open'] == df['max']) | (df['close'] == df['max']), df['close'], 0)
    df['supports'] = df['close'] - df['supports']
    df['resistances'] =  df['close'] - df['resistances']
    df['breakout_high'] = np.where((df['close'] > df['resistances']), df['max'] - df['resistances'], 0)
    df['breakout_high'] = df['breakout_high'].ewm(span=12, adjust=False).mean()
    df['breakout_low'] = np.where((df['close'] < df['supports']), df['supports'] - df['min'], 0)
    df['breakout_low'] = df['breakout_low'].ewm(span=12, adjust=False).mean()
    df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['WAD'] = 0
    price_diff = df['close'].diff()
    high_low_range = df['max'] - df['min']
    positive_flow = (price_diff > 0).astype(int)
    negative_flow = (price_diff < 0).astype(int)
    df['WAD'] = df['WAD'].shift(1) + (
    	(positive_flow * (df['close'] - df['min']) / high_low_range - negative_flow * (
    			df['max'] - df['close']) / high_low_range))

    MaFast_period = 3
    MaSlow_period = 6
    MaTrend_period = 133
    EMA_PERIOD = 30
    SMA_AUX_PERIOD = 15
    df['SMA_Fast'] = df['close'].rolling(window=MaFast_period).mean()
    df['SMA_Slow'] = df['close'].rolling(window=MaSlow_period).mean()
    df['EMA_Value'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    df['SMA_ValueAux'] = df['close'].rolling(window=SMA_AUX_PERIOD).mean()
    df['iprovmcad'] = df['SMA_Slow']-df['SMA_Fast']+df['SMA_ValueAux']
    df['EMA_Slow'] = df['close'].ewm(span=MaSlow_period, adjust=False).mean()
    df['EMA_Fast'] = df['close'].ewm(span=MaFast_period, adjust=False).mean()

    # Adicionando diferenças entre médias
    df['SMA_Fast_Slow_Diff'] = df['SMA_Fast'] - df['SMA_Slow']
    df['EMA_Fast_Slow_Diff'] = df['EMA_Fast'] - df['EMA_Slow']

    # Adicionando média móvel da diferença
    df['SMA_Fast_Slow_Diff_SMA'] = df['SMA_Fast_Slow_Diff'].rolling(window=SMA_AUX_PERIOD).mean()
    df['EMA_Fast_Slow_Diff_SMA'] = df['EMA_Fast_Slow_Diff'].rolling(window=SMA_AUX_PERIOD).mean()

    # Adicionando tendência baseada nas médias
    df['Trend_Slow'] = df['close'].rolling(window=MaTrend_period).mean()

    # Adicionando diferenças entre preços e médias
    df['Close_SMA_Fast_Diff'] = df['close'] - df['SMA_Fast']
    df['Close_SMA_Slow_Diff'] = df['close'] - df['SMA_Slow']
    df['Close_EMA_Fast_Diff'] = df['close'] - df['EMA_Fast']
    df['Close_EMA_Slow_Diff'] = df['close'] - df['EMA_Slow']
    df['Close_Trend_Diff'] = df['close'] - df['Trend_Slow']
    k_period = 14
    d_period = 3
    df['Lowest_Low'] = df['close'].rolling(window=k_period).min()
    df['Highest_High'] = df['close'].rolling(window=k_period).max()
    df['%K'] = ((df['close'] - df['Lowest_Low']) / (df['Highest_High'] - df['Lowest_Low'])) * 100
    df['%D'] = df['%K'].rolling(window=d_period).mean()
    df['LTA'] = df['close'].ewm(span=34, adjust=False).mean()
    df['LTB'] = df['close'].ewm(span=14, adjust=False).mean()
    diff_lta = (df['close'] - df['LTA']).abs()
    diff_ltb = (df['close'] - df['LTB']).abs()
    df['SMA2'] = df['close'].rolling(window=2).mean()
    df['LTA'] = diff_lta
    df['LTB'] = diff_lta
    df['SMA5'] = df['close'].rolling(window=5).mean()
    df['SMA10'] = df['close'].rolling(window=10).mean()
    df['SMA15'] = df['close'].rolling(window=15).mean()
    df['SMA30'] = df['close'].rolling(window=30).mean()
    df['SMA60'] = df['close'].rolling(window=60).mean()
    df['SMA2'] = np.log(1 + np.abs(df['SMA2'] - df['close'])) * np.sign(df['close'] - df['SMA60'])
    df['SMA5'] = np.log(1 + np.abs(df['SMA5'] - df['close'])) * np.sign(df['close'] - df['SMA60'])
    df['SMA10'] = np.log(1 + np.abs(df['SMA10'] - df['close'])) * np.sign(df['close'] - df['SMA60'])
    df['SMA15'] = np.log(1 + np.abs(df['SMA15'] - df['close'])) * np.sign(df['close'] - df['SMA60'])
    df['SMA30'] = np.log(1 + np.abs(df['SMA30'] - df['close'])) * np.sign(df['close'] - df['SMA60'])
    df['SMA60'] = np.log(1 + np.abs(df['SMA30'] - df['close'])) * np.sign(df['close'] - df['SMA60'])
    df['SMA9max'] = df['close'].rolling(window=9).mean()
    df['SMA9min'] = df['close'].rolling(window=9).mean()
    df['SMA9max'] = df['close']-df['SMA9max']
    df['SMA9min'] = df['close']-df['SMA9min']
    window = 14
    delta = df['close'].diff(1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    high_low = df['max'] - df['min']
    high_close_prev = abs(df['max'] - df['close'].shift(1))
    low_close_prev = abs(df['min'] - df['close'].shift(1))

    true_range = pd.DataFrame({'high_low': high_low, 'high_close_prev': high_close_prev, 'low_close_prev': low_close_prev})
    true_range_max = true_range.max(axis=1)

    atr = true_range_max.rolling(window=14).mean()

    df['ATR'] = atr
    df = df.dropna()
    df = df.drop(columns=['SMA60','EMA_9','close','max','min','Lowest_Low','Highest_High','%K',
                          'Trend_Slow','EMA_Fast','SMA_Slow','EMA_Value','SMA_ValueAux','EMA_Slow','SMA_Fast'])
    return df


def processanalizeotc(mt5):
	par = 'EURUSD'
	df = getcandles(mt5)
	df = pd.DataFrame(df)
	df['target'] = np.where((df['open'] < df['close'])&(df['max'] < df['close'].shift(-1)), 1,
	                        np.where(
		                        (df['open'] > df['close'])&(df['min'] > df['close'].shift(-1)),
		                        0,
		                        0.5))
	df['target'] = df['target'].shift(-1)
	df['volume'] = np.where((df['open'] < df['close']), df['volume'],
	                        np.where(
		                        (df['open'] > df['close']),
		                        -df['volume'],
		                        0))
	max_value = df['volume'].max()
	min_value = df['volume'].min()
	range_value = max_value - min_value
	df['volume'] = ((df['volume'] - min_value) / range_value) * 2 - 1
	df['diffcsx'] = df['open'] - df['close']
	df['diffcsx'] = df['diffcsx'].shift(1) - df['diffcsx']
	df['diffcs'] = df['close'].shift(1) - df['close']
	df['diffcs2'] = df['max'].shift(1) - df['max']
	df['diffcs3'] = df['close'].shift(1) - df['close']
	df['diffcs4'] = df['open'].shift(1) - df['open']
	df['diffcsmeadi1'] = df['diffcs'].ewm(span=12, adjust=False).mean()
	df['diffcsmeadi2'] = df['diffcs2'].ewm(span=12, adjust=False).mean()
	df['diffcsmeadi3'] = df['diffcs3'].ewm(span=12, adjust=False).mean()
	df['diffcsmeadi4'] = df['diffcs4'].ewm(span=12, adjust=False).mean()
	df['supports'] = np.where((df['open'] == df['min']) | (df['close'] == df['min']), df['close'], 0)
	df['resistances'] = np.where((df['open'] == df['max']) | (df['close'] == df['max']), df['close'], 0)
	df['supports'] = df['close'] - df['supports']
	df['resistances'] = df['close'] - df['resistances']
	df['breakout_high'] = np.where((df['close'] > df['resistances']), df['max'] - df['resistances'], 0)
	df['breakout_high'] = df['breakout_high'].ewm(span=12, adjust=False).mean()
	df['breakout_low'] = np.where((df['close'] < df['supports']), df['supports'] - df['min'], 0)
	df['breakout_low'] = df['breakout_low'].ewm(span=12, adjust=False).mean()
	df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
	df['WAD'] = 0
	price_diff = df['close'].diff()
	high_low_range = df['max'] - df['min']
	positive_flow = (price_diff > 0).astype(int)
	negative_flow = (price_diff < 0).astype(int)
	df['WAD'] = df['WAD'].shift(1) + (
		(positive_flow * (df['close'] - df['min']) / high_low_range - negative_flow * (
				df['max'] - df['close']) / high_low_range))
	MaFast_period = 3
	MaSlow_period = 6
	MaTrend_period = 133
	EMA_PERIOD = 30
	SMA_AUX_PERIOD = 15
	df['SMA_Fast'] = df['close'].rolling(window=MaFast_period).mean()
	df['SMA_Slow'] = df['close'].rolling(window=MaSlow_period).mean()
	df['EMA_Value'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
	df['SMA_ValueAux'] = df['close'].rolling(window=SMA_AUX_PERIOD).mean()
	df['iprovmcad'] = df['SMA_Slow'] - df['SMA_Fast'] + df['SMA_ValueAux']
	df['EMA_Slow'] = df['close'].ewm(span=MaSlow_period, adjust=False).mean()
	df['EMA_Fast'] = df['close'].ewm(span=MaFast_period, adjust=False).mean()

	# Adicionando diferenças entre médias
	df['SMA_Fast_Slow_Diff'] = df['SMA_Fast'] - df['SMA_Slow']
	df['EMA_Fast_Slow_Diff'] = df['EMA_Fast'] - df['EMA_Slow']

	# Adicionando média móvel da diferença
	df['SMA_Fast_Slow_Diff_SMA'] = df['SMA_Fast_Slow_Diff'].rolling(window=SMA_AUX_PERIOD).mean()
	df['EMA_Fast_Slow_Diff_SMA'] = df['EMA_Fast_Slow_Diff'].rolling(window=SMA_AUX_PERIOD).mean()

	# Adicionando tendência baseada nas médias
	df['Trend_Slow'] = df['close'].rolling(window=MaTrend_period).mean()

	# Adicionando diferenças entre preços e médias
	df['Close_SMA_Fast_Diff'] = df['close'] - df['SMA_Fast']
	df['Close_SMA_Slow_Diff'] = df['close'] - df['SMA_Slow']
	df['Close_EMA_Fast_Diff'] = df['close'] - df['EMA_Fast']
	df['Close_EMA_Slow_Diff'] = df['close'] - df['EMA_Slow']
	df['Close_Trend_Diff'] = df['close'] - df['Trend_Slow']
	k_period = 14
	d_period = 3
	df['Lowest_Low'] = df['close'].rolling(window=k_period).min()
	df['Highest_High'] = df['close'].rolling(window=k_period).max()
	df['%K'] = ((df['close'] - df['Lowest_Low']) / (df['Highest_High'] - df['Lowest_Low'])) * 100
	df['%D'] = df['%K'].rolling(window=d_period).mean()
	df['LTA'] = df['close'].ewm(span=34, adjust=False).mean()
	df['LTB'] = df['close'].ewm(span=14, adjust=False).mean()
	diff_lta = (df['close'] - df['LTA']).abs()
	diff_ltb = (df['close'] - df['LTB']).abs()
	df['SMA2'] = df['close'].rolling(window=2).mean()
	df['LTA'] = diff_lta
	df['LTB'] = diff_lta
	df['SMA5'] = df['close'].rolling(window=5).mean()
	df['SMA10'] = df['close'].rolling(window=10).mean()
	df['SMA15'] = df['close'].rolling(window=15).mean()
	df['SMA30'] = df['close'].rolling(window=30).mean()
	df['SMA60'] = df['close'].rolling(window=60).mean()
	df['SMA2'] = np.log(1 + np.abs(df['SMA2'] - df['close'])) * np.sign(df['close'] - df['SMA60'])
	df['SMA5'] = np.log(1 + np.abs(df['SMA5'] - df['close'])) * np.sign(df['close'] - df['SMA60'])
	df['SMA10'] = np.log(1 + np.abs(df['SMA10'] - df['close'])) * np.sign(df['close'] - df['SMA60'])
	df['SMA15'] = np.log(1 + np.abs(df['SMA15'] - df['close'])) * np.sign(df['close'] - df['SMA60'])
	df['SMA30'] = np.log(1 + np.abs(df['SMA30'] - df['close'])) * np.sign(df['close'] - df['SMA60'])
	df['SMA60'] = np.log(1 + np.abs(df['SMA30'] - df['close'])) * np.sign(df['close'] - df['SMA60'])
	df['SMA9max'] = df['close'].rolling(window=9).mean()
	df['SMA9min'] = df['close'].rolling(window=9).mean()
	df['SMA9max'] = df['close'] - df['SMA9max']
	df['SMA9min'] = df['close'] - df['SMA9min']
	window = 14
	delta = df['close'].diff(1)
	gain = np.where(delta > 0, delta, 0)
	loss = np.where(delta < 0, -delta, 0)
	avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
	avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()
	rs = avg_gain / avg_loss
	df['rsi'] = 100 - (100 / (1 + rs))
	high_low = df['max'] - df['min']
	high_close_prev = abs(df['max'] - df['close'].shift(1))
	low_close_prev = abs(df['min'] - df['close'].shift(1))
	true_range = pd.DataFrame(
		{'high_low': high_low, 'high_close_prev': high_close_prev, 'low_close_prev': low_close_prev})
	true_range_max = true_range.max(axis=1)

	atr = true_range_max.rolling(window=14).mean()
	df['ATR'] = atr
	df = df.dropna()
	df = df.drop(columns=['SMA60', 'EMA_9', 'close', 'max', 'min', 'Lowest_Low', 'Highest_High', '%K',
	                      'Trend_Slow', 'EMA_Fast', 'SMA_Slow', 'EMA_Value', 'SMA_ValueAux', 'EMA_Slow', 'SMA_Fast'])
	return df



# Crie um modelo XGBoost
checktempo2()
df = processanalizeotc(mt5)
y = df['target']
X = df.drop(columns=['target'])

#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

par = 'EURUSD'
y_pred = model.predict(X[-120:])
df = pd.DataFrame(y_pred)
df['predctions'] = df
todos_os_sinais = []
y_test = y[-120:]
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
predicted_classes = df['predctions'][:60]
print(predicted_classes)
hora_atual = datetime.now()
minutos_atuais = hora_atual.minute
minutos_atuais = minutos_atuais
minutos_proximo = (minutos_atuais // 1 + 1) * 1
velas_horario = []
hora_vela = hora_atual.replace(minute=minutos_proximo, second=0, microsecond=0)
decisoes = []
limiar_compra = 0.9
limiar_venda = 0.1
print(datetime.now().second)
for prev in predicted_classes:  # Use predicted_classes em vez de predictions
    if prev >= limiar_compra and prev <= 0.999:
        decisoes.append(f"CALL;{prev:.5f}")
    elif prev < limiar_venda and prev > 0.01:
        decisoes.append(f"PUT;{prev:.5f}")
    else:
        decisoes.append(None)  # Defina como None para não salvar os sinais "Esperar"
    velas_horario.append(hora_vela.strftime("%H:%M"))
    hora_vela += timedelta(minutes=1)
sinais_par = [f'M1;{par};{horario};{decisao};' for horario, decisao in zip(velas_horario, decisoes) if decisao]
todos_os_sinais.extend(sinais_par)
sinais_ordenados = sorted(todos_os_sinais, key=lambda x: datetime.strptime(x.split(';')[2], '%H:%M'))
with open('sinais.txt', 'w') as file:
                    file.write('\n'.join(sinais_ordenados))
# Plotar a importância das características
#xgb.plot_importance(model)
#plt.show()
limiar_compra = 0.75
limiar_venda = 0.35

# Crie arrays para sinalizar compra e venda com base nas previsões
sinal_compra = (y_pred[-60:] >= limiar_compra)&(y_pred[-60:] < 1)
sinal_venda = (y_pred[-60:] <= limiar_venda)&(y_pred[-60:] >0 )
#
## Crie um array de índices para representar o tempo
time_index = np.arange(len(df[-60:]))

plt.figure(figsize=(12, 6))
plt.plot(time_index, y_test[-60:], label='Valores Reais', color='blue')
plt.plot(time_index, y_pred[-60:], label='Previsões', color='red')
plt.scatter(time_index[sinal_compra], y_pred[-60:][sinal_compra], marker='^', color='green', label='Sinal de Compra')
plt.scatter(time_index[sinal_venda], y_pred[-60:][sinal_venda], marker='v', color='purple', label='Sinal de Venda')
plt.xlabel('Tempo')
plt.ylabel('Preço')
plt.legend()
plt.title('Previsões vs. Valores Reais com Sinais de Compra e Venda')
plt.show()
compra_correta = sinal_compra & (y_test[-60:] == 1)  # Compra correta
venda_correta = sinal_venda & (y_test[-60:] == 0)    # Venda correta
erro = (sinal_compra & (y_test[-60:] == 0)) | (sinal_venda & (y_test[-60:] == 1))  # Erro
num_compras_corretas = compra_correta.sum()
num_vendas_corretas = venda_correta.sum()
num_erros = erro.sum()
plt.figure(figsize=(12, 6))
plt.plot(time_index, y_test[-60:], label='Valores Reais', color='blue')
plt.plot(time_index, y_pred[-60:], label='Previsões', color='red')
plt.scatter(time_index[sinal_compra], y_pred[-60:][sinal_compra], marker='^', color='green', label='Sinal de Compra')
plt.scatter(time_index[sinal_venda], y_pred[-60:][sinal_venda], marker='v', color='purple', label='Sinal de Venda')
plt.xlabel('Tempo')
plt.ylabel('Preço')
plt.legend()
plt.title('Previsões vs. Valores Reais com Sinais de Compra e Venda')
plt.text(10, 1, f'Compras Corretas: {num_compras_corretas}')
plt.text(10, 0.9, f'Vendas Corretas: {num_vendas_corretas}')
plt.text(10, 0.8, f'Erros: {num_erros}')
plt.show()
