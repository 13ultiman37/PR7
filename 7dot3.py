import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("bitcoin.csv")
projection = 14
data['predict'] = data['close'].shift(-projection)
x = DataFrame(data, columns=['close'], )
y = DataFrame(data, columns=['predict'])
x = np.array(x, type(float))
y = np.array(y, type(float))
x = x[:-projection]
y = y[:-projection]
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', alpha=0.3)
plt.xlabel('close')
plt.ylabel('predict')
plt.show()
regression = LinearRegression()
regression.fit(x, y)

print("\n----------Линейная регрессия: ----------")
print(regression.coef_)
print("\n----------Перехват: ----------")
print(regression.intercept_)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.3, color='purple')
plt.plot(x, regression.predict(x), color='yellow', linewidth=3)
plt.xlabel('close')
plt.ylabel('predict')
plt.show()

print("\n----------Прогнозируемая цена: ----------")
print(regression.predict(data[['close']][-projection:]))
print("\n----------Точность прогнозируемой цены: ----------")
print(regression.score(x, y))
