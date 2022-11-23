import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("housePrice.csv", delimiter=',')
print(data.isna().sum())

data.Address.fillna('other', inplace=True)
data = data.drop([807, 2802, 570, 2171, 1604, 709])
data = data.drop_duplicates()

print("\n----------После нормализации: ----------")
print(data.isna().sum())
print('\n', data[data.duplicated()])

x = data['Area'].astype(float)
y = data['Price(USD)'].astype(float)

n = np.size(x)
m_x = np.mean(x)
m_y = np.mean(y)

SS_xy = np.sum(y * x) - n * m_y * m_x
SS_xx = np.sum(x * x) - n * m_x * m_x

b_1 = SS_xy / SS_xx
b_0 = m_y - b_1 * m_x

print("\n----------Линейная регрессия: ----------")
print(b_1, '\n', b_0)

plt.scatter(x, y, color="m", marker="o", s=30, alpha=0.3)
y_pred = b_0 + b_1 * x
plt.plot(x, y_pred, color="g")
plt.xlabel('x')
plt.ylabel('y')
plt.show()
