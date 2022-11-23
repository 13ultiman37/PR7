import numpy as np
import matplotlib.pyplot as plt

street = np.array([80, 98, 75, 91, 78])
garage = np.array([100, 82, 105, 89, 102])
print(np.corrcoef(street, garage)[0, 1])

plt.grid(True)
plt.title('Диаграмма рассеяния', fontsize=20)
plt.xlabel('Число машин на улице')
plt.ylabel('Число машин в гараже')
plt.scatter(street, garage, marker='o', color='crimson')
plt.show()
