import pandas as pd
import numpy as np

data = []

for i in range(120):

    # Mostly normal values
    heart_rate = int(np.random.normal(80, 10))  # around 80
    spo2 = int(np.random.normal(97, 2))         # around 97
    temp = round(np.random.normal(36.8, 0.5), 2)

    # Occasionally abnormal
    if np.random.rand() < 0.2:
        heart_rate = np.random.randint(100, 130)
        spo2 = np.random.randint(85, 92)
        temp = round(np.random.uniform(37.5, 39), 2)
        ecg = 1
        fall = np.random.choice([0,1])
        risk = 1
    else:
        ecg = 0
        fall = 0
        risk = 0

    data.append([heart_rate, spo2, temp, fall, ecg, risk])

df = pd.DataFrame(data, columns=[
    "heart_rate","spo2","temp","fall","ecg","risk"
])

df.to_csv("health_data.csv", index=False)