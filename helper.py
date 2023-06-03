import pandas as pd
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('Data-3/features_30_sec.csv')
data = df[['filename', 'label']]
data['from'] = 0.000
data['to'] = 30.000
data['filename'] = data['filename'].apply(lambda x: x[:-4])
data = data[['filename', 'from', 'to', 'label']]

counter = 0
for idx, row in data.iterrows():
    if counter < 0:
        counter += 1
        continue
    if counter > 999:
        break
    print(f'{row["filename"]}, {row["from"]}, {row["to"]}, "{row["label"]}"')
    counter += 1

