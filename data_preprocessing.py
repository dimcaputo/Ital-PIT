import pandas as pd
import numpy as np

df = pd.read_csv(filepath_or_buffer='training_data/xyz_distances.csv',
                 index_col='pose_id')

print(df.info())