import numpy as np
import OnlyMLFS

path = 'C:\Projects\OnlyMLFS\Wine_Quality_Data.csv'
con = data_importer
data = con.import_d(path)
data_df = import_data(data)
print(data_df.d_type())
