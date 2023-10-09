import pandas as pd
import json
import numpy as np

class data_importer:
    def __init__(self):
        self.d_type = None

    def import_d(self,file):
        if isinstance(file, pd.DataFrame):
            self.d_type = "Pandas DataFrame"
            data = file
        elif isinstance(file, np.ndarray):
            self.d_type = "Numpy Array"
            data =  pd.DataFrame(file)
        elif isinstance(file, str):
            if file.endswith(".csv"):
                self.d_type = "Comma Seperated Values"
                data =  pd.read_csv(file)
            elif file.endswith(".json"):
                self.d_type = "JSON"
                data = pd.read_json(file)
            elif file.endswith(".tsv"):
                self.d_type = "Tab Seperated Values"
                data =  pd.read_csv(file,sep="\t")
            else:
                return ValueError("Incompatible File Format!!")
        else:
            return ValueError("Incompatible File Format!!")
        return data
    
    def d_type(self):
        return self.d_type
        
