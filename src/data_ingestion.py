from config import Config

# Import Manupulation Libraries
import pandas as pd 
import numpy as np

def data_ingestion():
    # read the CSV file into a DAtaframe
    df = pd.read_csv(Config.filepath)

    return df

    