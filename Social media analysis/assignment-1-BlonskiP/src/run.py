from src.data_loader import Data_loader
import pandas as pd
from src.visualize import vizualize
pd.set_option('display.width', 100000)
loader = Data_loader()
covid_data = loader.load_data()

print(len(covid_data))
vizualize(covid_data)