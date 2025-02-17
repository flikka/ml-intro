# Using pandas, create a dataframe with 100 rows. The columsn should be temperature, wind, and if there is snow or not (true false)
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "temperature": np.random.randint(-20, 30, 1000),
    "wind": np.random.randint(0, 30, 1000),
    "precipitation": np.random.randint(0, 100, 1000),
})

# Make a column with slippery or not, true if temparature < 0 and precipitation > 0
df["slippery"] = (df["temperature"] < 0) & (df["precipitation"] > 0) & (df["wind"] < 25)
df.to_parquet("weather.parquet")