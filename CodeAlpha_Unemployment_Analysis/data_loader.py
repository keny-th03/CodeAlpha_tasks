import pandas as pd  

def load_and_clean_data():
    # Load datasets
    unemp = pd.read_csv("Unemployment in India.csv")
    unemp_covid = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")
    
    # Clean column names
    unemp.columns = unemp.columns.str.strip()
    unemp_covid.columns = unemp_covid.columns.str.strip()

    # Convert Date column
    unemp['Date'] = pd.to_datetime(unemp['Date'], dayfirst=True)
    unemp_covid['Date'] = pd.to_datetime(unemp_covid['Date'], dayfirst=True)

    # Drop missing values
    unemp.dropna(inplace=True)
    unemp_covid.dropna(inplace=True)

    # Add month column for seasonal analysis
    unemp['Month'] = unemp['Date'].dt.month

    return unemp, unemp_covid