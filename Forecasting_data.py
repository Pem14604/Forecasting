import pandas as pd
from pymongo import MongoClient

def basic_data():
    file_path =     dataset = pd.read_excel(file_path, encoding='latin1')
    dataset.insert(2, "DataBuffer", 1)

    def preprocess_daily(dataset):
        ts = dataset
        created = 'Created'
        ts['date'] = pd.to_datetime(ts[created].astype(str), errors = "coerce")
        ts['Date'] = ts['date'].dt.date
        ts.drop(created, axis=1, inplace=True)
        ts['Ticket_count'] = ts['DataBuffer'].groupby(ts['Date']).transform('sum')

        ts_daily = pd.DataFrame({"Ticket_count": ts.groupby(['Date']).size()}).reset_index()
        ts_daily = ts_daily.sort_values(['Date'], ascending=[True])
        ts_daily.reset_index(drop=True, inplace=True)
        return ts_daily

    ts_month = preprocess_daily(dataset)
    ts_month['Date'] = ts_month['Date'].astype(str)

    table = {}
    table["Opening Date"] = ts_month['Date'].values.tolist()
    table["Aggregate"] = ts_month['Ticket_count'].values.tolist()

    client = MongoClient()
    client = MongoClient('localhost', 27017)

    mydatabase = client['forecasting']
    mycollection = mydatabase['TableNew']

    rec = table
    rec = mydatabase.TableNew.insert(rec)

    dictionary = {}
    for i in mydatabase.TableNew.find():                             
        # myDATABASE = DATABASE_NAME, TABLENEW = TABLE_NAME
        
        dictionary = i

    Created = dictionary['Opening Date']
    Ticket_count = dictionary['Aggregate']
    dataset = pd.DataFrame({"Created": Created, "Ticket_count": Ticket_count})
    return dataset
