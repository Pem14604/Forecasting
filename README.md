# Forecasting

In this we will be forecasting time series and for this we have used ARIMA model as we getting best resuls on this, we have tried FBprophet, LSTM, SARIMA as well

**There are three files in thise repository:**

- Forescasting data.py: We are using this file to read excel file and creating entry into mongo DB as we need to extract data from mongo DB.
excel file shouuld have data entry in datetime format with column name "Created" and rest we will do take itself.

- Forecasting_function.py In this we are creating ARIMA model along with grid search to get the best values of p,q and d also we are dtetcting anomaly in the data so if you interested then you can use this to check the anomaly as well.

- Forecasting_service.py In this file we have created service for forecasting and anomaly detection as well.
