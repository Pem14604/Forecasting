from statsmodels.tsa.arima_model import ARIMA

import numpy as np
import pandas as pd


def best_params(dataset):
    p_values = [0, 1, 2, 4, 6, 8, 10]
    d_values = range(0, 3)
    q_values = range(0, 3)
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    model_support = ARIMA(dataset.Ticket_count.astype('float32'), order=order)  # column = DF.COLUMN
                    result_support = model_support.fit(disp=0)
                    y_pred_support = pd.Series(result_support.fittedvalues.astype(int), copy=True)
                    size = int(len(dataset) * 0.2)
                    y_pred_support = y_pred_support[-size:]
                    ts_actual = dataset.Ticket_count[-size:]

                    y_pred_graph = pd.Series(result_support.fittedvalues.astype(int), copy=True)
                    ts_actual_graph = dataset.Ticket_count[:]

                    def mean_absolute_percentage_error(y_true, y_pred):
                        y_true, y_pred = np.array(y_true), np.array(y_pred)
                        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

                    mea = mean_absolute_percentage_error(ts_actual, y_pred_support)
                    if mea < best_score:
                        best_score, best_cfg = mea, order
                except:
                    continue
    return (best_cfg)


def sarima_model_daily(dataset, forecast_days):
    preprocessData = dataset
    preprocessData['date'] = pd.to_datetime(preprocessData['Created'])
    preprocessData['Date'] = preprocessData['date'].dt.date

    preprocessData = preprocessData.groupby(['Date'])['Ticket_count'].sum().reset_index()
    from scipy import stats
    import numpy as np
    z = np.abs(stats.zscore(preprocessData.Ticket_count))
    preprocessData['z'] = z
    preprocessData = preprocessData.query('z <=1').reset_index()
    print (len(preprocessData))

    if (len(preprocessData)>=100):

        best_cfg = best_params(preprocessData)
        print(best_cfg)

        model_support = ARIMA(preprocessData.Ticket_count, order=best_cfg)  # column = DF.COLUMN
        result_support = model_support.fit(disp=0)
        y_pred_support = pd.Series(result_support.fittedvalues.astype(int), copy=True)
        size = int(len(preprocessData) * 0.2)
        y_pred_support = y_pred_support[-size:]
        ts_actual = preprocessData.Ticket_count[-size:]

        y_pred_graph = pd.Series(result_support.fittedvalues.astype(int), copy=True)
        ts_actual_graph = preprocessData.Ticket_count[:]

        def mean_absolute_percentage_error(y_true, y_pred):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        mea = mean_absolute_percentage_error(ts_actual, y_pred_support)
        accuracy = int(100 - mea)
        print('Accuracy = ' + str(accuracy) + '%')

        x = pd.date_range(start=preprocessData["Date"].values.tolist()[-1],
                          periods=forecast_days + 1).date.tolist()[1:]
        y = preprocessData['Date'].values.tolist()
        y.extend(x)

        df = preprocessData
        start = forecast_days
        start_support = len(df)
        end_support = start_support + int(start - 1)
        prediction = result_support.predict(start=start_support,
                                            end=end_support, exog=None, dynamic=False)
        pre = prediction.astype(int).values.tolist()

        # print("Next " + str(start) + " days" +  "forecasted support issue count = " + str(pre))

        df1 = pd.DataFrame({'Actual': ts_actual_graph, 'Predicted': y_pred_graph})
        df2 = pd.DataFrame({"Actual": [0 for i in range(start)], "Predicted": pre})

        df1 = df1.append(df2).reset_index()
        df1 = df1.drop('index', axis=1)
        df1['Period'] = y

        df1 = df1.tail(4 + start).reset_index()
        df1.drop("index", axis=1, inplace=True)
        df1['Period'] = df1['Period'].astype(str)
        df1_json = df1.to_json(orient="records")

    return df1_json


def sarima_model_weekly(dataset, start):
    # FUNCTION TO PERFORM S-ARIMA ON DF.COLUMN

    preprocessData = dataset
    preprocessData['date'] = pd.to_datetime(preprocessData['Created'])
    preprocessData['week'] = preprocessData['date'].dt.week
    preprocessData['year'] = preprocessData['date'].dt.year
    preprocessData.drop("Created", axis=1, inplace=True)
    preprocessData = preprocessData.groupby(['year', 'week'])['Ticket_count'].sum().reset_index()
    from scipy import stats
    import numpy as np
    z = np.abs(stats.zscore(preprocessData.Ticket_count))
    preprocessData['z'] = z
    preprocessData = preprocessData.query('z <=1').reset_index()
    print (len(preprocessData))

    if (len(preprocessData) >= 50):

        best_cfg = best_params(preprocessData)

        model_support = ARIMA(preprocessData.Ticket_count.astype("float32"), order=best_cfg)
        result_support = model_support.fit(disp=0)
        y_pred_support = pd.Series(result_support.fittedvalues, copy=True)
        size = len(preprocessData) * 0.2
        size = int(size)
        y_pred_support = y_pred_support[-size:]
        ts_actual = preprocessData.Ticket_count[-size:]

        y_pred_graph = pd.Series(result_support.fittedvalues.astype(int), copy=True)
        ts_actual_graph = preprocessData.Ticket_count[:]

        def mean_absolute_percentage_error(y_true, y_pred):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        mea = mean_absolute_percentage_error(ts_actual, y_pred_support)
        accuracy = int(100 - mea)
        print('Accuracy = ' + str(accuracy) + '%')

        start_support = len(preprocessData)
        end_support = start_support + int(start - 1)
        prediction = result_support.predict(start=start_support,
                                            end=end_support, exog=None, dynamic=False)
        pre = prediction.astype(int).values.tolist()

        # print("Next " + str(start) + " weeks " +  " forecasted support issue count = " + str(pre))

        df1 = pd.DataFrame({'Actual': ts_actual_graph, 'Predicted': y_pred_graph})
        df2 = pd.DataFrame({"Actual": [0 for i in range(start)], "Predicted": pre})

        df1 = df1.append(df2).reset_index()
        df1 = df1.drop('index', axis=1)

        # df1 = df1.fillna(0)
        week = preprocessData['week'].values.tolist()
        year = preprocessData['year'].values.tolist()

        for i in range(0, start):
            week.append(week[-1] + 1)
            year.append(year[-1])

        df1['week'] = week
        df1['year'] = year
        df1['week'] = df1['week'].astype(str)
        df1['year'] = df1['year'].astype(str)
        df1['Period'] = df1[['week', 'year']].apply(lambda x: 'th week-'.join(x), axis=1)
        df1.drop(["week", "year"], axis=1, inplace=True)
        df1 = df1.tail(4 + start).reset_index()
        df1.drop("index", axis=1, inplace=True)
        df1_json = df1.to_json(orient="records")
    else:
        df1_json = "Data Insufficient"
    return df1_json
