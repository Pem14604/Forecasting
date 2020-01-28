if __name__ == '__main__':
    import logging
    from flask import Flask, request, jsonify
    import Forecasting_data
    import Forecasting_function
    APP = Flask(import_name="forecasting")


    @APP.route("/get_forecasting", methods=['GET'])
    def get_forecasting():
        logging.basicConfig(filename='./app.log')


        try:
            num = request.args.get("num")
            num = int(num)
            frequency = request.args.get("frequency")
            dataset = Forecasting_data.basic_data()

            if(str.lower(frequency) == "weekly" and num < 3):
                df_json = Forecasting_function.sarima_model_weekly(dataset, num)
            elif(str.lower(frequency) == "daily" and num < 11):
                df_json = Forecasting_function.sarima_model_daily(dataset, num)
            else:
                df_json = jsonify({"ERROR":"Frequency - Weekly/Daily, Num (Weekly Range) - 1-2, Num(Daily Range) - 1-10"})

        except Exception as E:
            logging.exception(str(E))
            return jsonify({"Error": str(E), "Exception": str(E)}), 400

        return df_json
    APP.run(host="127.0.0.1", port="5001")
