from flask import Flask, render_template, request
from sklearn.externals import joblib
import os
import pandas as pd
import numpy as np

app = Flask(__name__, static_url_path='/static/')

@app.route('/')
def form():
    return render_template('index.html')


@app.route('/predict_gdp', methods=['POST', 'GET'])
def predict_gdp():
    
    # get the parameters
    life_expectancy = int(request.form['life_expectancy'])
    co2_per_capita = int(request.form['co2_per_capita'])
    fertility_rate = int(request.form['fertility_rate'])
    merch_export = int(request.form['merch_export'])

    # load the X_columns file
    X_columns = joblib.load('model/X_columns.joblib')
    print(X_columns)

    # generate a dataframe with zeros
    df_prediction = pd.DataFrame(np.zeros((1, len(X_columns))), columns=X_columns)
    print(df_prediction)

    # change the dataframe according to the inputs
    df_prediction.at[0, 'life_expectancy'] = (life_expectancy)
    df_prediction.at[0, 'co2_per_capita'] = (co2_per_capita)
    df_prediction.at[0, 'fertility_rate'] = (fertility_rate)
    df_prediction.at[0, 'merch_export'] = (merch_export)
    print(df_prediction)

    # load the model and predict
    model = joblib.load('model/model.joblib')
    prediction = model.predict(df_prediction.head(1).values)
    predicted_gdp = prediction.round(1)[0]

    return render_template('results.html',
                           life_expectancy=int(life_expectancy),
                           co2_per_capita=int(co2_per_capita),
                           fertility_rate=int(fertility_rate),
                           merch_export=int(merch_export),
                           predicted_gdp=int(predicted_gdp)
                           )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
