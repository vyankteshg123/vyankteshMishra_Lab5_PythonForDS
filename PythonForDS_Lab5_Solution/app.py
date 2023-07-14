import pickle
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

app = Flask(__name__)
# open and load the pickle file provided in read mode.
model = pickle.load(open(r'model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')
# In the above code, are returning the output of the function "render_template()".
# This function accepts an input of an HTML page.
# This function looks for the mentioned page under templates folder.
# This is the reason we created this index.html form under templates folder.


# Predict function to read the values from the UI and predict the price value.
@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    scaled_final_features = ss.fit_transform(final_features)
    prediction = model.predict(scaled_final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Price of the Car is {} dollars'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
    app.config['TEMPLATES_AUTO_RELOAD'] = True
