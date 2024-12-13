from flask import Flask, request, redirect, url_for, render_template
from model_api import DistilBertApi

app = Flask(__name__)
model = DistilBertApi()


@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html', preds_available=False)


@app.route('/api_predict', methods=['POST'])
def api_predict():
    text = request.form['title']
    predictions = model.predict_top_5(text)
    return render_template('index.html', preds_available=True, preds=predictions, title=text)


if __name__ == '__main__':
    # app.run(host='0.0.0.0')
    app.run(debug=True)
