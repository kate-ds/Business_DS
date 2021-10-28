# USAGE
# Start the server:
# 	python run_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
import numpy as np
import dill
import pandas as pd
dill._dill._reverse_typemap['ClassType'] = type
#import cloudpickle
import flask

# initialize our Flask application and the model
app = flask.Flask(__name__)
model = None


def load_model(model_path):
    # load the pre-trained model
    global model
    with open(model_path, 'rb') as f:
        model = dill.load(f)


def load_transactions(transactions_path):
    """Load and prepare transaction data"""
    transactions = pd.read_csv(transactions_path)
    users = transactions.groupby('customer_id')['mcc_code'].apply(list).reset_index()
    types = transactions.groupby('customer_id')['tr_type'].apply(list).reset_index()
    transactions = pd.merge(users, types,  on='customer_id', how='left')

    return transactions

@app.route("/", methods=["GET"])
def general():
    return "Welcome to gender prediction process"

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        customer_id = ""
        request_json = flask.request.get_json()
        if request_json["customer_id"]:
            customer_id = request_json["customer_id"]

        preds = model.predict_proba(user_transactions.loc[user_transactions['customer_id'] == customer_id])
        data["predictions"] = preds[:, 1][0]
        data["customer_id"] = customer_id
        # indicate that the request was a success
        data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading the model and Flask starting server..."
           "please wait until server has fully started"))
    modelpath = "app/models2/model.pkl"
    user_transactions = load_transactions('app/Data2/transactions.csv')
    load_model(modelpath)
    app.run()
