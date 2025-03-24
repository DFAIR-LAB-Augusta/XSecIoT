# src/core/listener.py
import io
import logging

import pandas as pd

from flask import Flask, request

logger = logging.getLogger(__name__)

app = Flask(__name__)
data_callback = None  # This will be set externally


@app.route('/', methods=['POST'])
def receive_csv():
    try:
        csv_data = request.data.decode('utf-8') 
        df = pd.read_csv(io.StringIO(csv_data))
        logging.info("Received CSV data:")

        # If a processing callback is set, call it
        if data_callback:
            data_callback(df)

        return "CSV received", 200
    except Exception as e:
        return f"Error: {e}", 400


def run_server(callback=None, host='127.0.0.1', port=2048):
    global data_callback
    data_callback = callback
    app.run(host=host, port=port)


if __name__ == '__main__':
    run_server()
    