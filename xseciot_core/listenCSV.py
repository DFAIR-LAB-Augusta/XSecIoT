from flask import Flask, request
import pandas as pd
import io

app = Flask(__name__)

@app.route('/', methods=['POST'])
def receive_csv():
    try:
        csv_data = request.data.decode('utf-8')  # Read the incoming CSV
        df = pd.read_csv(io.StringIO(csv_data))
        print("Received CSV data:")
        print(df)
        return "CSV received", 200
    except Exception as e:
        return f"Error: {e}", 400

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=2048)