from flask import Flask, redirect
from requests import post
import json

headers = {
    'Connection': 'keep-alive',
    'Host': 'user.xunfss.com'
}

app = Flask(__name__)

def collect_url_values(data, prefix='url', results=None):
    if results is None:
        results = []
 
    if isinstance(data, dict):
        for key, value in data.items():
            if key.startswith(prefix) and key != 'urlprivate':
                results.append(value)
            collect_url_values(value, prefix, results)
    elif isinstance(data, list):
        for item in data:
            collect_url_values(item, prefix, results)
 
    return results

def cycle_values():
    while True:
        data = post(url='https://user.xunfss.com/app/listapp.php', data={'a': 'get18', 'system': 'pc'}).json()
        print(data)
        for value in collect_url_values(data):
            yield value



value_generator = cycle_values()

@app.route('/')
def index():
    return redirect(next(value_generator))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9900)
