import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)
model = pickle.load(open('model.pkl' , 'rb'))

@app.route('/api', methods = ['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([[np.array(data['exp'])]])
    output = prediction[0]
    return jsonify(output)

# if request.method==[’GET’]:
#     url = ‘http://localhost:5000/api'
# plt.axis([0,11,0,130000])
# xvals = np.arange(1, 10, 0.24)
# yvals = 9450*xvals + 25792
# plt.plot(xvals,yvals)
# xscat = np.arange(1, 10, 0.24)

# plt.scatter(xscat, yscat, c=’red’, s=3)
# plt.ylabel(‘salary’)
# plt.xlabel(‘years’)
# bytes_image = io.BytesIO()
# plt.savefig(bytes_image, format=’png’)
# bytes_image.seek(0)
# return send_file(bytes_image , mimetype=’image/png’)

if __name__ == "__main__":
    app.run(port = 5000, debug=True)