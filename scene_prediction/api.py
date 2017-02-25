from flask import Flask, render_template, request
from ScenePreditNB import ScenePredictNB

app = Flask(__name__)


@app.route('/upload')
def upload_file():
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        nb = ScenePredictNB()
        result_scene = nb.predict(f.filename)
        return result_scene
    if request.method == "GET":
        return '''<html>
   <body>

      <form action = "http://10.244.25.36:1234/uploader" method = "POST"
         enctype = "multipart/form-data">
         <input type = "file" name = "file" />
         <input type = "submit"/>
      </form>

   </body>
</html>'''


if __name__ == '__main__':
    app.run(host='0.0.0.0.',port=1234,debug=True)