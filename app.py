import numpy as np
from flask import Flask, request, render_template, abort, make_response
import os
import uuid
import urllib
from PIL import Image
from loadmodel import *
import matplotlib.image as mpimg

def predict_attribute(model, path, display_img=True):
    print("filepath",path)
    predicted = model.predict(path)
    if display_img:
        size = 244,244
        img=Image.open(path)
        img.thumbnail(size,Image.ANTIALIAS)
        imgplot = mpimg.imread(path)
        plt.imshow(imgplot)
    return predicted[0]

#create dataframe
df = pd.DataFrame(columns=["Img_name",
            "Collar",
            #"Epaulette",
            "Hood",
            #"Lapel",
            "Neckline",
            "Pocket",
            "Sleeve"])

#predict image classification
def predictModel(img):
    model = get_model()
    pred_result =  model.predict(img)
    return pred_result

#add row to dataframe
def addPred(pred,name):
    global df
    grid = {"Img_name":name,
            "Collar":pred[1][0].item(),
            #"Epaulette":pred[1][1].item(),
            "Hood":pred[1][2].item(),
            #"Lapel":pred[1][3].item(),
            "Neckline":pred[1][4].item(),
            "Pocket":pred[1][5].item(),
            "Sleeve":pred[1][6].item()}
    df = df.append(grid, ignore_index=True)
    print(df)

def dftohtml():
    global df
    htmldf = df.drop_duplicates(subset=['Img_name'],ignore_index=True).to_html(justify='center')
    text_file = open("templates/pandas.html", "w")
    text_file.write(htmldf)
    text_file.close()

app = Flask(__name__, template_folder = 'templates')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

#create our "home" route using the "index.html" page
@app.route('/')
def home():
    return render_template('index.html')

#Set a post method to yield predictions on page
@app.route('/success.html' , methods = ['GET' , 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd() , 'static/images')
    if request.method == 'POST':
        if(request.form):
            link = request.form.get('link')
            try :
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename+".jpg"
                img_path = os.path.join(target_img , filename)
                output = open(img_path , "wb")
                output.write(resource.read())
                output.close()
                img = filename

                pred_result =  predictModel(img_path) #predict_attribute(model, img_path)

                predictions = {
                    "prob1":pred_result[1][0].item(),
                    #"prob2":pred_result[1][1].item(),
                    "prob3":pred_result[1][2].item(),
                    #"prob4":pred_result[1][3].item(),
                    "prob5":pred_result[1][4].item(),
                    "prob6":pred_result[1][5].item(),
                    "prob7":pred_result[1][6].item(),
                }

                addPred(pred_result,filename)

            except Exception as e : 
                print(str(e))
                error = 'This image from this site is not accesible or inappropriate input'

            if(len(error) == 0):
                return  render_template('success.html' , img  = img , predictions = predictions)
            else:
                return render_template('index.html' , error = error) 

            
        elif (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img , file.filename))
                img_path = os.path.join(target_img , file.filename)
                img = file.filename

                pred_result = predictModel(img_path)

                predictions = {
                    "prob1":pred_result[1][0].item(),
                    "prob2":pred_result[1][1].item(),
                    "prob3":pred_result[1][2].item(),
                    "prob4":pred_result[1][3].item(),
                    "prob5":pred_result[1][4].item(),
                    "prob6":pred_result[1][5].item(),
                    "prob7":pred_result[1][6].item(),
                }
                
                addPred(pred_result,file.filename)

            else:
                error = "Please upload images of jpg , jpeg and png extension only"

            if(len(error) == 0):
                return  render_template('success.html' , img  = img , predictions = predictions)
            else:
                return render_template('index.html' , error = error)

    else:
        return render_template('index.html') 

@app.route("/csv/")
def getCSV():
    # with open("outputs/Adjacency.csv") as fp:
    #     csv = fp.read()
    if not df.empty:
        csv = df.drop_duplicates(subset=['Img_name'],ignore_index=True).to_csv(index=False)
        response = make_response(csv)
        cd = 'attachment; filename=myPred.csv'
        response.headers['Content-Disposition'] = cd 
        response.mimetype='text/csv'    
        return response

    else:
        return abort(400, 'No predictions were made') 

@app.route("/pandas.html")
def getdf():
    if not df.empty:
        dftohtml()
        return render_template('pandas.html')
    else:
        return abort(400, 'No predictions were made') 

#Run app
if __name__ == "__main__":
    app.run(debug=True)

