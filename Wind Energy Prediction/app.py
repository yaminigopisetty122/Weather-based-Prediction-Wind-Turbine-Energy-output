from flask import Flask,render_template,request
import pickle,requests,numpy as np

app=Flask(__name__)

model=pickle.load(open("wind_model.pkl","rb"))

API_KEY= process.env.API_KEY;

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/windapi")
def windapi():
    return render_template("predict.html")

@app.route("/weather",methods=["POST"])
def weather():

    city=request.form["city"]
    print("=============")
    print("CITY RECEIVED:",city)
    print("=============")
    url=f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    data=requests.get(url).json()

    temp=data["main"]["temp"]
    hum=data["main"]["humidity"]
    pres=round(data["main"]["pressure"]*0.75006,2)
    wind=round(data["wind"]["speed"],2)

    return render_template("predict.html",
        city=city,
        temp=temp,
        hum=hum,
        pres=pres,
        wind=wind
    )

@app.route("/predict",methods=["POST"])
def predict():

    theo=float(request.form["theoretical"])
    ws=float(request.form["windspeed"])

    city=request.form["city"]
    temp=request.form["temp"]
    hum=request.form["hum"]
    pres=request.form["pres"]
    wind=request.form["wind"]

    X=np.array([[ws,theo]])
    pred=model.predict(X)[0]

    return render_template("predict.html",
        city=city,
        temp=temp,
        hum=hum,
        pres=pres,
        wind=wind,
        prediction=round(pred,2)
    )

if __name__=="__main__":
    app.run(debug=True)