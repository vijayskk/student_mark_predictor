import joblib

model = joblib.load('model.joblib')

g1 = float(input("Enter the first term mark (in 20) : "))
g2 = float(input("Enter the second term mark (in 20) ; "))
stime = float(input("Enter the studytime : "))
fail = float(input("Enter the failiure count : "))
abs = float(input("Enter the absence count : "))



pred = model.predict([[g1,g2,stime,fail,abs]])
print("The predicted score is :" , pred[0])
