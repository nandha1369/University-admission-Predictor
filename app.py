import pandas as pd
import numpy as np
import re
import joblib
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, render_template, redirect, url_for, session, flash
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mail import Mail, Message

app = Flask(__name__)
app.secret_key = '13'

def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(pattern, email)

# MySQL Configuration
app.config["MYSQL_HOST"] = "127.0.0.1"
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = ""
app.config["MYSQL_DB"] = "project"

mysql = MySQL(app)

# Flask-Mail Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your-email@gmail.com'
app.config['MAIL_PASSWORD'] = 'your-email-password'
app.config['MAIL_DEFAULT_SENDER'] = 'your-email@gmail.com'

mail = Mail(app)

# Load dataset (Use correct path)
dataset_path = r"C:\Users\Nandhakumar S\flask_project\data\Admission_Predict.csv"
data = pd.read_csv(dataset_path)

# Fix column names
data.columns = data.columns.str.strip()  # Removes extra spaces

# Select features and target
X = data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA']]
y = (data['Chance of Admit'] >= 0.5).astype(int)  # Convert probability to binary (1: chance, 0: no chance)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save model and scaler
model_dir = r"C:\Users\Nandhakumar S\flask_project\models"
os.makedirs(model_dir, exist_ok=True)

pickle.dump(model, open(os.path.join(model_dir, "model.pkl"), "wb"))
pickle.dump(scaler, open(os.path.join(model_dir, "scaler.pkl"), "wb"))

print("Model and scaler saved successfully!")


@app.route("/")
def home():
    return render_template("home.html")

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None  # To store email validation error

    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if not is_valid_email(email):
            error = "Enter a valid email."
        else:
            hashed_password = generate_password_hash(password)
            cursor = mysql.connection.cursor()

            try:
                cursor.execute("INSERT INTO user_id (Username, Email, Password) VALUES (%s, %s, %s)", 
                               (username, email, hashed_password))
                mysql.connection.commit()
                flash("Registration successful! Please log in.", "success")
                return redirect('/login')
            except Exception:
                error = "Email already exists!"  # Error for duplicate email
            finally:
                cursor.close()
    
    return render_template('register.html', error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        cursor = mysql.connection.cursor()

        cursor.execute("SELECT Username, Email, Password FROM user_id WHERE Email = %s", (email,))
        user = cursor.fetchone()
        cursor.close()

        if user and check_password_hash(user[2], password):
            session['loggedin'] = True
            session['username'] = user[0]
            flash("Login successful!", "success")
            return redirect(url_for('predict'))
        flash("Invalid email or password!", "danger")
    
    return render_template('login.html')

@app.route('/forgotpassword', methods=['GET', 'POST'])
def forgotpassword():
    if request.method == 'POST':
        email = request.form.get('email')

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM user_id WHERE Email = %s", (email,))
        user = cursor.fetchone()
        cursor.close()

        if user:
            # Send reset email
            msg = Message("Password Reset Request",
                          recipients=[email])
            msg.body = f"Click the link to reset your password: http://127.0.0.1:5000/resetpassword/{email}"
            mail.send(msg)

            flash("A password reset link has been sent to your email.", "success")
        else:
            flash("Email not found!", "danger")

        return redirect(url_for('forgotpassword'))

    return render_template('forgotpassword.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            features = [request.form.get(k) for k in ['gre_score', 'toefl_score', 'university_rating', 'sop', 'lor', 'cgpa']]
            
            if None in features or "" in features:
                flash("Please fill in all fields!", "danger")
                return render_template('predict.html')
            
            features = np.array([[float(f) for f in features]])
            scaled_features = scaler.transform(features)
            prediction = int(model.predict(scaled_features)[0])
            result = "Chance" if prediction == 1 else "No Chance"

            # Prepare data for Excel
            username = session.get('username', 'Guest')
            new_entry = {
                "Username": username,
                "GRE Score": features[0][0],
                "TOEFL Score": features[0][1],
                "University Rating": features[0][2],
                "SOP": features[0][3],
                "LOR": features[0][4],
                "CGPA": features[0][5],
                "Prediction": result
            }

            file_path = "data/Admission_Predict1.xlsx"

            # Append new data or create file if missing
            if os.path.exists(file_path):
                df = pd.read_excel(file_path)
                df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
            else:
                df = pd.DataFrame([new_entry])

            df.to_excel(file_path, index=False)

            flash(f"Prediction saved: {result}", "success")
   
            if prediction == 1:
                return redirect(url_for('chance'))
            else:
                return redirect(url_for('nochance'))
        except ValueError:
            flash("Invalid input. Please enter valid numbers.", "danger")
    
    return render_template('predict.html')

@app.route('/chance')
def chance():
    return render_template('chance.html')

@app.route('/nochance')
def nochance():
    return render_template('nochance.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)
