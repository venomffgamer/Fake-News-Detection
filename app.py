from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from google.oauth2 import id_token
from google.auth.transport import requests
import model  # Import your model.py
# import plotly.express as px
import os

app = Flask(__name__)
app.secret_key = "neg27coMh7cv9F5fue54WuGgKY9GVPMR"  # Set a secret key for session management

GOOGLE_CLIENT_ID = "567891971421-vt9bv8ceifgll96r9me58vto7v6gkdr7.apps.googleusercontent.com"

@app.route("/")
def home():
    user = session.get('user')  # Check if the user is logged in
    return render_template("index.html", user=user)

@app.route("/dashboard")
def dashboard():
    user = session.get('user')
    if not user:
        return redirect(url_for('home'))
    return render_template("dashboard.html", user=user)

@app.route("/login")
def login():
    return render_template("googlelogin.html")

# Route for the news prediction dashboard
@app.route("/news")
def news():
    return render_template("board.html")

@app.route("/visualization")
def visualization():    
    # Ensure the paths match the saved images
    return render_template("visualization.html")
    

@app.route('/oauth2callback', methods=['POST'])
def oauth2callback():
    try:
        # Get the token from the frontend
        data = request.get_json()
        token = data.get('credential')
        
        # Verify the token with Google
        idinfo = id_token.verify_oauth2_token(token, requests.Request(), GOOGLE_CLIENT_ID)
        
        # Extract user information
        user_name = idinfo.get('name', 'Unknown User')
        user_email = idinfo.get('email')
        
        # Save user info in the session
        session['user'] = {'name': user_name, 'email': user_email}
        
        # Respond with success and the user's name
        return jsonify({'status': 'success', 'name': user_name})
    except ValueError:
        return jsonify({'status': 'error', 'message': 'Invalid token'}), 400

@app.route("/logout")
def logout():
    session.pop('user', None)  # Clear user session
    return redirect(url_for('home'))

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle user input for fake news prediction and fetch real news if it's fake.
    """
    if request.method == "POST":
        # Get news content from form input
        news = request.form.get("news", "")
        
        # Call the manual_testing function in model.py
        result = model.manual_testing(news)  # Use your model logic
        
        print(result)  # Debugging line to check the structure of result
        
        # Send the prediction result and articles to the frontend
        return render_template("results.html", news=news, prediction=result["label"], articles=result["articles"])

if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0')
