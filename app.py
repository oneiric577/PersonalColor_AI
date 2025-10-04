from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import uuid
import numpy as np
from PIL import Image
import webbrowser
from threading import Timer
from retrain import retrain_model
import pandas as pd
from datetime import datetime
import datetime as dt
from tensorflow.keras.models import load_model  # ✅ 모델 로딩은 요청 시

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "default_secret")

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# DB 초기화
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route("/")
def main():
    if 'username' in session:
        return redirect(url_for('admin' if session.get('is_admin') else 'home'))
    return render_template('main.html')

@app.route("/home")
def home():
    if 'username' not in session or session.get('is_admin'):
        return redirect(url_for('login'))
    return render_template('home.html', username=session['username'])

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("SELECT password, is_admin FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        conn.close()
        if result and check_password_hash(result[0], password):
            session.update({"username": username, "is_admin": result[1]})
            return redirect(url_for('admin' if result[1] else 'home'))
        return "로그인 실패. 다시 시도해 주세요."
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            return "이미 존재하는 사용자입니다."
        finally:
            conn.close()
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("main"))

@app.route("/upload", methods=["POST"])
def upload():
    if "username" not in session or session.get('is_admin'):
        return redirect(url_for("login"))

    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for("home"))

    filename = secure_filename(f"{uuid.uuid4().hex}_{file.filename}")
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # ✅ 모델을 매번 로드 (최신 반영)
    model = load_model('personal_color_model.h5')

    # 이미지 전처리
    img = Image.open(filepath).convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # 예측
    prediction = model.predict(img_array)
    labels = ['spring_warm', 'summer_cool', 'autumn_warm', 'winter_cool']
    result = labels[np.argmax(prediction)]
    confidence = round(float(np.max(prediction)) * 100, 2)

    # 진단 결과에 따른 추천 멘트
    recommend_dict = {
        'spring_warm': '상큼하고 밝은 분위기를 더해주는 산뜻한 파스텔톤과 웜톤 컬러가 잘 어울립니다. 코랄, 피치, 라임, 누디 베이지, 밝은 카키 등 웜하고 투명한 색상의 옷을 선택해보세요.',
        'summer_cool': '차분하고 우아한 무드가 느껴지는 쿨톤 파스텔과 연한 컬러가 멋스럽습니다. 라벤더, 바이올렛, 분홍, 연한 블루, 머스타드, 뮤트 그레이 등 부드럽고 시원한 색상의 옷을 추천합니다',
        'autumn_warm': '고급스럽고 따뜻한 분위기를 주는 어스톤과 웜톤 컬러가 잘 어울립니다. 머스타드, 올리브, 벽돌 레드, 테라코타, 연한 브라운, 황금빛 골드 등 포근하고 깊은 색감의 옷을 선택해보세요.',
        'winter_cool': '강렬하고 세련된 느낌을 주는 쿨톤의 선명한 컬러가 멋집니다. 네온, 아이스 블루, 진한 레드, 에메랄드, 블랙, 화이트 등 선명하고 뚜렷한 색상의 옷을 추천합니다.'
    }
    recommend_text = recommend_dict.get(result, '')

    return render_template("result.html", username=session["username"], result=result,
                           confidence=confidence, image_url=url_for('static', filename='uploads/' + filename),
                           filename=filename, predicted=result, recommend_text=recommend_text)

@app.route("/feedback", methods=["POST"])
def feedback():
    if "username" not in session:
        return redirect(url_for("login"))

    # 데이터만 세션에 저장
    session["feedback_data"] = {
        "filename": request.form["filename"],
        "correct": request.form["correct"],
        "predicted": request.form["predicted"]
    }

    # CSV 파일이 없으면 생성
    if not os.path.exists("retrain_data.csv"):
        with open("retrain_data.csv", "w", encoding="utf-8") as f:
            f.write("filename,spring_warm,summer_cool,autumn_warm,winter_cool\n")

    return redirect(url_for("loading"))

@app.route("/admin")
def admin():
    if 'username' not in session or session.get('is_admin') != 1:
        return redirect(url_for('login'))
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, username FROM users WHERE is_admin = 0")
    users = cursor.fetchall()
    conn.close()
    return render_template('admin.html', users=users)

@app.route("/delete_users", methods=["POST"])
def delete_users():
    if 'username' not in session or session.get('is_admin') != 1:
        return redirect(url_for('login'))
    user_ids = request.form.getlist('user_ids')
    if user_ids:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        query = f"DELETE FROM users WHERE id IN ({','.join(['?']*len(user_ids))})"
        cursor.execute(query, user_ids)
        conn.commit()
        conn.close()
    return redirect(url_for('admin'))

@app.route("/loading")
def loading():
    if "feedback_data" not in session:
        return redirect(url_for("home"))
    return render_template("loading.html", feedback_data=session["feedback_data"])

@app.route("/retrain_async", methods=["POST"])
def retrain_async():
    try:
        if "feedback_data" not in session:
            print("Error: No feedback data in session")
            return "error: no feedback data"

        # 세션에서 피드백 데이터 가져오기
        feedback_data = session["feedback_data"]
        print(f"Feedback data: {feedback_data}")  # 디버깅용 로그

        filename = feedback_data["filename"]
        selected_label = feedback_data["correct"]
        predicted = feedback_data["predicted"]

        # 파일 존재 여부 확인
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if not os.path.exists(filepath):
            print(f"Error: Image file not found: {filepath}")
            return "error: image file not found"

        LABELS = ['spring_warm', 'summer_cool', 'autumn_warm', 'winter_cool']
        one_hot = [1 if label == selected_label else 0 for label in LABELS]

        # CSV 파일이 없으면 생성
        if not os.path.exists("retrain_data.csv"):
            with open("retrain_data.csv", "w", encoding="utf-8") as f:
                f.write("filename,spring_warm,summer_cool,autumn_warm,winter_cool\n")

        # 데이터 저장 (filename과 원-핫 인코딩만 저장)
        with open("retrain_data.csv", "a", encoding="utf-8") as f:
            f.write(f"{filename},{','.join(map(str, one_hot))}\n")
        
        print("Data saved to CSV successfully")  # 디버깅용 로그

        # 재학습 실행
        try:
            retrain_model()
            print("Model retraining completed successfully")  # 디버깅용 로그
        except Exception as e:
            print(f"Error during model retraining: {str(e)}")
            return f"error: model retraining failed - {str(e)}"
        
        # 세션에서 피드백 데이터 삭제
        session.pop("feedback_data", None)
        
        return "success"
    except Exception as e:
        print(f"Unexpected error in retrain_async: {str(e)}")
        return f"error: {str(e)}"

if __name__ == "__main__":
    port = 5000
    url = f"http://localhost:{port}"
    Timer(1, lambda: webbrowser.open(url)).start()
    app.run(port=port, debug=True)
