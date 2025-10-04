import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from datetime import datetime, timedelta

MODEL_PATH = "personal_color_model.h5"
CSV_PATH = "retrain_data.csv"
UPLOAD_DIR = "static/uploads/"
LABELS = ['spring_warm', 'summer_cool', 'autumn_warm', 'winter_cool']
MAX_TRAINING_DATA = 100  # 최대 학습 데이터 수
MIN_CONFIDENCE = 0.7     # 최소 신뢰도

def clean_old_data():
    """30일 이상 된 데이터 삭제"""
    if not os.path.exists(CSV_PATH):
        return
    
    current_time = datetime.now()
    cleaned_lines = []
    
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:  # filename, timestamp, 4 labels
                try:
                    timestamp = datetime.fromisoformat(parts[1])
                    if (current_time - timestamp) <= timedelta(days=30):
                        cleaned_lines.append(line)
                except:
                    continue
    
    with open(CSV_PATH, "w", encoding="utf-8") as f:
        f.writelines(cleaned_lines)

def retrain_model():
    try:
        if not os.path.exists(CSV_PATH):
            print("📂 retrain_data.csv 파일이 없습니다.")
            return

        # 데이터 읽기
        df = pd.read_csv(CSV_PATH)
        print(f"Loaded {len(df)} rows from CSV")  # 디버깅용 로그
        
        # 최근 MAX_TRAINING_DATA 개의 데이터만 사용
        df = df.tail(MAX_TRAINING_DATA)
        print(f"Using {len(df)} rows for training")  # 디버깅용 로그

        images, labels = [], []
        valid_count = 0

        for _, row in df.iterrows():
            img_path = os.path.join(UPLOAD_DIR, row['filename'])

            if not os.path.exists(img_path):
                print(f"❌ 이미지 파일 없음: {img_path}")
                continue

            try:
                img = load_img(img_path, target_size=(224, 224))
                img = img_to_array(img) / 255.0
                images.append(img)

                # 원-핫 인코딩 값만 사용
                label_vector = row[LABELS].values.astype(np.float32)
                labels.append(label_vector)
                valid_count += 1
                print(f"Processed image {valid_count}: {img_path} with label {LABELS[np.argmax(label_vector)]}")
            except Exception as e:
                print(f"⚠️ 이미지 처리 오류 ({img_path}): {e}")

        if valid_count < 5:
            print("🚫 유효한 재학습 데이터가 부족합니다.")
            return

        print(f"Prepared {valid_count} valid images for training")  # 디버깅용 로그

        X = np.array(images)
        y = np.array(labels)

        try:
            model = load_model(MODEL_PATH)
            print("Model loaded successfully")  # 디버깅용 로그
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            return

        # 학습률을 더 낮춰서 미세 조정
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),  # 학습률 더 낮춤
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

        print("🟡 모델 재학습 중...")
        # 더 많은 에포크와 작은 배치 사이즈로 학습
        history = model.fit(X, y, 
                          epochs=20,  # 에포크 수 증가
                          batch_size=2,  # 배치 사이즈 감소
                          validation_split=0.2,
                          verbose=1)  # 학습 진행상황 상세 출력
        
        # 학습 결과 출력
        print("\n=== 학습 결과 ===")
        print(f"최종 정확도: {history.history['accuracy'][-1]:.4f}")
        print(f"최종 검증 정확도: {history.history['val_accuracy'][-1]:.4f}")
        print(f"최종 손실: {history.history['loss'][-1]:.4f}")
        print(f"최종 검증 손실: {history.history['val_loss'][-1]:.4f}")
        
        model.save(MODEL_PATH)
        print("✅ 모델이 성공적으로 재학습되었습니다.")
    except Exception as e:
        print(f"Unexpected error in retrain_model: {str(e)}")
        raise  # 에러를 상위로 전파

if __name__ == "__main__":
    retrain_model()
