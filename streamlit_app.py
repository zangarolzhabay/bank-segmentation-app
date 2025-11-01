import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Сегментация клиентов", layout="centered")
st.title("🏦 Сегментация клиентов банка")

# Пытаемся загрузить файлы
err = []
model = None
scaler = None
try:
    model = joblib.load("kmeans_model.joblib")
except Exception as e:
    err.append(f"Не удалось загрузить kmeans_model.joblib: {e}")

try:
    scaler = joblib.load("scaler.joblib")
except Exception as e:
    err.append(f"Не удалось загрузить scaler.joblib: {e}")

if err:
    st.error("❌ Проблема с файлами модели.\n\n" + "\n\n".join(err))
    st.info("Положи оба файла рядом со streamlit_app.py и перезапусти: "
            "kmeans_model.joblib и scaler.joblib.")
    st.stop()

st.success("✅ Модель и scaler загружены.")

st.subheader("Ввод признаков")
balance = st.number_input("Баланс на счёте", min_value=0.0, step=100.0)
purchases = st.number_input("Сумма покупок", min_value=0.0, step=100.0)
cash_advance = st.number_input("Снятие наличных", min_value=0.0, step=100.0)
credit_limit = st.number_input("Кредитный лимит", min_value=0.0, step=100.0)
payments = st.number_input("Платежи", min_value=0.0, step=100.0)

if st.button("Определить сегмент"):
    X = np.array([[balance, purchases, cash_advance, credit_limit, payments]])
    X_scaled = scaler.transform(X)
    cluster = int(model.predict(X_scaled)[0])

    if cluster == 0:
        st.success("📌 Сегмент A — Базовый клиент (низкая активность)")
    else:
        st.success("💎 Сегмент B — Премиум клиент (высокая активность)")