import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error

# إنشاء بيانات وهمية
np.random.seed(42)
date_range = pd.date_range(start='2018-01-01', end='2025-12-31', freq='D')
data = pd.DataFrame({
    'ds': date_range,
    'y': 3000 + 500*np.sin(2*np.pi*date_range.dayofyear/365) + np.random.normal(0, 100, len(date_range))
})

# واجهة Streamlit
st.title("برنامج التنبؤ بالأحمال الكهربائية في العراق")
st.write("البيانات الأولية لاستهلاك الكهرباء:")
st.line_chart(data.set_index('ds')['y'])

# تقسيم البيانات للتدريب والاختبار
train = data[data['ds'] < '2025-01-01']
test = data[data['ds'] >= '2025-01-01']

# إنشاء نموذج Prophet
model = Prophet(yearly_seasonality=True, daily_seasonality=False)
model.fit(train)

# التنبؤ بالمستقبل
future = model.make_future_dataframe(periods=len(test), freq='D')
forecast = model.predict(future)

# تقييم النموذج
pred = forecast['yhat'][-len(test):].values
mae = mean_absolute_error(test['y'], pred)
rmse = np.sqrt(mean_squared_error(test['y'], pred))

st.write(f"MAE (Mean Absolute Error): {mae:.2f}")
st.write(f"RMSE (Root Mean Squared Error): {rmse:.2f}")

# رسم النتائج
st.write("رسم استهلاك الكهرباء الفعلي مقابل التنبؤ:")
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(train['ds'], train['y'], label='البيانات التاريخية')
ax.plot(test['ds'], test['y'], label='البيانات الفعلية (اختبار)')
ax.plot(test['ds'], pred, label='التنبؤ', linestyle='--')
ax.set_xlabel("التاريخ")
ax.set_ylabel("استهلاك الكهرباء (ميغاواط)")
ax.legend()
st.pyplot(fig)

# جدول التنبؤ بالأحمال القادمة
st.write("جدول التنبؤ:")
st.dataframe(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(30))
