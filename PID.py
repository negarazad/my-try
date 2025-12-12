import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# تنظیمات اولیه
# -------------------------------
NUM_CHANNELS = 3
TIME_STEPS = 200

# مقدار هدف SINR (مقدار مطلوب)
SINR_TARGET = 15  

# ضرایب PID
Kp = 0.8
Ki = 0.02
Kd = 0.3

# -------------------------------
# توابع مدل
# -------------------------------

def generate_interference(num_channels, T):
    """ تولید تداخل تصادفی زمان‌متغیر برای هر کانال """
    base = np.random.uniform(5, 25, num_channels)
    noise = np.random.normal(0, 2, (T, num_channels))
    interference = base + np.cumsum(noise, axis=0)
    return np.abs(interference)

def calculate_sinr(interference, power=30):
    """ محاسبه SINR ساده """
    return power / (interference + 1e-6)

# -------------------------------
# شبیه‌سازی PID + تخصیص کانال
# -------------------------------

interference = generate_interference(NUM_CHANNELS, TIME_STEPS)

integral = 0
prev_error = 0

selected_channel_history = []
sinr_history = []
pid_output_history = []

current_channel = 0

for t in range(TIME_STEPS):

    # SINR کانال فعلی
    sinr_t = calculate_sinr(interference[t, current_channel])
    
    # خطا نسبت به مقدار مطلوب
    error = SINR_TARGET - sinr_t

    # اجزای PID
    integral += error
    derivative = error - prev_error
    prev_error = error

    # خروجی PID
    u = Kp*error + Ki*integral + Kd*derivative

    # ذخیره جهت رسم
    pid_output_history.append(u)
    sinr_history.append(sinr_t)
    selected_channel_history.append(current_channel)

    # -----------------------------------------
    # تصمیم‌گیری تخصیص کانال:
    # اگر خروجی PID زیاد باشد → کانال را عوض کن
    # -----------------------------------------
    if abs(u) > 5:
        # کانالی پیدا کن که SINR بهترین باشد
        sinrs = calculate_sinr(interference[t, :])
        current_channel = np.argmax(sinrs)

# -------------------------------
# رسم نتایج
# -------------------------------
plt.figure(figsize=(12,6))
plt.plot(sinr_history, label="SINR")
plt.axhline(SINR_TARGET, color='r', linestyle='--', label="Target")
plt.legend()
plt.title("SINR Tracking with PID-based Channel Allocation")
plt.xlabel("Time")
plt.ylabel("SINR")
plt.show()

plt.figure(figsize=(12,4))
plt.plot(selected_channel_history)
plt.title("Selected Channel Over Time")
plt.xlabel("Time")
plt.ylabel("Channel Index")
plt.show()
