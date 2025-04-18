#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install ta')


# In[ ]:


# استيراد المكتبات اللازمة
import pandas as pd
import ta

# تحميل البيانات (مثال: من ملف CSV أو API مثل Binance أو CoinGecko)
# على سبيل المثال: data = pd.read_csv("path_to_your_data.csv")

# إذا كنت تستخدم بيانات تجريبية كمثال
data = pd.DataFrame({
    'close': [100, 102, 104, 103, 105, 107, 110, 109, 108, 110]
})

# حساب بعض المؤشرات الفنية باستخدام مكتبة ta
data['RSI'] = ta.momentum.RSIIndicator(data['close'], window=14).rsi()
data['SMA'] = ta.trend.sma_indicator(data['close'], window=50)
data['EMA'] = ta.trend.ema_indicator(data['close'], window=50)

# عرض البيانات بعد إضافة المؤشرات
print(data)


# In[ ]:


# استيراد المكتبات اللازمة
import pandas as pd
import ta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# تحميل البيانات (يمكنك استبدال هذا بمصدر بياناتك)
# على سبيل المثال: data = pd.read_csv("path_to_your_data.csv")
data = pd.DataFrame({
    'close': [100, 102, 104, 103, 105, 107, 110, 109, 108, 110]
})

# حساب بعض المؤشرات الفنية باستخدام مكتبة ta
data['RSI'] = ta.momentum.RSIIndicator(data['close'], window=14).rsi()
data['SMA'] = ta.trend.sma_indicator(data['close'], window=50)
data['EMA'] = ta.trend.ema_indicator(data['close'], window=50)

# إضافة العمود الذي يعبر عن التغير المستقبلي في السعر (Future Change)
data['Future_Price'] = data['close'].shift(-1)
data['Future_Change'] = (data['Future_Price'] - data['close']) / data['close']
data['Label'] = data['Future_Change'].apply(lambda x: 1 if x > 0 else 0)

# تقسيم البيانات إلى مجموعة تدريب واختبار
X = data[['RSI', 'SMA', 'EMA']]  # المؤشرات الفنية كميزات
y = data['Label']  # الهدف (شراء أو بيع)

# تقسيم البيانات إلى مجموعة تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب النموذج باستخدام Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# اختبار النموذج
y_pred = model.predict(X_test)

# حساب دقة النموذج
accuracy = accuracy_score(y_test, y_pred)
print(f"دقة النموذج: {accuracy:.2f}")

# التنبؤ بالقيم الجديدة
predictions = model.predict(X)
data['Predictions'] = predictions

# عرض البيانات مع التنبؤات
print(data)


# In[ ]:


# تأكد من أن البيانات موجودة
import pandas as pd

# عرض أسماء الأعمدة
print("أسماء الأعمدة قبل التعديل:", data.columns)

# تحويل جميع أسماء الأعمدة إلى الحروف الصغيرة
data.columns = [col.lower() for col in data.columns]

# تأكد من أن الأعمدة تحتوي على أسماء صحيحة
print("أسماء الأعمدة بعد التعديل:", data.columns)

# إزالة أي قيم مفقودة (NaN) في البيانات
data = data.dropna()

# الآن تأكد من أن البيانات تحتوي على الأعمدة اللازمة
if 'open' not in data.columns or 'close' not in data.columns or 'high' not in data.columns or 'low' not in data.columns:
    print("البيانات لا تحتوي على الأعمدة اللازمة.")
else:
    # حساب بعض المؤشرات الفنية باستخدام مكتبة ta
    data['RSI'] = ta.momentum.RSIIndicator(data['close'], window=14).rsi()
    data['SMA'] = ta.trend.sma_indicator(data['close'], window=50)
    data['EMA'] = ta.trend.ema_indicator(data['close'], window=50)
    data['MACD'] = ta.trend.MACD(data['close']).macd()

    # عرض البيانات لتأكد من إضافة الأعمدة بشكل صحيح
    print(data.head())


# In[ ]:


import pandas as pd
import ta

# تحميل البيانات (تأكد من أنك قد قمت بتحميل البيانات مسبقاً في المتغير 'data')
# data = pd.read_csv('your_data.csv')  # في حالة عدم تحميل البيانات

# تأكد من أن البيانات تحتوي على الأعمدة المطلوبة
print("أسماء الأعمدة في البيانات الأصلية:", data.columns)

# تحويل جميع أسماء الأعمدة إلى الحروف الصغيرة
data.columns = [col.lower() for col in data.columns]

# التحقق من أن الأعمدة المطلوبة موجودة
required_columns = ['open', 'close', 'high', 'low']
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    print(f"الأعمدة المفقودة: {missing_columns}")
else:
    # حساب المؤشرات الفنية باستخدام مكتبة ta
    data['RSI'] = ta.momentum.RSIIndicator(data['close'], window=14).rsi()
    data['SMA'] = ta.trend.sma_indicator(data['close'], window=50)
    data['EMA'] = ta.trend.ema_indicator(data['close'], window=50)

    # طباعة جزء من البيانات للتأكد
    print(data.head())


# In[ ]:


# استيراد المكتبات الضرورية
import requests
import pandas as pd

# تحميل بيانات العملات من CoinGecko
def load_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': '30'  # البيانات لمدة 30 يومًا
    }
    response = requests.get(url, params=params)
    data = response.json()
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# تحميل البيانات
data = load_data()

# عرض أول 5 أسطر من البيانات
print(data.head())


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import ta

# تعريف العملات التي سنعمل عليها
coins = ['ada', 'sol', 'dot', 'shiba-inu']

# دالة لجلب بيانات العملات من CoinGecko
def get_coin_data(coin_id, vs_currency='usd', days='30'):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
    params = {'vs_currency': vs_currency, 'days': days}
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# تحميل البيانات لكل عملة
data_ada = get_coin_data('cardano', vs_currency='usd', days='30')
data_sol = get_coin_data('solana', vs_currency='usd', days='30')
data_dot = get_coin_data('polkadot', vs_currency='usd', days='30')
data_shiba = get_coin_data('shiba-inu', vs_currency='usd', days='30')

# دمج البيانات الخاصة بكل العملات في DataFrame واحد
def add_technical_indicators(df):
    df['RSI'] = ta.momentum.RSIIndicator(df['price'], window=14).rsi()
    df['SMA'] = ta.trend.sma_indicator(df['price'], window=50)
    df['EMA'] = ta.trend.ema_indicator(df['price'], window=50)
    return df

data_ada = add_technical_indicators(data_ada)
data_sol = add_technical_indicators(data_sol)
data_dot = add_technical_indicators(data_dot)
data_shiba = add_technical_indicators(data_shiba)

# رسم البيانات مع المؤشرات الفنية
def plot_coin_data(df, coin_name):
    plt.figure(figsize=(14, 7))
    plt.plot(df['timestamp'], df['price'], label='Price')
    plt.plot(df['timestamp'], df['SMA'], label='SMA (50)', linestyle='--')
    plt.plot(df['timestamp'], df['EMA'], label='EMA (50)', linestyle='-.')
    plt.title(f'{coin_name} Price and Technical Indicators')
    plt.xlabel('Date')
    plt.ylabel('Price in USD')
    plt.legend()
    plt.show()

# رسم البيانات لكل عملة
plot_coin_data(data_ada, 'ADA')
plot_coin_data(data_sol, 'SOL')
plot_coin_data(data_dot, 'DOT')
plot_coin_data(data_shiba, 'SHIBA')

# تحليل الفرص بناءً على RSI
def analyze_opportunity(df):
    if df['RSI'].iloc[-1] > 70:
        print(f"فرصة بيع: {df['RSI'].iloc[-1]} RSI أعلى من 70.")
    elif df['RSI'].iloc[-1] < 30:
        print(f"فرصة شراء: {df['RSI'].iloc[-1]} RSI أقل من 30.")
    else:
        print(f"لا توجد فرصة واضحة حاليًا. RSI: {df['RSI'].iloc[-1]}")

# تحليل الفرص لكل عملة
analyze_opportunity(data_ada)
analyze_opportunity(data_sol)
analyze_opportunity(data_dot)
analyze_opportunity(data_shiba)


# In[ ]:


import random
import time

# تحديد رأس المال الأساسي (مبالغ وهمية للتوضيح)
capital = 1000  # رأس المال بالـ USD
position_size = 0.5  # نسبة الدخول في الصفقة، على سبيل المثال 50% من رأس المال
stop_loss_percent = 5  # نسبة الخسارة المسموح بها، 5%
take_profit_percent = 30  # نسبة الربح المرغوب فيها، 30%

# تحديد حالة الصفقة (افتراضيًا لا توجد صفقة)
in_trade = False
entry_price = 0
trade_direction = ""  # "buy" أو "sell"

# تعريف دالة للمحاكاة
def trade_strategy():
    global capital, in_trade, entry_price, trade_direction

    # استخدام العملة المناسبة وفقاً لـ RSI (المثال هنا على ADA)
    df = data_ada  # يمكن تغييرها للعملات الأخرى بناءً على التحليل

    # إذا كانت الصفقة غير نشطة
    if not in_trade:
        # تحليل الفرص
        if df['RSI'].iloc[-1] < 30:  # فرصة شراء
            print("شراء: RSI أقل من 30")
            entry_price = df['price'].iloc[-1]
            in_trade = True
            trade_direction = "buy"
            print(f"تم الشراء بسعر: {entry_price}")
        elif df['RSI'].iloc[-1] > 70:  # فرصة بيع
            print("بيع: RSI أكبر من 70")
            entry_price = df['price'].iloc[-1]
            in_trade = True
            trade_direction = "sell"
            print(f"تم البيع بسعر: {entry_price}")

    # إذا كانت الصفقة نشطة
    if in_trade:
        current_price = df['price'].iloc[-1]
        print(f"السعر الحالي: {current_price}")

        # حساب الخسارة أو الربح
        if trade_direction == "buy":
            price_change = (current_price - entry_price) / entry_price * 100
        elif trade_direction == "sell":
            price_change = (entry_price - current_price) / entry_price * 100

        # تحقق من الوصول إلى هدف الربح أو الخسارة
        if price_change >= take_profit_percent:
            print(f"تم الوصول إلى هدف الربح بنسبة {price_change}%. تم إغلاق الصفقة.")
            capital += (capital * (position_size * take_profit_percent / 100))  # زيادة رأس المال بالربح
            in_trade = False  # إغلاق الصفقة
        elif price_change <= -stop_loss_percent:
            print(f"تم الوصول إلى حد الخسارة بنسبة {stop_loss_percent}%. تم إغلاق الصفقة.")
            capital -= (capital * (position_size * stop_loss_percent / 100))  # تقليل رأس المال بالخسارة
            in_trade = False  # إغلاق الصفقة

    # انتظار فترة معينة بين الصفقات (مثلاً ساعتين)
    wait_for_next_trade()

# دالة للانتظار بين الصفقات
def wait_for_next_trade():
    print("انتظار 2 ساعة لتقييم السوق...")
    time.sleep(7200)  # الانتظار لمدة ساعتين (7200 ثانية)

# بدء الاستراتيجية
trade_strategy()


# In[ ]:


get_ipython().system('pip install ta')


# In[ ]:


import requests
import pandas as pd
import ta

# دالة تحميل البيانات من CoinGecko
def get_coin_data(coin_id, vs_currency='usd', days='30'):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
    params = {'vs_currency': vs_currency, 'days': days}
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# دالة حساب المؤشرات الفنية
def add_technical_indicators(df):
    df['RSI'] = ta.momentum.RSIIndicator(df['price'], window=14).rsi()
    df['SMA'] = ta.trend.sma_indicator(df['price'], window=50)
    df['EMA'] = ta.trend.ema_indicator(df['price'], window=50)
    return df

# تحميل بيانات عملة ADA وتجهيزها
data_ada = get_coin_data('cardano', vs_currency='usd', days='30')
data_ada = add_technical_indicators(data_ada)


# In[ ]:


# إعداد المتغيرات الأساسية
capital = 1000             # رأس المال بالدولار
position_size = 0.5        # نسبة الدخول في الصفقة (50%)
stop_loss_percent = 5      # نسبة الخسارة المسموح بها
take_profit_percent = 30   # نسبة الربح المطلوبة
in_trade = False           # حالة الصفقة: غير مفعلة بالبداية
entry_price = 0            # سعر الدخول المبدئي
trade_direction = ""       # اتجاه الصفقة: شراء أو بيع


# In[ ]:


# سجل الصفقات
trade_log = []

# استراتيجية التداول الذكية
def trade_strategy_v2(df, coin_name):
    global capital, in_trade, entry_price, trade_direction

    current_price = df['price'].iloc[-1]
    rsi = df['RSI'].iloc[-1]

    print(f"\n--- تحليل {coin_name.upper()} ---")
    print(f"السعر الحالي: {current_price:.2f} USD")
    print(f"RSI الحالي: {rsi:.2f}")

    if not in_trade:
        if rsi < 30:
            print("إشارة شراء")
            entry_price = current_price
            in_trade = True
            trade_direction = "buy"
            print(f"تم الشراء بسعر: {entry_price}")
            trade_log.append({'coin': coin_name, 'type': 'BUY', 'price': entry_price})
        elif rsi > 70:
            print("إشارة بيع")
            entry_price = current_price
            in_trade = True
            trade_direction = "sell"
            print(f"تم البيع بسعر: {entry_price}")
            trade_log.append({'coin': coin_name, 'type': 'SELL', 'price': entry_price})
        else:
            print("لا توجد إشارة دخول حالياً.")
    else:
        change = (current_price - entry_price) / entry_price * 100 if trade_direction == "buy" else (entry_price - current_price) / entry_price * 100
        print(f"التغير الحالي: {change:.2f}%")

        if change >= take_profit_percent:
            print("تحقق الربح! إغلاق الصفقة.")
            in_trade = False
            trade_log.append({'coin': coin_name, 'type': 'TP', 'price': current_price, 'change': change})
        elif change <= -stop_loss_percent:
            print("تحققت الخسارة! إغلاق الصفقة.")
            in_trade = False
            trade_log.append({'coin': coin_name, 'type': 'SL', 'price': current_price, 'change': change})
        else:
            print("الصفقة لا تزال مفتوحة...")

# تشغيل الاستراتيجية على ADA
trade_strategy_v2(data_ada, 'ada')

# طباعة سجل الصفقات
if trade_log:
    print("\n--- سجل الصفقات ---")
    for log in trade_log:
        print(log)


# In[ ]:


get_ipython().system('pip install ta')


# In[ ]:


# تحميل البيانات من CoinGecko
def get_coin_data(coin_id, vs_currency='usd', days='30'):
    import requests
    import pandas as pd
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
    params = {'vs_currency': vs_currency, 'days': days}
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# إضافة المؤشرات الفنية
def add_technical_indicators(df):
    import ta
    df['RSI'] = ta.momentum.RSIIndicator(df['price'], window=14).rsi()
    df['SMA'] = ta.trend.sma_indicator(df['price'], window=50)
    df['EMA'] = ta.trend.ema_indicator(df['price'], window=50)
    return df

# تحميل وتجهيز البيانات لعملة ADA
data_ada = get_coin_data('cardano', vs_currency='usd', days='30')
data_ada = add_technical_indicators(data_ada)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# تأكد إن البيانات فيها مؤشرات فنية محسوبة مسبقًا
df = data_ada.copy()
df = df.dropna()  # نحذف القيم المفقودة

# الهدف: هل السعر سيرتفع بعد الخطوة القادمة؟
df['future_price'] = df['price'].shift(-1)
df['future_change'] = df['future_price'] - df['price']
df['label'] = df['future_change'].apply(lambda x: 1 if x > 0 else 0)

# الميزات (features) والهدف (target)
X = df[['RSI', 'SMA', 'EMA']]
y = df['label']

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب النموذج
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# التقييم
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"دقة النموذج: {accuracy:.2f}")

# التنبؤ على كل البيانات (نقدر نستخدمه لاحقًا في البوت)
df['prediction'] = model.predict(X)

# عرض النتائج
print(df[['price', 'RSI', 'SMA', 'EMA', 'prediction']].tail())


# In[1]:


get_ipython().system('pip install xgboost ta')


# In[2]:


import pandas as pd
import numpy as np
import requests
import ta
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. تحميل بيانات ADA
def get_coin_data(coin_id='cardano', vs_currency='usd', days='30'):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
    params = {'vs_currency': vs_currency, 'days': days}
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# 2. إضافة مؤشرات فنية
def add_indicators(df):
    df['RSI'] = ta.momentum.RSIIndicator(df['price'], window=14).rsi()
    df['SMA'] = ta.trend.sma_indicator(df['price'], window=10)
    df['EMA'] = ta.trend.ema_indicator(df['price'], window=10)
    df['MACD'] = ta.trend.MACD(df['price']).macd()
    return df.dropna()

# 3. إعداد البيانات للتعلم
data = get_coin_data()
data = add_indicators(data)

# إنشاء العمود الهدف: هل السعر راح يرتفع؟
data['future_price'] = data['price'].shift(-1)
data['future_change'] = data['future_price'] - data['price']
data['label'] = data['future_change'].apply(lambda x: 1 if x > 0 else 0)
data = data.dropna()

# الميزات والهدف
features = ['RSI', 'SMA', 'EMA', 'MACD']
X = data[features]
y = data['label']

# 4. تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. تدريب النموذج باستخدام XGBoost
model = XGBClassifier(n_estimators=100, max_depth=3, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# 6. التقييم
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nدقة النموذج (XGBoost): {accuracy:.2f}")

# 7. توقع على كل البيانات
data['prediction'] = model.predict(X)

# 8. عرض آخر التوقعات
print("\nآخر التوقعات:")
print(data[['timestamp', 'price', 'RSI', 'MACD', 'prediction']].tail())


# In[3]:


# استيراد المكتبات
import pandas as pd
import numpy as np
import requests
import ta
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# تحميل البيانات من CoinGecko
def get_coin_data(coin_id='cardano', vs_currency='usd', days='90'):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
    params = {'vs_currency': vs_currency, 'days': days}
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# إضافة مؤشرات فنية
def add_indicators(df):
    bb = ta.volatility.BollingerBands(df['price'])
    df['RSI'] = ta.momentum.RSIIndicator(df['price']).rsi()
    df['SMA'] = ta.trend.sma_indicator(df['price'], window=10)
    df['EMA'] = ta.trend.ema_indicator(df['price'], window=10)
    df['MACD'] = ta.trend.MACD(df['price']).macd()
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    return df.dropna()

# تجهيز البيانات
data = get_coin_data()
data = add_indicators(data)

# إنشاء العمود الهدف للتعلم
data['future_price'] = data['price'].shift(-1)
data['future_change'] = data['future_price'] - data['price']
data['label'] = data['future_change'].apply(lambda x: 1 if x > 0 else 0)
data = data.dropna()

# اختيار الميزات
features = ['RSI', 'SMA', 'EMA', 'MACD', 'BB_upper', 'BB_lower']
X = data[features]
y = data['label']

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب النموذج
model = XGBClassifier(n_estimators=100, max_depth=3, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# تقييم النموذج
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nدقة النموذج: {accuracy:.2f}")

# تطبيق التوقعات على كامل البيانات
data['prediction'] = model.predict(X)

# إعداد المتغيرات الأساسية للتداول
capital = 1000
position_size = 0.5
stop_loss_percent = 5
take_profit_percent = 20
in_trade = False
entry_price = 0
trade_direction = ""
ai_trade_log = []

# استراتيجية التداول بالذكاء الاصطناعي
def ai_trade_strategy(df, coin_name):
    global capital, in_trade, entry_price, trade_direction

    current_row = df.iloc[-1]
    current_price = current_row['price']
    prediction = current_row['prediction']

    print(f"\n--- تحليل AI لعملة {coin_name.upper()} ---")
    print(f"السعر الحالي: {current_price:.3f} USD | توقع: {'شراء' if prediction == 1 else 'تجنّب'}")

    if not in_trade:
        if prediction == 1:
            print("إشارة شراء من الذكاء الاصطناعي")
            entry_price = current_price
            in_trade = True
            trade_direction = "buy"
            ai_trade_log.append({'coin': coin_name, 'type': 'BUY', 'price': entry_price})
    else:
        change = (current_price - entry_price) / entry_price * 100
        print(f"التغير الحالي: {change:.2f}%")

        if change >= take_profit_percent:
            print("تم تحقيق الربح! إغلاق الصفقة.")
            in_trade = False
            ai_trade_log.append({'coin': coin_name, 'type': 'TP', 'price': current_price, 'change': change})
        elif change <= -stop_loss_percent:
            print("تحققت الخسارة! إغلاق الصفقة.")
            in_trade = False
            ai_trade_log.append({'coin': coin_name, 'type': 'SL', 'price': current_price, 'change': change})
        else:
            print("الصفقة ما تزال مفتوحة...")

# تشغيل البوت
ai_trade_strategy(data, 'ADA')

# عرض سجل التداولات
if ai_trade_log:
    print("\n--- سجل التداولات ---")
    for log in ai_trade_log:
        print(log)


# In[4]:


# استيراد المكتبات
import pandas as pd
import numpy as np
import requests
import ta
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# تحميل البيانات من CoinGecko (180 يوم)
def get_coin_data(coin_id='cardano', vs_currency='usd', days='180'):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
    params = {'vs_currency': vs_currency, 'days': days}
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# إضافة مؤشرات فنية
def add_indicators(df):
    bb = ta.volatility.BollingerBands(df['price'])
    df['RSI'] = ta.momentum.RSIIndicator(df['price']).rsi()
    df['SMA'] = ta.trend.sma_indicator(df['price'], window=10)
    df['EMA'] = ta.trend.ema_indicator(df['price'], window=10)
    df['MACD'] = ta.trend.MACD(df['price']).macd()
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    return df.dropna()

# تحميل وتجهيز البيانات
data = get_coin_data()
data = add_indicators(data)

# إنشاء العمود الهدف للتعلم
data['future_price'] = data['price'].shift(-1)
data['future_change'] = data['future_price'] - data['price']
data['label'] = data['future_change'].apply(lambda x: 1 if x > 0 else 0)
data = data.dropna()

# اختيار الميزات (features)
features = ['RSI', 'SMA', 'EMA', 'MACD', 'BB_upper', 'BB_lower']
X = data[features]
y = data['label']

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب نموذج XGBoost
model = XGBClassifier(n_estimators=100, max_depth=3, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# تقييم النموذج
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nدقة النموذج (180 يوم): {accuracy:.2f}")

# التوقعات على كامل البيانات
data['prediction'] = model.predict(X)

# إعداد المتغيرات الأساسية للتداول
capital = 1000
position_size = 0.5
stop_loss_percent = 5
take_profit_percent = 20
in_trade = False
entry_price = 0
trade_direction = ""
ai_trade_log = []

# استراتيجية التداول بالذكاء الاصطناعي
def ai_trade_strategy(df, coin_name):
    global capital, in_trade, entry_price, trade_direction

    current_row = df.iloc[-1]
    current_price = current_row['price']
    prediction = current_row['prediction']

    print(f"\n--- تحليل AI لعملة {coin_name.upper()} ---")
    print(f"السعر الحالي: {current_price:.3f} USD | توقع: {'شراء' if prediction == 1 else 'تجنّب'}")

    if not in_trade:
        if prediction == 1:
            print("إشارة شراء من الذكاء الاصطناعي")
            entry_price = current_price
            in_trade = True
            trade_direction = "buy"
            ai_trade_log.append({'coin': coin_name, 'type': 'BUY', 'price': entry_price})
    else:
        change = (current_price - entry_price) / entry_price * 100
        print(f"التغير الحالي: {change:.2f}%")

        if change >= take_profit_percent:
            print("تم تحقيق الربح! إغلاق الصفقة.")
            in_trade = False
            ai_trade_log.append({'coin': coin_name, 'type': 'TP', 'price': current_price, 'change': change})
        elif change <= -stop_loss_percent:
            print("تحققت الخسارة! إغلاق الصفقة.")
            in_trade = False
            ai_trade_log.append({'coin': coin_name, 'type': 'SL', 'price': current_price, 'change': change})
        else:
            print("الصفقة ما تزال مفتوحة...")

# تشغيل البوت على ADA
ai_trade_strategy(data, 'ADA')

# عرض سجل التداولات
if ai_trade_log:
    print("\n--- سجل التداولات ---")
    for log in ai_trade_log:
        print(log)


# In[5]:


import requests

def get_coins_in_price_range(min_price=0.2, max_price=30, limit=20):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': 100,
        'page': 1
    }
    response = requests.get(url, params=params)
    data = response.json()

    selected = []
    for coin in data:
        price = coin['current_price']
        if min_price <= price <= max_price:
            selected.append({'id': coin['id'], 'symbol': coin['symbol'], 'price': price})
        if len(selected) >= limit:
            break
    return selected

# جرب نطبع أول العملات المناسبة
coins = get_coins_in_price_range()
print("العملات المختارة:")
for c in coins:
    print(f"{c['symbol'].upper()} - {c['price']}$")


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import ta
import time

# إعدادات التداول
capital = 1000
position_size = 0.5
stop_loss_percent = 5
take_profit_percent = 20
in_trade = False
entry_price = 0
trade_direction = ""
ai_trade_log = []

def get_coin_data(coin_id, vs_currency='usd', days='180'):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
    params = {'vs_currency': vs_currency, 'days': days}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return None
    data = response.json()
    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def add_indicators(df):
    bb = ta.volatility.BollingerBands(df['price'])
    df['RSI'] = ta.momentum.RSIIndicator(df['price']).rsi()
    df['SMA'] = ta.trend.sma_indicator(df['price'], window=10)
    df['EMA'] = ta.trend.ema_indicator(df['price'], window=10)
    df['MACD'] = ta.trend.MACD(df['price']).macd()
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    return df.dropna()

def run_ai_bot_on_coin(coin):
    global in_trade, entry_price, trade_direction

    print(f"\n=== العملة: {coin['symbol'].upper()} ({coin['id']}) ===")

    df = get_coin_data(coin['id'])
    if df is None or len(df) < 100:
        print("فشل في تحميل البيانات.")
        return

    df = add_indicators(df)
    df['future_price'] = df['price'].shift(-1)
    df['future_change'] = df['future_price'] - df['price']
    df['label'] = df['future_change'].apply(lambda x: 1 if x > 0 else 0)
    df = df.dropna()

    features = ['RSI', 'SMA', 'EMA', 'MACD', 'BB_upper', 'BB_lower']
    X = df[features]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(n_estimators=100, max_depth=3, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"دقة النموذج: {accuracy:.2f}")

    df['prediction'] = model.predict(X)
    current_row = df.iloc[-1]
    current_price = current_row['price']
    prediction = current_row['prediction']

    print(f"السعر الحالي: {current_price:.3f}$ | توقع: {'شراء' if prediction == 1 else 'تجنّب'}")

    if not in_trade and prediction == 1:
        print(" تم فتح صفقة شراء افتراضية ")
        entry_price = current_price
        in_trade = True
        trade_direction = "buy"
        ai_trade_log.append({'coin': coin['symbol'], 'type': 'BUY', 'price': entry_price})

    elif in_trade:
        change = (current_price - entry_price) / entry_price * 100
        if change >= take_profit_percent:
            print(" تم تحقيق الربح! إغلاق الصفقة. ")
            in_trade = False
            ai_trade_log.append({'coin': coin['symbol'], 'type': 'TP', 'price': current_price, 'change': change})
        elif change <= -stop_loss_percent:
            print(" تحققت الخسارة! إغلاق الصفقة. ")
            in_trade = False
            ai_trade_log.append({'coin': coin['symbol'], 'type': 'SL', 'price': current_price, 'change': change})
        else:
            print(f"الصفقة ما زالت مفتوحة... | التغير: {change:.2f}%")

# تشغيل البوت على كل عملة من القائمة
for coin in coins:
    run_ai_bot_on_coin(coin)
    time.sleep(1)  # تأخير بسيط لتجنب الحظر من CoinGecko

# طباعة سجل التداولات
print("\n--- سجل التداولات ---")
for log in ai_trade_log:
    print(log)


# In[ ]:


import time

# عدد ساعات التشغيل (مثلاً 24 ساعة)
run_hours = 24
delay_minutes = 60  # مدة الانتظار بين كل تشغيل (بالدقائق)

for i in range(run_hours):
    print(f"\n⏱️ تشغيل رقم {i+1} من {run_hours} (الساعة الحالية تقريبًا)")

    for coin in coins:
        run_ai_bot_on_coin(coin)

    print("\n📊 سجل التداول:")
    for log in ai_trade_log:
        print(log)

    if i < run_hours - 1:
        print(f"\n⏳ الانتظار {delay_minutes} دقيقة للتشغيل القادم...")
        time.sleep(delay_minutes * 60)


# In[ ]:


streamlit run app.py


# In[ ]:




