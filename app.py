import requests
import pandas as pd
import ta
import time

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

# إعداد المتغيرات الأساسية
capital = 1000             # رأس المال بالدولار
position_size = 0.5        # نسبة الدخول في الصفقة (50%)
stop_loss_percent = 5      # نسبة الخسارة المسموح بها
take_profit_percent = 30   # نسبة الربح المطلوبة
in_trade = False           # حالة الصفقة: غير مفعلة بالبداية
entry_price = 0            # سعر الدخول المبدئي
trade_direction = ""       # اتجاه الصفقة: شراء أو بيع

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
def main():
    # تحميل بيانات عملة ADA وتجهيزها
    data_ada = get_coin_data('cardano', vs_currency='usd', days='30')
    data_ada = add_technical_indicators(data_ada)

    # تنفيذ الاستراتيجية
    trade_strategy_v2(data_ada, 'cardano')

# تنفيذ البرنامج
if __name__ == "__main__":
    main()
