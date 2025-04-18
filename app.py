
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="نظام التداول الذكي", layout="wide")
st.title("نظام التداول الذكي - Smart Trading AI")

log_file = "trade_performance.csv"
if os.path.exists(log_file):
    trade_log = pd.read_csv(log_file)
else:
    trade_log = pd.DataFrame(columns=["Coin", "Type", "Price", "P/L %", "Balance"])

tabs = st.tabs(["لوحة الأداء", "تحليل السوق", "تشغيل النموذج", "سجل الصفقات"])

with tabs[0]:
    st.subheader("لوحة الأداء")
    if not trade_log.empty:
        latest_balance = trade_log['Balance'].iloc[-1]
        total_trades = len(trade_log)
        win_rate = (trade_log['Type'] == 'Profit').sum() / total_trades * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("الرصيد الحالي", f"${latest_balance:,.2f}")
        col2.metric("عدد الصفقات", total_trades)
        col3.metric("نسبة النجاح", f"{win_rate:.2f} %")
    else:
        st.info("لا توجد بيانات صفقات حالياً. يرجى تشغيل النموذج أولاً.")

with tabs[1]:
    st.subheader("تحليل السوق الفني")
    st.write("يعرض المؤشرات الفنية وإشارات الدخول الذكية (مثال توضيحي).")
    if os.path.exists("market_data.csv"):
        df = pd.read_csv("market_data.csv")
        st.line_chart(df['Close'])
        st.dataframe(df[df['Smart_Entry'] == 1].tail(10))
    else:
        st.warning("لا توجد بيانات للسوق. يرجى تشغيل النموذج أولاً.")

with tabs[2]:
    st.subheader("تشغيل النموذج الذكي")
    if st.button("ابدأ التحليل والتداول"):
        with st.spinner("جاري تنفيذ النموذج..."):
            from trading_model import trade_strategy
            trade_strategy()
        st.success("تم تنفيذ النموذج بنجاح.")

with tabs[3]:
    st.subheader("سجل الصفقات")
    if not trade_log.empty:
        st.dataframe(trade_log.tail(20))
        st.download_button("تحميل السجل بصيغة CSV", trade_log.to_csv(index=False), "trade_log.csv", "text/csv")
    else:
        st.info("لا توجد صفقات مسجلة بعد.")
