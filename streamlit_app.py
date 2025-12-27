import streamlit as st
import yfinance as yf
import pandas as pd
import twstock
import time
from datetime import datetime
import plotly.graph_objects as go
import os
import uuid
import csv

# --- 1. 網頁設定 ---
# 更新邏輯：修正顏色渲染報錯問題，版本維持 1.6，後綴修正為 c
VER = "ver 1.6c (Industry + Stability Fix)"
st.set_page_config(page_title=f"✨ 黑嚕嚕-旗鼓相當({VER})", layout="wide")

# --- 流量紀錄與後台功能 ---
LOG_FILE = "traffic_log.csv"

def get_remote_ip():
    """取得使用者 IP"""
    try:
        from streamlit.web.server.websocket_headers import _get_websocket_headers
        headers = _get_websocket_headers()
        if headers and "X-Forwarded-For" in headers:
            return headers["X-Forwarded-For"].split(",")[0]
    except:
        pass
    return "Unknown/Local"

def log_traffic():
    """流量紀錄"""
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())[:8]
        st.session_state["has_logged"] = False
    if not st.session_state["has_logged"]:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_ip = get_remote_ip()
        session_id = st.session_state["session_id"]
        file_exists = os.path.exists(LOG_FILE)
        with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["時間", "IP位址", "Session_ID", "頁面動作"])
            writer.writerow([current_time, user_ip, session_id, "進入首頁"])
        st.session_state["has_logged"] = True

log_traffic()

# --- 2. 核心功能區 ---

@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_list():
    """取得台股清單並保留產業別"""
    try:
        tse = twstock.twse
        otc = twstock.tpex
        stock_dict = {}
        exclude_industries = ["金融保險業", "存託憑證", "ETF", "受益證券"]
        for code, info in tse.items():
            if info.type == "股票" and info.group not in exclude_industries:
                stock_dict[f"{code}.TW"] = {"name": info.name, "code": code, "group": info.group}
        for code, info in otc.items():
            if info.type == "股票" and info.group not in exclude_industries:
                stock_dict[f"{code}.TWO"] = {"name": info.name, "code": code, "group": info.group}
        return stock_dict
    except:
        return {}

def calculate_kd_series(df, n=9):
    """計算 KD 指標序列"""
    low_min = df["Low"].rolling(window=n).min()
    high_max = df["High"].rolling(window=n).max()
    rsv = (df["Close"] - low_min) / (high_max - low_min) * 100
    rsv = rsv.fillna(50)
    k, d = 50, 50
    k_list, d_list = [], []
    for r in rsv:
        k = (2/3) * k + (1/3) * r
        d = (2/3) * d + (1/3) * k
        k_list.append(k)
        d_list.append(d)
    return pd.Series(k_list, index=df.index), pd.Series(d_list, index=df.index)

def detect_leg_kick_signal(stock_df, lookback=60, trigger_days=5, kd_threshold=20):
    """打腳發動核心邏輯"""
    if len(stock_df) < 30: return False, None
    recent_df = stock_df.tail(lookback).copy()
    k_series, d_series = calculate_kd_series(recent_df)
    t1 = k_series[k_series < kd_threshold].last_valid_index()
    if t1 is None: return False, None
    oversold_close = float(recent_df.loc[t1, "Close"])
    idx_list = list(recent_df.index)
    t1_pos = idx_list.index(t1)
    t_cross = None
    for i in range(t1_pos + 1, len(idx_list)):
        dt, prev_dt = idx_list[i], idx_list[i-1]
        if (k_series.loc[prev_dt] <= d_series.loc[prev_dt]) and (k_series.loc[dt] > d_series.loc[dt]):
            t_cross = dt
            break
    if t_cross is None: return False, None
    cross_pos = idx_list.index(t_cross)
    end_pos = min(cross_pos + trigger_days, len(idx_list) - 1)
    for i in range(cross_pos, end_pos + 1):
        dt = idx_list[i]
        if i == 0: continue
        curr = recent_df.iloc[i]; prev = recent_df.iloc[i-1]
        # 紅吞黑條件
        red_engulf = (prev["Close"] < prev["Open"]) and (curr["Close"] > curr["Open"]) and (curr["Open"] < prev["Close"]) and (curr["Close"] > prev["Open"])
        if red_engulf and (curr["Close"] > oversold_close) and (k_series.loc[dt] >= kd_threshold):
            return True, dt
    return False, None

def run_strategy_backtest(stock_dict, progress_bar, mode, min_vol_threshold):
    """策略回測模組"""
    results = []
    all_tickers = list(stock_dict.keys())
    BATCH_SIZE = 50
    total_batches = (len(all_tickers) // BATCH_SIZE) + 1
    for i, batch_idx in enumerate(range(0, len(all_tickers), BATCH_SIZE)):
        batch = all_tickers[batch_idx : batch_idx + BATCH_SIZE]
        try:
            data = yf.download(batch, period="1y", progress=False, auto_adjust=False)
            if data.empty: continue
            df_c = data["Close"]
            if isinstance(df_c, pd.Series): df_c = df_c.to_frame(name=batch[0])
            scan_window = df_c.index[-60:]
            for ticker in df_c.columns:
                try:
                    c_series = df_c[ticker].dropna()
                    if len(c_series) < 60: continue
                    stock_info = stock_dict.get(ticker, {})
                    full_ohlc = pd.DataFrame({"Open":data["Open"][ticker],"High":data["High"][ticker],"Low":data["Low"][ticker],"Close":data["Close"][ticker]}).dropna()
                    for date in scan_window:
                        if date not in c_series.index: continue
                        idx = c_series.index.get_loc(date)
                        vol = float(data["Volume"][ticker].iloc[idx])
                        if vol < (min_vol_threshold * 1000): continue
                        is_match = False
                        if mode == "🦵 打腳發動":
                            ok, t_dt = detect_leg_kick_signal(full_ohlc.loc[:date])
                            if ok and t_dt == date: is_match = True
                        elif mode == "🐎 多頭馬車":
                            m30 = c_series.rolling(30).mean().iloc[idx]
                            m60 = c_series.rolling(60).mean().iloc[idx]
                            if c_series.iloc[idx] > m30 > m60: is_match = True
                        if is_match:
                            results.append({
                                "月份": date.strftime("%m月"), "代號": ticker.split('.')[0], "名稱": stock_info.get("name"),
                                "產業": stock_info.get("group", "其他"), "訊號日期": date.strftime("%Y-%m-%d"),
                                "訊號價": round(c_series.iloc[idx], 2), "結果": "已驗證"
                            })
                            break
                except: continue
        except: pass
        progress_bar.progress((i+1)/total_batches)
    return pd.DataFrame(results)

def fetch_all_data(stock_dict, progress_bar):
    """即時下載最新股價數據"""
    all_tickers = list(stock_dict.keys())
    BATCH_SIZE = 50
    total_batches = (len(all_tickers) // BATCH_SIZE) + 1
    raw_data = []
    for i, batch_idx in enumerate(range(0, len(all_tickers), BATCH_SIZE)):
        batch = all_tickers[batch_idx : batch_idx + BATCH_SIZE]
        try:
            data = yf.download(batch, period="1y", progress=False, auto_adjust=False)
            df_c = data["Close"]
            for ticker in df_c.columns:
                try:
                    p = float(df_c[ticker].iloc[-1])
                    m200 = df_c[ticker].rolling(200).mean().iloc[-1]
                    info = stock_dict[ticker]
                    stock_df = pd.DataFrame({"Open":data["Open"][ticker],"High":data["High"][ticker],"Low":data["Low"][ticker],"Close":df_c[ticker]}).dropna()
                    leg_kick, _ = detect_leg_kick_signal(stock_df)
                    raw_data.append({
                        "代號": info["code"], "名稱": info["name"], "產業": info["group"], "完整代號": ticker,
                        "收盤價": p, "生命線": round(m200,2) if not pd.isna(m200) else 0,
                        "乖離率(%)": round((p-m200)/m200*100,2) if not pd.isna(m200) and m200!=0 else 0,
                        "成交量": int(data["Volume"][ticker].iloc[-1]), "打腳發動": leg_kick, 
                        "皇冠特選": (p > df_c[ticker].rolling(30).mean().iloc[-1] > df_c[ticker].rolling(60).mean().iloc[-1])
                    })
                except: continue
        except: pass
        progress_bar.progress((i+1)/total_batches)
    return pd.DataFrame(raw_data)

# --- 3. 介面區 ---
st.title(f"✨ {VER} 黑嚕嚕-旗鼓相當")

if "master_df" not in st.session_state:
    st.session_state["master_df"] = None
if "backtest_result" not in st.session_state:
    st.session_state["backtest_result"] = None

with st.sidebar:
    st.header("功能選單")
    CACHE_FILE = "stock_cache_v16.csv"
    if st.button("🔄 下載最新股價", type="primary"):
        stock_dict = get_stock_list()
        pb = st.progress(0, text="正在同步市場數據...")
        df_new = fetch_all_data(stock_dict, pb)
        if not df_new.empty:
            df_new.to_csv(CACHE_FILE, index=False)
            st.session_state["master_df"] = df_new
            st.rerun()
    
    if os.path.exists(CACHE_FILE) and st.session_state["master_df"] is None:
        st.session_state["master_df"] = pd.read_csv(CACHE_FILE)

    strategy_mode = st.radio("選擇策略", ["🛡️ 生命線保衛戰", "🐎 多頭馬車", "🦵 打腳發動"])
    min_vol = st.number_input("最低成交量(張)", 500, 10000, 1000)
    
    if st.button("🧪 執行策略回測"):
        stock_dict = get_stock_list()
        pb_bt = st.progress(0, text="正在驗證歷史訊號...")
        bt_df = run_strategy_backtest(stock_dict, pb_bt, strategy_mode, min_vol)
        st.session_state["backtest_result"] = bt_df

# 主畫面
if st.session_state["master_df"] is None:
    st.warning("👈 請先點擊左側 sidebar 下載最新股價開始挖掘標的。")
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        if os.path.exists("welcome.jpg"):
            st.image("welcome.jpg", use_container_width=True)
            st.markdown("<p style='text-align:center; font-size:1.2em;'>預祝心想事成，從從容容，紫氣東來! 🟣✨</p>", unsafe_allow_html=True)
else:
    df_res = st.session_state["master_df"].copy()
    df_res = df_res[df_res["成交量"] >= min_vol*1000]
    
    if strategy_mode == "🦵 打腳發動": df_res = df_res[df_res["打腳發動"]==True]
    elif strategy_mode == "🐎 多頭馬車": df_res = df_res[df_res["皇冠特選"]==True]
    
    st.subheader(f"🔍 今日篩選結果 ({strategy_mode}) - 找到 {len(df_res)} 檔")
    if not df_res.empty:
        df_res["成交量(張)"] = (df_res["成交量"]/1000).astype(int)
        show_cols = ["代號", "名稱", "產業", "收盤價", "生命線", "乖離率(%)", "成交量(張)"]
        
        # --- 核心修正：手動顏色渲染 (正紅負綠)，不調用 matplotlib ---
        def color_style(val):
            color = '#ff4b4b' if val > 0 else '#008000'
            return f'color: {color}; font-weight: bold'

        st.dataframe(df_res[show_cols].style.map(color_style, subset=["乖離率(%)"]), use_container_width=True, hide_index=True)
        
        sel_stock = st.selectbox("查看詳細指示器", df_res["名稱"].tolist())
        row_data = df_res[df_res["名稱"]==sel_stock].iloc[0]
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=row_data['收盤價'],
            title={'text': f"{row_data['名稱']} ({row_data['產業']})"},
            gauge={'axis': {'range': [None, row_data['收盤價']*1.2]}, 'bar': {'color': "#ff4b4b"}}
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)
    else:
        st.info("今日盤面沒有符合此策略的標的，換個策略試試看？")

if st.session_state["backtest_result"] is not None:
    st.divider()
    st.subheader("🧪 策略歷史回測報告")
    st.dataframe(st.session_state["backtest_result"], use_container_width=True, hide_index=True)
