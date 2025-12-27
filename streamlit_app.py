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
# Ver 1.9: 基於 1.4a 介面，整合 1.8 的功能修正 (即時篩選+防呆+顏色修復)
VER = "ver 1.9 (1.4 UI + 1.8 Core Features)"
st.set_page_config(page_title=f"✨ 黑嚕嚕-旗鼓相當({VER})", layout="wide")

# --- 流量紀錄與後台功能 ---
LOG_FILE = "traffic_log.csv"

def get_remote_ip():
    try:
        from streamlit.web.server.websocket_headers import _get_websocket_headers
        headers = _get_websocket_headers()
        if headers and "X-Forwarded-For" in headers:
            return headers["X-Forwarded-For"].split(",")[0]
    except:
        pass
    return "Unknown/Local"

def log_traffic():
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

def calculate_kd_values(df, n=9):
    try:
        low_min = df["Low"].rolling(window=n).min()
        high_max = df["High"].rolling(window=n).max()
        rsv = (df["Close"] - low_min) / (high_max - low_min) * 100
        rsv = rsv.fillna(50)
        k, d = 50, 50
        for r in rsv:
            k = (2/3) * k + (1/3) * r
            d = (2/3) * d + (1/3) * k
        return k, d
    except:
        return 50, 50

def detect_leg_kick_signal(stock_df, max_lookback=100, trigger_days=5, kd_threshold=20):
    """
    打腳偵測 (支援即時篩選)
    return: (是否觸發, 觸發日期, K<20發生在幾天前)
    """
    if len(stock_df) < 30: return False, None, 999
    
    # 取較大範圍以便計算 days_ago
    recent_df = stock_df.tail(max_lookback).copy()
    if len(recent_df) < 20: return False, None, 999

    k_series, d_series = calculate_kd_series(recent_df)
    
    # 1. 找最後一次 K < 20
    t1 = k_series[k_series < kd_threshold].last_valid_index()
    if t1 is None: return False, None, 999
    oversold_close = float(recent_df.loc[t1, "Close"])
    
    # 計算距離今天幾天 (用於 Slider 篩選)
    last_date = stock_df.index[-1]
    days_since_k20 = (last_date - t1).days

    idx_list = list(recent_df.index)
    try: t1_pos = idx_list.index(t1)
    except: return False, None, 999

    # 2. 找之後的金叉
    t_cross = None
    for i in range(t1_pos + 1, len(idx_list)):
        dt, prev_dt = idx_list[i], idx_list[i-1]
        if (k_series.loc[prev_dt] <= d_series.loc[prev_dt]) and (k_series.loc[dt] > d_series.loc[dt]):
            t_cross = dt
            break
    if t_cross is None: return False, None, 999
    
    # 3. 金叉後 5 天內觸發紅吞黑
    cross_pos = idx_list.index(t_cross)
    end_pos = min(cross_pos + trigger_days, len(idx_list) - 1)
    
    for i in range(cross_pos, end_pos + 1):
        dt = idx_list[i]
        if i == 0: continue
        curr = recent_df.iloc[i]; prev = recent_df.iloc[i-1]
        
        red_engulf = (prev["Close"] < prev["Open"]) and (curr["Close"] > curr["Open"]) and (curr["Open"] < prev["Close"]) and (curr["Close"] > prev["Open"])
        
        if red_engulf and (curr["Close"] > oversold_close) and (k_series.loc[dt] >= kd_threshold):
            return True, dt, days_since_k20
            
    return False, None, 999

def run_strategy_backtest(stock_dict, progress_bar, mode, min_vol_threshold, lookback_days):
    results = []
    all_tickers = list(stock_dict.keys())
    BATCH_SIZE = 50
    total_batches = (len(all_tickers) // BATCH_SIZE) + 1
    
    for i, batch_idx in enumerate(range(0, len(all_tickers), BATCH_SIZE)):
        batch = all_tickers[batch_idx : batch_idx + BATCH_SIZE]
        try:
            data = yf.download(batch, period="2y", progress=False, auto_adjust=False)
            if data.empty: continue
            df_c = data["Close"]
            if isinstance(df_c, pd.Series): df_c = df_c.to_frame(name=batch[0])
            
            scan_window = df_c.index[-60:]
            
            for ticker in df_c.columns:
                try:
                    c_series = df_c[ticker].dropna()
                    if len(c_series) < lookback_days + 10: continue
                    stock_info = stock_dict.get(ticker, {})
                    full_ohlc = pd.DataFrame({"Open":data["Open"][ticker],"High":data["High"][ticker],"Low":data["Low"][ticker],"Close":data["Close"][ticker]}).dropna()
                    
                    for date in scan_window:
                        if date not in c_series.index: continue
                        idx = c_series.index.get_loc(date)
                        if float(data["Volume"][ticker].iloc[idx]) < (min_vol_threshold * 1000): continue
                        
                        is_match = False
                        # 依據策略模式判斷
                        if mode == "🦵 打腳發動 (KD+紅吞)":
                            ok, t_dt, _ = detect_leg_kick_signal(full_ohlc.loc[:date], max_lookback=lookback_days)
                            if ok and t_dt == date: is_match = True
                        elif mode == "🐎 多頭馬車發動 (多頭排列)":
                            m30 = c_series.rolling(30).mean().iloc[idx]
                            m60 = c_series.rolling(60).mean().iloc[idx]
                            if c_series.iloc[idx] > m30 > m60: is_match = True
                        elif mode == "🔥 起死回生 (Da來守住)":
                            m200 = c_series.rolling(200).mean().iloc[idx]
                            # 簡易回測邏輯：站上生命線
                            if c_series.iloc[idx] > m200 and c_series.iloc[idx-1] < m200: is_match = True
                        
                        if is_match:
                            results.append({
                                "月份": date.strftime("%m月"), "代號": ticker.split('.')[0], "名稱": stock_info.get("name"),
                                "產業": stock_info.get("group", "其他"), "訊號日期": date.strftime("%Y-%m-%d"),
                                "訊號價": round(c_series.iloc[idx], 2), "最高漲幅(%)": 0.0, "結果": "已驗證"
                            })
                            break
                except: continue
        except: pass
        progress_bar.progress((i+1)/total_batches)
    return pd.DataFrame(results)

def fetch_all_data(stock_dict, progress_bar):
    """
    下載所有資料並計算所有策略指標 (避免切換策略時 KeyError)
    """
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
                    ma200_df = df_c[ticker].rolling(200).mean()
                    ma30_df = df_c[ticker].rolling(30).mean()
                    ma60_df = df_c[ticker].rolling(60).mean()
                    
                    m200 = float(ma200_df.iloc[-1])
                    m30 = float(ma30_df.iloc[-1])
                    m60 = float(ma60_df.iloc[-1])
                    prev_m200 = float(ma200_df.iloc[-21]) if len(ma200_df) > 21 else 0

                    if pd.isna(p) or pd.isna(m200) or m200 == 0: continue

                    info = stock_dict[ticker]
                    stock_df = pd.DataFrame({"Open":data["Open"][ticker],"High":data["High"][ticker],"Low":data["Low"][ticker],"Close":df_c[ticker]}).dropna()
                    
                    # 計算 KD (通用)
                    k_val, d_val = 50.0, 50.0
                    if len(stock_df) >= 9:
                        k_val, d_val = calculate_kd_values(stock_df)

                    # 1. 打腳策略 (Leg Kick) - 預先計算 days_ago
                    leg_kick, leg_date, k20_days = detect_leg_kick_signal(stock_df, max_lookback=100)
                    
                    # 2. 起死回生 (Treasure)
                    is_treasure = False
                    if len(stock_df) >= 8:
                        rc = stock_df["Close"].iloc[-8:]
                        rm = ma200_df.iloc[-8:]
                        if (rc.iloc[-1] > rm.iloc[-1]) and (rc.iloc[:-1] < rm.iloc[:-1]).any():
                            is_treasure = True

                    # 3. 皇冠特選 (Royal)
                    is_royal = (p > m30 > m60 > m200)

                    # 4. 基礎指標
                    bias = ((p - m200) / m200) * 100
                    ma_trend = "⬆️向上" if m200 >= prev_m200 else "⬇️向下"

                    raw_data.append({
                        "代號": info["code"], "名稱": info["name"], "產業": info["group"], "完整代號": ticker,
                        "收盤價": p, "生命線": round(m200,2), 
                        "MA30": round(m30,2), "MA60": round(m60,2),
                        "乖離率(%)": round(bias, 2), 
                        "abs_bias": abs(bias),  # 修正：補回絕對值
                        "生命線趨勢": ma_trend,   # 修正：補回趨勢
                        "成交量": int(data["Volume"][ticker].iloc[-1]), 
                        "昨日成交量": int(data["Volume"][ticker].iloc[-2]),
                        "K值": k_val, "D值": d_val,
                        "打腳發動": leg_kick, "k20_days_ago": k20_days,
                        "皇冠特選": is_royal, "浴火重生": is_treasure # 修正：補回所有策略 flag
                    })
                except: continue
        except: pass
        progress_bar.progress((i+1)/total_batches)
    return pd.DataFrame(raw_data)

def plot_stock_chart(ticker, name):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        
        df["200MA"] = df["Close"].rolling(200).mean()
        df["30MA"] = df["Close"].rolling(30).mean()
        df["60MA"] = df["Close"].rolling(60).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="收盤價", line=dict(color="#00CC96")))
        fig.add_trace(go.Scatter(x=df.index, y=df["30MA"], name="30MA", line=dict(color="#AB63FA", dash="dot")))
        fig.add_trace(go.Scatter(x=df.index, y=df["60MA"], name="60MA", line=dict(color="#19D3F3", dash="dot")))
        fig.add_trace(go.Scatter(x=df.index, y=df["200MA"], name="生命線", line=dict(color="#FFA15A", width=2)))
        
        fig.update_layout(title=f"📊 {name} ({ticker})", height=450, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    except: st.error("圖表載入失敗")

# --- 3. 介面顯示區 ---
st.title(f"✨ {VER} 黑嚕嚕-旗鼓相當")
st.markdown("---")

if "master_df" not in st.session_state: st.session_state["master_df"] = None
if "backtest_result" not in st.session_state: st.session_state["backtest_result"] = None

with st.sidebar:
    st.header("資料庫管理")
    CACHE_FILE = "stock_cache_v19.csv"

    # 重置按鈕 (保留 1.7 的位置)
    if st.button("🚨 強制重置系統", type="primary"):
        st.cache_data.clear(); st.session_state.clear()
        if os.path.exists(CACHE_FILE): os.remove(CACHE_FILE)
        st.rerun()

    if os.path.exists(CACHE_FILE) and st.session_state["master_df"] is None:
        try:
            df_loaded = pd.read_csv(CACHE_FILE)
            # 防呆檢查：確保關鍵欄位都存在
            req_cols = ["k20_days_ago", "浴火重生", "abs_bias", "打腳發動"]
            if not all(col in df_loaded.columns for col in req_cols):
                st.error("⚠️ 資料結構過舊，請點擊上方 **「🚨 強制重置系統」**")
                st.session_state["master_df"] = None 
            else:
                st.session_state["master_df"] = df_loaded
                st.success("⚡ 歷史資料已載入")
        except: pass

    st.divider()
    st.header("1. 策略設定")
    strategy_mode = st.radio("選擇策略", ["🛡️ 生命線保衛戰", "🔥 起死回生", "🐎 多頭馬車發動 (多頭排列)", "🦵 打腳發動 (KD+紅吞)"])
    
    leg_kick_days_filter = 60
    if strategy_mode == "🦵 打腳發動 (KD+紅吞)":
        st.markdown("---")
        st.info("💡 調整下方滑桿，可即時過濾結果")
        leg_kick_days_filter = st.slider("🦵 前置搜尋天數 (K<20)", 20, 100, 60, step=5)
        st.markdown("---")

    min_vol_input = st.number_input("最低成交量 (張)", value=1000, step=100)
    bias_threshold = st.slider("乖離率範圍 (±%)", 0.5, 5.0, 2.5, step=0.1)
    
    # 策略細部選項 (保留 1.4a 的介面邏輯)
    filter_trend_up = False; filter_trend_down = False; filter_kd = False; filter_vol_double = False
    if strategy_mode == "🛡️ 生命線保衛戰":
        c1, c2 = st.columns(2)
        with c1: filter_trend_up = st.checkbox("生命線向上")
        with c2: filter_trend_down = st.checkbox("生命線向下")
        filter_kd = st.checkbox("KD 黃金交叉")
        filter_vol_double = st.checkbox("出量 (今日 > 昨日x1.5)")
    elif strategy_mode in ["🔥 起死回生", "🐎 多頭馬車發動 (多頭排列)"]:
        filter_vol_double = st.checkbox("出量確認")

    st.divider()
    
    if st.button("🔄 下載最新股價 (開市用)", type="secondary"):
        stock_dict = get_stock_list()
        # 動畫回歸
        ph = st.empty()
        with ph: st.markdown("""<div style="text-align: center; font-size: 40px; animation: blink 1s infinite;">🎁💰✨</div>
            <style>@keyframes blink { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }</style>
            <div style="text-align: center;">正在挖掘寶藏中...</div>""", unsafe_allow_html=True)
        
        pb = st.progress(0)
        df_new = fetch_all_data(stock_dict, pb)
        ph.empty()
        
        if not df_new.empty:
            df_new.to_csv(CACHE_FILE, index=False)
            st.session_state["master_df"] = df_new
            st.rerun()
            
    if st.button("🧪 執行策略回測"):
        stock_dict = get_stock_list()
        pb_bt = st.progress(0, text="正在驗證歷史訊號...")
        bt_df = run_strategy_backtest(stock_dict, pb_bt, strategy_mode, min_vol_input, leg_kick_days_filter)
        st.session_state["backtest_result"] = bt_df

# 主畫面 - 日常篩選
if st.session_state["master_df"] is not None:
    df = st.session_state["master_df"].copy()
    
    # 確保欄位存在 (KeyError 防護)
    if "生命線" not in df.columns or "k20_days_ago" not in df.columns:
        st.error("⚠️ 資料結構不符，請執行「🚨 強制重置系統」！")
        st.stop()

    # 基礎過濾
    df = df[df["成交量"] >= min_vol_input * 1000]

    # 策略分流
    if strategy_mode == "🔥 起死回生":
        df = df[df["浴火重生"] == True]
    elif strategy_mode == "🐎 多頭馬車發動 (多頭排列)":
        df = df[df["皇冠特選"] == True]
    elif strategy_mode == "🦵 打腳發動 (KD+紅吞)":
        df = df[df["打腳發動"] == True]
        df = df[df["k20_days_ago"] <= leg_kick_days_filter] # 即時過濾
    else:
        # 生命線保衛戰
        df = df[df["abs_bias"] <= bias_threshold]
        if filter_trend_up: df = df[df["生命線趨勢"] == "⬆️向上"]
        if filter_trend_down: df = df[df["生命線趨勢"] == "⬇️向下"]
        if filter_kd: df = df[df["K值"] > df["D值"]]

    if filter_vol_double:
        df = df[df["成交量"] > (df["昨日成交量"] * 1.5)]

    if df.empty:
        st.warning("⚠️ 找不到符合條件的股票！")
    else:
        st.markdown(f"""<div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center; border: 2px solid #ff4b4b;">
            <h2 style="color: #333; margin:0;">🔍 根據共篩選出 <span style="color: #ff4b4b; font-size: 1.5em;">{len(df)}</span> 檔股票</h2>
        </div><br>""", unsafe_allow_html=True)

        df["成交量(張)"] = (df["成交量"] / 1000).astype(int)
        df["KD值"] = df.apply(lambda x: f"K:{int(x['K值'])} D:{int(x['D值'])}", axis=1)
        df["選股標籤"] = df["代號"].astype(str) + " " + df["名稱"].astype(str)

        # 顯示欄位
        display_cols = ["代號", "名稱", "產業", "收盤價", "生命線", "乖離率(%)", "位置", "KD值", "成交量(張)"]
        if strategy_mode == "🐎 多頭馬車發動 (多頭排列)":
            display_cols = ["代號", "名稱", "產業", "收盤價", "MA30", "MA60", "生命線", "KD值", "成交量(張)"]
        elif strategy_mode == "🦵 打腳發動 (KD+紅吞)":
            df = df.rename(columns={"k20_days_ago": "前置K<20(天前)"})
            display_cols = ["代號", "名稱", "產業", "收盤價", "乖離率(%)", "前置K<20(天前)", "KD值", "成交量(張)"]

        df = df.sort_values(by="成交量", ascending=False)

        # 1.4a 的 Tab 介面
        tab1, tab2 = st.tabs(["📋 篩選結果列表", "📊 日趨勢圖"])

        with tab1:
            # 使用 style.map 修復顏色 (Ver 1.8 修正)
            def color_bias(val):
                return f'color: {"#ff4b4b" if val > 0 else "#008000"}; font-weight: bold'
            
            try:
                # 針對乖離率上色
                st.dataframe(df[display_cols].style.map(color_bias, subset=["乖離率(%)"]), use_container_width=True, hide_index=True)
            except:
                st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

        with tab2:
            st.markdown("### 🔍 個股趨勢圖")
            # 修正：被動觸發線圖 (Ver 1.8)
            # 使用 index=None，預設不選取任何股票
            selected_stock_label = st.selectbox("請選擇一檔股票：", df["選股標籤"].tolist(), index=None, placeholder="點此選擇以查看線圖...")
            
            if selected_stock_label:
                selected_row = df[df["選股標籤"] == selected_stock_label].iloc[0]
                plot_stock_chart(selected_row["完整代號"], selected_row["名稱"])

                c1, c2, c3 = st.columns(3)
                c1.metric("收盤價", f"{selected_row['收盤價']:.2f}")
                c2.metric("成交量", f"{selected_row['成交量(張)']} 張")
                c3.metric("KD", selected_row["KD值"])
            else:
                st.info("👈 請從下拉選單中選擇一檔股票來顯示線圖")

else:
    st.warning("👈 請先點擊左側 sidebar 的 **「🔄 下載最新股價」** 按鈕開始挖寶！")
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        if os.path.exists("welcome.jpg"):
            st.markdown("""<div style="text-align: center; font-size: 1.1em; margin-bottom: 20px;">
                這是數年來的經驗收納<br>此工具僅供參考，不代表投資建議<br>預祝心想事成，從從容容，紫氣東來! 🟣✨</div>""", unsafe_allow_html=True)
            st.image("welcome.jpg", width=420)
        else:
            st.info("💡 尚未偵測到 welcome.jpg")

# 回測結果區
if st.session_state["backtest_result"] is not None:
    bt_df = st.session_state["backtest_result"]
    st.markdown("---")
    st.subheader(f"🧪 策略回測報告")
    
    if not bt_df.empty:
        # 簡單上色
        def style_ret(val): return f'color: {"red" if val > 0 else "green"}'
        try:
            st.dataframe(bt_df.style.map(style_ret, subset=["最高漲幅(%)"]), use_container_width=True)
        except:
            st.dataframe(bt_df, use_container_width=True)
    else:
        st.info("無回測數據")
