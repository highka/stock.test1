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
# 更新邏輯：新增產業類別功能，版本升級至 1.4
# 修改內部資料傳遞邏輯，後綴標註 a
VER = "ver 1.4a (Industry Support + Backtest Optimized)"
st.set_page_config(page_title=f"✨ 黑嚕嚕-旗鼓相當({VER})", layout="wide")

# --- 流量紀錄與後台功能 ---
LOG_FILE = "traffic_log.csv"

def get_remote_ip():
    """嘗試取得使用者 IP (針對 Streamlit Cloud)"""
    try:
        from streamlit.web.server.websocket_headers import _get_websocket_headers
        headers = _get_websocket_headers()
        if headers and "X-Forwarded-For" in headers:
            return headers["X-Forwarded-For"].split(",")[0]
    except:
        pass
    return "Unknown/Local"

def log_traffic():
    """紀錄使用者訪問"""
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
    """取得台股清單 (排除金融/ETF，保留產業資訊)"""
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
    """回傳整條 K / D 序列"""
    low_min = df["Low"].rolling(window=n).min()
    high_max = df["High"].rolling(window=n).max()
    rsv = (df["Close"] - low_min) / (high_max - low_min) * 100
    rsv = rsv.fillna(50)

    k_list, d_list = [], []
    k, d = 50, 50
    for r in rsv:
        k = (2/3) * k + (1/3) * r
        d = (2/3) * d + (1/3) * k
        k_list.append(k)
        d_list.append(d)

    return pd.Series(k_list, index=df.index), pd.Series(d_list, index=df.index)

def _is_red_engulf_black(prev_open, prev_close, curr_open, curr_close):
    """紅吞黑邏輯"""
    prev_is_black = prev_close < prev_open
    curr_is_red = curr_close > curr_open
    engulf = (curr_open < prev_close) and (curr_close > prev_open)
    return prev_is_black and curr_is_red and engulf

def detect_leg_kick_signal(stock_df, lookback=60, trigger_days=5, kd_threshold=20):
    """打腳發動判斷邏輯"""
    if len(stock_df) < max(lookback + 2, 30):
        return False, None

    recent_df = stock_df.tail(lookback).copy()
    k_series, d_series = calculate_kd_series(recent_df)

    # 1) 最後一次 K < 20
    t1 = k_series[k_series < kd_threshold].last_valid_index()
    if t1 is None:
        return False, None
    oversold_close = float(recent_df.loc[t1, "Close"])

    # 2) t1 之後找第一次 KD 金叉
    idx_list = list(recent_df.index)
    t1_pos = idx_list.index(t1)
    t_cross = None
    for i in range(t1_pos + 1, len(idx_list)):
        dt = idx_list[i]
        prev_dt = idx_list[i - 1]
        if (k_series.loc[prev_dt] <= d_series.loc[prev_dt]) and (k_series.loc[dt] > d_series.loc[dt]):
            t_cross = dt
            break
    if t_cross is None:
        return False, None

    # 3) 金叉後 trigger_days 內找觸發
    cross_pos = idx_list.index(t_cross)
    end_pos = min(cross_pos + trigger_days, len(idx_list) - 1)

    for i in range(cross_pos, end_pos + 1):
        dt = idx_list[i]
        if i == 0: continue
        if float(k_series.loc[dt]) < kd_threshold: continue

        prev_row = recent_df.iloc[i - 1]
        curr_row = recent_df.iloc[i]
        if _is_red_engulf_black(prev_row["Open"], prev_row["Close"], curr_row["Open"], curr_row["Close"]) and (curr_row["Close"] > oversold_close):
            return True, dt

    return False, None

# --- 策略回測核心 ---
def run_strategy_backtest(stock_dict, progress_bar, mode, use_trend_up, use_treasure, use_vol, use_royal, use_leg_kick, min_vol_threshold):
    results = []
    all_tickers = list(stock_dict.keys())
    BATCH_SIZE = 50
    total_batches = (len(all_tickers) // BATCH_SIZE) + 1
    OBSERVE_DAYS = 20 if use_royal else 10

    for i, batch_idx in enumerate(range(0, len(all_tickers), BATCH_SIZE)):
        batch = all_tickers[batch_idx : batch_idx + BATCH_SIZE]
        try:
            data = yf.download(batch, period="2y", interval="1d", progress=False, auto_adjust=False)
            if data.empty: continue
            
            df_c = data["Close"]
            df_o = data["Open"]
            df_v = data["Volume"]
            df_l = data["Low"]
            df_h = data["High"]

            # 轉為 DataFrame 處理單檔情況
            if isinstance(df_c, pd.Series):
                df_c = df_c.to_frame(name=batch[0]); df_o = df_o.to_frame(name=batch[0])
                df_v = df_v.to_frame(name=batch[0]); df_l = df_l.to_frame(name=batch[0]); df_h = df_h.to_frame(name=batch[0])

            ma200_df = df_c.rolling(window=200).mean()
            ma30_df = df_c.rolling(window=30).mean()
            ma60_df = df_c.rolling(window=60).mean()
            scan_window = df_c.index[-90:]

            for ticker in df_c.columns:
                try:
                    c_series = df_c[ticker].dropna()
                    if len(c_series) < 200: continue
                    
                    stock_info = stock_dict.get(ticker, {})
                    stock_name = stock_info.get("name", ticker)
                    industry = stock_info.get("group", "未知")

                    full_ohlc = pd.DataFrame({"Open": df_o[ticker], "Close": df_c[ticker], "High": df_h[ticker], "Low": df_l[ticker]}).dropna()

                    for date in scan_window:
                        if date not in c_series.index: continue
                        idx = c_series.index.get_loc(date)
                        if idx < 200: continue

                        close_p = float(c_series.iloc[idx])
                        vol = float(df_v[ticker].iloc[idx])
                        prev_vol = float(df_v[ticker].iloc[idx-1]) if idx > 0 else 1.0
                        ma200_val = float(ma200_df[ticker].iloc[idx])
                        
                        if ma200_val == 0 or vol < (min_vol_threshold * 1000): continue

                        is_match = False
                        if use_leg_kick:
                            sub_df = full_ohlc.loc[:date]
                            ok, trig_dt = detect_leg_kick_signal(sub_df)
                            if ok and trig_dt == date: is_match = True
                        elif use_royal:
                            ma30_v, ma60_v = ma30_df[ticker].iloc[idx], ma60_df[ticker].iloc[idx]
                            if close_p > ma30_v > ma60_v > ma200_val: is_match = True
                        else:
                            # 其他策略判斷...
                            if use_treasure:
                                recent_c = c_series.iloc[idx-7:idx+1]
                                recent_m = ma200_df[ticker].iloc[idx-7:idx+1]
                                if recent_c.iloc[-1] > recent_m.iloc[-1] and (recent_c.iloc[:-1] < recent_m.iloc[:-1]).any():
                                    is_match = True
                            else:
                                if ma200_val * 0.90 <= float(df_l[ticker].iloc[idx]) <= ma200_val * 1.03 and close_p > ma200_val:
                                    is_match = True

                        if is_match:
                            days_after = len(c_series) - 1 - idx
                            res_status = "觀察中"; prof = 0.0; is_watching = True
                            
                            if days_after >= 1:
                                if use_royal:
                                    # 馬車專用結算邏輯
                                    check_df = full_ohlc.iloc[idx+1 : idx+1+OBSERVE_DAYS]
                                    if (check_df["High"] >= close_p * 1.10).any():
                                        res_status = "Win (止盈) 🐎"; prof = 10.0; is_watching = False
                                    elif (check_df["Close"] < ma200_df[ticker].iloc[idx+1:idx+1+OBSERVE_DAYS]).any():
                                        res_status = "Loss (破線) 🛑"; is_watching = False # 簡化計算
                                    elif days_after >= OBSERVE_DAYS:
                                        prof = (c_series.iloc[idx+OBSERVE_DAYS] - close_p)/close_p*100
                                        res_status = "Win (期滿)" if prof > 0 else "Loss (期滿)"; is_watching = False
                                else:
                                    if days_after >= OBSERVE_DAYS:
                                        max_p = float(df_h[ticker].iloc[idx+1:idx+1+OBSERVE_DAYS].max())
                                        prof = (max_p - close_p)/close_p*100
                                        res_status = "驗證成功 🏆" if prof > 3.0 else ("Win" if prof > 0 else "Loss"); is_watching = False
                                    else:
                                        prof = (c_series.iloc[-1] - close_p)/close_p*100

                            results.append({
                                "月份": date.strftime("%m月") if not is_watching else "👀 關注中",
                                "代號": ticker.split('.')[0], "名稱": stock_name, "產業": industry,
                                "訊號日期": date.strftime("%Y-%m-%d"), "訊號價": round(close_p, 2),
                                "最高漲幅(%)": round(prof, 2), "結果": res_status
                            })
                            if use_royal: break
                except: continue
        except: pass
        progress_bar.progress((i+1)/total_batches, text=f"深度回測中...({int((i+1)/total_batches*100)}%)")
    return pd.DataFrame(results)

def fetch_all_data(stock_dict, progress_bar):
    all_tickers = list(stock_dict.keys())
    BATCH_SIZE = 50
    total_batches = (len(all_tickers) // BATCH_SIZE) + 1
    raw_data = []

    for i, batch_idx in enumerate(range(0, len(all_tickers), BATCH_SIZE)):
        batch = all_tickers[batch_idx : batch_idx + BATCH_SIZE]
        try:
            data = yf.download(batch, period="1y", progress=False, auto_adjust=False)
            df_c = data["Close"]; df_v = data["Volume"]; df_h = data["High"]; df_l = data["Low"]; df_o = data["Open"]
            
            ma200_df = df_c.rolling(window=200).mean()
            ma30_df = df_c.rolling(window=30).mean()
            ma60_df = df_c.rolling(window=60).mean()

            for ticker in df_c.columns:
                try:
                    p = float(df_c[ticker].iloc[-1])
                    m200 = float(ma200_df[ticker].iloc[-1])
                    if pd.isna(m200) or m200 == 0: continue
                    
                    info = stock_dict[ticker]
                    stock_df = pd.DataFrame({"Open":df_o[ticker], "Close":df_c[ticker], "High":df_h[ticker], "Low":df_l[ticker]}).dropna()
                    k_v, d_v = 0.0, 0.0
                    if len(stock_df) >= 20:
                        ks, ds = calculate_kd_series(stock_df)
                        k_v, d_v = ks.iloc[-1], ds.iloc[-1]
                    
                    leg_kick, leg_date = detect_leg_kick_signal(stock_df)

                    raw_data.append({
                        "代號": info["code"], "名稱": info["name"], "產業": info["group"], "完整代號": ticker,
                        "收盤價": p, "生命線": m200, "MA30": float(ma30_df[ticker].iloc[-1]), "MA60": float(ma60_df[ticker].iloc[-1]),
                        "生命線趨勢": "向上" if m200 > ma200_df[ticker].iloc[-21] else "向下",
                        "乖離率(%)": (p-m200)/m200*100, "成交量": int(df_v[ticker].iloc[-1]), "昨日成交量": int(df_v[ticker].iloc[-2]),
                        "K值": k_v, "D值": d_v, "浴火重生": (p > m200 and (df_c[ticker].iloc[-8:-1] < ma200_df[ticker].iloc[-8:-1]).any()),
                        "皇冠特選": (p > ma30_df[ticker].iloc[-1] > ma60_df[ticker].iloc[-1] > m200),
                        "打腳發動": leg_kick, "打腳日期": leg_date.strftime("%Y-%m-%d") if leg_date else ""
                    })
                except: continue
        except: pass
        progress_bar.progress((i+1)/total_batches, text=f"採礦中...{int((i+1)/total_batches*100)}%")
    return pd.DataFrame(raw_data)

def plot_stock_chart(ticker, name):
    df = yf.download(ticker, period="1y", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df["200MA"] = df["Close"].rolling(200).mean()
    df["30MA"] = df["Close"].rolling(30).mean()
    plot_df = df.tail(120)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Close"], name="收盤價", line=dict(color="#00CC96")))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["200MA"], name="200MA", line=dict(color="#FFA15A", width=2)))
    fig.update_layout(title=f"{name} ({ticker})", height=450, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# --- 3. 介面區 ---
st.title(f"✨ {VER} 黑嚕嚕-旗鼓相當")
st.markdown("---")

CACHE_FILE = "stock_data_cache_v14.csv"
if "master_df" not in st.session_state:
    if os.path.exists(CACHE_FILE):
        st.session_state["master_df"] = pd.read_csv(CACHE_FILE)
        st.session_state["last_update"] = "已載入快取"
    else:
        st.session_state["master_df"] = None
        st.session_state["last_update"] = None

with st.sidebar:
    st.header("數據控制台")
    if st.button("🔄 下載最新股價", type="primary"):
        stock_dict = get_stock_list()
        pb = st.progress(0)
        df = fetch_all_data(stock_dict, pb)
        if not df.empty:
            df.to_csv(CACHE_FILE, index=False)
            st.session_state["master_df"] = df
            st.session_state["last_update"] = datetime.now().strftime("%H:%M:%S")
            st.rerun()
    
    if st.button("🚨 重置"):
        if os.path.exists(CACHE_FILE): os.remove(CACHE_FILE)
        st.session_state.clear()
        st.rerun()

    st.divider()
    strategy_mode = st.radio("選擇策略", ["🛡️ 生命線保衛戰", "🔥 起死回生", "🐎 多頭馬車", "🦵 打腳發動"])
    min_vol = st.number_input("最低成交量(張)", 500, 10000, 1000)
    bias_val = st.slider("乖離率範圍", 0.0, 10.0, 2.5)

    if st.button("🧪 執行策略回測"):
        stock_dict = get_stock_list()
        pb = st.progress(0)
        res_df = run_strategy_backtest(stock_dict, pb, strategy_mode, False, 
                                       (strategy_mode=="🔥 起死回生"), False, 
                                       (strategy_mode=="🐎 多頭馬車"), 
                                       (strategy_mode=="🦵 打腳發動"), min_vol)
        st.session_state["backtest_result"] = res_df
        st.success("回測完成")

# 顯示回測結果
if "backtest_result" in st.session_state:
    bt_df = st.session_state["backtest_result"]
    st.subheader(f"📊 回測報告 ({strategy_mode})")
    col1, col2 = st.columns(2)
    win_count = len(bt_df[bt_df["結果"].str.contains("Win|成功")])
    col1.metric("獲利次數", win_count)
    col2.metric("總訊號數", len(bt_df))
    st.dataframe(bt_df, use_container_width=True, hide_index=True)

# 顯示即時篩選
if st.session_state["master_df"] is not None:
    df = st.session_state["master_df"].copy()
    df = df[df["成交量"] >= min_vol*1000]
    
    if strategy_mode == "🔥 起死回生": df = df[df["浴火重生"]==True]
    elif strategy_mode == "🐎 多頭馬車": df = df[df["皇冠特選"]==True]
    elif strategy_mode == "🦵 打腳發動": df = df[df["打腳發動"]==True]
    else: df = df[abs(df["乖離率(%)"]) <= bias_val]

    st.subheader(f"🔍 今日篩選結果 (共 {len(df)} 檔)")
    if not df.empty:
        df["成交量(張)"] = (df["成交量"]/1000).astype(int)
        show_cols = ["代號", "名稱", "產業", "收盤價", "生命線", "乖離率(%)", "成交量(張)"]
        st.dataframe(df[show_cols].style.background_gradient(subset=["乖離率(%)"]), use_container_width=True, hide_index=True)
        
        sel = st.selectbox("查看趨勢圖", df["代號"].tolist())
        row = df[df["代號"]==sel].iloc[0]
        plot_stock_chart(row["完整代號"], row["名稱"])
    else:
        st.info("目前無符合條件標的")
else:
    st.warning("👈 請先下載資料")
