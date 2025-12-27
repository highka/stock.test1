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
# 更新：加入自訂打腳天數 (Max 60, Step 5) + 穩定性修復
VER = "ver 1.5 (Custom LegKick Days + Stability Fix)"
st.set_page_config(page_title=f"✨ 黑嚕嚕-旗鼓相當({VER})", layout="wide")

# --- 流量紀錄與後台功能 ---
LOG_FILE = "traffic_log.csv"

def get_remote_ip():
    """嘗試取得使用者 IP"""
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
    """取得台股清單 (排除金融/ETF)"""
    try:
        tse = twstock.twse
        otc = twstock.tpex
        stock_dict = {}
        exclude_industries = ["金融保險業", "存託憑證"]

        for code, info in tse.items():
            if info.type == "股票":
                if info.group not in exclude_industries:
                    stock_dict[f"{code}.TW"] = {"name": info.name, "code": code, "group": info.group}

        for code, info in otc.items():
            if info.type == "股票":
                if info.group not in exclude_industries:
                    stock_dict[f"{code}.TWO"] = {"name": info.name, "code": code, "group": info.group}
        return stock_dict
    except:
        return {}

def calculate_kd_values(df, n=9):
    """回傳最後一筆 K, D（舊版用）"""
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

    k_series = pd.Series(k_list, index=df.index)
    d_series = pd.Series(d_list, index=df.index)
    return k_series, d_series

def _is_red_engulf_black(prev_open, prev_close, curr_open, curr_close):
    """紅吞黑判斷"""
    prev_is_black = prev_close < prev_open
    curr_is_red = curr_close > curr_open
    engulf = (curr_open < prev_close) and (curr_close > prev_open)
    return prev_is_black and curr_is_red and engulf

def detect_leg_kick_signal(stock_df, lookback_days, trigger_days=5, kd_threshold=20):
    """
    ✅ 打腳發動核心邏輯 (支援自訂 lookback_days)
    """
    if len(stock_df) < max(lookback_days + 2, 30):
        return False, None

    # 使用使用者設定的天數來切分資料
    recent_df = stock_df.tail(lookback_days).copy()
    if len(recent_df) < 20:
        return False, None

    k_series, d_series = calculate_kd_series(recent_df)

    # 1) 最後一次 K < 20
    t1 = k_series[k_series < kd_threshold].last_valid_index()
    if t1 is None:
        return False, None
    oversold_close = float(recent_df.loc[t1, "Close"])

    # 2) t1 之後找第一次 KD 金叉
    idx_list = list(recent_df.index)
    
    try:
        t1_pos = idx_list.index(t1)
    except ValueError:
        return False, None # 防呆

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
        if i == 0:
            continue

        # K >= 20
        if float(k_series.loc[dt]) < kd_threshold:
            continue

        prev_row = recent_df.iloc[i - 1]
        curr_row = recent_df.iloc[i]

        prev_open, prev_close = float(prev_row["Open"]), float(prev_row["Close"])
        curr_open, curr_close = float(curr_row["Open"]), float(curr_row["Close"])

        if _is_red_engulf_black(prev_open, prev_close, curr_open, curr_close) and (curr_close > oversold_close):
            return True, dt

    return False, None

# --- 策略回測核心函數 ---
def run_strategy_backtest(
    stock_dict,
    progress_bar,
    mode,
    use_trend_up,
    use_treasure,
    use_vol,
    use_royal,
    use_leg_kick,
    min_vol_threshold,
    leg_kick_lookback # 新增參數
):
    results = []
    all_tickers = list(stock_dict.keys())
    BATCH_SIZE = 50
    total_batches = (len(all_tickers) // BATCH_SIZE) + 1
    OBSERVE_DAYS = 20 if use_royal else 10

    for i, batch_idx in enumerate(range(0, len(all_tickers), BATCH_SIZE)):
        batch = all_tickers[batch_idx : batch_idx + BATCH_SIZE]
        try:
            # 下載較長的歷史資料以確保 lookback 足夠
            data = yf.download(batch, period="2y", interval="1d", progress=False, auto_adjust=False)
            if data.empty: continue

            try:
                df_o = data["Open"]; df_c = data["Close"]; df_v = data["Volume"]
                df_l = data["Low"]; df_h = data["High"]
            except KeyError: continue

            if isinstance(df_c, pd.Series):
                df_o = df_o.to_frame(name=batch[0]); df_c = df_c.to_frame(name=batch[0])
                df_v = df_v.to_frame(name=batch[0]); df_l = df_l.to_frame(name=batch[0]); df_h = df_h.to_frame(name=batch[0])

            ma200_df = df_c.rolling(window=200).mean()
            ma30_df = df_c.rolling(window=30).mean()
            ma60_df = df_c.rolling(window=60).mean()

            scan_window = df_c.index[-90:] 

            for ticker in df_c.columns:
                try:
                    c_series = df_c[ticker].dropna()
                    if len(c_series) < 200: continue

                    o_series = df_o[ticker].reindex(c_series.index).dropna()
                    v_series = df_v[ticker].reindex(c_series.index).dropna()
                    l_series = df_l[ticker].reindex(c_series.index).dropna()
                    h_series = df_h[ticker].reindex(c_series.index).dropna()

                    ma200_series = ma200_df[ticker].reindex(c_series.index)
                    ma30_series = ma30_df[ticker].reindex(c_series.index)
                    ma60_series = ma60_df[ticker].reindex(c_series.index)

                    stock_info = stock_dict.get(ticker, {})
                    stock_name = stock_info.get("name", ticker)
                    stock_industry = stock_info.get("group", "")
                    total_len = len(c_series)

                    full_ohlc = pd.DataFrame({"Open": o_series, "Close": c_series, "High": h_series, "Low": l_series}).dropna()

                    for date in scan_window:
                        if date not in c_series.index: continue
                        idx = c_series.index.get_loc(date)
                        if idx < 200: continue

                        close_p = float(c_series.iloc[idx])
                        vol = float(v_series.iloc[idx]) if date in v_series.index else 0.0
                        prev_vol = float(v_series.iloc[idx - 1]) if idx - 1 >= 0 else 0.0
                        ma200_val = float(ma200_series.iloc[idx]) if not pd.isna(ma200_series.iloc[idx]) else 0.0

                        if ma200_val == 0: continue
                        if vol < (min_vol_threshold * 1000): continue
                        if prev_vol == 0: prev_vol = 1.0

                        is_match = False

                        # --- 🦵 打腳回測 ---
                        if use_leg_kick:
                            sub_df = full_ohlc.loc[:date].copy()
                            # 傳入自訂天數
                            ok, trig_dt = detect_leg_kick_signal(sub_df, leg_kick_lookback, trigger_days=5, kd_threshold=20)
                            if ok and trig_dt == date:
                                is_match = True

                        # --- 🐎 多頭馬車發動 ---
                        elif use_royal:
                            ma30_val = float(ma30_series.iloc[idx])
                            ma60_val = float(ma60_series.iloc[idx])
                            if (close_p > ma30_val) and (ma30_val > ma60_val) and (ma60_val > ma200_val):
                                is_match = True

                        # --- 🔥 起死回生 / 🛡️ 生命線保衛戰 ---
                        else:
                            low_p = float(l_series.iloc[idx])
                            ma_val_20ago = float(ma200_series.iloc[idx - 20])

                            if use_trend_up and (ma200_val <= ma_val_20ago): continue
                            if use_vol and (vol <= prev_vol * 1.5): continue

                            if use_treasure:
                                start_idx = idx - 7
                                if start_idx < 0: continue
                                recent_c = c_series.iloc[start_idx : idx + 1]
                                recent_ma = ma200_series.iloc[start_idx : idx + 1]
                                cond_today_up = recent_c.iloc[-1] > recent_ma.iloc[-1]
                                past_c = recent_c.iloc[:-1]
                                past_ma = recent_ma.iloc[:-1]
                                if cond_today_up and (past_c < past_ma).any():
                                    is_match = True
                            else:
                                cond_near = (low_p <= ma200_val * 1.03) and (low_p >= ma200_val * 0.90)
                                if cond_near and (close_p > ma200_val):
                                    is_match = True

                        if not is_match: continue

                        # ---- 命中後：統一出結果 ----
                        month_str = date.strftime("%m月")
                        days_after_signal = total_len - 1 - idx
                        final_profit_pct = 0.0
                        result_status = "觀察中"
                        is_watching = False

                        if days_after_signal < 1:
                            is_watching = True
                        elif use_royal:
                            is_watching = True
                            current_price = float(c_series.iloc[-1])
                            final_profit_pct = (current_price - close_p) / close_p * 100
                            check_days = min(days_after_signal, OBSERVE_DAYS)
                            for d in range(1, check_days + 1):
                                day_idx = idx + d
                                day_high = float(h_series.iloc[day_idx])
                                day_close = float(c_series.iloc[day_idx])
                                day_ma200 = float(ma200_series.iloc[day_idx])
                                if day_high >= close_p * 1.10:
                                    final_profit_pct = 10.0
                                    result_status = "Win (止盈出場) 🐎"
                                    is_watching = False
                                    break
                                if day_close < day_ma200:
                                    final_profit_pct = (day_close - close_p) / close_p * 100
                                    result_status = "Loss (破線停損) 🛑"
                                    is_watching = False
                                    break
                            if is_watching:
                                if days_after_signal >= OBSERVE_DAYS:
                                    end_close = float(c_series.iloc[idx + OBSERVE_DAYS])
                                    final_profit_pct = (end_close - close_p) / close_p * 100
                                    result_status = "Win (期滿獲利)" if final_profit_pct > 0 else "Loss (期滿虧損)"
                                    is_watching = False
                        else:
                            if days_after_signal < OBSERVE_DAYS:
                                current_price = float(c_series.iloc[-1])
                                final_profit_pct = (current_price - close_p) / close_p * 100
                                is_watching = True
                            else:
                                future_highs = h_series.iloc[idx + 1 : idx + 1 + OBSERVE_DAYS]
                                max_price = float(future_highs.max())
                                final_profit_pct = (max_price - close_p) / close_p * 100
                                if final_profit_pct > 3.0: result_status = "驗證成功 🏆"
                                elif final_profit_pct > 0: result_status = "Win (反彈)"
                                else: result_status = "Loss 📉"

                        results.append({
                            "月份": "👀 關注中" if is_watching else month_str,
                            "代號": ticker.replace(".TW", "").replace(".TWO", ""),
                            "名稱": stock_name,
                            "產業": stock_industry,
                            "訊號日期": date.strftime("%Y-%m-%d"),
                            "訊號價": round(close_p, 2),
                            "最高漲幅(%)": round(final_profit_pct, 2),
                            "結果": "觀察中" if is_watching else result_status,
                        })
                        if use_royal: break
                except: continue
        except: pass
        progress_bar.progress((i + 1) / total_batches, text=f"深度回測中...({int((i+1)/total_batches*100)}%)")

    if not results:
        return pd.DataFrame(columns=["月份", "代號", "名稱", "產業", "訊號日期", "訊號價", "最高漲幅(%)", "結果"])
    return pd.DataFrame(results)

def fetch_all_data(stock_dict, progress_bar, status_text, leg_kick_lookback):
    """即時下載並篩選，支援自訂打腳天數"""
    if not stock_dict: return pd.DataFrame()
    all_tickers = list(stock_dict.keys())
    BATCH_SIZE = 50
    total_batches = (len(all_tickers) // BATCH_SIZE) + 1
    raw_data_list = []

    for i, batch_idx in enumerate(range(0, len(all_tickers), BATCH_SIZE)):
        batch = all_tickers[batch_idx : batch_idx + BATCH_SIZE]
        try:
            data = yf.download(batch, period="1y", interval="1d", progress=False, auto_adjust=False)
            if not data.empty:
                try:
                    df_o = data["Open"]; df_c = data["Close"]; df_h = data["High"]
                    df_l = data["Low"]; df_v = data["Volume"]
                except KeyError: continue

                if isinstance(df_c, pd.Series):
                    df_o = df_o.to_frame(name=batch[0]); df_c = df_c.to_frame(name=batch[0])
                    df_h = df_h.to_frame(name=batch[0]); df_l = df_l.to_frame(name=batch[0]); df_v = df_v.to_frame(name=batch[0])

                ma200_df = df_c.rolling(window=200).mean()
                ma30_df = df_c.rolling(window=30).mean()
                ma60_df = df_c.rolling(window=60).mean()

                last_price_s = df_c.iloc[-1]
                last_ma200_s = ma200_df.iloc[-1]
                last_ma30_s = ma30_df.iloc[-1]
                last_ma60_s = ma60_df.iloc[-1]
                prev_ma200_s = ma200_df.iloc[-21]
                last_vol_s = df_v.iloc[-1]; prev_vol_s = df_v.iloc[-2]
                
                recent_c_df = df_c.iloc[-8:]; recent_ma_df = ma200_df.iloc[-8:]

                for ticker in df_c.columns:
                    try:
                        price = float(last_price_s[ticker]); ma200 = float(last_ma200_s[ticker])
                        if pd.isna(price) or pd.isna(ma200) or ma200 == 0: continue

                        ma_trend = "⬆️向上" if ma200 >= float(prev_ma200_s[ticker]) else "⬇️向下"
                        
                        # 準備 K 線資料
                        stock_df = pd.DataFrame({
                            "Open": df_o[ticker], "Close": df_c[ticker],
                            "High": df_h[ticker], "Low": df_l[ticker]
                        }).dropna()

                        k_val, d_val = 0.0, 0.0
                        is_leg_kick = False
                        leg_kick_date = None

                        if len(stock_df) >= 20:
                            k_series, d_series = calculate_kd_series(stock_df)
                            k_val, d_val = float(k_series.iloc[-1]), float(d_series.iloc[-1])
                            
                            # 🦵 打腳：傳入自訂天數
                            is_leg_kick, leg_kick_date = detect_leg_kick_signal(stock_df, leg_kick_lookback, trigger_days=5, kd_threshold=20)
                        else:
                            if len(stock_df) >= 9: k_val, d_val = calculate_kd_values(stock_df)

                        # 起死回生
                        is_treasure = False
                        my_rc = recent_c_df[ticker]; my_rm = recent_ma_df[ticker]
                        if len(my_rc) >= 8 and (my_rc.iloc[-1] > my_rm.iloc[-1]) and (my_rc.iloc[:-1] < my_rm.iloc[:-1]).any():
                            is_treasure = True
                        
                        # 皇冠特選
                        ma30 = float(last_ma30_s[ticker]); ma60 = float(last_ma60_s[ticker])
                        is_royal = (price > ma30) and (ma30 > ma60) and (ma60 > ma200)

                        stock_info = stock_dict.get(ticker)
                        if not stock_info: continue
                        
                        bias = ((price - ma200) / ma200) * 100

                        raw_data_list.append({
                            "代號": stock_info["code"], "名稱": stock_info["name"], "產業": stock_info["group"],
                            "完整代號": ticker, "收盤價": float(price), "生命線": float(ma200),
                            "MA30": float(ma30), "MA60": float(ma60), "生命線趨勢": ma_trend,
                            "乖離率(%)": float(bias), "abs_bias": abs(float(bias)),
                            "成交量": int(last_vol_s[ticker]), "昨日成交量": int(prev_vol_s[ticker]),
                            "K值": float(k_val), "D值": float(d_val),
                            "位置": "🟢生命線上" if price >= ma200 else "🔴生命線下",
                            "浴火重生": is_treasure, "皇冠特選": is_royal,
                            "打腳發動": is_leg_kick,
                            "打腳日期": leg_kick_date.strftime("%Y-%m-%d") if leg_kick_date else "",
                        })
                    except: continue
        except: pass
        progress_bar.progress((i + 1) / total_batches, text=f"挖掘中...({int((i+1)/total_batches*100)}%)")
        time.sleep(0.2)
    return pd.DataFrame(raw_data_list)

def plot_stock_chart(ticker, name):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        df = df[df["Volume"] > 0].dropna()
        if df.empty: return

        df["200MA"] = df["Close"].rolling(window=200).mean()
        df["30MA"] = df["Close"].rolling(window=30).mean()
        df["60MA"] = df["Close"].rolling(window=60).mean()
        plot_df = df.tail(120).copy()
        plot_df["DateStr"] = plot_df.index.strftime("%Y-%m-%d")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df["DateStr"], y=plot_df["Close"], mode="lines", name="收盤價", line=dict(color="#00CC96", width=2.5)))
        fig.add_trace(go.Scatter(x=plot_df["DateStr"], y=plot_df["30MA"], mode="lines", name="30MA", line=dict(color="#AB63FA", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=plot_df["DateStr"], y=plot_df["60MA"], mode="lines", name="60MA", line=dict(color="#19D3F3", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=plot_df["DateStr"], y=plot_df["200MA"], mode="lines", name="200MA", line=dict(color="#FFA15A", width=3)))
        fig.update_layout(title=f"📊 {name} ({ticker})", yaxis_title="價格", height=500, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    except: st.error("繪圖失敗")

# --- 3. 介面顯示區 ---
st.title(f"✨ {VER} 黑嚕嚕-旗鼓相當")
st.markdown("---")

if "master_df" not in st.session_state: st.session_state["master_df"] = None
if "last_update" not in st.session_state: st.session_state["last_update"] = None
if "backtest_result" not in st.session_state: st.session_state["backtest_result"] = None

with st.sidebar:
    st.header("資料庫管理")
    CACHE_FILE = "stock_data_cache.csv"

    if st.button("🚨 強制重置系統"):
        st.cache_data.clear(); st.session_state.clear()
        if os.path.exists(CACHE_FILE): os.remove(CACHE_FILE)
        st.rerun()

    if st.session_state["master_df"] is None and os.path.exists(CACHE_FILE):
        try:
            st.session_state["master_df"] = pd.read_csv(CACHE_FILE)
            mod_time = os.path.getmtime(CACHE_FILE)
            st.session_state["last_update"] = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
            st.success(f"⚡ 已載入 ({st.session_state['last_update']})")
        except: st.error("快取讀取失敗")

    # --- 策略與參數設定 ---
    st.divider()
    st.header("1. 策略參數")
    
    strategy_mode = st.radio(
        "選擇策略：",
        ("🛡️ 生命線保衛戰 (反彈/支撐)", "🔥 起死回生 (Da來守住)", "🐎 多頭馬車發動 (多頭排列)", "🦵 打腳發動 (KD+紅吞)")
    )

    # 預設變數
    leg_kick_days = 60 # 預設值

    if strategy_mode == "🦵 打腳發動 (KD+紅吞)":
        st.info("參數說明：設定往回推幾天內曾出現超賣(K<20)")
        # 新增滑桿：範圍 20~60，間距 5
        leg_kick_days = st.slider("🦵 KD前置搜尋天數 (Lookback)", 20, 60, 60, step=5)

    min_vol_input = st.number_input("最低成交量 (張)", value=1000, step=100)
    bias_threshold = st.slider("乖離率範圍 (±%)", 0.5, 5.0, 2.5, step=0.1)

    # 細部條件
    filter_trend_up = False; filter_trend_down = False; filter_kd = False; filter_vol_double = False
    if strategy_mode == "🛡️ 生命線保衛戰 (反彈/支撐)":
        c1, c2 = st.columns(2)
        with c1: filter_trend_up = st.checkbox("生命線向上")
        with c2: filter_trend_down = st.checkbox("生命線向下")
        filter_kd = st.checkbox("KD 黃金交叉")
        filter_vol_double = st.checkbox("出量 (今日>昨日x1.5)")
    elif strategy_mode == "🔥 起死回生 (Da來守住)":
        filter_vol_double = st.checkbox("出量確認")
    elif strategy_mode == "🐎 多頭馬車發動 (多頭排列)":
        filter_vol_double = st.checkbox("出量確認")

    st.markdown("---")
    
    # 下載按鈕 (傳入 leg_kick_days)
    if st.button("🔄 下載最新股價 (開市用)", type="primary"):
        stock_dict = get_stock_list()
        if not stock_dict: st.error("無法取得清單")
        else:
            ph = st.empty(); bar = st.progress(0, text="準備下載...")
            with ph: st.markdown("""<div style="text-align:center;font-size:30px;">⏳</div>""", unsafe_allow_html=True)
            
            # 傳遞參數
            df = fetch_all_data(stock_dict, bar, st.empty(), leg_kick_days)
            
            if not df.empty:
                df.to_csv(CACHE_FILE, index=False)
                st.session_state["master_df"] = df
                st.session_state["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.success(f"更新完成！({len(df)} 檔)")
            else: st.error("連線阻擋，請 Reboot App。")
            ph.empty(); bar.empty()

    if st.session_state["last_update"]: st.caption(f"Update: {st.session_state['last_update']}")

    # 回測按鈕 (傳入 leg_kick_days)
    if st.button("🧪 策略回測"):
        st.info("正在調閱歷史檔案... ⏳")
        stock_dict = get_stock_list()
        bt_progress = st.progress(0, text="初始化...")
        
        use_treasure_p = (strategy_mode == "🔥 起死回生 (Da來守住)")
        use_royal_p = (strategy_mode == "🐎 多頭馬車發動 (多頭排列)")
        use_legkick_p = (strategy_mode == "🦵 打腳發動 (KD+紅吞)")

        bt_df = run_strategy_backtest(
            stock_dict, bt_progress, strategy_mode,
            filter_trend_up, use_treasure_p, filter_vol_double,
            use_royal_p, use_legkick_p, min_vol_input,
            leg_kick_days # 傳入參數
        )
        st.session_state["backtest_result"] = bt_df
        bt_progress.empty()
        st.success("回測完成！")

# 主畫面 - 回測報告
if st.session_state["backtest_result"] is not None:
    bt_df = st.session_state["backtest_result"]
    st.markdown("---")
    st.subheader(f"🧪 策略回測報告：{strategy_mode}")
    if "結果" in bt_df.columns:
        df_w = bt_df[bt_df["結果"] == "觀察中"].copy()
        df_h = bt_df[bt_df["結果"] != "觀察中"].copy()
    else: df_h = bt_df.copy(); df_w = bt_df.iloc[0:0]

    if not df_w.empty:
        st.markdown("""<div style="background:#fff8dc;padding:10px;border-left:5px solid #ffa500;">
        <b>👀 關注中訊號</b></div><br>""", unsafe_allow_html=True)
        # 穩定顯示 (style.map)
        def color_pos(val): return f'color: {"red" if val > 0 else "green"}'
        st.dataframe(df_w[["代號","名稱","產業","訊號日期","訊號價","最高漲幅(%)"]].style.map(color_pos, subset=["最高漲幅(%)"]), use_container_width=True, hide_index=True)
    else: st.info("目前無進行中訊號。")

    if not df_h.empty:
        st.markdown("### 📜 已結算數據")
        win_count = len(df_h[df_h["結果"].str.contains("Win|成功")])
        win_rate = int(win_count / len(df_h) * 100)
        c1, c2 = st.columns(2)
        c1.metric("總次數", len(df_h))
        c2.metric("勝率", f"{win_rate}%")
        st.dataframe(df_h, use_container_width=True)

# 主畫面 - 篩選結果
if st.session_state["master_df"] is not None:
    df = st.session_state["master_df"].copy()
    if "生命線" not in df.columns: st.error("請重置系統！"); st.stop()
    
    df = df[df["成交量"] >= min_vol_input*1000]
    
    if strategy_mode == "🔥 起死回生 (Da來守住)": df = df[df["浴火重生"]==True]
    elif strategy_mode == "🐎 多頭馬車發動 (多頭排列)": df = df[df["皇冠特選"]==True]
    elif strategy_mode == "🦵 打腳發動 (KD+紅吞)": df = df[df["打腳發動"]==True]
    else:
        df = df[df["abs_bias"] <= bias_threshold]
        if filter_trend_up: df = df[df["生命線趨勢"]=="⬆️向上"]
        if filter_trend_down: df = df[df["生命線趨勢"]=="⬇️向下"]
        if filter_kd: df = df[df["K值"] > df["D值"]]

    if filter_vol_double: df = df[df["成交量"] > df["昨日成交量"]*1.5]

    if df.empty:
        st.warning("⚠️ 無符合條件標的")
    else:
        st.markdown(f"""<div style="background:#f0f2f6;padding:10px;text-align:center;border:2px solid #ff4b4b;">
        <h3>🔍 篩選出 <span style="color:#ff4b4b;">{len(df)}</span> 檔</h3></div><br>""", unsafe_allow_html=True)
        
        df["成交量(張)"] = (df["成交量"]/1000).astype(int)
        df["KD值"] = df.apply(lambda x: f"K:{int(x['K值'])} D:{int(x['D值'])}", axis=1)
        df["選股標籤"] = df["代號"].astype(str) + " " + df["名稱"].astype(str)
        
        cols = ["代號","名稱","產業","收盤價","生命線","乖離率(%)","位置","KD值","成交量(張)"]
        if strategy_mode == "🐎 多頭馬車發動 (多頭排列)": cols = ["代號","名稱","產業","收盤價","MA30","MA60","生命線","KD值","成交量(張)"]
        
        df = df.sort_values(by="成交量", ascending=False)
        
        t1, t2 = st.tabs(["📋 列表", "📊 圖表"])
        with t1:
            # 穩定著色 (style.map) - 避免 matplotlib 依賴問題
            def color_bias(val): return f'color: {"red" if val > 0 else "green"}; font-weight: bold'
            try:
                st.dataframe(df[cols].style.map(color_bias, subset=["乖離率(%)"]), use_container_width=True, hide_index=True)
            except:
                st.dataframe(df[cols], use_container_width=True, hide_index=True)

        with t2:
            sel = st.selectbox("選擇股票：", df["選股標籤"].tolist())
            row = df[df["選股標籤"]==sel].iloc[0]
            plot_stock_chart(row["完整代號"], row["名稱"])
            c1, c2, c3 = st.columns(3)
            c1.metric("價", row['收盤價']); c2.metric("量", f"{row['成交量(張)']}張"); c3.metric("KD", row['KD值'])

else:
    st.warning("👈 請點擊左側 **「🔄 下載最新股價」**")
    c1, c2, c3 = st.columns([1, 3, 1])
    with c2:
        if os.path.exists("welcome.jpg"):
            st.markdown("""<div style="text-align:center;margin-bottom:20px;">
            預祝心想事成，從從容容，紫氣東來! 🟣✨</div>""", unsafe_allow_html=True)
            st.image("welcome.jpg", width=600) # 放大圖片
        else: st.info("💡 尚未偵測到 welcome.jpg")
