import streamlit as st
import yfinance as yf
import pandas as pd
import twstock
import time
import random
import gc  # 引入垃圾回收機制 (System)
from datetime import datetime, timedelta
import plotly.graph_objects as go
import os
import uuid
import csv

# --- 1. 網頁設定 ---
VER = "ver 2.4 (固若金湯版)"
st.set_page_config(page_title=f"✨ 黑嚕嚕-旗鼓相當({VER})", layout="wide")

# --- 流量紀錄與後台功能 ---
LOG_FILE = "traffic_log.csv"

def get_remote_ip():
    """嘗試取得使用者 IP"""
    try:
        if hasattr(st, "context") and st.context.headers:
            headers = st.context.headers
            if "X-Forwarded-For" in headers:
                return headers["X-Forwarded-For"].split(",")[0]
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
        exclude_industries = ["金融保險業", "存託憑證"]
        for code, info in tse.items():
            if info.type == "股票" and info.group not in exclude_industries:
                stock_dict[f"{code}.TW"] = {"name": info.name, "code": code, "group": info.group}
        for code, info in otc.items():
            if info.type == "股票" and info.group not in exclude_industries:
                stock_dict[f"{code}.TWO"] = {"name": info.name, "code": code, "group": info.group}
        return stock_dict
    except:
        return {}

def test_connection():
    try:
        test_ticker = "2330.TW"
        data = yf.download(test_ticker, period="5d", progress=False, threads=False)
        if not data.empty:
            return True, f"✅ 連線成功！成功抓取 {test_ticker} (資料筆數: {len(data)})"
        else:
            return False, f"❌ 連線失敗！抓取 {test_ticker} 回傳空值 (可能是 IP 被鎖)"
    except Exception as e:
        return False, f"❌ 連線錯誤: {str(e)}"

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

def calculate_kd_series(df, n=9):
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
    prev_is_black = prev_close < prev_open
    curr_is_red = curr_close > curr_open
    engulf = (curr_open <= prev_close) and (curr_close > prev_open)
    return prev_is_black and curr_is_red and engulf

# --- 新增策略：🛡️ 固若金湯 (破底翻突破) ---
def detect_solid_defense_signal(stock_df, k_series, lookback=60):
    if len(stock_df) < 15: # 至少需要一些天數來成型
        return False, {}

    recent_df = stock_df.tail(lookback).copy()
    idx_list = list(recent_df.index)
    today_idx = idx_list[-1]

    # 1. 找最低點 (到昨天為止的最低點，因為今天要突破)
    past_df = recent_df.iloc[:-1]
    if past_df.empty: return False, {}
    
    t_low = past_df['Low'].idxmin()
    t_low_pos = idx_list.index(t_low)

    if t_low_pos == len(idx_list) - 1: 
        return False, {} # 最低點太近，沒有時間形成箱型和守住訊號

    Daa = float(recent_df.loc[t_low, 'Low'])
    Da = float(recent_df.loc[t_low, 'Close'])

    # 2. 找箱頂 (最低點之後，到昨天為止的最高點)
    box_period = recent_df.iloc[t_low_pos + 1 : -1]
    if box_period.empty:
        box_top = Da # 防呆
    else:
        box_top = float(box_period['High'].max())

    # 3. 檢查是否有「守住」訊號
    held_support = False
    hold_date = None
    hold_type = ""

    for i in range(t_low_pos + 1, len(idx_list)):
        curr_dt = idx_list[i]
        prev_dt = idx_list[i - 1]

        curr_o, curr_c = float(recent_df.loc[curr_dt, 'Open']), float(recent_df.loc[curr_dt, 'Close'])
        prev_o, prev_c = float(recent_df.loc[prev_dt, 'Open']), float(recent_df.loc[prev_dt, 'Close'])

        if curr_dt in k_series.index:
            curr_k = float(k_series.loc[curr_dt])
        else:
            curr_k = 0.0

        is_engulf = _is_red_engulf_black(prev_o, prev_c, curr_o, curr_c)

        if is_engulf and curr_k > 20:
            if curr_c > Da:
                held_support = True
                hold_date = curr_dt
                hold_type = "守住Da"
            elif curr_c > Daa:
                held_support = True
                hold_date = curr_dt
                hold_type = "守住Daa"

    if not held_support:
        return False, {}

    # 4. 檢查今天是否為進場發動點 (拉出一根紅棒，且突破箱型頂部)
    today_o, today_c = float(recent_df.loc[today_idx, 'Open']), float(recent_df.loc[today_idx, 'Close'])
    is_red_today = today_c > today_o
    is_breakout = today_c > box_top

    if is_red_today and is_breakout:
        return True, {
            "最低點日期": t_low,
            "Da": Da,
            "Daa": Daa,
            "箱頂": box_top,
            "守住日期": hold_date,
            "守住類型": hold_type
        }

    return False, {}

def detect_leg_kick_signal(stock_df, lookback=60, trigger_days=3, kd_threshold=20):
    if len(stock_df) < max(lookback + 2, 30): return False, None, None, None
    recent_df = stock_df.tail(lookback).copy()
    if len(recent_df) < 20: return False, None, None, None

    k_series, d_series = calculate_kd_series(recent_df)
    t1 = k_series[k_series < kd_threshold].last_valid_index()
    if t1 is None: return False, None, None, None
    oversold_close = float(recent_df.loc[t1, "Close"])

    idx_list = list(recent_df.index)
    t1_pos = idx_list.index(t1)
    t_cross = None
    for i in range(t1_pos + 1, len(idx_list)):
        dt = idx_list[i]
        prev_dt = idx_list[i - 1]
        if (k_series.loc[prev_dt] <= d_series.loc[prev_dt]) and (k_series.loc[dt] > d_series.loc[dt]):
            t_cross = dt
            break
    
    if t_cross is None: return False, None, t1, None

    cross_pos = idx_list.index(t_cross)
    end_pos = min(cross_pos + trigger_days, len(idx_list) - 1)

    for i in range(cross_pos, end_pos + 1):
        dt = idx_list[i]
        if i == 0: continue
        if float(k_series.loc[dt]) < kd_threshold: continue

        prev_row = recent_df.iloc[i - 1]
        curr_row = recent_df.iloc[i]
        prev_open, prev_close = float(prev_row["Open"]), float(prev_row["Close"])
        curr_open, curr_close = float(curr_row["Open"]), float(curr_row["Close"])

        if _is_red_engulf_black(prev_open, prev_close, curr_open, curr_close) and (curr_close > oversold_close):
            return True, dt, t1, t_cross

    return False, None, t1, t_cross

def detect_w_bottom_signal(stock_df, k_series, d_series, lookback=60):
    if len(stock_df) < 30: return False, None, None
    valid_idx = stock_df.index.intersection(k_series.index)
    if len(valid_idx) < 30: return False, None, None
    target_k = k_series.loc[valid_idx].tail(lookback)
    
    k_under_20 = target_k[target_k < 20]
    if k_under_20.empty: return False, None, None
    t_left = k_under_20.last_valid_index()
    if valid_idx.get_loc(t_left) > len(valid_idx) - 5: return False, None, None
    left_low = float(stock_df.loc[t_left, "Close"]) 
    
    t_today = valid_idx[-1]
    structure_mask = (valid_idx > t_left) & (valid_idx < t_today)
    structure_period = stock_df.loc[structure_mask]
    if structure_period.empty: return False, None, None
    
    t_peak = structure_period["High"].idxmax()
    peak_k = float(k_series.loc[t_peak])
    
    if peak_k >= 80: return False, None, None
        
    curr_row = stock_df.iloc[-1]
    prev_row = stock_df.iloc[-2]
    curr_open, curr_close = float(curr_row["Open"]), float(curr_row["Close"])
    prev_open, prev_close = float(prev_row["Open"]), float(prev_row["Close"])
    
    if not _is_red_engulf_black(prev_open, prev_close, curr_open, curr_close): return False, None, None
    if curr_close <= left_low: return False, None, None

    return True, t_left, t_peak

def run_strategy_backtest(
    stock_dict,
    progress_bar,
    mode,
    use_trend_up,
    use_treasure,
    use_vol,
    use_solid_defense,
    use_leg_kick,
    use_w_bottom,
    min_vol_threshold,
):
    results = []
    all_tickers = list(stock_dict.keys())
    BATCH_SIZE = 40
    total_batches = (len(all_tickers) // BATCH_SIZE) + 1
    OBSERVE_DAYS = 30 

    for i, batch_idx in enumerate(range(0, len(all_tickers), BATCH_SIZE)):
        batch = all_tickers[batch_idx : batch_idx + BATCH_SIZE]
        try:
            data = yf.download(batch, period="2y", interval="1d", progress=False, auto_adjust=False, threads=False)
            if data.empty: 
                time.sleep(8)
                data = yf.download(batch, period="2y", interval="1d", progress=False, auto_adjust=False, threads=False)
                if data.empty: continue
                
            try:
                df_o, df_c = data["Open"], data["Close"]
                df_v, df_l, df_h = data["Volume"], data["Low"], data["High"]
            except KeyError: continue

            if isinstance(df_c, pd.Series):
                df_o = df_o.to_frame(name=batch[0])
                df_c = df_c.to_frame(name=batch[0])
                df_v = df_v.to_frame(name=batch[0])
                df_l = df_l.to_frame(name=batch[0])
                df_h = df_h.to_frame(name=batch[0])

            ma200_df = df_c.rolling(window=200).mean()
            ma30_df = df_c.rolling(window=30).mean()
            ma60_df = df_c.rolling(window=60).mean()
            scan_window = df_c.index[-90:]

            for ticker in df_c.columns:
                try:
                    c_series = df_c[ticker].dropna()
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

                    full_ohlc = pd.DataFrame({
                        "Open": o_series, "Close": c_series, "High": h_series, "Low": l_series
                    }).dropna()

                    k_full, d_full = calculate_kd_series(full_ohlc)

                    for date in scan_window:
                        if date not in c_series.index: continue
                        idx = c_series.index.get_loc(date)
                        if idx < 200: continue

                        close_p = float(c_series.iloc[idx])
                        vol = float(v_series.iloc[idx]) if date in v_series.index else 0.0
                        prev_vol = float(v_series.iloc[idx - 1]) if idx - 1 >= 0 else 0.0
                        ma200_val = float(ma200_series.iloc[idx]) if not pd.isna(ma200_series.iloc[idx]) else 0.0

                        if ma200_val == 0 or vol < (min_vol_threshold * 1000): continue
                        if prev_vol == 0: prev_vol = 1.0

                        is_match = False
                        detail_info = {} 
                        stop_loss_price = 0.0
                        target_price = 0.0

                        if use_solid_defense:
                            sub_df = full_ohlc.loc[:date].copy()
                            sd_ok, sd_det = detect_solid_defense_signal(sub_df, k_full, lookback=60)
                            if sd_ok:
                                is_match = True
                                detail_info["最低點日期"] = sd_det["最低點日期"].strftime("%m-%d")
                                detail_info["守住日期"] = sd_det["守住日期"].strftime("%m-%d")
                                detail_info["守住類型"] = sd_det["守住類型"]
                                detail_info["箱頂"] = round(sd_det["箱頂"], 2)
                                stop_loss_price = sd_det["Daa"]
                                target_price = close_p + (close_p - stop_loss_price) * 1.5

                        elif use_w_bottom:
                            sub_df = full_ohlc.loc[:date].copy()
                            w_ok, t_left, t_peak = detect_w_bottom_signal(sub_df, k_full, d_full, lookback=60)
                            if w_ok:
                                is_match = True
                                detail_info["左腳日期"] = t_left.strftime("%m-%d")
                                left_low_p = float(sub_df.loc[t_left, "Low"])
                                stop_loss_price = left_low_p
                                neck_high_p = float(sub_df.loc[t_peak, "High"])
                                amplitude = neck_high_p - left_low_p
                                target_price = close_p + (2 * amplitude)

                        elif use_leg_kick:
                            sub_df = full_ohlc.loc[:date].copy()
                            ok, trig_dt, t_low, t_cross = detect_leg_kick_signal(sub_df, lookback=60, trigger_days=3, kd_threshold=20)
                            if ok and trig_dt == date:
                                is_match = True
                                detail_info["KD低點"] = t_low.strftime("%m-%d") if t_low else ""
                                detail_info["KD金叉"] = t_cross.strftime("%m-%d") if t_cross else ""
                                try:
                                    left_low_p = float(sub_df.loc[t_low, "Low"])
                                except:
                                    left_low_p = float(sub_df.loc[t_low, "Close"])
                                stop_loss_price = left_low_p
                                amplitude = close_p - left_low_p
                                target_price = close_p + (2 * amplitude)

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
                                cond_past_down = (past_c < past_ma).any()
                                if cond_today_up and cond_past_down: is_match = True
                            else:
                                cond_near = (low_p <= ma200_val * 1.03) and (low_p >= ma200_val * 0.90)
                                cond_up = (close_p > ma200_val)
                                if cond_near and cond_up: is_match = True

                        if not is_match: continue

                        month_str = date.strftime("%m月")
                        days_after_signal = total_len - 1 - idx
                        final_profit_pct = 0.0
                        result_status = "觀察中"
                        is_watching = False
                        if days_after_signal < 1: is_watching = True
                        else:
                            MAX_HOLD_DAYS = 30
                            check_days = min(days_after_signal, MAX_HOLD_DAYS)
                            is_watching = True
                            
                            for d in range(1, check_days + 1):
                                curr_idx = idx + d
                                if curr_idx >= len(c_series): break
                                curr_c = float(c_series.iloc[curr_idx])
                                curr_h = float(h_series.iloc[curr_idx])
                                curr_k = float(k_full.iloc[curr_idx])
                                curr_d = float(d_full.iloc[curr_idx])
                                prev_k = float(k_full.iloc[curr_idx - 1])
                                prev_d = float(d_full.iloc[curr_idx - 1])
                                
                                if stop_loss_price > 0 and curr_c < stop_loss_price:
                                    final_profit_pct = (curr_c - close_p) / close_p * 100
                                    is_watching = False
                                    result_status = "Loss (破防守) 🛑"
                                    break
                                
                                if target_price > 0 and curr_h >= target_price:
                                    final_profit_pct = (target_price - close_p) / close_p * 100
                                    is_watching = False
                                    result_status = "Win (達標停利) 🎯"
                                    break
                                    
                                if (prev_k > 80) and (prev_k >= prev_d) and (curr_k < curr_d):
                                    final_profit_pct = (curr_c - close_p) / close_p * 100
                                    is_watching = False
                                    result_status = "Win (KD>80死叉) 📉"
                                    break
                            
                            if is_watching:
                                current_price = float(c_series.iloc[-1])
                                final_profit_pct = (current_price - close_p) / close_p * 100
                                if days_after_signal >= MAX_HOLD_DAYS:
                                    end_close = float(c_series.iloc[idx + MAX_HOLD_DAYS])
                                    final_profit_pct = (end_close - close_p) / close_p * 100
                                    result_status = "Win (期滿)" if final_profit_pct > 0 else "Loss (期滿)"
                                    is_watching = False
                                else:
                                    result_status = "觀察中"

                        record = {
                            "月份": "👀 關注中" if is_watching else month_str,
                            "代號": ticker.replace(".TW", "").replace(".TWO", ""),
                            "名稱": stock_name,
                            "產業": stock_industry,
                            "訊號日期": date.strftime("%Y-%m-%d"),
                            "訊號價": round(close_p, 2),
                            "損益(%)": round(final_profit_pct, 2),
                            "結果": "觀察中" if is_watching else result_status,
                        }
                        if use_solid_defense:
                            record["最低點"] = detail_info.get("最低點日期", "")
                            record["守住日期"] = detail_info.get("守住日期", "")
                            record["守住類型"] = detail_info.get("守住類型", "")
                            record["箱頂"] = detail_info.get("箱頂", "")
                        if use_leg_kick:
                            record["KD低點"] = detail_info.get("KD低點", "")
                            record["KD金叉"] = detail_info.get("KD金叉", "")
                        if use_w_bottom:
                            record["左腳"] = detail_info.get("左腳日期", "")
                        results.append(record)
                except: continue
        except: pass
        progress = (i + 1) / total_batches
        progress_bar.progress(progress, text=f"深度回測中 (計算分月數據)...({int(progress*100)}%)")
        time.sleep(random.uniform(0.8, 1.5))
        gc.collect() 
    return pd.DataFrame(results)

def fetch_all_data(stock_dict, progress_bar, status_text, debug_container=None):
    if not stock_dict: 
        st.error("❌ 無法取得股票清單 (twstock 阻擋或失敗)！")
        return pd.DataFrame()
        
    all_tickers = list(stock_dict.keys())
    BATCH_SIZE = 40 
    total_batches = (len(all_tickers) // BATCH_SIZE) + 1
    raw_data_list = []
    
    debug_logs = []
    log_area = None
    if debug_container:
        log_area = debug_container.empty()

    for i, batch_idx in enumerate(range(0, len(all_tickers), BATCH_SIZE)):
        batch = all_tickers[batch_idx : batch_idx + BATCH_SIZE]
        try:
            data = yf.download(batch, period="1y", interval="1d", progress=False, auto_adjust=False, threads=False)
            
            msg = f"Batch {i+1}: 嘗試下載 {len(batch)} 檔"
            if data.empty:
                msg += " ❌ (Empty Response)"
                time.sleep(8) 
                data = yf.download(batch, period="1y", interval="1d", progress=False, auto_adjust=False, threads=False)
                if data.empty:
                    msg += " -> 重試失敗"
                else:
                    msg += f" -> 重試成功 ({len(data.columns)})"
            else:
                msg += f" ✅ ({len(data.columns)} 筆資料)"
            
            debug_logs.append(msg)
            if log_area:
                log_area.text("\n".join(debug_logs[-10:]))

            if not data.empty:
                try:
                    df_o, df_c = data["Open"], data["Close"]
                    df_h, df_l, df_v = data["High"], data["Low"], data["Volume"]
                except KeyError: continue

                if isinstance(df_c, pd.Series):
                    df_o = df_o.to_frame(name=batch[0])
                    df_c = df_c.to_frame(name=batch[0])
                    df_h = df_h.to_frame(name=batch[0])
                    df_l = df_l.to_frame(name=batch[0])
                    df_v = df_v.to_frame(name=batch[0])

                ma200_df = df_c.rolling(window=200).mean()
                ma30_df = df_c.rolling(window=30).mean()
                ma60_df = df_c.rolling(window=60).mean()
                
                last_price_series = df_c.iloc[-1]
                last_ma200_series = ma200_df.iloc[-1]
                last_ma30_series = ma30_df.iloc[-1]
                last_ma60_series = ma60_df.iloc[-1]
                prev_ma200_series = ma200_df.iloc[-21]
                last_vol_series = df_v.iloc[-1]
                prev_vol_series = df_v.iloc[-2]
                recent_close_df = df_c.iloc[-8:]
                recent_ma200_df = ma200_df.iloc[-8:]
                current_market_date = df_c.index[-1]

                for ticker in df_c.columns:
                    try:
                        price = float(last_price_series[ticker])
                        ma200 = float(last_ma200_series[ticker])
                        ma30 = float(last_ma30_series[ticker])
                        ma60 = float(last_ma60_series[ticker])
                        prev_ma200 = float(prev_ma200_series[ticker])
                        vol = float(last_vol_series[ticker])
                        prev_vol = float(prev_vol_series[ticker])

                        if pd.isna(price) or pd.isna(ma200) or ma200 == 0: continue
                        ma_trend = "⬆️向上" if ma200 >= prev_ma200 else "⬇️向下"

                        is_treasure = False
                        my_recent_c = recent_close_df[ticker]
                        my_recent_ma = recent_ma200_df[ticker]
                        if len(my_recent_c) >= 8:
                            cond_today_up = my_recent_c.iloc[-1] > my_recent_ma.iloc[-1]
                            cond_past_down = (my_recent_c.iloc[:-1] < my_recent_ma.iloc[:-1]).any()
                            if cond_today_up and cond_past_down: is_treasure = True

                        stock_df = pd.DataFrame({
                            "Open": df_o[ticker], "Close": df_c[ticker],
                            "High": df_h[ticker], "Low": df_l[ticker],
                        }).dropna()

                        k_val, d_val = 0.0, 0.0
                        
                        is_solid_defense = False
                        sd_details = {}
                        is_leg_kick = False
                        leg_kick_date = None
                        t_low = None
                        t_cross = None
                        is_w_bottom = False
                        w_left_date = None
                        w_peak_date = None

                        if len(stock_df) >= 20:
                            k_series, d_series = calculate_kd_series(stock_df)
                            k_val = float(k_series.iloc[-1])
                            d_val = float(d_series.iloc[-1])

                            # 固若金湯策略
                            sd_ok, sd_det = detect_solid_defense_signal(stock_df, k_series, lookback=60)
                            if sd_ok:
                                is_solid_defense = True
                                sd_details = sd_det

                            # 蓄勢待發
                            is_leg_kick, leg_kick_date, t_low, t_cross = detect_leg_kick_signal(stock_df, lookback=60, trigger_days=3, kd_threshold=20)
                            if is_leg_kick:
                                day_diff = (current_market_date - leg_kick_date).days
                                if day_diff > 5: is_leg_kick = False

                            # 光神腳
                            w_ok, t_left, t_peak = detect_w_bottom_signal(stock_df, k_series, d_series, lookback=60)
                            if w_ok:
                                is_w_bottom = True
                                w_left_date = t_left
                                w_peak_date = t_peak
                        else:
                            if len(stock_df) >= 9: k_val, d_val = calculate_kd_values(stock_df)

                        bias = ((price - ma200) / ma200) * 100
                        stock_info = stock_dict.get(ticker)
                        if not stock_info: continue

                        raw_data_list.append({
                            "代號": stock_info["code"],
                            "名稱": stock_info["name"],
                            "產業": stock_info["group"],
                            "完整代號": ticker,
                            "收盤價": float(price),
                            "生命線": float(ma200),
                            "MA30": float(ma30),
                            "MA60": float(ma60),
                            "生命線趨勢": ma_trend,
                            "乖離率(%)": float(bias),
                            "abs_bias": abs(float(bias)),
                            "成交量": int(vol),
                            "昨日成交量": int(prev_vol),
                            "K值": float(k_val),
                            "D值": float(d_val),
                            "位置": "🟢生命線上" if price >= ma200 else "🔴生命線下",
                            "浴火重生": is_treasure,
                            "固若金湯": is_solid_defense,
                            "最低點日期": sd_details.get("最低點日期", "").strftime("%Y-%m-%d") if sd_details else "",
                            "守住日期": sd_details.get("守住日期", "").strftime("%Y-%m-%d") if sd_details else "",
                            "守住類型": sd_details.get("守住類型", ""),
                            "箱頂": round(sd_details.get("箱頂", 0.0), 2) if sd_details else 0.0,
                            "蓄勢待發": is_leg_kick,
                            "蓄勢日期": leg_kick_date.strftime("%Y-%m-%d") if leg_kick_date else "",
                            "KD低點": t_low.strftime("%Y-%m-%d") if t_low else "",
                            "KD金叉": t_cross.strftime("%Y-%m-%d") if t_cross else "",
                            "光神腳": is_w_bottom,
                            "左腳日期": w_left_date.strftime("%Y-%m-%d") if w_left_date else "",
                            "中高日期": w_peak_date.strftime("%Y-%m-%d") if w_peak_date else "",
                        })
                    except: continue
        except Exception as e:
            debug_logs.append(f"Batch {i+1} Error: {str(e)}")
            if log_area: log_area.text("\n".join(debug_logs[-10:]))
            pass
            
        current_progress = (i + 1) / total_batches
        progress_bar.progress(current_progress, text=f"系統正在努力挖掘寶藏中...({int(current_progress*100)}%)")
        time.sleep(random.uniform(0.8, 1.5))
        gc.collect() 
    return pd.DataFrame(raw_data_list)

def plot_stock_chart(ticker, name, strategy_mode=""):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        df = df[df["Volume"] > 0].dropna()
        if df.empty:
            st.error("無法取得有效數據")
            return

        df["200MA"] = df["Close"].rolling(window=200).mean()
        df["30MA"] = df["Close"].rolling(window=30).mean()
        df["60MA"] = df["Close"].rolling(window=60).mean()
        
        plot_df = df.tail(150).copy() 
        plot_df["DateStr"] = plot_df.index.strftime("%Y-%m-%d")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df["DateStr"], y=plot_df["Close"], mode="lines", name="收盤價", line=dict(color="#00CC96", width=2.5)))
        fig.add_trace(go.Scatter(x=plot_df["DateStr"], y=plot_df["30MA"], mode="lines", name="30MA(月線)", line=dict(color="#AB63FA", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=plot_df["DateStr"], y=plot_df["60MA"], mode="lines", name="60MA(季線)", line=dict(color="#19D3F3", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=plot_df["DateStr"], y=plot_df["200MA"], mode="lines", name="200MA(生命線)", line=dict(color="#FFA15A", width=3)))

        fig.update_layout(
            title=f"📊 {name} ({ticker}) 股價 vs 均線排列",
            yaxis_title="價格", height=500, hovermode="x unified",
            xaxis=dict(type="category", tickangle=-45, nticks=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e: st.error(f"繪圖失敗: {e}")

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
        st.cache_data.clear()
        st.session_state.clear()
        if os.path.exists(CACHE_FILE): os.remove(CACHE_FILE)
        st.success("系統已重置！請重新點擊更新股價。")
        st.rerun()

    if st.session_state["master_df"] is None and os.path.exists(CACHE_FILE):
        try:
            df_cache = pd.read_csv(CACHE_FILE)
            st.session_state["master_df"] = df_cache
            mod_time = os.path.getmtime(CACHE_FILE)
            st.session_state["last_update"] = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
            st.success(f"⚡ 已快速載入上次資料 ({st.session_state['last_update']})")
        except Exception as e: st.error(f"讀取快取失敗: {e}")

    if st.button("🩺 測試連線 (Check IP)"):
        ok, msg = test_connection()
        if ok: st.success(msg)
        else: st.error(msg)

    if st.button("🔄 下載最新股價 (開市用)", type="primary"):
        stock_dict = get_stock_list()
        if not stock_dict: st.error("無法取得股票清單")
        else:
            placeholder_emoji = st.empty()
            with placeholder_emoji:
                st.markdown("""<div style="text-align: center; font-size: 40px; animation: blink 1s infinite;">🎁💰✨</div>
                    <style>@keyframes blink { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }</style>
                    <div style="text-align: center;">連線下載中 (Batch=40, Period=1Y)...</div>""", unsafe_allow_html=True)
            
            debug_container = st.expander("🕵️ 下載詳細日誌 (Debug Log)", expanded=True)
            
            status_text = st.empty()
            progress_bar = st.progress(0, text="準備下載...")
            df = fetch_all_data(stock_dict, progress_bar, status_text, debug_container)
            
            if not df.empty:
                df.to_csv(CACHE_FILE, index=False)
                st.session_state["master_df"] = df
                st.session_state["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.success(f"更新完成！共 {len(df)} 檔資料")
            else:
                st.error("⛔ 連線資料庫阻擋。請查看上方日誌了解詳情。")
                with st.expander("🆘 Reboot App (點我展開)"): st.info("請點擊右上角「⋮」->「Reboot App」")
            placeholder_emoji.empty()
            progress_bar.empty()

    if st.session_state["last_update"]: st.caption(f"最後更新：{st.session_state['last_update']}")
    st.divider()
    with st.expander("🔐 管理員後台"):
        admin_pwd = st.text_input("請輸入管理密碼", type="password", key="admin_pwd_input")
        if admin_pwd == "admin1133":
            if os.path.exists(LOG_FILE):
                st.markdown("### 🚦 流量統計")
                log_df = pd.read_csv(LOG_FILE)
                st.metric("總點擊", len(log_df))
                st.dataframe(log_df.sort_values(by="時間", ascending=False), use_container_width=True)
                with open(LOG_FILE, "rb") as f: st.download_button("📥 下載 Log", f, file_name="traffic_log.csv")
            else: st.info("尚無紀錄")
        elif admin_pwd: st.error("密碼錯誤")
    st.divider()

    st.header("2. 即時篩選器")
    bias_threshold = st.slider("乖離率範圍 (±%)", 0.5, 5.0, 2.5, step=0.1)
    min_vol_input = st.number_input("最低成交量 (張)", value=1000, step=100)
    st.subheader("策略選擇")
    
    # 變更策略名稱
    strategy_mode = st.radio("選擇篩選策略：", (
        "🛡️ 生命線保衛戰 (反彈/支撐)", 
        "🔥 起死回生 (Da來守住)", 
        "🛡️ 固若金湯 (破底翻突破)", 
        "🏹 蓄勢待發 (KD+紅吞)", 
        "⚡ 光神腳 (紅吞+左腳KD<80)"
    ))
    
    st.caption("細部條件：")
    filter_trend_up = False
    filter_trend_down = False
    filter_kd = False
    filter_vol_double = False

    if strategy_mode == "🛡️ 生命線保衛戰 (反彈/支撐)":
        c1, c2 = st.columns(2)
        with c1: filter_trend_up = st.checkbox("生命線向上", value=False)
        with c2: filter_trend_down = st.checkbox("生命線向下", value=False)
        filter_kd = st.checkbox("KD 黃金交叉", value=False)
        filter_vol_double = st.checkbox("出量 (今日 > 昨日x1.5)", value=False)
    elif strategy_mode == "🔥 起死回生 (Da來守住)":
        st.info("ℹ️ 過去7日跌破，今日站回生命線。")
        filter_vol_double = st.checkbox("出量確認", value=False)
    elif strategy_mode == "🛡️ 固若金湯 (破底翻突破)":
        st.info("條件：尋找波段底部 Da(收盤價)/Daa(最低價)。出現紅吞且 KD>20 確認守住底部後，今日以紅 K 強勢突破反彈高點（箱頂）。")
    elif strategy_mode == "🏹 蓄勢待發 (KD+紅吞)":
        st.info("條件：K<20後金叉，金叉後3日內發動(K>=20, 紅吞黑)。")
    elif strategy_mode == "⚡ 光神腳 (紅吞+左腳KD<80)":
        st.info("條件：左腳(K<20)；頸線(波段高點) K<80；紅吞黑發動。")

    st.divider()
    st.caption("⚠️ 回測將使用上方「最低成交量」過濾。")
    if st.button("🧪 策略回測"):
        st.info("阿吉正在調閱歷史檔案... ⏳")
        stock_dict = get_stock_list()
        bt_progress = st.progress(0, text="回測中...")
        use_treasure_param = (strategy_mode == "🔥 起死回生 (Da來守住)")
        use_solid_defense_param = (strategy_mode == "🛡️ 固若金湯 (破底翻突破)")
        use_legkick_param = (strategy_mode == "🏹 蓄勢待發 (KD+紅吞)")
        use_w_bottom_param = (strategy_mode == "⚡ 光神腳 (紅吞+左腳KD<80)")

        bt_df = run_strategy_backtest(
            stock_dict, bt_progress, mode=strategy_mode,
            use_trend_up=filter_trend_up, use_treasure=use_treasure_param,
            use_vol=filter_vol_double, use_solid_defense=use_solid_defense_param,
            use_leg_kick=use_legkick_param, use_w_bottom=use_w_bottom_param,
            min_vol_threshold=min_vol_input,
        )
        st.session_state["backtest_result"] = bt_df
        bt_progress.empty()
        st.success("回測完成！")

    # 🔥 開發日誌上鎖
    with st.expander("📅 系統開發日誌"):
        log_pwd = st.text_input("請輸入密碼以查看日誌", type="password", key="dev_log_pwd")
        if log_pwd == "1103":
            st.write(f"**🕒 系統時間:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            st.markdown("---")
            st.markdown("""
                ### Ver 2.4 (固若金湯版)
                * **新策略上線**：移除舊有多頭馬車，替換為全新的「固若金湯 (破底翻突破)」策略。
                * **嚴謹底部邏輯**：智能找尋區間底部 (Da/Daa)，確認「守住訊號(紅吞+KD>20)」後，精準抓取帶量紅K突破箱型頂部的進場起漲點。
                * **資訊面板擴充**：策略資訊列詳細揭露「落底時間」、「箱型頂部價格」、「守住類型 (Da/Daa)」。
                """)
        elif log_pwd != "":
            st.error("密碼錯誤")

# 主畫面 - 日常篩選
if st.session_state["master_df"] is not None:
    df = st.session_state["master_df"].copy()
    if "生命線" not in df.columns:
        st.error("⚠️ 資料結構已更新！請點擊 **「🚨 強制重置系統」**。")
        st.stop()

    df = df[df["成交量"] >= (min_vol_input * 1000)]

    if strategy_mode == "🔥 起死回生 (Da來守住)":
        df = df[df["浴火重生"] == True]
    elif strategy_mode == "🛡️ 固若金湯 (破底翻突破)":
        df = df[df["固若金湯"] == True] if "固若金湯" in df.columns else df.iloc[0:0]
    elif strategy_mode == "🏹 蓄勢待發 (KD+紅吞)":
        df = df[df["蓄勢待發"] == True] if "蓄勢待發" in df.columns else df.iloc[0:0]
    elif strategy_mode == "⚡ 光神腳 (紅吞+左腳KD<80)":
        df = df[df["光神腳"] == True] if "光神腳" in df.columns else df.iloc[0:0]
    else:
        df = df[df["abs_bias"] <= bias_threshold]
        if filter_trend_up: df = df[df["生命線趨勢"].str.contains("向上")]
        elif filter_trend_down: df = df[df["生命線趨勢"].str.contains("向下")]
        if filter_kd: df = df[df["K值"] > df["D值"]]

    if filter_vol_double: df = df[df["成交量"] > (df["昨日成交量"] * 1.5)]

    if len(df) == 0: st.warning("⚠️ 找不到符合條件的股票！")
    else:
        st.markdown(f"""<div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center; border: 2px solid #ff4b4b;">
                <h2 style="color: #333; margin:0;">🔍 根據共篩選出 <span style="color: #ff4b4b; font-size: 1.5em;">{len(df)}</span> 檔股票</h2></div><br>""", unsafe_allow_html=True)

        df["成交量(張)"] = (df["成交量"] / 1000).astype(int)
        df["KD值"] = df.apply(lambda x: f"K:{int(x['K值'])} D:{int(x['D值'])}", axis=1)
        df["選股標籤"] = df["代號"].astype(str) + " " + df["名稱"].astype(str)

        # 固定版面顯示欄位
        fixed_display_cols = ["代號", "名稱", "產業", "收盤價", "生命線", "乖離率(%)", "位置", "KD值", "成交量(張)"]
        if strategy_mode == "🛡️ 固若金湯 (破底翻突破)":
            fixed_display_cols = ["代號", "名稱", "產業", "收盤價", "箱頂", "守住類型", "生命線", "KD值", "成交量(張)"]

        for col in fixed_display_cols:
            if col not in df.columns:
                if col in ['名稱', '產業', '位置', 'KD值', '守住類型']: df[col] = "-"
                else: df[col] = 0

        df = df.sort_values(by="成交量", ascending=False)
        final_df_to_show = df[fixed_display_cols]

        tab1, tab2 = st.tabs(["📋 篩選結果列表", "📊 日趨勢圖"])

        with tab1:
            def highlight_row(row):
                return ["background-color: #e6fffa; color: black"] * len(row) if row["收盤價"] > row["生命線"] else ["background-color: #fff0f0; color: black"] * len(row)
            try:
                st.dataframe(final_df_to_show.style.apply(highlight_row, axis=1), use_container_width=True, hide_index=True)
            except:
                st.dataframe(final_df_to_show, use_container_width=True, hide_index=True)

        with tab2:
            st.markdown("### 🔍 個股趨勢圖")
            selected_stock_label = st.selectbox("請選擇一檔股票：", df["選股標籤"].tolist())
            selected_row = df[df["選股標籤"] == selected_stock_label].iloc[0]
            
            plot_stock_chart(selected_row["完整代號"], selected_row["名稱"], strategy_mode)

            # ✅ 在詳細頁揭露隱藏資訊
            if strategy_mode == "🛡️ 固若金湯 (破底翻突破)":
                st.markdown("---")
                st.caption("🛡️ 固若金湯策略詳細數據:")
                c1, c2, c3 = st.columns(3)
                with c1: st.info(f"📉 落底點 (Da/Daa)\n\n**{selected_row.get('最低點日期', '-')}**")
                with c2: st.warning(f"🛡️ 守住確認日\n\n**{selected_row.get('守住日期', '-')} ({selected_row.get('守住類型', '-')})**")
                with c3: st.success(f"🚀 突破箱頂價位\n\n**{selected_row.get('箱頂', '-')}**")

            elif strategy_mode == "🏹 蓄勢待發 (KD+紅吞)":
                st.markdown("---")
                st.caption("🏹 蓄勢待發策略詳細數據:")
                k_col1, k_col2, k_col3 = st.columns(3)
                kick_date = selected_row.get("蓄勢日期", "-")
                low_date = selected_row.get("KD低點", "-")
                with k_col1: st.info(f"📉 KD落底日\n\n**{low_date}**")
                with k_col3: st.success(f"🚀 發動攻擊日\n\n**{kick_date}**")
            
            elif strategy_mode == "⚡ 光神腳 (紅吞+左腳KD<80)":
                st.markdown("---")
                st.caption("⚡ 光神腳策略數據:")
                w_col1, w_col2 = st.columns(2)
                w_left = selected_row.get("左腳日期", "-")
                w_peak = selected_row.get("中高日期", "-")
                with w_col1: st.info(f"🦶 左腳落底\n\n**{w_left}**")
                with w_col2: st.warning(f"⛰️ 頸線高點\n\n**{w_peak}**")

else:
    st.warning("👈 請先點擊左側 sidebar 的 **「🔄 下載最新股價」** 按鈕開始挖寶！")
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        if os.path.exists("welcome.jpg"):
            st.markdown("""<div style="text-align: center; font-size: 1.1em; margin-bottom: 20px;">
                這是數年來的經驗收納<br>此工具僅供參考，不代表投資建議<br>預祝心想事成，從從容容，紫氣東來! 🟣✨</div>""", unsafe_allow_html=True)
            st.image("welcome.jpg", width=420)
        else: st.info("💡 尚未偵測到 welcome.jpg")
