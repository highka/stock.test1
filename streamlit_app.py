import streamlit as st
import yfinance as yf
import pandas as pd
import twstock
import time
import random
import requests
import gc  # 引入垃圾回收機制
from datetime import datetime, timedelta
import plotly.graph_objects as go
import os
import uuid
import csv

# --- 1. 網頁設定 ---
VER = "ver 3.8 (Debug Mode + Connection Check)"
st.set_page_config(page_title=f"✨ 黑嚕嚕-旗鼓相當({VER})", layout="wide")

# --- 流量紀錄與後台功能 ---
LOG_FILE = "traffic_log.csv"

def get_remote_ip():
    """嘗試取得使用者 IP (兼容新舊版 Streamlit)"""
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

def get_req_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Connection": "keep-alive"
    })
    return session

# 🔥 新增：連線測試函式
def test_connection():
    session = get_req_session()
    try:
        test_ticker = "2330.TW"
        data = yf.download(test_ticker, period="5d", progress=False, session=session, threads=False)
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

def _is_gap_up_attack(prev_close, prev_high, curr_open, curr_close):
    is_gap_up = curr_open > prev_close 
    is_red = curr_close > curr_open    
    break_high = curr_close > prev_high 
    return is_gap_up and is_red and break_high

def _is_bearish_engulfing(prev_open, prev_close, curr_open, curr_close):
    prev_is_red = prev_close > prev_open
    curr_is_black = curr_close < curr_open
    engulf = (curr_open >= prev_close) and (curr_close <= prev_open)
    return prev_is_red and curr_is_black and engulf

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
    if len(stock_df) < 30: return False, None, None, None, 0
    valid_idx = stock_df.index.intersection(k_series.index)
    if len(valid_idx) < 30: return False, None, None, None, 0
    target_k = k_series.loc[valid_idx].tail(lookback)
    target_price = stock_df.loc[valid_idx].tail(lookback)
    
    k_under_20 = target_k[target_k < 20]
    if k_under_20.empty: return False, None, None, None, 0
    
    left_leg_candidates = target_price.loc[k_under_20.index]
    t_left = left_leg_candidates["Low"].idxmin()
    t_left_pos = valid_idx.get_loc(t_left)
    
    if t_left_pos > len(valid_idx) - 5: return False, None, None, None, 0
    left_low = float(stock_df.loc[t_left, "Low"])
    
    end_scan_pos = len(valid_idx) - 2
    t_peak = None
    peak_k_val = 0.0
    
    for i in range(t_left_pos + 1, end_scan_pos):
        curr_dt = valid_idx[i]
        prev_dt = valid_idx[i-1]
        curr_row = stock_df.loc[curr_dt]
        prev_row = stock_df.loc[prev_dt]
        if _is_bearish_engulfing(prev_row['Open'], prev_row['Close'], curr_row['Open'], curr_row['Close']):
            t_peak = curr_dt
            peak_k_val = float(k_series.loc[curr_dt])
            break 
    
    if t_peak is None: return False, None, None, None, 0
    
    t_peak_pos = valid_idx.get_loc(t_peak)
    t_today = valid_idx[-1]
    if t_peak_pos >= len(valid_idx) - 2: return False, None, None, None, 0
    
    right_leg_mask = (valid_idx > t_peak) & (valid_idx < t_today)
    right_leg_period = stock_df.loc[right_leg_mask]
    
    if right_leg_period.empty: return False, None, None, None, 0
    t_right = right_leg_period["Low"].idxmin()
    right_low = float(stock_df.loc[t_right, "Low"])
    
    if right_low <= left_low * 0.99: return False, None, None, None, 0
        
    curr_row = stock_df.iloc[-1]
    prev_row = stock_df.iloc[-2]
    curr_open = float(curr_row["Open"])
    curr_close = float(curr_row["Close"])
    prev_open = float(prev_row["Open"])
    prev_close = float(prev_row["Close"])
    prev_high = float(prev_row["High"])
    
    cond_engulf = _is_red_engulf_black(prev_open, prev_close, curr_open, curr_close)
    cond_gap = _is_gap_up_attack(prev_close, prev_high, curr_open, curr_close)
    
    if not (cond_engulf or cond_gap): return False, None, None, None, 0
    if curr_close <= right_low: return False, None, None, None, 0

    return True, t_left, t_right, t_peak, peak_k_val

def run_strategy_backtest(
    stock_dict,
    progress_bar,
    mode,
    use_trend_up,
    use_treasure,
    use_vol,
    use_royal,
    use_leg_kick,
    use_w_bottom,
    min_vol_threshold,
):
    # 回測函式維持原樣，但加上 threads=False 確保穩定
    results = []
    all_tickers = list(stock_dict.keys())
    BATCH_SIZE = 15
    total_batches = (len(all_tickers) // BATCH_SIZE) + 1
    OBSERVE_DAYS = 30 
    session = get_req_session()

    for i, batch_idx in enumerate(range(0, len(all_tickers), BATCH_SIZE)):
        batch = all_tickers[batch_idx : batch_idx + BATCH_SIZE]
        try:
            data = yf.download(batch, period="2y", interval="1d", progress=False, auto_adjust=False, session=session, threads=False)
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

                        if use_w_bottom:
                            sub_df = full_ohlc.loc[:date].copy()
                            w_ok, t_left, t_right, t_peak, peak_k = detect_w_bottom_signal(sub_df, k_full, d_full, lookback=60)
                            if w_ok:
                                is_match = True
                                detail_info["左腳日期"] = t_left.strftime("%m-%d")
                                detail_info["右腳日期"] = t_right.strftime("%m-%d")
                                detail_info["頸線日期"] = t_peak.strftime("%m-%d")
                                detail_info["頸線K值"] = int(peak_k)
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

                        elif use_royal:
                            ma30_val = float(ma30_series.iloc[idx])
                            ma60_val = float(ma60_series.iloc[idx])
                            if (close_p > ma30_val) and (ma30_val > ma60_val) and (ma60_val > ma200_val):
                                is_match = True
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
                            if stop_loss_price == 0: stop_loss_price = ma200_val * 0.95
                            if target_price == 0: target_price = close_p * 1.15

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
                                
                                if curr_c < stop_loss_price:
                                    final_profit_pct = (curr_c - close_p) / close_p * 100
                                    is_watching = False
                                    result_status = "Loss (破左腳) 🛑"
                                    break
                                
                                if curr_h >= target_price:
                                    final_profit_pct = (target_price - close_p) / close_p * 100
                                    is_watching = False
                                    result_status = "Win (達標2倍幅) 🎯"
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
                        if use_leg_kick:
                            record["KD低點"] = detail_info.get("KD低點", "")
                            record["KD金叉"] = detail_info.get("KD金叉", "")
                        if use_w_bottom:
                            record["左腳"] = detail_info.get("左腳日期", "")
                            record["右腳"] = detail_info.get("右腳日期", "")
                            peak_k = detail_info.get("頸線K值", 0)
                            record["頸線"] = f"{detail_info.get('頸線日期', '')} (K:{peak_k})"
                        results.append(record)
                        if use_royal: break
                except: continue
        except: pass
        progress = (i + 1) / total_batches
        progress_bar.progress(progress, text=f"深度回測中 (計算分月數據)...({int(progress*100)}%)")
        time.sleep(1.5)
        gc.collect() 
    return pd.DataFrame(raw_data_list)

def fetch_all_data(stock_dict, progress_bar, status_text, debug_container=None):
    if not stock_dict: return pd.DataFrame()
    all_tickers = list(stock_dict.keys())
    
    # 🔥 穩定模式：極小批次 + 單線程
    BATCH_SIZE = 15 
    total_batches = (len(all_tickers) // BATCH_SIZE) + 1
    raw_data_list = []
    
    # Debug 日誌初始化
    debug_logs = []
    log_area = None
    if debug_container:
        log_area = debug_container.empty()
    
    session = get_req_session()

    for i, batch_idx in enumerate(range(0, len(all_tickers), BATCH_SIZE)):
        batch = all_tickers[batch_idx : batch_idx + BATCH_SIZE]
        try:
            # 🔥 穩定模式：threads=False
            data = yf.download(batch, period="1y", interval="1d", progress=False, auto_adjust=False, session=session, threads=False)
            
            # Debug 訊息
            msg = f"Batch {i+1}: 嘗試下載 {len(batch)} 檔"
            if data.empty:
                msg += " ❌ (Empty Response)"
                # 自動重試
                time.sleep(3)
                data = yf.download(batch, period="1y", interval="1d", progress=False, auto_adjust=False, session=session, threads=False)
                if data.empty:
                    msg += " -> 重試失敗"
                else:
                    msg += f" -> 重試成功 ({len(data.columns)})"
            else:
                msg += f" ✅ ({len(data.columns)} 筆資料)"
            
            debug_logs.append(msg)
            if log_area:
                log_area.text("\n".join(debug_logs[-10:])) # 只顯示最後 10 行

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

                        is_royal = (price > ma30) and (ma30 > ma60) and (ma60 > ma200)

                        stock_df = pd.DataFrame({
                            "Open": df_o[ticker], "Close": df_c[ticker],
                            "High": df_h[ticker], "Low": df_l[ticker],
                        }).dropna()

                        k_val, d_val = 0.0, 0.0
                        is_leg_kick = False
                        leg_kick_date = None
                        t_low = None
                        t_cross = None
                        is_w_bottom = False
                        w_left_date = None
                        w_right_date = None
                        w_peak_date = None
                        peak_k_val = 0.0

                        if len(stock_df) >= 20:
                            k_series, d_series = calculate_kd_series(stock_df)
                            k_val = float(k_series.iloc[-1])
                            d_val = float(d_series.iloc[-1])

                            for day_offset in range(3):
                                target_idx = len(stock_df) - day_offset
                                if target_idx < 30: continue
                                sub_df = stock_df.iloc[:target_idx]
                                
                                if not is_leg_kick:
                                    kick_ok, trig_dt, t_l, t_c = detect_leg_kick_signal(sub_df, lookback=60, trigger_days=3, kd_threshold=20)
                                    if kick_ok and trig_dt == sub_df.index[-1]:
                                        is_leg_kick = True
                                        leg_kick_date = trig_dt
                                        t_low = t_l
                                        t_cross = t_c

                                if not is_w_bottom:
                                    w_ok, t_l, t_r, t_p, p_k = detect_w_bottom_signal(sub_df, k_series, d_series, lookback=60)
                                    if w_ok:
                                        is_w_bottom = True
                                        w_left_date = t_l
                                        w_right_date = t_r
                                        w_peak_date = t_p
                                        peak_k_val = p_k
                                        break 
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
                            "皇冠特選": is_royal,
                            "蓄勢待發": is_leg_kick,
                            "蓄勢日期": leg_kick_date.strftime("%Y-%m-%d") if leg_kick_date else "",
                            "KD低點": t_low.strftime("%Y-%m-%d") if t_low else "",
                            "KD金叉": t_cross.strftime("%Y-%m-%d") if t_cross else "",
                            "光神腳": is_w_bottom,
                            "左腳日期": w_left_date.strftime("%Y-%m-%d") if w_left_date else "",
                            "右腳日期": w_right_date.strftime("%Y-%m-%d") if w_right_date else "",
                            "頸線日期": w_peak_date.strftime("%Y-%m-%d") if w_peak_date else "",
                            "頸線K值": int(peak_k_val)
                        })
                    except: continue
        except Exception as e:
            debug_logs.append(f"Batch {i+1} Error: {str(e)}")
            if log_area: log_area.text("\n".join(debug_logs[-10:]))
            pass
            
        current_progress = (i + 1) / total_batches
        progress_bar.progress(current_progress, text=f"系統正在努力挖掘寶藏中...({int(current_progress*100)}%)")
        time.sleep(1.5)
        gc.collect() # 🧹 垃圾回收
    return pd.DataFrame(raw_data_list)

def plot_stock_chart(ticker, name, points_dict=None):
    try:
        session = get_req_session()
        # 🔥 繪圖也要 threads=False
        df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=False, session=session, threads=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        df = df[df["Volume"] > 0].dropna()
        if df.empty:
            st.error("無法取得有效數據")
            return

        df["200MA"] = df["Close"].rolling(window=200).mean()
        df["30MA"] = df["Close"].rolling(window=30).mean()
        df["60MA"] = df["Close"].rolling(window=60).mean()
        plot_df = df.tail(120).copy()
        plot_df["DateStr"] = plot_df.index.strftime("%Y-%m-%d")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df["DateStr"], y=plot_df["Close"], mode="lines", name="收盤價", line=dict(color="#00CC96", width=2.5)))
        fig.add_trace(go.Scatter(x=plot_df["DateStr"], y=plot_df["30MA"], mode="lines", name="30MA(月線)", line=dict(color="#AB63FA", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=plot_df["DateStr"], y=plot_df["60MA"], mode="lines", name="60MA(季線)", line=dict(color="#19D3F3", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=plot_df["DateStr"], y=plot_df["200MA"], mode="lines", name="200MA(生命線)", line=dict(color="#FFA15A", width=3)))

        if points_dict:
            for label, date_str in points_dict.items():
                if date_str and date_str != "-" and date_str in plot_df["DateStr"].values:
                    row = plot_df[plot_df["DateStr"] == date_str].iloc[0]
                    if "腳" in label or "低" in label:
                        y_val = row["Low"]
                        symbol, color, pos = "triangle-up", "red", "bottom center"
                    elif "頸" in label or "高" in label:
                        y_val = row["High"]
                        symbol, color, pos = "triangle-down", "blue", "top center"
                    elif "發動" in label or "蓄勢" in label:
                        y_val = row["Close"]
                        symbol, color, pos = "star", "gold", "top center"
                    else:
                        y_val = row["Close"]
                        symbol, color, pos = "circle", "gray", "top center"

                    fig.add_trace(go.Scatter(
                        x=[date_str], y=[y_val],
                        mode="markers+text",
                        name=label,
                        text=[label],
                        textposition=pos,
                        marker=dict(symbol=symbol, size=12, color=color),
                        showlegend=False
                    ))

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

    # 🔥 側邊欄新增：連線測試按鈕
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
                    <div style="text-align: center;">連線下載中 (Batch=15)...</div>""", unsafe_allow_html=True)
            
            # 🔥 新增：偵錯日誌區
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
        admin_pwd = st.text_input("請輸入管理密碼", type="password")
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
    strategy_mode = st.radio("選擇篩選策略：", ("🛡️ 生命線保衛戰 (反彈/支撐)", "🔥 起死回生 (Da來守住)", "🐎 多頭馬車發動 (多頭排列)", "🏹 蓄勢待發 (KD+紅吞)", "⚡ 光神腳 (紅吞+左腳KD<20)"))
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
    elif strategy_mode == "🐎 多頭馬車發動 (多頭排列)":
        st.info("ℹ️ 股價 > 30MA > 60MA > 200MA")
        filter_vol_double = st.checkbox("出量確認", value=False)
    elif strategy_mode == "🏹 蓄勢待發 (KD+紅吞)":
        st.info("條件：K<20後金叉，金叉後3日內發動(K>=20, 紅吞黑)。")
    elif strategy_mode == "⚡ 光神腳 (紅吞+左腳KD<20)":
        st.info("條件：左腳(K<20)；頸線(第一次黑吞)；右腳底底高。")

    st.divider()
    st.caption("⚠️ 回測將使用上方「最低成交量」過濾。")
    if st.button("🧪 策略回測"):
        st.info("阿吉正在調閱歷史檔案... ⏳")
        stock_dict = get_stock_list()
        bt_progress = st.progress(0, text="回測中...")
        use_treasure_param = (strategy_mode == "🔥 起死回生 (Da來守住)")
        use_royal_param = (strategy_mode == "🐎 多頭馬車發動 (多頭排列)")
        use_legkick_param = (strategy_mode == "🏹 蓄勢待發 (KD+紅吞)")
        use_w_bottom_param = (strategy_mode == "⚡ 光神腳 (紅吞+左腳KD<20)")

        bt_df = run_strategy_backtest(
            stock_dict, bt_progress, mode=strategy_mode,
            use_trend_up=filter_trend_up, use_treasure=use_treasure_param,
            use_vol=filter_vol_double, use_royal=use_royal_param,
            use_leg_kick=use_legkick_param, use_w_bottom=use_w_bottom_param,
            min_vol_threshold=min_vol_input,
        )
        st.session_state["backtest_result"] = bt_df
        bt_progress.empty()
        st.success("回測完成！")

    with st.expander("📅 系統開發日誌"):
        st.write(f"**🕒 重啟時間:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        st.markdown("---")
        st.markdown("""
            ### Ver 3.8 (Debug Mode + Connection Check)
            * **新增**：下載日誌區 (Debug Log)，即時顯示每一批次的下載狀態。
            * **新增**：連線測試按鈕 (Check IP)，快速確認是否被鎖。
            """)

# 主畫面 - 日常篩選
if st.session_state["master_df"] is not None:
    df = st.session_state["master_df"].copy()
    if "生命線" not in df.columns:
        st.error("⚠️ 資料結構已更新！請點擊 **「🚨 強制重置系統」**。")
        st.stop()

    df = df[df["成交量"] >= (min_vol_input * 1000)]

    if strategy_mode == "🔥 起死回生 (Da來守住)":
        df = df[df["浴火重生"] == True]
    elif strategy_mode == "🐎 多頭馬車發動 (多頭排列)":
        df = df[df["皇冠特選"] == True] if "皇冠特選" in df.columns else df.iloc[0:0]
    elif strategy_mode == "🏹 蓄勢待發 (KD+紅吞)":
        df = df[df["蓄勢待發"] == True] if "蓄勢待發" in df.columns else df.iloc[0:0]
    elif strategy_mode == "⚡ 光神腳 (紅吞+左腳KD<20)":
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
        if strategy_mode == "🐎 多頭馬車發動 (多頭排列)":
            fixed_display_cols = ["代號", "名稱", "產業", "收盤價", "MA30", "MA60", "生命線", "KD值", "成交量(張)"]

        for col in fixed_display_cols:
            if col not in df.columns:
                if col in ['名稱', '產業', '位置', 'KD值']: df[col] = "-"
                else: df[col] = 0

        df = df.sort_values(by="成交量", ascending=False)
        final_df_to_show = df[fixed_display_cols]

        tab1, tab2 = st.tabs(["📋 篩選結果列表", "📊 日趨勢圖"])

        with tab1:
            def highlight_row(row):
                return ["background-color: #e6fffa; color: black"] * len(row) if row["收盤價"] > row["生命線"] else ["background-color: #fff0f0; color: black"] * len(row)
            st.dataframe(final_df_to_show.style.apply(highlight_row, axis=1), use_container_width=True, hide_index=True)

        with tab2:
            st.markdown("### 🔍 個股趨勢圖")
            selected_stock_label = st.selectbox("請選擇一檔股票：", df["選股標籤"].tolist())
            selected_row = df[df["選股標籤"] == selected_stock_label].iloc[0]
            
            # 🔥 準備標記點
            points_to_plot = {}
            if strategy_mode == "⚡ 光神腳 (紅吞+左腳KD<20)":
                points_to_plot = {
                    "🦶 左腳": selected_row.get("左腳日期", ""),
                    "⛰️ 頸線": selected_row.get("頸線日期", ""),
                    "🦶 右腳": selected_row.get("右腳日期", ""),
                    "🚀 發動": datetime.now().strftime("%Y-%m-%d")
                }
            elif strategy_mode == "🏹 蓄勢待發 (KD+紅吞)":
                points_to_plot = {
                    "📉 KD低點": selected_row.get("KD低點", ""),
                    "🚀 發動": selected_row.get("蓄勢日期", "")
                }

            plot_stock_chart(selected_row["完整代號"], selected_row["名稱"], points_to_plot)

            # ✅ 在詳細頁揭露隱藏資訊
            if strategy_mode == "🏹 蓄勢待發 (KD+紅吞)":
                st.markdown("---")
                st.caption("🏹 蓄勢待發策略詳細數據:")
                k_col1, k_col2, k_col3 = st.columns(3)
                kick_date = selected_row.get("蓄勢日期", "-")
                low_date = selected_row.get("KD低點", "-")
                with k_col1: st.info(f"📉 KD落底日\n\n**{low_date}**")
                with k_col3: st.success(f"🚀 發動攻擊日\n\n**{kick_date}**")
            
            elif strategy_mode == "⚡ 光神腳 (紅吞+左腳KD<20)":
                st.markdown("---")
                st.caption("⚡ 光神腳策略數據:")
                w_col1, w_col2, w_col3 = st.columns(3)
                w_left = selected_row.get("左腳日期", "-")
                w_right = selected_row.get("右腳日期", "-")
                w_peak = selected_row.get("頸線日期", "-")
                peak_k = selected_row.get("頸線K值", 0)
                
                with w_col1: st.info(f"🦶 左腳落底\n\n**{w_left}**")
                
                # 頸線判斷與顯示
                peak_text = f"⛰️ 頸線(黑吞)\n\n**{w_peak}**\n\n(K: {peak_k})"
                if peak_k >= 80:
                    with w_col2: st.error(peak_text + "\n🔥 過熱")
                else:
                    with w_col2: st.warning(peak_text + "\n❄️ 正常")
                    
                with w_col3: st.success(f"🦶 右腳確認\n\n**{w_right}**")

else:
    st.warning("👈 請先點擊左側 sidebar 的 **「🔄 下載最新股價」** 按鈕開始挖寶！")
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        if os.path.exists("welcome.jpg"):
            st.markdown("""<div style="text-align: center; font-size: 1.1em; margin-bottom: 20px;">
                這是數年來的經驗收納<br>此工具僅供參考，不代表投資建議<br>預祝心想事成，從從容容，紫氣東來! 🟣✨</div>""", unsafe_allow_html=True)
            st.image("welcome.jpg", width=420)
        else: st.info("💡 尚未偵測到 welcome.jpg")
