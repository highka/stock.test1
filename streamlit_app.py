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
VER = "ver 1.2 (Rename + MA30 + LegKick v2)"
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
    """
    回傳整條 K / D 序列，用於打腳策略掃描
    df 需要至少包含 'High','Low','Close'
    """
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

def detect_leg_kick_signal(stock_df, k_series, lookback=60):
    """
    ✅ 打腳發動 (你指定的新版本)：
    1) 最近 lookback 根內，找「最後一次 KD < 20」的 t1
    2) 在 t1 之後，找「第一次 KD >= 20」的 t_cross
    3) 從 t_cross（含）之後開始，任一天只要出現「紅吞黑」
       且當天收盤價 > t1 那天收盤價（價格要比前次 KD<20 高）
       就觸發
    4) 找到就回傳 (True, 觸發日期)，否則 (False, None)
    """
    if len(stock_df) < lookback + 2:
        return False, None

    recent_df = stock_df.tail(lookback).copy()
    recent_k = k_series.reindex(recent_df.index)

    # 1) 找最後一次 KD < 20 的日期 t1
    t1 = recent_k[recent_k < 20].last_valid_index()
    if t1 is None:
        return False, None

    oversold_close = float(recent_df.loc[t1, "Close"])

    # 2) 找 t1 後第一次 KD >= 20 的 t_cross
    k_after_t1 = recent_k[recent_k.index > t1]
    t_cross = k_after_t1[k_after_t1 >= 20].first_valid_index()
    if t_cross is None:
        return False, None

    # 3) 從 t_cross（含）後任一天找紅吞黑
    future_index = [idx for idx in recent_df.index if idx >= t_cross]
    for dt in future_index:
        pos = recent_df.index.get_loc(dt)
        if pos == 0:
            continue

        prev_row = recent_df.iloc[pos - 1]
        curr_row = recent_df.iloc[pos]

        prev_open, prev_close = float(prev_row["Open"]), float(prev_row["Close"])
        curr_open, curr_close = float(curr_row["Open"]), float(curr_row["Close"])

        prev_is_black = prev_close < prev_open
        curr_is_red = curr_close > curr_open

        # 紅吞黑：當天實體包住前一天黑K實體
        engulf = (curr_open < prev_close) and (curr_close > prev_open)

        # 價格比前次 KD<20 那天收盤更高
        if prev_is_black and curr_is_red and engulf and (curr_close > oversold_close):
            return True, dt

    return False, None

# --- 策略回測核心函數 ---
def run_strategy_backtest(stock_dict, progress_bar, use_trend_up, use_treasure, use_vol, use_royal, min_vol_threshold):
    results = []
    all_tickers = list(stock_dict.keys())

    BATCH_SIZE = 50
    total_batches = (len(all_tickers) // BATCH_SIZE) + 1

    OBSERVE_DAYS = 20 if use_royal else 10

    for i, batch_idx in enumerate(range(0, len(all_tickers), BATCH_SIZE)):
        batch = all_tickers[batch_idx : batch_idx + BATCH_SIZE]
        try:
            data = yf.download(batch, period="2y", interval="1d", progress=False, auto_adjust=False)
            if not data.empty:
                try:
                    df_c = data["Close"]
                    df_v = data["Volume"]
                    df_l = data["Low"]
                    df_h = data["High"]
                except KeyError:
                    continue

                if isinstance(df_c, pd.Series):
                    df_c = df_c.to_frame(name=batch[0])
                    df_v = df_v.to_frame(name=batch[0])
                    df_l = df_l.to_frame(name=batch[0])
                    df_h = df_h.to_frame(name=batch[0])

                ma200_df = df_c.rolling(window=200).mean()
                if use_royal:
                    ma30_df = df_c.rolling(window=30).mean()
                    ma60_df = df_c.rolling(window=60).mean()

                scan_window = df_c.index[-90:]

                for ticker in df_c.columns:
                    try:
                        c_series = df_c[ticker]
                        v_series = df_v[ticker]
                        l_series = df_l[ticker]
                        h_series = df_h[ticker]
                        ma200_series = ma200_df[ticker]

                        if use_royal:
                            ma30_series = ma30_df[ticker]
                            ma60_series = ma60_df[ticker]

                        stock_name = stock_dict.get(ticker, {}).get("name", ticker)
                        total_len = len(c_series)

                        for date in scan_window:
                            if pd.isna(ma200_series.get(date)):
                                continue
                            if date not in c_series.index:
                                continue

                            idx = c_series.index.get_loc(date)
                            if idx < 200:
                                continue

                            close_p = float(c_series.iloc[idx])
                            vol = float(v_series.iloc[idx])
                            prev_vol = float(v_series.iloc[idx - 1])
                            ma200_val = float(ma200_series.iloc[idx])

                            if vol < (min_vol_threshold * 1000):
                                continue
                            if ma200_val == 0 or prev_vol == 0:
                                continue

                            is_match = False

                            if use_royal:
                                ma30_val = float(ma30_series.iloc[idx])
                                ma60_val = float(ma60_series.iloc[idx])
                                if (close_p > ma30_val) and (ma30_val > ma60_val) and (ma60_val > ma200_val):
                                    is_match = True
                            else:
                                low_p = float(l_series.iloc[idx])
                                ma_val_20ago = float(ma200_series.iloc[idx - 20])

                                if use_trend_up and (ma200_val <= ma_val_20ago):
                                    continue
                                if use_vol and (vol <= prev_vol * 1.5):
                                    continue

                                if use_treasure:
                                    start_idx = idx - 7
                                    if start_idx < 0:
                                        continue
                                    recent_c = c_series.iloc[start_idx : idx + 1]
                                    recent_ma = ma200_series.iloc[start_idx : idx + 1]
                                    cond_today_up = recent_c.iloc[-1] > recent_ma.iloc[-1]
                                    past_c = recent_c.iloc[:-1]
                                    past_ma = recent_ma.iloc[:-1]
                                    cond_past_down = (past_c < past_ma).any()

                                    if cond_today_up and cond_past_down:
                                        is_match = True
                                else:
                                    cond_near = (low_p <= ma200_val * 1.03) and (low_p >= ma200_val * 0.90)
                                    cond_up = (close_p > ma200_val)
                                    if cond_near and cond_up:
                                        is_match = True

                            if is_match:
                                month_str = date.strftime("%m月")
                                days_after_signal = total_len - 1 - idx

                                final_profit_pct = 0.0
                                result_status = "觀察中"
                                is_watching = False

                                if days_after_signal < 1:
                                    is_watching = True
                                    final_profit_pct = 0.0

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
                                            result_status = "觀察中"

                                else:
                                    if days_after_signal < OBSERVE_DAYS:
                                        current_price = float(c_series.iloc[-1])
                                        final_profit_pct = (current_price - close_p) / close_p * 100
                                        is_watching = True
                                    else:
                                        future_highs = h_series.iloc[idx + 1 : idx + 1 + OBSERVE_DAYS]
                                        max_price = float(future_highs.max())
                                        final_profit_pct = (max_price - close_p) / close_p * 100

                                        if final_profit_pct > 3.0:
                                            result_status = "驗證成功 🏆"
                                        elif final_profit_pct > 0:
                                            result_status = "Win (反彈)"
                                        else:
                                            result_status = "Loss 📉"

                                results.append({
                                    "月份": "👀 關注中" if is_watching else month_str,
                                    "代號": ticker.replace(".TW", "").replace(".TWO", ""),
                                    "名稱": stock_name,
                                    "訊號日期": date.strftime("%Y-%m-%d"),
                                    "訊號價": round(close_p, 2),
                                    "最高漲幅(%)": round(final_profit_pct, 2),
                                    "結果": "觀察中" if is_watching else result_status,
                                })

                                if use_royal:
                                    break

                    except:
                        continue
        except:
            pass

        progress = (i + 1) / total_batches
        progress_bar.progress(progress, text=f"深度回測中 (計算分月數據)...({int(progress*100)}%)")

    if not results:
        return pd.DataFrame(columns=["月份", "代號", "名稱", "訊號日期", "訊號價", "最高漲幅(%)", "結果"])

    return pd.DataFrame(results)

def fetch_all_data(stock_dict, progress_bar, status_text):
    if not stock_dict:
        return pd.DataFrame()

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
                    df_o = data["Open"]
                    df_c = data["Close"]
                    df_h = data["High"]
                    df_l = data["Low"]
                    df_v = data["Volume"]
                except KeyError:
                    continue

                if isinstance(df_c, pd.Series):
                    df_c = df_c.to_frame(name=batch[0])
                    df_h = df_h.to_frame(name=batch[0])
                    df_l = df_l.to_frame(name=batch[0])
                    df_v = df_v.to_frame(name=batch[0])
                    df_o = df_o.to_frame(name=batch[0])

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

                for ticker in df_c.columns:
                    try:
                        price = float(last_price_series[ticker])
                        ma200 = float(last_ma200_series[ticker])
                        ma30 = float(last_ma30_series[ticker])
                        ma60 = float(last_ma60_series[ticker])
                        prev_ma200 = float(prev_ma200_series[ticker])

                        vol = float(last_vol_series[ticker])
                        prev_vol = float(prev_vol_series[ticker])

                        if pd.isna(price) or pd.isna(ma200) or ma200 == 0:
                            continue

                        ma_trend = "⬆️向上" if ma200 >= prev_ma200 else "⬇️向下"

                        # 起死回生（原浴火重生）判斷：過去7日內曾跌破，但今日站回生命線
                        is_treasure = False
                        my_recent_c = recent_close_df[ticker]
                        my_recent_ma = recent_ma200_df[ticker]
                        if len(my_recent_c) >= 8:
                            cond_today_up = my_recent_c.iloc[-1] > my_recent_ma.iloc[-1]
                            past_c = my_recent_c.iloc[:-1]
                            past_ma = my_recent_ma.iloc[:-1]
                            cond_past_down = (past_c < past_ma).any()
                            if cond_today_up and cond_past_down:
                                is_treasure = True

                        # 🐎 多頭馬車發動（原皇冠特選）：股價 > 30MA > 60MA > 200MA
                        is_royal = False
                        if (price > ma30) and (ma30 > ma60) and (ma60 > ma200):
                            is_royal = True

                        # 準備 K 線資料
                        stock_df = pd.DataFrame({
                            "Open": df_o[ticker],
                            "Close": df_c[ticker],
                            "High": df_h[ticker],
                            "Low": df_l[ticker],
                        }).dropna()

                        k_val, d_val = 0, 0
                        is_leg_kick = False
                        leg_kick_date = None

                        if len(stock_df) >= 20:
                            k_series, d_series = calculate_kd_series(stock_df)
                            k_val = float(k_series.iloc[-1])
                            d_val = float(d_series.iloc[-1])

                            # 打腳：觀察 60 根（新規則）
                            is_leg_kick, leg_kick_date = detect_leg_kick_signal(stock_df, k_series, lookback=60)
                        else:
                            if len(stock_df) >= 9:
                                k_val, d_val = calculate_kd_values(stock_df)

                        bias = ((price - ma200) / ma200) * 100
                        stock_info = stock_dict.get(ticker)
                        if not stock_info:
                            continue

                        raw_data_list.append({
                            "代號": stock_info["code"],
                            "名稱": stock_info["name"],
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
                            "浴火重生": is_treasure,     # 欄位保留不改（避免舊快取/欄位斷裂）
                            "皇冠特選": is_royal,        # 欄位保留不改（避免舊快取/欄位斷裂）
                            "打腳發動": is_leg_kick,
                            "打腳日期": leg_kick_date.strftime("%Y-%m-%d") if leg_kick_date else "",
                        })
                    except:
                        continue
        except:
            pass

        current_progress = (i + 1) / total_batches
        progress_bar.progress(current_progress, text=f"系統正在努力挖掘寶藏中...({int(current_progress*100)}%)")
        time.sleep(0.3)

    return pd.DataFrame(raw_data_list)

def plot_stock_chart(ticker, name):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=False)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

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
        fig.add_trace(go.Scatter(x=plot_df["DateStr"], y=plot_df["Close"], mode="lines", name="收盤價",
                                 line=dict(color="#00CC96", width=2.5)))
        fig.add_trace(go.Scatter(x=plot_df["DateStr"], y=plot_df["30MA"], mode="lines", name="30MA(月線)",
                                 line=dict(color="#AB63FA", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=plot_df["DateStr"], y=plot_df["60MA"], mode="lines", name="60MA(季線)",
                                 line=dict(color="#19D3F3", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=plot_df["DateStr"], y=plot_df["200MA"], mode="lines", name="200MA(生命線)",
                                 line=dict(color="#FFA15A", width=3)))

        fig.update_layout(
            title=f"📊 {name} ({ticker}) 股價 vs 均線排列",
            yaxis_title="價格",
            height=500,
            hovermode="x unified",
            xaxis=dict(type="category", tickangle=-45, nticks=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"繪圖失敗: {e}")

# --- 3. 介面顯示區 ---
st.title(f"✨ {VER} 黑嚕嚕-旗鼓相當")
st.markdown("---")

if "master_df" not in st.session_state:
    st.session_state["master_df"] = None
if "last_update" not in st.session_state:
    st.session_state["last_update"] = None
if "backtest_result" not in st.session_state:
    st.session_state["backtest_result"] = None

with st.sidebar:
    st.header("資料庫管理")

    CACHE_FILE = "stock_data_cache.csv"

    if st.button("🚨 強制重置系統"):
        st.cache_data.clear()
        st.session_state.clear()
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        st.success("系統已重置！請重新點擊更新股價。")
        st.rerun()

    if st.session_state["master_df"] is None and os.path.exists(CACHE_FILE):
        try:
            df_cache = pd.read_csv(CACHE_FILE)
            st.session_state["master_df"] = df_cache
            mod_time = os.path.getmtime(CACHE_FILE)
            st.session_state["last_update"] = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
            st.success(f"⚡ 已快速載入上次資料 ({st.session_state['last_update']})")
        except Exception as e:
            st.error(f"讀取快取失敗: {e}")

    if st.button("🔄 下載最新股價 (開市用)", type="primary"):
        stock_dict = get_stock_list()

        if not stock_dict:
            st.error("無法取得股票清單，請稍後再試或按上方重置按鈕。")
        else:
            placeholder_emoji = st.empty()
            with placeholder_emoji:
                st.markdown(
                    """<div style="text-align: center; font-size: 40px; animation: blink 1s infinite;">🎁💰✨</div>
                    <style>@keyframes blink { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }</style>
                    <div style="text-align: center;">連線下載中 (Batch=50)...</div>""",
                    unsafe_allow_html=True,
                )

            status_text = st.empty()
            progress_bar = st.progress(0, text="準備下載...")

            df = fetch_all_data(stock_dict, progress_bar, status_text)

            if not df.empty:
                df.to_csv(CACHE_FILE, index=False)
                st.session_state["master_df"] = df
                st.session_state["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.success(f"更新完成！共 {len(df)} 檔資料")
            else:
                st.error("⛔ 由於申請次數過多，連線資料庫阻擋。")
                with st.expander("🆘 嘗試解決方案：Reboot App (點我展開)"):
                    st.info(
                        """
                        **請嘗試「重啟應用程式」來更換連線環境：**
                        1. 點擊網頁右上角的 **「⋮」** (三個點按鈕)。
                        2. 選擇 **「Reboot App」** (或 Clear Cache and Rerun)。
                        3. 等待網頁重新載入後，再試一次。
                        """
                    )

            placeholder_emoji.empty()
            progress_bar.empty()

    if st.session_state["last_update"]:
        st.caption(f"最後更新：{st.session_state['last_update']}")

    st.divider()

    with st.expander("🔐 管理員後台"):
        admin_pwd = st.text_input("請輸入管理密碼", type="password")
        if admin_pwd == "admin1133":
            if os.path.exists(LOG_FILE):
                st.markdown("### 🚦 流量統計 (最近紀錄)")
                log_df = pd.read_csv(LOG_FILE)
                total_visits = len(log_df)
                unique_users = log_df["Session_ID"].nunique()
                st.metric("總點擊次數", total_visits)
                st.metric("獨立訪客數 (Session)", unique_users)

                st.dataframe(log_df.sort_values(by="時間", ascending=False), use_container_width=True)

                with open(LOG_FILE, "rb") as f:
                    st.download_button("📥 下載完整 Log (CSV)", f, file_name="traffic_log.csv", mime="text/csv")
            else:
                st.info("尚無流量紀錄。")
        elif admin_pwd:
            st.error("密碼錯誤")

    st.divider()

    st.header("2. 即時篩選器")
    bias_threshold = st.slider("乖離率範圍 (±%)", 0.5, 5.0, 2.5, step=0.1)
    min_vol_input = st.number_input("最低成交量 (張)", value=1000, step=100)

    st.subheader("策略選擇")

    strategy_mode = st.radio(
        "選擇篩選策略：",
        (
            "🛡️ 生命線保衛戰 (反彈/支撐)",
            "🔥 起死回生 (Da來守住)",
            "🐎 多頭馬車發動 (多頭排列)",
            "🦵 打腳發動 (KD+紅吞)",
        ),
    )

    st.caption("細部條件：")

    filter_trend_up = False
    filter_trend_down = False
    filter_kd = False
    filter_vol_double = False
    filter_royal = False
    filter_treasure = False

    if strategy_mode == "🛡️ 生命線保衛戰 (反彈/支撐)":
        col1, col2 = st.columns(2)
        with col1:
            filter_trend_up = st.checkbox("生命線向上", value=False)
        with col2:
            filter_trend_down = st.checkbox("生命線向下", value=False)
        filter_kd = st.checkbox("KD 黃金交叉", value=False)
        filter_vol_double = st.checkbox("出量 (今日 > 昨日x1.5)", value=False)

    elif strategy_mode == "🔥 起死回生 (Da來守住)":
        filter_treasure = True
        st.info("ℹ️ 尋找：過去7日內曾跌破，但今日站回生命線的個股。")
        filter_vol_double = st.checkbox("出量確認", value=False)

    elif strategy_mode == "🐎 多頭馬車發動 (多頭排列)":
        filter_royal = True
        st.info("ℹ️ 條件：股價 > 30MA > 60MA > 200MA (多頭強勢股)")
        st.markdown(
            """
            **回測規則 (更嚴格)：**
            * **停利**：20天內任一天觸及 +10%
            * **停損**：收盤價跌破 200MA
            """
        )
        filter_vol_double = st.checkbox("出量確認", value=False)

    elif strategy_mode == "🦵 打腳發動 (KD+紅吞)":
        st.info("條件：前一次 KD<20，之後 KD>=20；KD>=20 之後任一天出現紅吞黑，且收盤價高於前次 KD<20 的收盤價，即觸發。")

    st.divider()

    st.caption("⚠️ 回測將使用上方設定的「最低成交量」進行過濾。")
    if st.button("🧪 策略回測"):
        if strategy_mode == "🦵 打腳發動 (KD+紅吞)":
            st.warning("『打腳發動』策略目前僅支援即時篩選，尚未開放回測。")
        else:
            st.info("阿吉正在調閱過去2年的歷史檔案，進行深度驗證... (請稍候) ⏳")
            stock_dict = get_stock_list()
            bt_progress = st.progress(0, text="初始化回測...")

            use_treasure_param = True if strategy_mode == "🔥 起死回生 (Da來守住)" else False
            use_royal_param = True if strategy_mode == "🐎 多頭馬車發動 (多頭排列)" else False

            bt_df = run_strategy_backtest(
                stock_dict,
                bt_progress,
                use_trend_up=filter_trend_up,
                use_treasure=use_treasure_param,
                use_vol=filter_vol_double,
                use_royal=use_royal_param,
                min_vol_threshold=min_vol_input,
            )

            st.session_state["backtest_result"] = bt_df
            bt_progress.empty()
            st.success("回測完成！請查看下方結果。")

    with st.expander("📅 系統開發日誌"):
        st.write(f"**🕒 系統最後重啟時間:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        st.markdown("---")
        st.markdown(
            """
            ### Ver 1.2
            * 介面改名：生命線保衛戰 / 起死回生 / 🐎 多頭馬車發動
            * 打腳策略更新：KD>=20 後任一天紅吞黑才觸發
            * 均線顯示：30MA / 60MA / 200MA
            """
        )

# 主畫面 - 回測報告
if st.session_state["backtest_result"] is not None:
    bt_df = st.session_state["backtest_result"]
    st.markdown("---")

    s_name = "🛡️ 生命線保衛戰"
    if "strategy_mode" in locals():
        if strategy_mode == "🔥 起死回生 (Da來守住)":
            s_name = "🔥 起死回生"
        elif strategy_mode == "🐎 多頭馬車發動 (多頭排列)":
            s_name = "🐎 多頭馬車發動"

    st.subheader(f"🧪 策略回測報告：{s_name}")

    if "結果" in bt_df.columns:
        df_history = bt_df[bt_df["結果"] != "觀察中"].copy()
        df_watching = bt_df[bt_df["結果"] == "觀察中"].copy()
    else:
        df_history = bt_df.copy()
        df_watching = bt_df.iloc[0:0]

    if not df_watching.empty:
        st.markdown(
            """
            <div style="background-color: #fff8dc; padding: 15px; border-radius: 10px; border: 2px solid #ffa500; margin-bottom: 20px;">
                <h3 style="color: #d2691e; margin:0;">👀 旺來關注中 (進行中訊號)</h3>
                <p style="color: #666; margin:5px 0 0 0;">這些股票尚未觸發停利(+10%)或停損(破線)。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        df_watching = df_watching.sort_values(by="訊號日期", ascending=False)
        st.dataframe(
            df_watching[["代號", "名稱", "訊號日期", "訊號價", "最高漲幅(%)"]].style.background_gradient(cmap="Reds", subset=["最高漲幅(%)"]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("👀 目前沒有符合「關注中」的股票。")

    st.markdown("---")
    st.markdown("### 📜 歷史驗證數據 (已結算)")

    if len(df_history) > 0 and "月份" in df_history.columns:
        months = sorted(df_history["月份"].unique())
        tabs = st.tabs(["📊 總覽"] + months)

        with tabs[0]:
            win_df = df_history[df_history["結果"].astype(str).str.contains("Win") | df_history["結果"].astype(str).str.contains("驗證成功")]
            win_count = len(win_df)
            total_count = len(df_history)
            win_rate = int((win_count / total_count) * 100) if total_count > 0 else 0
            avg_max_ret = round(df_history["最高漲幅(%)"].mean(), 2)

            col1, col2, col3 = st.columns(3)
            col1.metric("總已結算次數", total_count)
            col2.metric("獲利機率", f"{win_rate}%")
            col3.metric("平均損益(%)", f"{avg_max_ret}%")
            st.dataframe(df_history, use_container_width=True)

        for i, m in enumerate(months):
            with tabs[i + 1]:
                m_df = df_history[df_history["月份"] == m]
                m_win = len(m_df[m_df["結果"].astype(str).str.contains("Win") | m_df["結果"].astype(str).str.contains("驗證成功")])
                m_total = len(m_df)
                m_rate = int((m_win / m_total) * 100) if m_total > 0 else 0
                m_avg = round(m_df["最高漲幅(%)"].mean(), 2) if m_total > 0 else 0

                c1, c2, c3 = st.columns(3)
                c1.metric(f"{m} 結算次數", m_total)
                c2.metric(f"{m} 獲利機率", f"{m_rate}%")
                c3.metric(f"{m} 平均損益", f"{m_avg}%")

                def color_ret(val):
                    return f'color: {"red" if val > 0 else "green"}'

                st.dataframe(m_df.style.map(color_ret, subset=["最高漲幅(%)"]), use_container_width=True)
    else:
        st.warning("在此回測期間內，沒有歷史股票符合條件。")

    st.markdown("---")

# 主畫面 - 日常篩選
if st.session_state["master_df"] is not None:
    df = st.session_state["master_df"].copy()

    if "生命線" not in df.columns:
        st.error("⚠️ 資料結構已更新！請點擊 **「🚨 強制重置系統」** 後重新下載。")
        st.stop()

    # 基礎過濾
    df = df[df["成交量"] >= (min_vol_input * 1000)]

    # 策略分流篩選
    if strategy_mode == "🔥 起死回生 (Da來守住)":
        df = df[df["浴火重生"] == True]

    elif strategy_mode == "🐎 多頭馬車發動 (多頭排列)":
        if "皇冠特選" in df.columns:
            df = df[df["皇冠特選"] == True]
        else:
            df = df[(df["收盤價"] > df["MA30"]) & (df["MA30"] > df["MA60"]) & (df["MA60"] > df["生命線"])]

    elif strategy_mode == "🦵 打腳發動 (KD+紅吞)":
        if "打腳發動" in df.columns:
            df = df[df["打腳發動"] == True]
        else:
            st.warning("目前資料尚未計算『打腳發動』欄位，請先重新下載最新股價。")
            df = df.iloc[0:0]

    else:
        # 🛡️ 生命線保衛戰
        df = df[df["abs_bias"] <= bias_threshold]
        if filter_trend_up:
            df = df[df["生命線趨勢"] == "⬆️向上"]
        elif filter_trend_down:
            df = df[df["生命線趨勢"] == "⬇️向下"]
        if filter_kd:
            df = df[df["K值"] > df["D值"]]

    if filter_vol_double:
        df = df[df["成交量"] > (df["昨日成交量"] * 1.5)]

    if len(df) == 0:
        st.warning("⚠️ 找不到符合條件的股票！")
    else:
        st.markdown(
            f"""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center; border: 2px solid #ff4b4b;">
                <h2 style="color: #333; margin:0;">🔍 根據共篩選出 <span style="color: #ff4b4b; font-size: 1.5em;">{len(df)}</span> 檔股票</h2>
            </div>
            <br>
            """,
            unsafe_allow_html=True,
        )

        df["成交量(張)"] = (df["成交量"] / 1000).astype(int)
        df["KD值"] = df.apply(lambda x: f"K:{int(x['K值'])} D:{int(x['D值'])}", axis=1)
        df["選股標籤"] = df["代號"].astype(str) + " " + df["名稱"].astype(str)

        display_cols = ["代號", "名稱", "收盤價", "生命線", "乖離率(%)", "位置", "KD值", "成交量(張)"]
        if strategy_mode == "🐎 多頭馬車發動 (多頭排列)":
            display_cols = ["代號", "名稱", "收盤價", "MA30", "MA60", "生命線", "KD值", "成交量(張)"]

        df = df.sort_values(by="成交量", ascending=False)

        tab1, tab2 = st.tabs(["📋 篩選結果列表", "📊 日趨勢圖"])

        with tab1:
            def highlight_row(row):
                return ["background-color: #e6fffa; color: black"] * len(row) if row["收盤價"] > row["生命線"] else ["background-color: #fff0f0; color: black"] * len(row)

            st.dataframe(df[display_cols].style.apply(highlight_row, axis=1), use_container_width=True, hide_index=True)

        with tab2:
            st.markdown("### 🔍 個股趨勢圖")
            selected_stock_label = st.selectbox("請選擇一檔股票：", df["選股標籤"].tolist())
            selected_row = df[df["選股標籤"] == selected_stock_label].iloc[0]
            plot_stock_chart(selected_row["完整代號"], selected_row["名稱"])

            c1, c2, c3 = st.columns(3)
            c1.metric("收盤價", f"{selected_row['收盤價']:.2f}")
            c2.metric("成交量", f"{selected_row['成交量(張)']} 張")
            c3.metric("KD", selected_row["KD值"])

else:
    st.warning("👈 請先點擊左側 sidebar 的 **「🔄 下載最新股價」** 按鈕開始挖寶！")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if os.path.exists("welcome.jpg"):
            st.markdown(
                """<div style="text-align: center; font-size: 1.1em; margin-bottom: 20px;">
                這是數年來的經驗收納<br>此工具僅供參考，不代表投資建議<br>預祝心想事成，從從容容，紫氣東來! 🟣✨</div>""",
                unsafe_allow_html=True,
            )
            sub_c1, sub_c2, sub_c3 = st.columns([1, 1, 1])
            with sub_c2:
                st.image("welcome.jpg", width=180)
        else:
            st.info("💡 尚未偵測到 welcome.jpg")
