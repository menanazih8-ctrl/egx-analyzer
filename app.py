import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title="محلل الأسهم المصرية", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .block-container {direction: rtl; text-align: right; padding-top: 1.5rem;}
    h1, h2, h3, p, label, .stMarkdown {direction: rtl; text-align: right;}
    </style>
    """,
    unsafe_allow_html=True,
)

POPULAR = {
    "البنك التجاري الدولي": "COMI.CA",
    "هيرميس": "HRHO.CA",
    "طلعت مصطفى": "TMGH.CA",
    "إيسترن كومباني": "EAST.CA",
    "السويدي إليكتريك": "SWDY.CA",
    "المصرية للاتصالات": "ETEL.CA",
    "أبوقير للأسمدة": "ABUK.CA",
    "فوري": "FWRY.CA",
    "أوراسكوم للإنشاء": "ORAS.CA",
    "بالم هيلز": "PHDC.CA",
    "مصر الجديدة للإسكان": "HELI.CA",
    "جهينة": "JUFO.CA",
    "بنك فيصل": "FAITA.CA",
    "القلعة": "CCAP.CA",
    "أخرى (اكتب الرمز)": "",
}

PERIODS = {"6 شهور": "6mo", "سنة": "1y", "سنتين": "2y"}


# ---------------------------------------------------------------- data
@st.cache_data(ttl=900, show_spinner=False)
def load(symbol: str, period: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


# ---------------------------------------------------------- indicators
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["Close"]
    for n in (20, 50, 200):
        d[f"EMA{n}"] = c.ewm(span=n, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - 100 / (1 + rs)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_SIG"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_HIST"] = d["MACD"] - d["MACD_SIG"]

    tr = pd.concat(
        [d["High"] - d["Low"], (d["High"] - c.shift()).abs(), (d["Low"] - c.shift()).abs()],
        axis=1,
    ).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["VOL_AVG"] = d["Volume"].rolling(20).mean()
    return d


def pivots(d: pd.DataFrame, w: int = 5):
    highs, lows = [], []
    h, l = d["High"].values, d["Low"].values
    for i in range(w, len(d) - w):
        if h[i] == h[i - w : i + w + 1].max():
            highs.append(float(h[i]))
        if l[i] == l[i - w : i + w + 1].min():
            lows.append(float(l[i]))
    return highs, lows


# ------------------------------------------------------------ analysis
def analyze(d: pd.DataFrame) -> dict:
    last = d.iloc[-1]
    price = float(last["Close"])
    atr = float(last["ATR"])
    rsi = float(last["RSI"])

    highs, lows = pivots(d)
    supports = sorted([x for x in lows if x < price], reverse=True)
    resists = sorted([x for x in highs if x > price])
    support = supports[0] if supports else float(d["Low"].tail(30).min())
    resistance = resists[0] if resists else float(d["High"].tail(30).max())

    score, reasons = 0.0, []

    if price > last["EMA50"]:
        score += 1
        reasons.append("✅ السعر فوق متوسط 50 يوم")
    else:
        score -= 1
        reasons.append("❌ السعر تحت متوسط 50 يوم")

    if last["EMA20"] > last["EMA50"]:
        score += 1
        reasons.append("✅ المتوسط 20 فوق 50 (اتجاه قصير صاعد)")
    else:
        score -= 1
        reasons.append("❌ المتوسط 20 تحت 50 (اتجاه قصير هابط)")

    if len(d) >= 200:
        if last["EMA50"] > last["EMA200"]:
            score += 1
            reasons.append("✅ المتوسط 50 فوق 200 (اتجاه طويل صاعد)")
        else:
            score -= 1
            reasons.append("❌ المتوسط 50 تحت 200 (اتجاه طويل هابط)")

    if rsi > 70:
        score -= 1
        reasons.append(f"⚠️ RSI = {rsi:.0f} تشبع شرائي، احتمال تصحيح")
    elif rsi >= 50:
        score += 1
        reasons.append(f"✅ RSI = {rsi:.0f} زخم إيجابي")
    elif rsi < 30:
        score += 0.5
        reasons.append(f"⚠️ RSI = {rsi:.0f} تشبع بيعي، احتمال ارتداد")
    else:
        score -= 0.5
        reasons.append(f"❌ RSI = {rsi:.0f} زخم ضعيف")

    if last["MACD_HIST"] > 0:
        score += 1
        reasons.append("✅ MACD إيجابي")
    else:
        score -= 1
        reasons.append("❌ MACD سلبي")

    vol_ratio = float(last["Volume"] / last["VOL_AVG"]) if last["VOL_AVG"] else 1.0
    up_day = last["Close"] >= last["Open"]
    if vol_ratio > 1.3 and up_day:
        score += 1
        reasons.append(f"✅ حجم التداول {vol_ratio:.1f}x المتوسط مع صعود")
    elif vol_ratio > 1.3 and not up_day:
        score -= 1
        reasons.append(f"❌ حجم التداول {vol_ratio:.1f}x المتوسط مع هبوط")
    else:
        reasons.append(f"➖ حجم التداول عادي ({vol_ratio:.1f}x)")

    if score >= 3:
        verdict, color = "شراء", "green"
    elif score >= 1:
        verdict, color = "انتظار / مراقبة", "orange"
    else:
        verdict, color = "تجنب", "red"

    confidence = int(min(100, max(0, 50 + score * 10)))

    # --- خطة الدخول والخروج
    breakout = resistance * 1.003
    dist_to_res = (resistance - price) / price
    if verdict == "شراء" and dist_to_res > 0.03:
        entry_lo, entry_hi = price - 0.5 * atr, price + 0.25 * atr
        entry_type = "دخول عند السعر الحالي أو عند تراجع بسيط"
    else:
        entry_lo, entry_hi = breakout, breakout + 0.25 * atr
        entry_type = "دخول عند اختراق المقاومة بحجم تداول مرتفع"
    entry = (entry_lo + entry_hi) / 2

    stop = max(entry - 2.5 * atr, min(entry - 1.0 * atr, support - 0.3 * atr))
    if stop >= entry:
        stop = entry - 2 * atr
    risk = entry - stop
    t1, t2, t3 = entry + 1.5 * risk, entry + 2.5 * risk, entry + 3.5 * risk

    return dict(
        price=price, atr=atr, rsi=rsi, support=support, resistance=resistance,
        score=score, verdict=verdict, color=color, confidence=confidence,
        reasons=reasons, entry_lo=entry_lo, entry_hi=entry_hi, entry=entry,
        entry_type=entry_type, stop=stop, risk=risk, t1=t1, t2=t2, t3=t3,
        vol_ratio=vol_ratio,
    )


# --------------------------------------------------------------- chart
def make_chart(d: pd.DataFrame, a: dict) -> go.Figure:
    d = d.tail(150)
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
    )
    fig.add_trace(
        go.Candlestick(
            x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
            name="السعر",
        ),
        row=1, col=1,
    )
    for n, col in ((20, "#f59e0b"), (50, "#3b82f6"), (200, "#a855f7")):
        fig.add_trace(
            go.Scatter(x=d.index, y=d[f"EMA{n}"], name=f"EMA{n}", line=dict(width=1.2, color=col)),
            row=1, col=1,
        )
    for y, name, col in (
        (a["resistance"], "مقاومة", "#ef4444"),
        (a["support"], "دعم", "#22c55e"),
        (a["stop"], "وقف الخسارة", "#dc2626"),
        (a["t1"], "هدف 1", "#16a34a"),
        (a["t2"], "هدف 2", "#15803d"),
    ):
        fig.add_hline(y=y, line_dash="dash", line_color=col, line_width=1,
                      annotation_text=f"{name} {y:.2f}", row=1, col=1)

    fig.add_trace(go.Scatter(x=d.index, y=d["RSI"], name="RSI", line=dict(color="#0ea5e9")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="gray", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="gray", row=2, col=1)

    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in d["MACD_HIST"]]
    fig.add_trace(go.Bar(x=d.index, y=d["MACD_HIST"], name="MACD", marker_color=colors), row=3, col=1)

    fig.update_layout(
        height=720, xaxis_rangeslider_visible=False, showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


# ------------------------------------------------------------------ UI
st.title("📈 محلل الأسهم المصرية")
st.caption("تحليل فني آلي: اتجاه، زخم، دعم ومقاومة، منطقة دخول، وقف خسارة، وأهداف.")

col1, col2 = st.columns([2, 1])
with col1:
    choice = st.selectbox("اختر السهم", list(POPULAR.keys()))
    symbol = POPULAR[choice]
    if not symbol:
        symbol = st.text_input("اكتب رمز السهم (مثال: COMI.CA)", "COMI.CA").strip().upper()
with col2:
    period_label = st.selectbox("الفترة", list(PERIODS.keys()), index=1)

if st.button("حلل السهم 🔍", type="primary", use_container_width=True):
    with st.spinner("جاري جلب البيانات وتحليلها..."):
        raw = load(symbol, PERIODS[period_label])

    if raw.empty or len(raw) < 60:
        st.error(
            "مقدرتش أجيب بيانات كافية للسهم ده. اتأكد من الرمز (لازم ينتهي بـ .CA) "
            "أو جرب سهم تاني. بيانات البورصة المصرية في Yahoo أحياناً بتكون ناقصة."
        )
        st.stop()

    d = add_indicators(raw)
    a = analyze(d)

    st.subheader(f"{symbol}")
    last_date = d.index[-1].strftime("%Y-%m-%d")
    st.caption(f"آخر تحديث للبيانات: {last_date}")

    m1, m2, m3 = st.columns(3)
    m1.metric("آخر سعر", f"{a['price']:.2f}")
    m2.metric("RSI", f"{a['rsi']:.0f}")
    m3.metric("درجة الثقة", f"{a['confidence']}%")

    st.markdown(f"### الخلاصة: :{a['color']}[{a['verdict']}]")

    st.markdown("#### 🎯 خطة التداول")
    if a["verdict"] == "تجنب":
        st.caption("⚠️ التقييم الحالي سلبي، فالخطة دي افتراضية وتنفع بس لو السهم غيّر اتجاهه واخترق المقاومة.")
    elif a["verdict"] != "شراء":
        st.caption("⏳ الأفضل تنتظر تأكيد الاختراق قبل الدخول.")
    st.markdown(
        f"""
| البند | القيمة |
|---|---|
| نوع الدخول | {a['entry_type']} |
| منطقة الدخول | {a['entry_lo']:.2f} – {a['entry_hi']:.2f} |
| وقف الخسارة | {a['stop']:.2f} ({(a['stop']/a['entry']-1)*100:.1f}%) |
| الهدف الأول | {a['t1']:.2f} (+{(a['t1']/a['entry']-1)*100:.1f}%) |
| الهدف الثاني | {a['t2']:.2f} (+{(a['t2']/a['entry']-1)*100:.1f}%) |
| الهدف الثالث | {a['t3']:.2f} (+{(a['t3']/a['entry']-1)*100:.1f}%) |
| أقرب دعم | {a['support']:.2f} |
| أقرب مقاومة | {a['resistance']:.2f} |
| عائد/مخاطرة (هدف 1) | 1 : 1.5 |
"""
    )

    st.plotly_chart(make_chart(d, a), use_container_width=True)

    st.markdown("#### 🧠 أسباب التقييم")
    for r in a["reasons"]:
        st.markdown(f"- {r}")

st.divider()
st.warning(
    "⚠️ هذه الأداة للأغراض التعليمية فقط وليست نصيحة استثمارية أو توصية بالشراء أو البيع. "
    "التحليل الفني لا يضمن نتائج، والتداول فيه مخاطرة. القرار والمسؤولية عليك."
)
