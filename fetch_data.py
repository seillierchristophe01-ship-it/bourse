import sys, io, time, json, datetime as dt, requests
import numpy as np, pandas as pd, yfinance as yf
from concurrent.futures import ThreadPoolExecutor

H = {"User-Agent": "Mozilla/5.0"}
# Segments de Tokyo à analyser : ajoute "スタンダード" (Standard) et/ou "グロース" (Growth) pour élargir
TOKYO_SEGMENTS = ["プライム"]
JPX_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xlsx"
NIKKEI_URL = "https://indexes.nikkei.co.jp/en/nkave/index/component"
W = {"tendance": 25, "rsi": 10, "macd": 10, "bollinger": 5, "volume": 5,
     "valorisation": 15, "croissance": 15, "potentiel": 15}

def tables(url):
    return pd.read_html(io.StringIO(requests.get(url, headers=H, timeout=30).text))

def find(url, code_cols, name_cols):
    for t in tables(url):
        c = next((x for x in t.columns if str(x).strip() in code_cols), None)
        n = next((x for x in t.columns if str(x).strip() in name_cols), None)
        if c is not None and n is not None and len(t) >= 20:
            return list(zip(t[c].astype(str), t[n].astype(str)))
    raise ValueError("table not found")

def nikkei225():
    ts = [t for t in tables(NIKKEI_URL) if {"Code", "Company Name"} <= set(map(str, t.columns))]
    t_all = pd.concat(ts, ignore_index=True).dropna(subset=["Code", "Company Name"])
    return [(str(c).strip().upper() + ".T", str(n), "Tokyo")
            for c, n in zip(t_all["Code"], t_all["Company Name"])
            if len(str(c).strip()) == 4 and str(c).strip().isalnum()]

def universe(market):
    u, errs = [], []
    if market == "newyork":
        for page in ["List_of_S%26P_500_companies", "List_of_S%26P_400_companies", "List_of_S%26P_600_companies"]:
            try:
                for s, n in find("https://en.wikipedia.org/wiki/" + page, ["Symbol", "Ticker symbol"], ["Security", "Company"]):
                    u.append((s.replace(".", "-"), n, "New York"))
            except Exception as e: errs.append(f"{page}: {e}")
    elif market == "paris":
        try:
            for s, n in find("https://en.wikipedia.org/wiki/CAC_40", ["Ticker"], ["Company"]):
                u.append((s if "." in s else s + ".PA", n, "Paris"))
        except Exception as e: errs.append(f"CAC 40: {e}")
        try:
            for line in open("tickers_paris.txt", encoding="utf-8"):
                t = line.split("#")[0].strip().upper()
                if t: u.append((t if "." in t else t + ".PA", t, "Paris"))
        except Exception as e: errs.append(f"tickers_paris.txt: {e}")
    elif market == "tokyo":
        try:
            x = pd.read_excel(io.BytesIO(requests.get(JPX_URL, headers=H, timeout=60).content))
            for c, n, g in zip(x[x.columns[1]], x[x.columns[2]], x[x.columns[3]]):
                c = str(c).strip().upper()
                if len(c) == 4 and c.isalnum() and "内国" in str(g) and any(k in str(g) for k in TOKYO_SEGMENTS):
                    u.append((c + ".T", str(n), "Tokyo"))
            if len(u) < 500: raise ValueError(f"seulement {len(u)} titres")
        except Exception as e:
            errs.append(f"Liste JPX indisponible ({e}) : repli sur le Nikkei 225")
            u = []
            try: u = nikkei225()
            except Exception as e2: errs.append(f"Nikkei 225: {e2}")
    else: errs.append("marché inconnu")
    return u, errs

def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(v) for v in o]
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)): return None
    return o

ip = lambda x, xp, fp: float(np.interp(x, xp, fp))

def indicators(df):
    c, v = df["Close"], df["Volume"]
    p = c.iloc[-1]
    sma50, sma200 = c.rolling(50).mean().iloc[-1], c.rolling(200).mean().iloc[-1]
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1/14).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1/14).mean()
    rsi = 100 - 100 / (1 + up.iloc[-1] / dn.iloc[-1]) if dn.iloc[-1] else 100
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean(); hist = macd - macd.ewm(span=9).mean()
    m20, s20 = c.rolling(20).mean().iloc[-1], c.rolling(20).std().iloc[-1]
    pb = (p - (m20 - 2*s20)) / (4*s20) if s20 else 0.5
    vr = v.rolling(20).mean().iloc[-1] / max(v.rolling(60).mean().iloc[-1], 1)
    dirn = np.sign(p / c.iloc[-21] - 1) if len(c) > 21 else 0
    s = {}
    t = [p > sma50, p > sma200] if not np.isnan(sma200) else [p > sma50]
    if not np.isnan(sma200): t.append(sma50 > sma200)
    s["tendance"] = 10 * sum(t) / len(t)
    s["rsi"] = ip(rsi, [0, 30, 50, 65, 75, 100], [2, 4, 6, 9, 6, 2])
    h = hist.iloc[-1] / p * 100
    s["macd"] = float(np.clip(5 + np.clip(h * 10, -4, 4) + (1 if hist.iloc[-1] > hist.iloc[-6] else -1), 0, 10))
    s["bollinger"] = ip(pb, [0, .2, .5, .8, 1, 1.2], [2, 4, 6, 8, 6, 3])
    s["volume"] = float(np.clip(5 + dirn * np.clip((vr - 1) * 10, -4, 4), 0, 10))
    return p, s

def info(sym):
    for k in range(2):
        try:
            r = yf.Ticker(sym).info
            if r: return sym, r
        except Exception: pass
        time.sleep(1.5)
    return sym, {}

def main(market):
    u, errs = universe(market)
    meta = {s: (n, m) for s, n, m in u}
    syms = list(meta)
    frames = {}
    for i in range(0, len(syms), 100):
        ch = syms[i:i+100]
        try:
            d = yf.download(ch, period="1y", group_by="ticker", auto_adjust=True, progress=False, threads=True)
            for s in ch:
                try:
                    x = d[s].dropna(subset=["Close"])
                    if len(x) >= 60: frames[s] = x
                except Exception: pass
        except Exception as e: errs.append(f"download {i}: {e}")
    with ThreadPoolExecutor(8) as ex: infos = dict(ex.map(info, list(frames)))
    out = []
    for s, df in frames.items():
        try:
            p, sc = indicators(df); f = infos.get(s, {})
            pe, g = f.get("trailingPE"), f.get("earningsGrowth")
            sc["valorisation"] = 5 if pe is None else (2 if pe <= 0 else ip(pe, [5, 12, 20, 30, 50, 100], [8, 9, 7, 5, 3, 1]))
            sc["croissance"] = 5 if g is None else ip(g, [-.3, 0, .1, .25, .5], [1, 4, 6, 8, 10])
            tm, tl, th = f.get("targetMeanPrice"), f.get("targetLowPrice"), f.get("targetHighPrice")
            pot = None
            if tm:
                pot = {"mid": tm/p - 1, "lo": tl/p - 1 if tl else None, "hi": th/p - 1 if th else None}
                sc["potentiel"] = ip(pot["mid"], [-.2, 0, .1, .25, .5], [1, 4, 6, 8, 10])
            else: sc["potentiel"] = 5
            glob = sum(sc[k] * W[k] for k in W) / sum(W.values())
            out.append({"t": s, "name": f.get("shortName") or meta[s][0], "mkt": meta[s][1], "price": round(float(p), 2),
                        "scores": {k: round(v, 1) for k, v in sc.items()}, "global": round(glob, 1),
                        "trend": "Hausse" if glob >= 6.5 else "Baisse" if glob <= 4.5 else "Neutre", "pot": pot})
        except Exception as e: errs.append(f"{s}: {e}")
    miss = sum(1 for f in infos.values() if not f.get("trailingPE") and not f.get("targetMeanPrice"))
    errs.insert(0, f"{market}: fondamentaux absents pour {miss}/{len(frames)} titres (notes neutres à 5)")
    out.sort(key=lambda r: -r["global"])
    payload = clean({"generated": dt.datetime.utcnow().isoformat() + "Z", "weights": W, "errors": errs[:50], "rows": out})
    json.dump(payload, open(f"docs/data_{market}.json", "w"), allow_nan=False)

main(sys.argv[1])
