"""
=====================================================================
 CHANTIER PRO : Budget, Probabilités & Fiabilité structurale
=====================================================================
Lancement local :
    pip install -r requirements.txt
    streamlit run app_chantier.py

Modules :
  1. Tableau de bord financier (FCFA) : budget, dépenses, EAC, écarts, statuts
  2. Probabilités : lois Normale / Log-Normale / Bêta-PERT / Uniforme,
     Monte-Carlo (coût ou délai), P(dépassement) avec IC 95 %, PDF / CDF
  3. Poutre béton armé (Eurocode 2 simplifié) + fiabilité : indice β de
     Cornell, Pf = Φ(-β), vérification Monte-Carlo, alerte critique
  4. Guide de déploiement GitHub + Streamlit Community Cloud

Présentation : tout ce qui est dessiné (anneaux, jauges, compteurs, barres) est du
HTML/CSS/SVG animé, sans bibliothèque. Les graphiques statistiques sont des graphiques
Plotly animés (chargés depuis le CDN Plotly : une connexion internet est nécessaire).
Les calculs (make_dist, simulate, mrd, beam_calc, beam_mc) ne sont pas modifiés.
=====================================================================
"""
import html
import json
import math
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components
from plotly.offline import get_plotlyjs_version
from scipy import stats

st.set_page_config(page_title="Chantier Pro", page_icon=":material/construction:", layout="wide", initial_sidebar_state="collapsed")

# ---------------------------------------------------------------------
# THÈMES
# Chaque thème règle ses couleurs (cartes, en-tête, graphiques). Le mode clair / sombre se choisit comme avant,
# dans le menu ⋮ → Settings de Streamlit : chaque thème a ses deux versions.
# Pour en ajouter un : copier un bloc de THEMES, changer les couleurs, lui donner une clé et un nom.
# ---------------------------------------------------------------------
def _ld(clair, sombre):
    return f"light-dark({clair},{sombre})"


T1_VARS = {
    "page": _ld("#F6F7F9", "#0B1220"), "ink": _ld("#1F2937", "#EAF0FA"),
    "mut": _ld("#5B6472", "#9DB0CC"), "line": _ld("rgba(15,23,42,.12)", "rgba(255,255,255,.09)"),
    "card": _ld("#FFFFFF", "#121B2E"), "card-sh": "none", "hov-sh": "0 12px 28px rgba(0,0,0,.16)",
    "track": "color-mix(in srgb,CanvasText 8%,transparent)",
    "accent": "#FF8A3D", "accent-t": _ld("#C2410C", "#FF8A3D"),
    "tab-bg": "color-mix(in srgb,var(--accent) 16%,transparent)", "tab-line": "color-mix(in srgb,var(--accent) 42%,transparent)",
    "mark-a": "#FF8A3D", "mark-b": "#FBBF24", "mark-ink": "#0B1220",
    "mark-bg": "linear-gradient(135deg,var(--mark-a),var(--mark-b))",
    "mark-sh": "0 8px 26px color-mix(in srgb,var(--mark-a) 38%,transparent)",
    "sec-bar": "linear-gradient(180deg,var(--mark-a),var(--mark-b))",
    "hero-a": "#0B1220", "hero-b": "#14234A", "hero-c": "#8A3410",
    "hero-bg": "linear-gradient(125deg,var(--hero-a) 0%,var(--hero-b) 52%,var(--hero-c) 135%)",
    "hero-ink": "#fff", "hero-mut": "rgba(255,255,255,.84)", "hero-line": "transparent", "hero-sh": "0 14px 40px rgba(2,8,23,.25)",
    "hero-g1": "rgba(249,115,22,.46)", "hero-g2": "rgba(45,212,191,.22)",
    "pill-bg": "rgba(255,255,255,.12)", "pill-line": "rgba(255,255,255,.2)", "pill-fg": "#fff",
    "hc-a": "#1B2A4A", "hc-b": "#101A33", "hc-g": "rgba(45,212,191,.18)",
    "hc-bg": "linear-gradient(160deg,var(--hc-a),var(--hc-b))", "hc-ink": "#EAF0FA", "hc-mut": "#9DB0CC", "hc-sub": "#C9D4E6",
    "hc-line": "rgba(255,255,255,.09)", "hc-sh": "0 14px 40px rgba(2,8,23,.24)", "hc-track": "rgba(255,255,255,.1)",
    "hc-side-bg": "rgba(255,255,255,.05)", "hc-side-line": "rgba(255,255,255,.08)",
    "gm": "#fff", "gm-soft": "rgba(255,255,255,.75)",
    "chip-ok-bg": "rgba(45,212,191,.16)", "chip-ok-fg": "#5EEAD4", "chip-warn-bg": "rgba(251,191,36,.16)", "chip-warn-fg": "#FCD34D",
    "chip-bad-bg": "rgba(255,107,107,.16)", "chip-bad-fg": "#FF9A9A",
    "al-ok": "color-mix(in srgb,var(--good) 11%,var(--page))", "al-warn": "color-mix(in srgb,var(--warn) 13%,var(--page))",
    "al-crit": "color-mix(in srgb,var(--bad) 14%,var(--page))",
    "r1": "#FF8A3D", "r2": "#2DD4BF", "spent": "#5B9CFF", "fc": "#FF8A3D",
    "c-blue": _ld("#2a78d6", "#5B9CFF"), "c-aqua": _ld("#1baf7a", "#2DD4BF"),
    "glow1": "rgba(255,138,61,.20)", "glow2": "rgba(91,156,255,.10)",
    "good-t": _ld("#0f766e", "#5EEAD4"), "warn-t": _ld("#8a5a00", "#FCD34D"), "bad-t": _ld("#b91c1c", "#FF9A9A"),
    "r-card": "22px", "r-hero": "26px", "r-hero-sm": "22px", "r-alert": "18px", "r-side": "20px", "r-mark": "16px",
}
_BLEU = _ld("#1a4bdb", "#589bf3")          # --plexa-primary : bleu en clair, bleu ciel en sombre
_FAINT = _ld("rgba(26,75,219,.1)", "rgba(88,155,243,.1)")     # --plexa-primary-faint
_BG3 = _ld("#eef1f8", "#181c21")           # --plexa-bg3

THEMES = {
    "t1": {
        "name": "Thème 1 · Chantier", "label": "Thème 1", "icon": "construction", "mode": "Dark",
        "font": "https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap",
        "fontfam": 'Manrope, "Segoe UI", system-ui, sans-serif',
        "level": {"ok": "#2DD4BF", "warn": "#FBBF24", "crit": "#FF6B6B"},          # bon / vigilance / critique (maquette)
        "chart": {
            "light": dict(text="#1F2937", muted="#5B6472", grid="rgba(0,0,0,.07)", axis="rgba(0,0,0,.22)", tip="#FFFFFF",
                          gray="#7a869c", c1="#2a78d6", c2="#eb6834", c3="#1baf7a", c4="#eda100"),
            "dark": dict(text="#E5E7EB", muted="#A3AEC2", grid="rgba(255,255,255,.08)", axis="rgba(255,255,255,.22)", tip="#0F172A",
                         gray="#5B6B8C", c1="#5B9CFF", c2="#FF8A3D", c3="#2DD4BF", c4="#FBBF24"),
        },
        "vars": T1_VARS,
        "extra": '.stApp :is(p,span,div,label,button,input,textarea,li,td,th,h1,h2,h3,h4,h5,h6,a,summary,kbd)'
                 ':not([data-testid="stIconMaterial"]){font-family:Manrope,"Segoe UI",system-ui,sans-serif;}',
    },
    # Insights, à l'identique : mêmes valeurs que --plexa-* de libs/shared-ui/src/styles/plexa-global.css
    # (fond, cartes, bordures, textes, bleu primaire, menthe, succès / alerte / danger, rayons, police Inter). Rien de dégradé.
    "t2": {
        "name": "Thème 2 · Insights", "label": "Thème 2", "icon": "insights", "mode": "Dark",
        "fontfam": 'Inter, "Segoe UI", system-ui, sans-serif',
        "font": "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap",
        "level": {"ok": "#10b981", "warn": "#f59e0b", "crit": "#f5405a"},          # --plexa-success / warning / danger
        "chart": {                       # couleurs utilisées par les graphiques d'Insights : primaire, warning, success, violet
            "light": dict(text="#060d1a", muted="#44546a", grid="rgba(15,23,42,.08)", axis="rgba(15,23,42,.25)", tip="#FFFFFF",
                          gray="#64748B", c1="#1a4bdb", c2="#F59E0B", c3="#059669", c4="#7C3AED"),
            "dark": dict(text="#FFFFFF", muted="#CBD5E1", grid="rgba(255,255,255,.08)", axis="rgba(255,255,255,.22)", tip="#111417",
                         gray="#94A3B8", c1="#589bf3", c2="#F59E0B", c3="#4ade80", c4="#A78BFA"),
        },
        "vars": {**T1_VARS,
                 "page": _ld("#eaeef5", "#06080a"), "ink": _ld("#060d1a", "#ffffff"),             # --plexa-bg / --plexa-text1
                 "mut": _ld("#44546a", "#cbd5e1"), "line": _ld("rgba(15,23,42,.12)", "rgba(255,255,255,.06)"),   # text3 / border
                 "card": _ld("#ffffff", "#111417"),                                              # --plexa-bg2
                 "card-sh": "0 1px 2px rgba(15,23,42,.08),0 1px 3px rgba(15,23,42,.1)", "hov-sh": "0 4px 20px rgba(0,0,0,.09)",
                 "track": _BG3,
                 "accent": _BLEU, "accent-t": _BLEU, "tab-bg": _FAINT, "tab-line": "transparent",
                 "mark-a": _BLEU, "mark-b": _BLEU, "mark-ink": _ld("#ffffff", "#06080a"), "mark-bg": _BLEU, "mark-sh": "none",
                 "sec-bar": _BLEU,
                 "hero-g1": "transparent", "hero-g2": "transparent",
                 "hero-bg": "var(--card)", "hero-ink": "var(--ink)", "hero-mut": "var(--mut)", "hero-line": "var(--line)", "hero-sh": "var(--card-sh)",
                 "pill-bg": _FAINT, "pill-line": "transparent", "pill-fg": _BLEU,
                 "hc-g": "transparent", "hc-bg": "var(--card)", "hc-ink": "var(--ink)", "hc-mut": "var(--mut)", "hc-sub": "var(--mut)",
                 "hc-line": "var(--line)", "hc-sh": "var(--card-sh)", "hc-track": _BG3, "hc-side-bg": _BG3, "hc-side-line": "var(--line)",
                 "gm": "var(--ink)", "gm-soft": "color-mix(in srgb,var(--ink) 70%,transparent)",
                 "chip-ok-bg": "rgba(16,185,129,.1)", "chip-ok-fg": _ld("#047857", "#4ade80"),
                 "chip-warn-bg": "rgba(245,158,11,.1)", "chip-warn-fg": _ld("#b45309", "#fbbf24"),
                 "chip-bad-bg": "rgba(245,64,90,.1)", "chip-bad-fg": _ld("#be123c", "#fb7185"),
                 "al-ok": _ld("#d1fae5", "#1a2e28"), "al-warn": _ld("#fef3c7", "#2e2618"), "al-crit": _ld("#fee2e2", "#2e1a20"),   # --toast-bg-*
                 "r1": _BLEU, "r2": _ld("#00c899", "#00e5b0"), "spent": _BLEU, "fc": "#f59e0b",   # menthe (--plexa-mint-dim / mint)
                 "c-blue": _BLEU, "c-aqua": _ld("#00c899", "#00e5b0"),
                 "glow1": "transparent", "glow2": "transparent",
                 "good-t": _ld("#047857", "#4ade80"), "warn-t": _ld("#b45309", "#fbbf24"), "bad-t": _ld("#be123c", "#fb7185"),
                 "r-card": "16px", "r-hero": "16px", "r-hero-sm": "16px", "r-alert": "12px", "r-side": "12px", "r-mark": "12px"},
        # Police Inter (celle d'Insights) sur tout le texte, sauf les icônes ; curseurs, boutons radio et cases de Streamlit
        # (orange dans config.toml) recolorés en bleu.
        "extra": '.stApp :is(p,span,div,label,button,input,textarea,li,td,th,h1,h2,h3,h4,h5,h6,a,summary,kbd)'
                 ':not([data-testid="stIconMaterial"]){font-family:Inter,"Segoe UI",system-ui,sans-serif;}'
                 'div[data-testid="stSlider"],div[data-testid="stSelectSlider"],div[data-testid="stRadio"],'
                 'div[data-testid="stCheckbox"],div[data-testid="stButtonGroup"]{filter:hue-rotate(200deg);}',
    },
}
THEME_KEY = (st.session_state.get("theme") or st.session_state.get("_theme") or st.query_params.get("theme") or "t2")   # par défaut : Insights
if THEME_KEY not in THEMES:
    THEME_KEY = "t2"
TH = THEMES[THEME_KEY]
LEVEL_C = TH["level"]


def _theme_css():
    base = {"mut": "color-mix(in srgb,var(--ink) 62%,transparent)", "line": "color-mix(in srgb,var(--ink) 13%,transparent)",
            "good": LEVEL_C["ok"], "warn": LEVEL_C["warn"], "bad": LEVEL_C["crit"]}
    base.update(TH["vars"])
    return ":root{" + ";".join(f"--{k}:{v}" for k, v in base.items()) + "}" + TH["extra"]


# ---------------------------------------------------------------------
# APPARENCE  (le thème de base est dans .streamlit/config.toml)
# Les couleurs des cartes s'adaptent toutes seules au mode clair / sombre :
# on s'appuie sur "Canvas" / "CanvasText" (fond et texte du thème affiché) et sur light-dark().
# ---------------------------------------------------------------------
st.markdown(
    """<style>
    @property --ip{syntax:'<integer>';initial-value:0;inherits:false}
    @property --dp{syntax:'<integer>';initial-value:0;inherits:false}
    __THEME__
    /* lueurs de fond : une couche fixe, jamais redessinée pendant le défilement */
    .stApp{color:var(--ink);background-color:var(--page) !important;
        background-image:radial-gradient(900px 520px at 90% -8%,var(--glow1),transparent 62%),
        radial-gradient(760px 520px at -8% 6%,var(--glow2),transparent 60%) !important;}
    header[data-testid="stHeader"]{background:var(--page) !important;border-bottom:1px solid var(--line);pointer-events:none !important;}
    header[data-testid="stHeader"] button, header[data-testid="stHeader"] a, header[data-testid="stHeader"] [role="button"]{pointer-events:auto;}
    .block-container{padding-top:76px !important;}

    div[data-testid="stAppDeployButton"], .stAppDeployButton{display:none;}
    @keyframes rise{from{opacity:0;transform:translateY(18px)}}
    @keyframes pop{from{opacity:0;transform:scale(.6)}}
    @keyframes cnt{from{--ip:0;--dp:0}}
    @keyframes rg{from{stroke-dashoffset:100}}
    @keyframes growx{from{transform:scaleX(0)}}
    @keyframes drift{from{transform:translate(0,0) scale(1)}to{transform:translate(-40px,30px) scale(1.15)}}
    @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(208,59,59,.55)}70%{box-shadow:0 0 0 16px rgba(208,59,59,0)}100%{box-shadow:0 0 0 0 rgba(208,59,59,0)}}

    /* --- en-tête --- */
    .hero{position:relative;overflow:hidden;isolation:isolate;border-radius:var(--r-hero);padding:28px 34px;margin:0 0 18px;color:var(--hero-ink);
        background:var(--hero-bg);border:1px solid var(--hero-line);box-shadow:var(--hero-sh);}
    .hero::before,.hero::after{content:"";position:absolute;z-index:-1;border-radius:50%;width:420px;height:420px;will-change:transform;}
    .hero::before{right:-90px;top:-150px;background:radial-gradient(circle,var(--hero-g1),transparent 64%);animation:drift 10s ease-in-out infinite alternate;}
    .hero::after{left:-130px;bottom:-210px;background:radial-gradient(circle,var(--hero-g2),transparent 64%);animation:drift 12s ease-in-out infinite alternate-reverse;}
    .hero-top{display:flex;align-items:center;gap:16px;}
    .mark{flex:none;width:50px;height:50px;border-radius:var(--r-mark);display:grid;place-items:center;color:var(--mark-ink);
        background:var(--mark-bg);box-shadow:var(--mark-sh);animation:pop .8s cubic-bezier(.2,1.4,.4,1) both;}
    .hero-title{font-size:2.1rem;font-weight:800;letter-spacing:-.03em;line-height:1.08;}
    .hero-sub{font-size:1rem;color:var(--hero-mut);margin-top:4px;}
    .pills{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px;}
    .pills span{display:inline-flex;align-items:center;gap:7px;background:var(--pill-bg);border:1px solid var(--pill-line);color:var(--pill-fg);border-radius:999px;padding:5px 13px;
        font-size:.8rem;font-weight:600;animation:rise .7s cubic-bezier(.2,.8,.2,1) both;animation-delay:calc(var(--i)*80ms + 300ms);}

    /* --- titres de section --- */
    .sec{display:flex;gap:12px;margin:28px 0 12px;}
    .sec::before{content:"";flex:none;width:5px;border-radius:9px;background:var(--sec-bar);}
    .sec h3{margin:0;padding:0;font-size:1.15rem;font-weight:800;letter-spacing:-.015em;}
    .sec p{margin:2px 0 0;font-size:.88rem;color:var(--mut);}

    /* --- nombres qui comptent de 0 à leur valeur --- */
    .cnt{counter-reset:ip var(--ip) dp var(--dp);animation:cnt 1.5s cubic-bezier(.2,.8,.2,1) both;animation-delay:calc(var(--i,0)*70ms + 150ms);}
    .cnt0::before{content:counter(ip);}
    .cnt1::before{content:counter(ip) "," counter(dp);}
    .cnt2::before{content:counter(ip) "," counter(dp,decimal-leading-zero);}

    /* --- grandes cartes sombres : anneaux et jauges --- */
    .hcard{position:relative;overflow:hidden;display:flex;align-items:center;gap:30px;flex-wrap:wrap;padding:26px 30px;margin:6px 0 14px;
        border-radius:var(--r-hero);color:var(--hc-ink);background:var(--hc-bg);border:1px solid var(--hc-line);
        box-shadow:var(--hc-sh);animation:rise .8s cubic-bezier(.2,.8,.2,1) both;}
    .hcard::after{content:"";position:absolute;right:-110px;top:-130px;width:340px;height:340px;border-radius:50%;
        background:radial-gradient(circle,var(--hc-g),transparent 65%);pointer-events:none;}
    .hc-body{position:relative;z-index:1;flex:1 1 260px;min-width:0;}
    .hc-side{position:relative;z-index:1;flex:1 1 300px;min-width:0;max-width:520px;margin-left:auto;padding:18px 20px;border-radius:var(--r-side);
        background:var(--hc-side-bg);border:1px solid var(--hc-side-line);}
    .tk{position:relative;height:16px;border-radius:99px;background:var(--hc-track);margin:34px 0 10px;}
    .tk.tt{margin-bottom:36px;}
    .tk i{position:absolute;top:0;bottom:0;border-radius:99px;transform-origin:left;animation:growx 1.6s cubic-bezier(.2,.8,.2,1) both;}
    .tk .sp{left:0;background:linear-gradient(90deg,var(--spent),var(--r1));}
    .tk .fc{background:repeating-linear-gradient(135deg,var(--fc) 0 6px,color-mix(in srgb,var(--fc) 58%,transparent) 6px 12px);animation-delay:.35s;border-radius:0 99px 99px 0;}
    .tk .bm{position:absolute;top:-12px;bottom:-12px;width:3px;margin-left:-1px;border-radius:3px;background:var(--gm);}
    .tk .bm span{position:absolute;bottom:100%;left:50%;transform:translateX(-50%);white-space:nowrap;font-size:.72rem;font-weight:700;margin-bottom:4px;}
    .tk .pt{position:absolute;top:-5px;bottom:-5px;width:2px;margin-left:-1px;background:var(--gm-soft);}
    .tk .pt span{position:absolute;top:100%;left:50%;transform:translateX(-50%);margin-top:7px;font-size:.7rem;font-weight:700;color:var(--hc-sub);}
    .tk-leg{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:.82rem;color:var(--hc-sub);}
    .tk-leg span{display:flex;align-items:center;gap:7px;}.tk-leg i{width:10px;height:10px;border-radius:3px;flex:none;}
    .hc-lab{font-size:.72rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:var(--hc-mut);}
    .hc-big{font-size:3.1rem;font-weight:800;letter-spacing:-.04em;line-height:1.05;margin-top:6px;}
    .hc-big small{font-size:1.05rem;font-weight:600;color:var(--hc-mut);letter-spacing:0;margin-left:6px;}
    .hc-title{font-size:1.5rem;font-weight:800;letter-spacing:-.02em;line-height:1.2;margin:10px 0 2px;}
    .hc-note{font-size:.88rem;color:var(--hc-mut);margin-top:8px;line-height:1.5;}
    .hc-leg{display:grid;gap:7px;margin-top:16px;font-size:.88rem;max-width:340px;}
    .hc-leg span{display:flex;align-items:center;gap:9px;}
    .hc-leg i{width:11px;height:11px;border-radius:50%;flex:none;}
    .hc-leg b{margin-left:auto;font-weight:800;}
    .chip{display:inline-flex;align-items:center;gap:7px;padding:6px 13px;border-radius:999px;font-size:.84rem;font-weight:700;margin-top:12px;}
    .chip.ok{background:var(--chip-ok-bg);color:var(--chip-ok-fg);}
    .chip.warn{background:var(--chip-warn-bg);color:var(--chip-warn-fg);}
    .chip.bad{background:var(--chip-bad-bg);color:var(--chip-bad-fg);}
    .ringw{position:relative;flex:none;width:176px;height:176px;}
    .ringw svg{display:block;width:100%;height:100%;}
    .rt{fill:none;stroke:var(--hc-track);}
    .rg{fill:none;stroke-linecap:round;stroke-dasharray:100;stroke-dashoffset:calc(100 - var(--v));animation:rg 1.7s cubic-bezier(.2,.8,.2,1) both;}
    .ringc{position:absolute;inset:0;display:grid;place-content:center;text-align:center;line-height:1;}
    .ringc b{font-size:2.3rem;font-weight:800;letter-spacing:-.04em;}
    .ringc > span{font-size:.7rem;color:var(--hc-mut);margin-top:6px;letter-spacing:.04em;}
    .gw{position:relative;flex:none;width:min(300px,100%);}
    .gw svg{display:block;width:100%;height:auto;overflow:visible;}
    .gz{fill:none;stroke-width:16;opacity:.3;}
    .gv{fill:none;stroke-width:16;stroke-linecap:round;stroke-dasharray:100;stroke-dashoffset:calc(100 - var(--v));
        animation:rg 1.8s cubic-bezier(.2,.8,.2,1) .25s both;}
    .gm{stroke:var(--gm);stroke-width:3.5;stroke-linecap:round;}
    .gtxt{position:absolute;left:0;right:0;bottom:2px;text-align:center;line-height:1;}
    .gtxt b{font-size:2.9rem;font-weight:800;letter-spacing:-.04em;}
    .gtxt > span{display:block;font-size:.74rem;color:var(--hc-mut);margin-top:7px;}
    .meter{margin-top:16px;}
    .m-head{display:flex;justify-content:space-between;align-items:baseline;font-size:.88rem;color:var(--hc-mut);}
    .m-head b{font-size:1.45rem;color:var(--hc-ink);font-weight:800;letter-spacing:-.02em;}
    .m-track{height:12px;border-radius:99px;background:var(--hc-track);overflow:hidden;margin-top:8px;}
    .m-track i{display:block;height:100%;border-radius:99px;transform-origin:left;animation:growx 1.5s cubic-bezier(.2,.8,.2,1) .4s both;}
    .m-sub{font-size:.78rem;color:var(--hc-mut);margin-top:6px;}

    /* --- indicateurs --- */
    .kpis{display:grid;grid-template-columns:repeat(var(--cols,6),minmax(0,1fr));gap:12px;margin:8px 0 12px;}
    .kpi{min-width:0;padding:15px 17px;border-radius:var(--r-card);background:var(--card);border:1px solid var(--line);box-shadow:var(--card-sh);
        animation:rise .8s cubic-bezier(.2,.8,.2,1) both;animation-delay:calc(var(--i,0)*70ms);transition:transform .2s,box-shadow .2s;}
    .kpi:hover{transform:translateY(-3px);box-shadow:var(--hov-sh);}
    .kl{font-size:.7rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:var(--mut);}
    .kv{font-size:1.65rem;font-weight:800;letter-spacing:-.025em;line-height:1.15;margin-top:7px;white-space:nowrap;}
    .kv small{font-size:.78rem;font-weight:600;color:var(--mut);letter-spacing:0;margin-left:4px;}
    .kd{display:inline-flex;align-items:center;gap:5px;margin-top:9px;padding:3px 10px;border-radius:999px;font-size:.74rem;font-weight:700;}
    .kd.ok{background:color-mix(in srgb,var(--good) 16%,transparent);color:var(--good-t);}
    .kd.warn{background:color-mix(in srgb,var(--warn) 20%,transparent);color:var(--warn-t);}
    .kd.bad{background:color-mix(in srgb,var(--bad) 16%,transparent);color:var(--bad-t);}
    .kd.mut{background:color-mix(in srgb,CanvasText 8%,transparent);color:var(--mut);}

    /* --- alertes --- */
    .alert{display:flex;gap:14px;align-items:flex-start;padding:16px 20px;margin:12px 0;border-radius:var(--r-alert);line-height:1.5;
        border:1px solid;border-left-width:6px;animation:rise .7s cubic-bezier(.2,.8,.2,1) both;}
    .a-ic{flex:none;width:34px;height:34px;border-radius:50%;display:grid;place-items:center;font-weight:800;color:#fff;}
    .alert b{font-size:1.04rem;}
    .a-ok{background:var(--al-ok);border-color:var(--good);}
    .a-ok .a-ic{background:var(--good);}.a-ok b{color:var(--good-t);}
    .a-warn{background:var(--al-warn);border-color:var(--warn);}
    .a-warn .a-ic{background:var(--warn);color:#1f1400;}.a-warn b{color:var(--warn-t);}
    .a-crit{background:var(--al-crit);border-color:var(--bad);animation:rise .7s cubic-bezier(.2,.8,.2,1) both,pulse 1.8s 1s infinite;}
    .a-crit .a-ic{background:var(--bad);}.a-crit b{color:var(--bad-t);}
    .ico{flex:none;vertical-align:-.2em;}
    .st-key-theme{position:absolute;width:1px;height:1px;overflow:hidden;opacity:0;pointer-events:none;margin:0 !important;padding:0 !important;}
    div[data-testid="stElementContainer"]:has(iframe[srcdoc*="chantier-bridge"]){position:absolute;width:0;height:0;overflow:hidden;margin:0 !important;}
    .foot{text-align:center;font-size:.8rem;color:var(--mut);margin:36px 0 6px;}

    /* --- cartes et graphiques --- */
    .card{padding:18px 20px;margin:6px 0 10px;border-radius:var(--r-card);background:var(--card);border:1px solid var(--line);box-shadow:var(--card-sh);animation:rise .8s cubic-bezier(.2,.8,.2,1) both;}
    .chart-title{font-weight:800;font-size:1rem;line-height:1.3;letter-spacing:-.01em;margin:2px 2px 0;}
    .chart-sub{font-size:.82rem;color:var(--mut);margin:3px 2px 6px;line-height:1.4;}
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.cc-mark){border-radius:var(--r-card);border-color:var(--line);background:var(--card);box-shadow:var(--card-sh);}
    div[data-testid="stExpander"] details{border-radius:var(--r-alert);border:1px solid var(--line);background:var(--card);}
    div[data-testid="stNumberInput"] input, div[data-baseweb="select"] > div{min-height:44px;}
    /* barres horizontales (écarts par lot, contribution à la variance) */
    .hb{display:grid;gap:11px;margin-top:10px;}
    .hb-row{display:grid;grid-template-columns:minmax(84px,170px) 1fr minmax(78px,auto);gap:12px;align-items:center;
        animation:rise .7s cubic-bezier(.2,.8,.2,1) both;animation-delay:calc(var(--i,0)*70ms);}
    .hb-l{font-size:.86rem;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
    .hb-t{position:relative;height:12px;border-radius:99px;background:var(--track);}
    .hb-t i{position:absolute;top:0;bottom:0;border-radius:99px;animation:growx 1.3s cubic-bezier(.2,.8,.2,1) both;animation-delay:calc(var(--i,0)*70ms + 250ms);}
    .hb.dv .hb-t::after{content:"";position:absolute;left:50%;top:-4px;bottom:-4px;width:1px;background:var(--mut);}
    .hb.dv .hb-row{grid-template-columns:minmax(84px,170px) 1fr 168px;}
    .hb-v{font-size:.86rem;text-align:right;white-space:nowrap;}
    .hb-v.ok{color:var(--good-t);}.hb-v.bad{color:var(--bad-t);}.hb-v.mut{color:var(--mut);}
    .hb-cap{display:flex;justify-content:space-between;font-size:.74rem;color:var(--mut);margin-top:10px;}
    /* cartes des lots */
    .lots{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px;margin:6px 0 10px;}
    .lot{padding:16px 18px;border-radius:var(--r-card);background:var(--card);border:1px solid var(--line);box-shadow:var(--card-sh);
        animation:rise .8s cubic-bezier(.2,.8,.2,1) both;animation-delay:calc(var(--i,0)*70ms);transition:transform .2s,box-shadow .2s;}
    .lot:hover{transform:translateY(-3px);box-shadow:var(--hov-sh);}
    .lot-head{display:flex;align-items:center;justify-content:space-between;gap:10px;}
    .lot-name{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:800;font-size:1rem;letter-spacing:-.01em;}
    .lot-main{display:flex;align-items:flex-end;justify-content:space-between;gap:10px;margin-top:14px;}
    .lot-k{font-size:.7rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:var(--mut);}
    .lot-v{font-size:1.55rem;font-weight:800;letter-spacing:-.025em;line-height:1.15;margin-top:3px;white-space:nowrap;}
    .lot-v small{font-size:.78rem;font-weight:600;color:var(--mut);margin-left:4px;letter-spacing:0;}
    .lot-d{display:inline-flex;align-items:center;gap:4px;padding:3px 10px;border-radius:999px;font-size:.78rem;font-weight:700;white-space:nowrap;}
    .lot-d.ok{background:color-mix(in srgb,var(--good) 16%,transparent);color:var(--good-t);}
    .lot-d.warn{background:color-mix(in srgb,var(--warn) 20%,transparent);color:var(--warn-t);}
    .lot-d.bad{background:color-mix(in srgb,var(--bad) 16%,transparent);color:var(--bad-t);}
    .lb{position:relative;height:12px;border-radius:99px;background:var(--track);margin:16px 0 9px;}
    .lb i{position:absolute;top:0;bottom:0;transform-origin:left;border-radius:99px;animation:growx 1.3s cubic-bezier(.2,.8,.2,1) both;
        animation-delay:calc(var(--i,0)*70ms + 250ms);}
    .lb .sp{left:0;background:var(--c-blue);}
    .lb .fc{background:repeating-linear-gradient(135deg,var(--fc) 0 5px,color-mix(in srgb,var(--fc) 55%,transparent) 5px 10px);
        border-radius:0 99px 99px 0;animation-delay:calc(var(--i,0)*70ms + 500ms);}
    .lb .bm{position:absolute;top:-4px;bottom:-4px;width:3px;margin-left:-1px;border-radius:3px;background:var(--ink);}
    .lb-leg{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:.78rem;color:var(--mut);}
    .lb-leg span{display:flex;align-items:center;gap:6px;}
    .lb-leg i{width:9px;height:9px;border-radius:3px;flex:none;}
    .lb-leg i.bmk{width:3px;height:11px;border-radius:2px;background:var(--ink);}
    .lot-note{font-size:.78rem;color:var(--mut);margin-top:12px;line-height:1.4;}
    .lot-chip{flex:none;display:inline-flex;align-items:center;gap:4px;font-size:.72rem;font-weight:700;padding:3px 11px;border-radius:999px;white-space:nowrap;}
    .lot-chip.ok{background:color-mix(in srgb,var(--good) 16%,transparent);color:var(--good-t);}
    .lot-chip.warn{background:color-mix(in srgb,var(--warn) 20%,transparent);color:var(--warn-t);}
    .lot-chip.bad{background:color-mix(in srgb,var(--bad) 16%,transparent);color:var(--bad-t);}
    .lot-line{display:flex;justify-content:space-between;font-size:.84rem;margin-top:13px;}
    .bar{height:8px;border-radius:99px;background:var(--track);overflow:hidden;margin-top:6px;}
    .bar>i{display:block;height:100%;border-radius:99px;transform-origin:left;animation:growx 1.3s cubic-bezier(.2,.8,.2,1) both;
        animation-delay:calc(var(--i,0)*70ms + 250ms);}
    .lot-foot{display:flex;flex-wrap:wrap;justify-content:space-between;gap:4px 12px;margin-top:15px;font-size:.84rem;color:var(--mut);}
    .lot-foot b{font-size:.95rem;color:var(--ink);}

    /* --- bulles d'aide --- */
    .tip{display:inline-grid;place-items:center;width:15px;height:15px;margin-left:6px;border-radius:50%;border:1.5px solid currentColor;
        font-size:10px;font-weight:800;line-height:1;opacity:.65;cursor:help;position:relative;text-transform:none;letter-spacing:0;
        vertical-align:middle;outline:none;}
    .tip::before{content:"i";}
    .tip::after{content:attr(data-tip);display:none;position:absolute;z-index:60;top:calc(100% + 8px);left:-10px;width:230px;padding:10px 12px;
        border-radius:10px;background:var(--ink);color:var(--page);font-size:.76rem;font-weight:500;line-height:1.4;text-align:left;
        white-space:normal;box-shadow:0 8px 24px rgba(0,0,0,.28);}
    .tip:hover,.tip:focus{opacity:1;}
    .tip:hover::after,.tip:focus::after{display:block;}
    .kpi:hover,.kpi:focus-within{position:relative;z-index:30;}
    .kpi:nth-child(even) .tip::after{left:auto;right:-10px;}
    /* --- bloc « à surveiller » --- */
    .watch{padding:16px 20px;margin:6px 0 14px;border-radius:var(--r-card);background:var(--card);border:1px solid var(--line);
        box-shadow:var(--card-sh);animation:rise .8s cubic-bezier(.2,.8,.2,1) both;}
    .w-head{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:6px;}
    .w-head b{font-size:1rem;font-weight:800;letter-spacing:-.01em;}
    .w-head a{font-size:.8rem;font-weight:700;color:var(--accent-t);text-decoration:none;}
    .wrow{display:flex;align-items:center;gap:12px;padding:9px 6px;border-radius:12px;color:inherit;text-decoration:none;
        animation:rise .7s cubic-bezier(.2,.8,.2,1) both;animation-delay:calc(var(--i,0)*80ms + 150ms);}
    a.wrow:hover{background:var(--track);}
    .w-ic{flex:none;width:28px;height:28px;border-radius:50%;display:grid;place-items:center;}
    .w-ic.ok{background:color-mix(in srgb,var(--good) 18%,transparent);color:var(--good-t);}
    .w-ic.warn{background:color-mix(in srgb,var(--warn) 22%,transparent);color:var(--warn-t);}
    .w-ic.bad{background:color-mix(in srgb,var(--bad) 18%,transparent);color:var(--bad-t);}
    .w-t{flex:1;min-width:0;font-size:.9rem;line-height:1.4;}
    .w-go{flex:none;color:var(--mut);}
    /* --- formulaires en cartes, scénarios --- */
    .fh{display:flex;align-items:center;gap:10px;margin:0 0 4px;}
    .fh-ic{width:30px;height:30px;border-radius:var(--r-mark);display:grid;place-items:center;background:var(--tab-bg);color:var(--accent-t);}
    .fh b{font-size:1rem;font-weight:800;letter-spacing:-.01em;}
    .fh-h{margin-left:auto;font-size:.74rem;color:var(--mut);}
    .pc-head{display:flex;align-items:center;gap:10px;}
    .pc-head b{font-size:1.02rem;font-weight:800;letter-spacing:-.01em;}
    .pc-ic{width:30px;height:30px;border-radius:50%;display:grid;place-items:center;flex:none;}
    .pc-ic.ok{background:color-mix(in srgb,var(--good) 18%,transparent);color:var(--good-t);}
    .pc-ic.warn{background:color-mix(in srgb,var(--warn) 22%,transparent);color:var(--warn-t);}
    .pc-ic.crit{background:color-mix(in srgb,var(--bad) 18%,transparent);color:var(--bad-t);}
    .pc-facts{font-size:.8rem;color:var(--mut);margin:2px 2px 8px;}
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.pc-on){border-color:var(--accent) !important;box-shadow:0 0 0 1px var(--accent);}
    /* --- schéma de la poutre --- */
    @keyframes bsdraw{from{stroke-dashoffset:1}}
    .bsch{display:flex;flex-wrap:wrap;gap:12px 24px;align-items:center;justify-content:center;margin-top:6px;}
    .bsch svg{display:block;height:auto;}
    .bsch .el{flex:2 1 340px;min-width:0;}
    .bsch .sc{flex:1 1 220px;max-width:300px;min-width:0;}
    .bs-d{stroke-dasharray:1;animation:bsdraw 1.4s cubic-bezier(.4,0,.2,1) both;}
    .bs-a{animation:rise .7s cubic-bezier(.2,.8,.2,1) both;animation-delay:calc(var(--i,0)*45ms + 300ms);}
    .bs-r{transform-box:fill-box;transform-origin:center;animation:pop .7s cubic-bezier(.2,1.4,.4,1) both;animation-delay:calc(var(--i,0)*70ms + 600ms);}
    .bt{font-size:14px;font-weight:600;fill:var(--mut);}
    .bt.b{fill:var(--ink);font-weight:800;}
    /* --- page Déploiement --- */
    .step{display:flex;gap:14px;align-items:flex-start;margin:26px 0 8px;}
    .step-n{flex:none;width:36px;height:36px;border-radius:50%;display:grid;place-items:center;font-weight:800;background:var(--mark-bg);color:var(--mark-ink);}
    .step h4{margin:0;padding:0;font-size:1.05rem;font-weight:800;letter-spacing:-.01em;}
    .step p{margin:3px 0 0;font-size:.9rem;color:var(--mut);line-height:1.5;}
    .files{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 4px 50px;}
    .files span{display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:999px;background:var(--pill-bg);color:var(--pill-fg);font-size:.8rem;font-weight:600;border:1px solid var(--pill-line);}

    /* --- barre du haut : marque à gauche, onglets au centre, réglages à droite --- */
    .topbar{display:flex;align-items:center;gap:12px;font-weight:800;font-size:1.25rem;letter-spacing:-.02em;}
    .topbar .mark{width:34px;height:34px;border-radius:11px;}
    .block-container > div[data-testid="stVerticalBlock"]{flex-direction:row !important;flex-wrap:wrap;align-items:center;gap:16px !important;}
    .block-container > div[data-testid="stVerticalBlock"] > div{flex:0 0 100%;min-width:0;order:5;}
    .block-container > div[data-testid="stVerticalBlock"] > div:has(.topbar){flex:1 1 0;order:1;}
    .block-container > div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stPopover"]){flex:1 1 0;order:3;display:flex;justify-content:flex-end;}
    .block-container > div[data-testid="stVerticalBlock"] > div[data-testid="stTabs"],
    .block-container > div[data-testid="stVerticalBlock"] > div[data-testid="stTabs"] > div{display:contents;}
    .block-container > div[data-testid="stVerticalBlock"] > div[data-testid="stTabs"] > div > div[role="tabpanel"]{flex:0 0 100%;order:4;min-width:0;}
    div[data-testid="stPopover"] > div > button, div[data-testid="stPopover"] button{width:44px;height:44px;padding:0;border-radius:14px;
        border:1px solid var(--line);background:var(--card);color:var(--ink);box-shadow:var(--card-sh);justify-content:center;}
    div[data-testid="stPopover"] button > *:nth-child(n+2), div[data-testid="stPopover"] button > div > *:nth-child(n+2){display:none;}
    /* la barre du haut reste visible quand on fait défiler (juste sous la barre de Streamlit, 3,75 rem) */
    .block-container > div[data-testid="stVerticalBlock"]::after{display:none;}
    .block-container > div[data-testid="stVerticalBlock"]::before{display:none !important;}
    .block-container > div[data-testid="stVerticalBlock"]::before{content:"";position:sticky;top:0;flex:0 0 100%;height:3.75rem;margin-bottom:-3.75rem;
        order:0;z-index:98;pointer-events:none;background:var(--page);box-shadow:0 0 0 100vmax var(--page);clip-path:inset(0 -100vmax);border-bottom:1px solid var(--line);}
    .block-container > div[data-testid="stVerticalBlock"] > div:has(.topbar){position:fixed;top:5px;left:5rem;z-index:1000001;height:50px;margin:0 !important;display:flex;align-items:center;flex:none;width:auto;}
    .block-container > div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stPopover"]){position:fixed;top:5px;right:56px;z-index:1000001;height:50px;margin:0 !important;display:flex;align-items:center;flex:none;width:auto;}
    @media (max-width:1100px){ .block-container > div[data-testid="stVerticalBlock"] > div:has(.topbar){left:1.6rem;} }
    @media (max-width:640px){ .block-container > div[data-testid="stVerticalBlock"] > div:has(.topbar){left:.8rem;} }
    @media (min-width:641px) and (max-width:860px){
      .block-container > div[data-testid="stVerticalBlock"] > div[data-testid="stTabs"] > div > div[role="tablist"]{left:auto !important;right:112px;transform:none !important;}
      .block-container > div[data-testid="stVerticalBlock"] > div[data-testid="stTabs"] > div > div[role="tablist"] [role="tab"]{padding:8px 12px !important;}
    }
    div[data-testid="stTabPanel"] div[data-testid="stTabs"] [role="tablist"], div[role="tabpanel"] div[data-testid="stTabs"] > div > div[role="tablist"]{
        position:sticky;top:3.75rem;z-index:90;background:var(--page);padding-top:6px;}
    @media (min-width:641px){
      .block-container > div[data-testid="stVerticalBlock"] > div[data-testid="stTabs"] > div > div[role="tablist"]{
        order:2;flex:0 0 auto;width:auto;gap:6px;padding:5px;border-radius:18px;background:var(--card);border:1px solid var(--line);box-shadow:var(--card-sh);
        position:fixed;top:5px;left:50%;transform:translateX(-50%);z-index:1000001;box-sizing:border-box;height:50px;align-items:center;}
      .block-container > div[data-testid="stVerticalBlock"] > div[data-testid="stTabs"] > div > div[role="tablist"] [role="tab"]{
        border-radius:13px;padding:8px 18px;color:var(--mut);}
      .block-container > div[data-testid="stVerticalBlock"] > div[data-testid="stTabs"] > div > div[role="tablist"] [role="tab"][aria-selected="true"]{
        color:var(--accent-t);}
    }
    /* --- glossaire (page Aide) --- */
    .gl{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px;margin:6px 0 10px;}
    .gl div{padding:12px 14px;border-radius:var(--r-alert);background:var(--card);border:1px solid var(--line);box-shadow:var(--card-sh);font-size:.84rem;line-height:1.45;color:var(--mut);}
    .gl b{display:block;color:var(--ink);font-size:.9rem;margin-bottom:2px;}

    /* --- onglets : boutons arrondis --- */
    div[data-testid="stTabs"] .react-aria-SelectionIndicator, div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] [data-baseweb="tab-border"]{display:none !important;}
    div[data-testid="stTabs"] [role="tablist"]{gap:6px;box-shadow:none !important;border-bottom:none !important;}
    div[data-testid="stTabs"] [role="tab"]{border-radius:999px;padding:9px 18px;font-weight:700;border:1px solid transparent;transition:background .2s,color .2s;}
    div[data-testid="stTabs"] [role="tab"] p{font-size:.98rem;}
    div[data-testid="stTabs"] [role="tab"][aria-selected="true"]{background:var(--tab-bg);
        border-color:var(--tab-line);color:var(--accent-t);}

    /* =========================== TABLETTE =========================== */
    @media (min-width:641px) and (max-width:1100px){
      .block-container{padding-left:1.6rem !important;padding-right:1.6rem !important;}
      div[data-testid="stHorizontalBlock"]{flex-wrap:wrap;gap:.8rem !important;}
      div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]{
        flex:1 1 calc(50% - .8rem) !important;min-width:calc(50% - .8rem) !important;width:auto !important;}
      .kpis{grid-template-columns:repeat(3,minmax(0,1fr));}
      .hero-title{font-size:1.8rem;}
    }

    /* =========================== TÉLÉPHONE =========================== */
    @media (max-width:640px){
      .block-container{padding:76px .8rem 7rem .8rem !important;}
      .hero{padding:20px 20px;border-radius:var(--r-hero-sm);margin-bottom:10px;}
      .hero-title{font-size:1.5rem;}
      .mark{width:42px;height:42px;border-radius:calc(var(--r-mark) - 2px);}
      .hero-sub{font-size:.9rem;}
      .pills{flex-wrap:nowrap;overflow-x:auto;margin-top:14px;padding-bottom:2px;scrollbar-width:none;}
      .pills span{white-space:nowrap;}
      .sec{margin:22px 0 10px;}
      .hcard{padding:20px 18px;gap:18px;justify-content:center;}
      .hc-body,.hc-side{flex-basis:100%;max-width:none;}
      .hc-big{font-size:2.5rem;}
      .kpis{grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;}
      .kpi{padding:13px 14px;}
      .kv{font-size:1.32rem;}
      .kpi:last-child:nth-child(odd){grid-column:1 / -1;}
      .hb-row{grid-template-columns:minmax(70px,38%) 1fr auto;gap:8px;}
      .hb.dv .hb-row{grid-template-columns:minmax(64px,32%) 1fr 74px;}
      .hb .pc{display:none;}
      div[data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"] div[data-testid="stHorizontalBlock"]) > div[data-testid="stColumn"]{
        flex:1 1 100% !important;min-width:100% !important;}
      .hb.dv .hb-v{font-size:.76rem;}
      div[data-testid="stHorizontalBlock"]:has(div[data-testid="stNumberInput"]){flex-wrap:wrap;gap:.6rem !important;}
      div[data-testid="stHorizontalBlock"]:has(div[data-testid="stNumberInput"]) > div[data-testid="stColumn"]{
        flex:1 1 calc(50% - .6rem) !important;min-width:calc(50% - .6rem) !important;width:auto !important;}
      /* navigation du bas flottante, comme une application mobile (onglets principaux seulement) */
      .block-container > div[data-testid="stVerticalBlock"] > div[data-testid="stTabs"] > div > div[role="tablist"]{
        position:fixed;left:10px;right:10px;bottom:calc(10px + env(safe-area-inset-bottom,0px));z-index:999;display:flex;gap:2px;overflow:visible;
        padding:6px;border-radius:24px;border:1px solid var(--line);
        /* "Canvas" prend la couleur de fond du thème affiché (clair ou sombre) */
        background:var(--page);background:color-mix(in srgb,var(--page) 90%,transparent);backdrop-filter:blur(16px);
        box-shadow:0 10px 30px rgba(0,0,0,.3);}
      .block-container > div[data-testid="stVerticalBlock"] > div[data-testid="stTabs"] > div > div[role="tablist"] [role="tab"]{
        flex:1 1 0;min-width:0;height:auto;padding:11px 2px;justify-content:center;border-radius:18px;}
      .block-container > div[data-testid="stVerticalBlock"] > div[data-testid="stTabs"] > div > div[role="tablist"] [role="tab"] p{
        font-size:.78rem;white-space:nowrap;}
      .block-container > div[data-testid="stVerticalBlock"] > div[data-testid="stTabs"] > div > div[role="tablist"] button{display:none;}
    }
    @media (prefers-reduced-motion:reduce){
      .hero::before,.hero::after,.mark,.pills span,.hcard,.kpi,.cnt,.rg,.gv,.m-track i,.alert,.card,.hb-row,.hb-t i,.lot,.bar>i,.lb i,.bsch *,.wrow,.watch{animation:none !important;}
    }
    </style>""".replace("__THEME__", _theme_css()),
    unsafe_allow_html=True,
)

DISTS = ["Normale", "Log-Normale", "Bêta-PERT", "Uniforme"]
FONT = TH["fontfam"]


# ---------------------------------------------------------------------
# UTILITAIRES
# ---------------------------------------------------------------------
def _stretch(fn, *args, **kwargs):
    """Compatibilité entre versions de Streamlit (width='stretch' / use_container_width)."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


def esc(x):
    return html.escape(str(x))


_ICONS = {
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "x": '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "up": '<path d="M7 7h10v10"/><path d="M7 17 17 7"/>',
    "down": '<path d="m7 7 10 10"/><path d="M17 7v10H7"/>',
    "minus": '<path d="M5 12h14"/>',
    "left": '<path d="m15 18-6-6 6-6"/>',
    "right": '<path d="m9 18 6-6-6-6"/>',
    "pie": '<path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/>',
    "hardhat": '<path d="M10 10V5a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1v5"/><path d="M14 6a6 6 0 0 1 6 6v3"/>'
               '<path d="M4 15v-3a6 6 0 0 1 6-6"/><rect x="2" y="15" width="20" height="4" rx="1"/>',
    "wallet": '<path d="M19 7V4a1 1 0 0 0-1-1H5a2 2 0 0 0 0 4h15a1 1 0 0 1 1 1v4h-3a2 2 0 0 0 0 4h3a1 1 0 0 0 1-1v-2a1 1 0 0 0-1-1"/>'
              '<path d="M3 5v14a2 2 0 0 0 2 2h15a1 1 0 0 0 1-1v-4"/>',
    "dice": '<rect width="12" height="12" x="2" y="10" rx="2" ry="2"/><path d="m17.92 14 3.5-3.5a2.24 2.24 0 0 0 0-3l-5-4.92a2.24 2.24 0 0 0-3 0L10 6"/>'
            '<path d="M6 18h.01"/><path d="M10 14h.01"/><path d="M15 6h.01"/><path d="M18 9h.01"/>',
    "ruler": '<path d="M21.3 15.3a2.4 2.4 0 0 1 0 3.4l-2.6 2.6a2.4 2.4 0 0 1-3.4 0L2.7 8.7a2.41 2.41 0 0 1 0-3.4l2.6-2.6a2.41 2.41 0 0 1 3.4 0Z"/>'
             '<path d="m14.5 12.5 2-2"/><path d="m11.5 9.5 2-2"/><path d="m8.5 6.5 2-2"/><path d="m17.5 15.5 2-2"/>',
    "layers": '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/>'
              '<path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>',
    "weight": '<circle cx="12" cy="5" r="3"/><path d="M6.5 8a2 2 0 0 0-1.905 1.46L2.1 18.5A2 2 0 0 0 4 21h16a2 2 0 0 0 1.925-2.54L19.4 9.5A2 2 0 0 0 17.5 8Z"/>',
    "rebar": '<path d="M3 18h18"/><circle cx="7" cy="13" r="1.6"/><circle cx="12" cy="13" r="1.6"/><circle cx="17" cy="13" r="1.6"/><path d="M4 6h16v12"/>',
    "shield": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>'
              '<path d="m9 12 2 2 4-4"/>',
}


def ico(name, size=16):
    """Petite icône SVG (même couleur que le texte autour)."""
    return (f'<svg class="ico" viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="currentColor" '
            f'stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{_ICONS[name]}</svg>')


def num(x, dec=1):
    """Nombre au format français : 1 234,5"""
    return f"{x:,.{dec}f}".replace(",", " ").replace(".", ",")


def pc(x, dec=1):
    """Pourcentage à la française : 0,771 -> « 77,1 % »."""
    return f"{num(x * 100, dec)} %"


def fcfa(x):
    return f"{x:,.0f}".replace(",", " ") + " FCFA"


def fcfa_m(x):
    return f"{num(x / 1e6)} M FCFA"


def cnt(v, dec=0, sign=""):
    """Nombre qui se décompte de 0 à sa valeur à l'affichage (animation CSS).
    La vraie valeur reste lue par les lecteurs d'écran. Le signe se passe à part (v est pris en valeur absolue)."""
    q = int(round(abs(float(v)) * 10 ** dec))
    ip, dp = divmod(q, 10 ** dec)
    return (f'<span class="cv" aria-label="{sign}{num(abs(float(v)), dec)}">{sign}'
            f'<span class="cnt cnt{dec}" aria-hidden="true" style="--ip:{ip};--dp:{dp}"></span></span>')


def signed(v):
    return "−" if v < 0 else "+"


def banner(level, title, msg=""):
    icon = {"ok": "check", "warn": "alert", "crit": "x"}[level]
    st.markdown(f'<div class="alert a-{level}"><span class="a-ic" aria-hidden="true">{ico(icon, 18)}</span>'
                f'<div><b>{title}</b><br>{msg}</div></div>', unsafe_allow_html=True)


def hero():
    """Barre du haut : la marque (les onglets et les réglages viennent se ranger sur la même ligne, voir le CSS)."""
    st.markdown(f'<div class="topbar"><span class="mark">{ico("hardhat", 19)}</span>Chantier Pro</div>', unsafe_allow_html=True)


def section(title, caption=""):
    """Titre de section avec une courte explication en dessous."""
    cap = f"<p>{caption}</p>" if caption else ""
    st.markdown(f'<div class="sec"><div><h3>{title}</h3>{cap}</div></div>', unsafe_allow_html=True)


GLOSS = {          # explications en langage simple, affichées dans une bulle (icône « i »)
    "Budget (BAC)": "Le budget total prévu pour le chantier, fixé avant de commencer.",
    "Dépenses engagées": "L'argent déjà dépensé ou promis jusqu'ici.",
    "Avancement pondéré": "La part du chantier réellement faite. Un gros lot compte plus qu'un petit.",
    "Reste à dépenser": "Le budget moins les dépenses engagées. Négatif : le budget est déjà dépassé.",
    "Prévision finale (EAC)": "Le coût total attendu à la fin du chantier, si l'on continue au même rythme de dépense.",
    "CPI (EV/AC)": "Rendement des dépenses : valeur du travail fait divisée par l'argent dépensé. Sous 1, on dépense plus que ce que le travail vaut.",
    "Moyenne E[X]": "Le résultat moyen des chantiers simulés.",
    "Écart-type σ": "Mesure de l'incertitude : plus il est grand, plus les résultats possibles sont dispersés.",
    "Médiane P50": "Une chance sur deux de faire mieux, une chance sur deux de faire pire.",
    "P95 (VaR)": "Dans 95 % des cas, le résultat reste en dessous de cette valeur : le scénario très défavorable mais plausible.",
    "Probabilité de dépasser le seuil": "La part des chantiers simulés dont le coût (ou le délai) dépasse le seuil d'alerte.",
    "Moyenne": "La valeur moyenne de la loi.",
    "Écart-type": "Mesure de l'incertitude : plus il est grand, plus les valeurs sont dispersées.",
    "Asymétrie": "Positive : les mauvaises surprises (valeurs hautes) sont plus fréquentes que les bonnes.",
    "P(X ≤ x₀)": "La probabilité que la valeur reste sous x₀.",
    "P(X > x₀)": "La probabilité de dépasser x₀.",
    "Verdict de sécurité": "Ce que le calcul conclut : la poutre résiste-t-elle avec une marge suffisante ?",
    "MEd (ELU)": "Le moment de flexion maximal que la poutre doit supporter, avec les charges majorées de sécurité.",
    "MRd": "Le moment de flexion maximal que la poutre peut supporter.",
    "As requis / retenu": "La section d'acier tendu demandée par le calcul, et celle réellement mise en place.",
    "Flèche": "L'affaissement de la poutre en son milieu, à long terme.",
    "β de Cornell": "Indice de fiabilité : plus il est élevé, plus la rupture est improbable. Cible : 3,8 (norme EN 1990).",
    "Pf (Cornell / MC)": "Probabilité de rupture en flexion, estimée par la formule de Cornell et par la simulation Monte-Carlo.",
}


def _tip(texte):
    """Petite bulle d'aide « i » (survol ou toucher)."""
    if not texte:
        return ""
    return f'<span class="tip" tabindex="0" role="note" aria-label="{esc(texte)}" data-tip="{esc(texte)}"></span>'


def _lab(label):
    return f'<div class="hc-lab">{label}{_tip(GLOSS.get(label))}</div>'


def kpi_grid(items, cols=None):
    """Rangée d'indicateurs. items : (libellé, valeur_html, détail, ton, icône) ; ton : ok / warn / bad / mut."""
    cards = []
    for i, (label, value, sub, tone, icon) in enumerate(items):
        pill = f'<div class="kd {tone}">{ico(icon, 13) if icon else ""}{sub}</div>' if sub else ""
        cards.append(f'<div class="kpi" style="--i:{i}"><div class="kl">{label}{_tip(GLOSS.get(label))}</div><div class="kv">{value}</div>{pill}</div>')
    st.markdown(f'<div class="kpis" style="--cols:{cols or len(items)}">{"".join(cards)}</div>', unsafe_allow_html=True)


def unit(u):
    return f"<small>{u}</small>"


# ---- anneaux, jauges, barres (HTML / SVG animés) ----------------------
def rings(outer, inner, centre_html, centre_label):
    """Deux anneaux concentriques : l'extérieur = budget consommé, l'intérieur = avancement."""
    def arc(r, v, color, w, delay):
        v = max(0.0, min(1.0, v)) * 100
        track = f'<circle class="rt" cx="88" cy="88" r="{r}" stroke-width="{w}"/>'
        if v < 0.4:
            return track
        return (track + f'<circle class="rg" cx="88" cy="88" r="{r}" stroke-width="{w}" pathLength="100" '
                f'transform="rotate(-90 88 88)" style="stroke:{color};--v:{v:.2f};animation-delay:{delay}ms"/>')
    col_out = LEVEL_C["crit"] if outer > 1 else "var(--r1)"
    return (f'<div class="ringw"><svg viewBox="0 0 176 176" role="img" aria-label="Budget consommé et avancement">'
            f'{arc(74, outer, col_out, 13, 200)}{arc(55, inner, "var(--r2)", 11, 450)}</svg>'
            f'<div class="ringc"><b>{centre_html}</b><span>{centre_label}</span></div></div>')


ARC = "M 20 110 A 100 100 0 0 1 220 110"


def _pt(f, r):
    th = math.pi * (1 - f)
    return 120 + r * math.cos(th), 110 - r * math.sin(th)


def gauge(frac, color, zones, mark=None, big="", sub="", label=""):
    """Demi-cercle de jauge. zones : (début, fin, couleur) en fractions de 0 à 1 ; mark : position d'un repère blanc."""
    frac = max(0.0, min(1.0, frac))
    z = "".join(f'<path class="gz" d="{ARC}" pathLength="100" stroke="{c}" '
                f'style="stroke-dasharray:{(e - s) * 100:.2f} 100;stroke-dashoffset:{-s * 100:.2f}"/>'
                for s, e, c in zones if e > s)
    tick = ""
    if mark is not None:
        (x1, y1), (x2, y2) = _pt(max(0, min(1, mark)), 80), _pt(max(0, min(1, mark)), 120)
        tick = f'<line class="gm" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>'
    arc = f'<path class="gv" d="{ARC}" pathLength="100" stroke="{color}" style="--v:{frac * 100:.2f}"/>' if frac > 0.004 else ""
    return (f'<div class="gw"><svg viewBox="0 0 240 130" role="img" aria-label="{esc(label)}">{z}{arc}{tick}</svg>'
            f'<div class="gtxt"><b>{big}</b><span>{sub}</span></div></div>')


def meter(label, pct, value_html, sub):
    over = pct > 100
    color = LEVEL_C["crit"] if over else (LEVEL_C["warn"] if pct > 90 else "var(--r2)")
    return (f'<div class="meter"><div class="m-head"><span>{label}</span><b>{value_html}</b></div>'
            f'<div class="m-track" role="img" aria-label="{pct:.0f} pour cent"><i style="width:{min(pct, 100):.1f}%;background:{color}"></i></div>'
            f'<div class="m-sub">{sub}</div></div>')


def budget_track(bac, ac, eac):
    """Barre : argent déjà dépensé, reste à dépenser prévu, et repère du budget."""
    top = max(bac, eac, ac) * 1.04 or 1.0
    sp, fc = ac / top * 100, max(0.0, eac - ac) / top * 100
    return ('<div class="hc-side"><div class="hc-lab">Où va-t-on par rapport au budget ?</div>'
            f'<div class="tk" role="img" aria-label="Dépensé {fcfa_m(ac)}, prévision {fcfa_m(eac)}, budget {fcfa_m(bac)}">'
            f'<i class="sp" style="width:{sp:.2f}%"></i><i class="fc" style="left:{sp:.2f}%;width:{fc:.2f}%"></i>'
            f'<div class="bm" style="left:{bac / top * 100:.2f}%"><span>Budget {num(bac / 1e6)} M</span></div></div>'
            '<div class="tk-leg">'
            f'<span><i style="background:var(--spent)"></i>Dépensé {num(ac / 1e6)} M</span>'
            f'<span><i style="background:var(--fc)"></i>Reste prévu {num(max(0.0, eac - ac) / 1e6)} M</span></div></div>')


def risk_track(vals, seuil, u):
    """Étendue des résultats simulés (P1 à P99), avec la médiane, le P95 et le seuil d'alerte."""
    p1, p50, p95, p99 = vals
    lo, hi = min(p1, seuil), max(p99, seuil)
    pad = (hi - lo) * 0.07 or 1.0
    lo, hi = lo - pad, hi + pad

    def pos(v):
        return (v - lo) / (hi - lo) * 100
    ticks = "".join(f'<div class="pt" style="left:{pos(v):.2f}%"><span>{n}</span></div>' for n, v in (("P50", p50), ("P95", p95)))
    return ('<div class="hc-side"><div class="hc-lab">Où tombe le seuil parmi les résultats simulés ?</div>'
            f'<div class="tk tt" role="img" aria-label="Seuil {num(seuil)} {u}, médiane {num(p50)}, P95 {num(p95)}">'
            f'<i class="sp" style="left:{pos(p1):.2f}%;width:{pos(p99) - pos(p1):.2f}%"></i>{ticks}'
            f'<div class="bm" style="left:{pos(seuil):.2f}%"><span>Seuil {num(seuil)}</span></div></div>'
            f'<div class="tk-leg"><span><i style="background:var(--spent)"></i>98 % des chantiers simulés</span>'
            f'<span>P50 {num(p50)} · P95 {num(p95)} {u}</span></div></div>')


def chip(level, text):
    icon = {"ok": "check", "warn": "alert", "crit": "x"}[level]
    cls = {"ok": "ok", "warn": "warn", "crit": "bad"}[level]
    return f'<span class="chip {cls}">{ico(icon, 15)}{text}</span>'


def hbars(rows, diverging=False, color="var(--c-blue)"):
    """Barres horizontales animées. rows : (libellé, valeur, texte affiché, ton)."""
    mx = max((abs(v) for _, v, _, _ in rows), default=0) or 1.0
    out = []
    for i, (lab, v, txt, tone) in enumerate(rows):
        w = abs(v) / mx * (50 if diverging else 100)
        if diverging:
            col = "var(--bad)" if v > 0 else "var(--good)"
            side, origin = ("left:50%", "left") if v >= 0 else ("right:50%", "right")
            bar = f'<i style="{side};width:{w:.1f}%;background:{col};transform-origin:{origin}"></i>'
        else:
            bar = f'<i style="left:0;width:{w:.1f}%;background:{color}"></i>'
        out.append(f'<div class="hb-row" style="--i:{i}"><span class="hb-l" title="{esc(lab)}">{esc(lab)}</span>'
                   f'<div class="hb-t">{bar}</div><b class="hb-v {tone}">{txt}</b></div>')
    return f'<div class="hb{" dv" if diverging else ""}">{"".join(out)}</div>'


def lot_cards(ed):
    """Une carte par lot : prévision finale, barre budget / dépensé / prévision, avancement."""
    cartes = []
    for i, r in enumerate(ed.itertuples()):
        cls = "ok" if r.EcartPct <= 0 else ("warn" if r.EcartPct <= 5 else "bad")
        icon, label = {"ok": ("check", "Dans le budget"), "warn": ("alert", "Vigilance"), "bad": ("x", "Dérive")}[cls]
        used, prog = max(0.0, float(r.Consomme)), max(0.0, min(float(r.Avancement), 100.0))
        top = max(r.Budget, r.EAC, r.Depense) * 1.04 or 1.0
        sp, fc, bm = r.Depense / top * 100, max(0.0, r.EAC - r.Depense) / top * 100, r.Budget / top * 100
        arrow = "up" if r.Ecart > 0 else ("down" if r.Ecart < 0 else "minus")
        signe = "" if abs(r.EcartPct) < 0.05 else signed(r.EcartPct)
        cartes.append(
            f'<div class="lot" id="lot-{r.Index}" style="--i:{i}"><div class="lot-head"><span class="lot-name" title="{esc(r.Lot)}">{esc(r.Lot)}</span>'
            f'<span class="lot-chip {cls}">{ico(icon, 12)} {label}</span></div>'
            f'<div class="lot-main"><div><div class="lot-k">Prévision finale</div>'
            f'<div class="lot-v">{num(r.EAC / 1e6)}<small>M FCFA</small></div></div>'
            f'<span class="lot-d {cls}">{ico(arrow, 12)} {signe}{num(abs(r.EcartPct))} %</span></div>'
            f'<div class="lb" role="img" aria-label="Dépensé {fcfa_m(r.Depense)}, prévision {fcfa_m(r.EAC)}, budget {fcfa_m(r.Budget)}">'
            f'<i class="sp" style="width:{sp:.2f}%"></i><i class="fc" style="left:{sp:.2f}%;width:{fc:.2f}%"></i>'
            f'<b class="bm" style="left:{bm:.2f}%"></b></div>'
            f'<div class="lb-leg"><span><i style="background:var(--c-blue)"></i>Dépensé {num(r.Depense / 1e6)} M</span>'
            f'<span><i class="bmk"></i>Budget {num(r.Budget / 1e6)} M</span></div>'
            f'<div class="lot-line"><span>Avancement</span><b>{num(prog, 0)} %</b></div>'
            f'<div class="bar"><i style="width:{prog:.0f}%;background:var(--c-aqua)"></i></div>'
            f'<div class="lot-note">{num(used, 0)} % du budget consommé pour {num(prog, 0)} % d\'avancement</div></div>'
        )
    return '<div class="lots">' + "".join(cartes) + "</div>"


# ---- graphiques animés (Plotly dans une petite page, chargé quand il devient visible) ----
def _rgba(hexa, a):
    h = hexa.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def _pal(dark=None):
    """Couleurs des graphiques : celles du thème choisi, dans le mode (clair / sombre) affiché par Streamlit."""
    if dark is None:
        try:
            dark = st.context.theme.type != "light"
        except Exception:
            dark = False
    return dict(dark=dark, **TH["chart"]["dark" if dark else "light"])


_PAGE = """<!doctype html><html><head><meta charset="utf-8">__FONTLINK__
<style>
html,body{margin:0;padding:0;background:transparent;overflow:hidden;color:__TEXT__;font:13px/1.4 __FONT__}
#c{width:100%;height:__H__px;border-radius:14px}
.sk{background:linear-gradient(90deg,rgba(128,128,128,.07),rgba(128,128,128,.17),rgba(128,128,128,.07));background-size:220% 100%;animation:sh 1.3s linear infinite}
@keyframes sh{to{background-position:-220% 0}}
.err{padding:20px;opacity:.7}
</style></head><body><div id="c" class="sk"></div><script>
(function(){
var FIGS=__FIGS__, KINDS=__KINDS__, SRC="__SRC__", PYMODE="__PYMODE__", el=document.getElementById('c'), booted=false, ready=false;
// Mode clair / sombre de Streamlit lu dans la page parente (sinon : celui connu au calcul).
function modeNow(){
  try{var cs=getComputedStyle(parent.document.querySelector('.stApp')).colorScheme; if(cs) return cs.indexOf('dark')>=0?'dark':'light';}catch(e){}
  return PYMODE;
}
var mode=modeNow(), FIG=FIGS[mode];
var ORIG=JSON.parse(JSON.stringify(FIG.data));   // Plotly modifie FIG.data en place : on garde les vraies valeurs à part
var reduce=window.matchMedia&&matchMedia('(prefers-reduced-motion: reduce)').matches;
var ease=function(t){return 1-Math.pow(1-t,3)}, inout=function(t){return t<.5?2*t*t:1-Math.pow(-2*t+2,2)/2};
function partial(k,t){
  var tr=ORIG[k.i], n, key, src;
  if(k.kind==='bar'||k.kind==='barh'){
    key=k.kind==='bar'?'y':'x'; src=tr[key]; n=src.length; var st=.55, o={};
    o[key]=src.map(function(v,j){return v==null?v:v*ease(Math.min(1,Math.max(0,(t-st*j/Math.max(n-1,1))/(1-st))))});
    return o;
  }
  if(k.kind==='line'){ n=tr.y.length; var m=Math.max(1,Math.round(n*inout(t))); return {x:tr.x.slice(0,m),y:tr.y.slice(0,m)}; }
  var wk=k.kind==='wfh'?'x':'y', wv=tr[wk], sw=.7, o2={}; n=wv.length;   // cascade : les écarts se construisent l'un après l'autre
  o2[wk]=wv.map(function(v,j){return tr.measure[j]!=='relative'?v:v*ease(Math.min(1,Math.max(0,(t-sw*j/Math.max(n-1,1))/(1-sw))))});
  return o2;
}
function apply(t){
  var groups={};
  KINDS.forEach(function(k){
    var p=partial(k,t), g=groups[k.kind]=groups[k.kind]||{idx:[],up:{}};
    g.idx.push(k.i);
    for(var key in p){(g.up[key]=g.up[key]||[]).push(p[key]);}
  });
  for(var kind in groups){Plotly.restyle(el,groups[kind].up,groups[kind].idx);}
}
function play(){
  if(reduce){apply(1);return;}
  var D=1250,t0=performance.now();
  (function f(now){var t=Math.min(1,(now-t0)/D);apply(t);if(t<1)requestAnimationFrame(f);})(t0);
}
function boot(){
  if(booted)return; booted=true;
  var s=document.createElement('script'); s.src=SRC;
  s.onerror=function(){el.className='err';el.textContent='Graphique animé indisponible : connexion internet requise.';};
  s.onload=function(){
    Plotly.newPlot(el,FIG.data,FIG.layout,{displayModeBar:false,responsive:true}).then(function(){
      var fl=el._fullLayout,u={};
      ['xaxis','yaxis'].forEach(function(a){if(fl[a]&&fl[a].range){u[a+'.range']=fl[a].range.slice();u[a+'.autorange']=false;}});
      return Plotly.relayout(el,u);
    }).then(function(){el.classList.remove('sk');ready=true;apply(0);requestAnimationFrame(play);});
  };
  document.head.appendChild(s);
}
function retheme(m){
  var lay=JSON.parse(JSON.stringify(FIGS[m].layout)), cl=el._fullLayout;
  ['xaxis','yaxis'].forEach(function(a){lay[a]=lay[a]||{};if(cl[a]&&cl[a].range){lay[a].range=cl[a].range.slice();lay[a].autorange=false;}});
  Plotly.react(el,JSON.parse(JSON.stringify(FIGS[m].data)),lay,{displayModeBar:false,responsive:true});
}
setInterval(function(){var m=modeNow();if(ready&&m!==mode){mode=m;retheme(m);}},600);
if('IntersectionObserver' in window){
  var io=new IntersectionObserver(function(es){if(es.some(function(e){return e.isIntersecting})){io.disconnect();boot();}},{threshold:.15});
  io.observe(el);
}else{boot();}
})();
</script></body></html>"""


def _plain(fig):
    """Remplace les tableaux numpy par des listes (le script d'animation lit les valeurs une à une)."""
    for tr in fig.data:
        for a in ("x", "y", "measure"):
            try:
                v = getattr(tr, a)
            except AttributeError:
                continue
            if v is not None:
                setattr(tr, a, np.asarray(v).tolist())
    return fig


def anim_plot(fig, height=360, hover="x unified"):
    """Affiche un graphique Plotly animé (barres qui montent, courbes qui se tracent, cascade qui se construit).
    L'animation se lance quand le graphique devient visible (ouverture de l'onglet, défilement)."""
    P = _pal()
    n_series = sum(1 for tr in fig.data if tr.showlegend is not False and getattr(tr, "name", None))
    fig.update_layout(
        template="none", height=height, autosize=True, dragmode=False, separators=", ",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=[P["c1"], P["c2"], P["c3"], P["c4"]],
        font=dict(family=FONT, size=12, color=P["text"]), hovermode=hover,
        margin=dict(l=8, r=14, t=44 if n_series > 1 else 14, b=8),
        hoverlabel=dict(bgcolor=P["tip"], bordercolor=P["axis"], font=dict(color=P["text"], size=12)),
        legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", font=dict(color=P["muted"])),
        xaxis=dict(gridcolor=P["grid"], zeroline=False, linecolor=P["axis"], automargin=True, tickfont=dict(color=P["muted"])),
        yaxis=dict(gridcolor=P["grid"], zeroline=False, automargin=True, tickfont=dict(color=P["muted"])),
    )
    _plain(fig)
    kinds = []
    for i, tr in enumerate(fig.data):
        if tr.type == "bar":
            kinds.append({"i": i, "kind": "barh" if tr.orientation == "h" else "bar"})
        elif tr.type == "scatter":
            kinds.append({"i": i, "kind": "line"})
        elif tr.type == "waterfall":
            kinds.append({"i": i, "kind": "wfh" if tr.orientation == "h" else "wf"})
    # La même figure dans l'autre mode : on échange les couleurs du thème une à une (le texte JSON ne change pas autrement).
    Q = _pal(not P["dark"])
    cles = ("text", "muted", "grid", "axis", "tip", "gray", "c1", "c2", "c3", "c4")
    echange = {f'"{P[k]}"': f'"{Q[k]}"' for k in cles}
    echange.update({f'"{_rgba(P["c1"], .25)}"': f'"{_rgba(Q["c1"], .25)}"'})
    cur = pio.to_json(fig)
    autre = re.sub("|".join(re.escape(k) for k in echange), lambda m: echange[m.group(0)], cur)
    figs = '{"%s":%s,"%s":%s}' % ("dark" if P["dark"] else "light", cur, "light" if P["dark"] else "dark", autre)
    page = (_PAGE.replace("__FIGS__", figs.replace("</", "<\\/")).replace("__PYMODE__", "dark" if P["dark"] else "light")
            .replace("__KINDS__", str(kinds).replace("'", '"'))
            .replace("__SRC__", f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js")
            .replace("__TEXT__", P["text"]).replace("__FONTLINK__", f'<link rel="stylesheet" href="{TH["font"]}">' if TH["font"] else "")
            .replace("__FONT__", FONT).replace("__H__", str(height)))
    if hasattr(st, "iframe"):
        st.iframe(page, height=height + 2)
    else:                                   # Streamlit plus ancien
        components.html(page, height=height + 2)


_BRIDGE = """<!--chantier-bridge--><script>
(function(){
  var P = window.parent, D;
  try { D = P.document; } catch (e) { return; }
  var CUR = "__CUR__", ITEMS = __ITEMS__, FONT = "__FONT__", GID = "chantierThemes";
  if (P.__chantierObs) { try { P.__chantierObs.disconnect(); } catch (e) {} }
  var old = D.querySelector('[data-testid="' + GID + '"]'); if (old) old.remove();
  if (FONT && !D.getElementById("chantier-font")) {
    var l = D.createElement("link"); l.id = "chantier-font"; l.rel = "stylesheet"; l.href = FONT; D.head.appendChild(l);
  }
  function texts(el) { var w = D.createTreeWalker(el, 4), a = [], n; while ((n = w.nextNode())) a.push(n); return a; }
  function paint(b, sel) {     // ne touche à rien si c'est déjà bon (sinon le surveillant tournerait en boucle)
    var a = sel ? "true" : "false", c = sel ? b._on : b._off;
    if (b.getAttribute("aria-checked") !== a) b.setAttribute("aria-checked", a);
    if (b.className !== c) b.className = c;
  }
  function choose(it) {
    var m = D.querySelector('[data-testid="stMainMenuItem-theme-' + it.mode + '"]');
    if (m && m.getAttribute("aria-checked") !== "true") m.click();          // clair / sombre natif de Streamlit
    if (it.key === CUR) return;
    var ctl = D.querySelector(".st-key-theme");
    var t = ctl && [].slice.call(ctl.querySelectorAll("button")).find(function (x) { return x.textContent.trim() === it.name; });
    if (t) { t.click(); return; }                                           // relance le script avec le nouveau thème
    var u = new URL(P.location.href); u.searchParams.set("theme", it.key); P.location.href = u.toString();
  }
  function build() {
    var sw = D.querySelector('[data-testid="stThemeSwitcher"]'); if (!sw) return;
    var cur = D.querySelector('[data-testid="' + GID + '"]');
    var btns = [].slice.call(sw.querySelectorAll("button"));
    var off = btns.find(function (b) { return b.getAttribute("aria-checked") === "false"; });
    var on = btns.find(function (b) { return b.getAttribute("aria-checked") === "true"; });
    if (!off) return;
    if (cur && cur.nextElementSibling === sw) {      // déjà là : on recopie les styles actuels (ils changent avec le clair / sombre)
      [].slice.call(cur.querySelectorAll("button")).forEach(function (b) {
        b._off = off.className; b._on = on ? on.className : off.className; paint(b, b.getAttribute("aria-checked") === "true");
      });
      return;
    }
    if (cur) cur.remove();
    var grp = sw.cloneNode(false); grp.setAttribute("data-testid", GID); grp.setAttribute("aria-label", "Thèmes de Chantier Pro"); grp.style.marginBottom = "4px";
    ITEMS.forEach(function (it) {
      var b = off.cloneNode(true); b.setAttribute("data-testid", "stMainMenuItem-theme-" + it.key); b.removeAttribute("fdprocessedid");
      var icon = b.querySelector('[data-testid="stIconMaterial"]');
      texts(b).forEach(function (n) { if (icon && icon.contains(n)) n.textContent = it.icon; else if (n.textContent.trim()) n.textContent = it.label; });
      b.title = it.name; b._off = off.className; b._on = on ? on.className : off.className;
      paint(b, it.key === CUR);
      b.addEventListener("click", function (ev) { ev.preventDefault(); ev.stopPropagation(); choose(it); }, true);
      grp.appendChild(b);
    });
    sw.parentNode.insertBefore(grp, sw);
  }
  var raf = 0, obs = new P.MutationObserver(function () { if (raf) return; raf = P.requestAnimationFrame(function () { raf = 0; build(); }); });
  obs.observe(D.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["class"] }); P.__chantierObs = obs; build();
})();
</script>"""


def _theme_bridge():
    """Ajoute « Chantier / Insights » dans le menu ⋮ de Streamlit, juste au-dessus de System / Light / Dark.
    Streamlit n'a pas de moyen officiel pour cela : ce petit script (invisible) copie les boutons du menu."""
    items = [dict(key=k, name=t["name"], label=t["label"], icon=t["icon"], mode=t["mode"]) for k, t in THEMES.items()]
    page = (_BRIDGE.replace("__CUR__", THEME_KEY).replace("__ITEMS__", json.dumps(items, ensure_ascii=False))
            .replace("__FONT__", TH["font"]))
    if hasattr(st, "iframe"):
        st.iframe(page, height=1)
    else:                                       # Streamlit plus ancien
        components.html(page, height=1)


def chart(fig, title, sub="", height=360, hover="x unified"):
    """Carte contenant un titre, une phrase de lecture et le graphique animé."""
    with st.container(border=True):
        head = f'<span class="cc-mark"></span><div class="chart-title">{esc(title)}</div>'
        if sub:
            head += f'<div class="chart-sub">{esc(sub)}</div>'
        st.markdown(head, unsafe_allow_html=True)
        anim_plot(fig, height, hover)


# ---------------------------------------------------------------------
# DONNÉES DE TEST (budget total = 90 000 000 FCFA)
# ---------------------------------------------------------------------
DEMO = pd.DataFrame({
    "Lot": ["Terrassement", "Fondations", "Gros œuvre", "Charpente & toiture",
            "Électricité", "Plomberie", "Finitions", "Aménagements ext."],
    "Budget": [8_000_000, 14_000_000, 32_000_000, 9_000_000, 7_000_000, 6_000_000, 10_000_000, 4_000_000],
    "Depense": [8_400_000, 14_800_000, 21_500_000, 3_200_000, 1_800_000, 1_200_000, 0, 0],
    "Avancement": [100, 100, 62, 35, 25, 20, 0, 0],
    "Loi": ["Normale", "Log-Normale", "Bêta-PERT", "Bêta-PERT", "Log-Normale", "Uniforme", "Bêta-PERT", "Uniforme"],
    "Copt": [8_000_000, 13_600_000, 30_000_000, 8_200_000, 6_000_000, 5_000_000, 9_000_000, 3_600_000],
    "Cprob": [8_200_000, 14_000_000, 32_000_000, 9_000_000, 6_600_000, 5_600_000, 9_800_000, 4_000_000],
    "Cpess": [8_600_000, 15_000_000, 37_000_000, 11_000_000, 8_200_000, 7_000_000, 12_500_000, 5_200_000],
    "Dopt": [20, 35, 100, 25, 30, 25, 35, 15],
    "Dprob": [25, 40, 120, 30, 35, 30, 40, 20],
    "Dpess": [35, 55, 160, 45, 50, 45, 60, 30],
})
NUM = {"Budget", "Depense", "Avancement", "Copt", "Cprob", "Cpess", "Dopt", "Dprob", "Dpess"}


# ---------------------------------------------------------------------
# LOIS DE PROBABILITÉ  (a = optimiste, m = probable, b = pessimiste)
# ---------------------------------------------------------------------
def make_dist(name, a, m, b):
    """Retourne une loi scipy 'gelée'.
    Normale     : N(m, σ=(b-a)/6)                    (±3σ ≈ [a, b])
    Log-Normale : moyenne (a+4m+b)/6, σ=(b-a)/6, paramètres par la méthode des moments
    Bêta-PERT   : α=1+4(m-a)/(b-a), β=1+4(b-m)/(b-a) sur [a, b]
    Uniforme    : U(a, b)"""
    w = max(b - a, 1e-9)
    if name == "Normale":
        return stats.norm(m, w / 6)
    if name == "Log-Normale":
        mean, sd = (a + 4 * m + b) / 6, w / 6
        k = 1 + (sd / mean) ** 2
        return stats.lognorm(s=math.sqrt(math.log(k)), scale=mean / math.sqrt(k))
    if name == "Bêta-PERT":
        return stats.beta(1 + 4 * (m - a) / w, 1 + 4 * (b - m) / w, loc=a, scale=w)
    return stats.uniform(a, w)


@st.cache_data(show_spinner=False)
def simulate(params, n, seed):
    """Tirages Monte-Carlo indépendants : matrice (n, nb_lots)."""
    rng = np.random.default_rng(seed)
    return np.column_stack([make_dist(*p).rvs(size=n, random_state=rng) for p in params])


# ---------------------------------------------------------------------
# 1. TABLEAU DE BORD FINANCIER
# ---------------------------------------------------------------------
def watch_card(ed, cpi):
    """Les points d'attention du budget, en phrases. Chaque ligne renvoie vers la carte du lot."""
    pos = ed[ed["Ecart"] > 0].sort_values("Ecart", ascending=False)
    tot = float(pos["Ecart"].sum())
    rows = []
    for k, (i, r) in enumerate(pos.head(3).iterrows()):
        cls = "bad" if r["EcartPct"] > 5 else "warn"
        part = f" ({num(r['Ecart'] / tot * 100, 0)} % de la dérive prévue)" if len(pos) > 1 else ""
        rows.append(f'<a class="wrow" style="--i:{k}" href="#lot-{i}"><span class="w-ic {cls}">{ico("alert" if cls == "warn" else "x", 15)}</span>'
                    f'<span class="w-t"><b>{esc(r["Lot"])}</b> : +{num(r["Ecart"] / 1e6)} M FCFA au-dessus du budget{part}</span>'
                    f'<span class="w-go">{ico("right", 16)}</span></a>')
    if not rows:
        rows.append(f'<div class="wrow" style="--i:0"><span class="w-ic ok">{ico("check", 15)}</span>'
                    f'<span class="w-t">Aucun lot ne dépasse son budget prévu.</span></div>')
    k = len(rows)
    if cpi < 1:
        rows.append(f'<div class="wrow" style="--i:{k}"><span class="w-ic warn">{ico("alert", 15)}</span><span class="w-t">'
                    f'<b>Rendement des dépenses : {num(cpi, 2)}</b>. Pour 1 FCFA dépensé, il ne se fait que {num(cpi, 2)} FCFA de travail prévu.</span></div>')
    else:
        rows.append(f'<div class="wrow" style="--i:{k}"><span class="w-ic ok">{ico("check", 15)}</span><span class="w-t">'
                    f'<b>Rendement des dépenses : {num(cpi, 2)}</b>. Les dépenses produisent au moins autant de travail que prévu.</span></div>')
    st.markdown('<div class="watch"><div class="w-head"><b>À surveiller</b><a href="#lots-editor">Modifier les lots</a></div>'
                + "".join(rows) + "</div>", unsafe_allow_html=True)


def tab_dashboard():
    P = _pal()
    cfg = {
        "Lot": st.column_config.TextColumn("Lot", required=True),
        "Budget": st.column_config.NumberColumn("Budget (FCFA)", min_value=0, step=100_000, format="%d"),
        "Depense": st.column_config.NumberColumn("Dépensé (FCFA)", min_value=0, step=100_000, format="%d"),
        "Avancement": st.column_config.NumberColumn("Avancement (%)", min_value=0, max_value=100, step=1),
        "Loi": st.column_config.SelectboxColumn("Loi de probabilité", options=DISTS),
        "Copt": st.column_config.NumberColumn("Coût opt.", format="%d"),
        "Cprob": st.column_config.NumberColumn("Coût prob.", format="%d"),
        "Cpess": st.column_config.NumberColumn("Coût pess.", format="%d"),
        "Dopt": st.column_config.NumberColumn("Durée opt. (j)"),
        "Dprob": st.column_config.NumberColumn("Durée prob. (j)"),
        "Dpess": st.column_config.NumberColumn("Durée pess. (j)"),
    }
    res = st.container()                        # les résultats s'affichent ici, au-dessus du tableau des lots
    box = st.container(border=True)
    with box:
        st.markdown('<span class="cc-mark" id="lots-editor"></span><div class="chart-title">Vos lots</div>'
                    '<div class="chart-sub">Modifiez, ajoutez ou supprimez un lot : tout le reste de la page se met à jour. '
                    'Les colonnes a / m / b sont les estimations optimiste, probable et pessimiste utilisées par les simulations.</div>',
                    unsafe_allow_html=True)
        ed = _stretch(st.data_editor, DEMO, num_rows="dynamic", hide_index=True, key="ed_lots", column_config=cfg)
    ed = ed.dropna(subset=["Lot"])
    ed = ed[ed["Lot"].astype(str).str.strip() != ""].copy()
    for c in NUM:
        ed[c] = pd.to_numeric(ed[c], errors="coerce").fillna(0)
    ed["Loi"] = ed["Loi"].where(ed["Loi"].isin(DISTS), "Bêta-PERT")
    ed = ed.reset_index(drop=True)
    st.session_state["lots_cur"] = ed
    if ed.empty:
        st.warning("Ajoutez au moins un lot dans le devis.")
        return

    av = ed["Avancement"] / 100
    ed["Consomme"] = np.where(ed["Budget"] > 0, ed["Depense"] / ed["Budget"].replace(0, np.nan) * 100, 0)
    ed["EAC"] = np.where(ed["Avancement"] >= 10, ed["Depense"] / np.maximum(av, 1e-9),
                         np.maximum(ed["Budget"], ed["Depense"]))
    ed["Ecart"] = ed["EAC"] - ed["Budget"]
    ed["EcartPct"] = np.where(ed["Budget"] > 0, ed["Ecart"] / ed["Budget"].replace(0, np.nan) * 100, 0)
    ed["Statut"] = np.select([ed["EcartPct"] <= 0, ed["EcartPct"] <= 5], ["OK", "Vigilance"], "Dérive")

    bac, ac = ed["Budget"].sum(), ed["Depense"].sum()
    ev = (ed["Budget"] * av).sum()
    cpi = ev / ac if ac > 0 else 1.0
    eac = ed["EAC"].sum()
    with res:
        _budget_view(ed, bac, ac, ev, cpi, eac)


def _budget_view(ed, bac, ac, ev, cpi, eac):
    P = _pal()
    conso, avanc = (ac / bac, ev / bac) if bac else (0.0, 0.0)
    over = eac > bac
    pct = (eac - bac) / bac * 100 if bac else 0.0

    # --- ligne 1 : prévision finale à gauche, les six indicateurs à droite ---
    c_hero, c_kpi = st.columns([2, 3])
    with c_hero:
        st.markdown(
            '<div class="hcard">'
            + rings(conso, avanc, cnt(conso * 100, 0, "") + "<small> %</small>", "consommé")
            + '<div class="hc-body">' + _lab("Prévision finale (EAC)")
            + f'<div class="hc-big">{cnt(eac / 1e6, 1)}<small>M FCFA</small></div>'
            + chip("warn" if over else "ok",
                   f"{signed(pct)}{num(abs(pct))} % au-dessus du budget de {num(bac / 1e6)} M FCFA" if over
                   else f"Dans le budget de {num(bac / 1e6)} M FCFA · marge {num((bac - eac) / 1e6)} M")
            + '<div class="hc-leg">'
            + f'<span><i style="background:{LEVEL_C["crit"] if conso > 1 else "var(--r1)"}"></i>Budget consommé<b>{num(conso * 100, 0)} %</b></span>'
            + f'<span><i style="background:var(--r2)"></i>Avancement pondéré<b>{num(avanc * 100, 0)} %</b></span></div>'
            + '</div></div>',
            unsafe_allow_html=True)
    with c_kpi:
        kpi_grid([
            ("Budget (BAC)", cnt(bac / 1e6, 1) + unit("M FCFA"), "", "mut", ""),
            ("Dépenses engagées", cnt(ac / 1e6, 1) + unit("M FCFA"), f"{pc(ac / bac, 0)} du budget" if bac else "", "mut", "pie"),
            ("Avancement pondéré", cnt(ev / bac * 100, 1) + unit("%") if bac else "n.d.", "", "mut", ""),
            ("Reste à dépenser", cnt(abs(bac - ac) / 1e6, 1, "−" if bac < ac else "") + unit("M FCFA"), "", "mut", ""),
            ("Prévision finale (EAC)", cnt(eac / 1e6, 1) + unit("M FCFA"),
             f"{signed(eac - bac)}{num(abs(eac - bac) / 1e6)} M", "bad" if eac > bac else "ok", "up" if eac > bac else "down"),
            ("CPI (EV/AC)", cnt(cpi, 2), "sous contrôle" if cpi >= 1 else "dérive", "ok" if cpi >= 1 else "bad", "check" if cpi >= 1 else "up"),
        ], cols=3)
    if eac > bac:
        banner("warn", f"Dépassement prévisionnel de {fcfa(eac - bac)} ({pc((eac - bac) / bac)})",
               "Extrapolation à performance de coût constante (EAC = BAC / CPI par lot).")
    else:
        banner("ok", "Prévision dans le budget", f"Marge prévisionnelle : {fcfa(bac - eac)}.")

    # --- ligne 2 : budget, dépensé, prévision par lot | écart prévu par lot ---
    n_lots = len(ed)
    haut_g = max(340, 40 * n_lots + 130)
    g1, g2 = st.columns([3, 2])
    fig = go.Figure()
    fig.add_bar(y=ed["Lot"], x=ed["Budget"] / 1e6, orientation="h", name="Budget", marker_color=P["gray"], opacity=.4, width=.8)
    fig.add_bar(y=ed["Lot"], x=ed["Depense"] / 1e6, orientation="h", name="Dépensé", marker_color=P["c1"], width=.4)
    fig.add_scatter(y=ed["Lot"], x=ed["EAC"] / 1e6, mode="markers", name="Prévision (EAC)",
                    marker=dict(symbol="diamond", size=13, color=P["c2"], line=dict(color=P["tip"], width=2)))
    fig.update_layout(barmode="overlay", yaxis=dict(autorange="reversed"), xaxis_title="M FCFA")
    with g1:
        chart(fig, "Budget, dépensé et prévision par lot",
              "Barre grise : le budget. Barre colorée : déjà dépensé. Losange : coût final prévu. Losange au-delà de la barre grise = le lot dérive.",
              haut_g, "y unified")
    with g2:
        rows = []
        for r in ed.sort_values("Ecart", ascending=False).itertuples():
            e = r.Ecart / 1e6
            tone = "bad" if r.Ecart > 0 else ("ok" if r.Ecart < 0 else "mut")
            icon = "up" if r.Ecart > 0 else ("down" if r.Ecart < 0 else "minus")
            txt = (f"{ico(icon, 12)} {signed(e)}{num(abs(e))} M<span class=\"pc\"> · {signed(r.EcartPct)}{num(abs(r.EcartPct))} %</span>"
                   if abs(e) >= 0.05 else f"{ico(icon, 12)} Au budget")
            rows.append((r.Lot, e, txt, tone))
        st.markdown('<div class="card"><div class="chart-title">Écart prévu par lot</div>'
                    '<div class="chart-sub">Dépassement attendu, en millions de FCFA, du plus gros écart au plus petit.</div>'
                    + hbars(rows, diverging=True)
                    + f'<div class="hb-cap"><span>{ico("left", 12)} sous le budget</span><span>au-dessus du budget {ico("right", 12)}</span></div></div>',
                    unsafe_allow_html=True)

    # --- ligne 3 : cascade | points d'attention ---
    w1, w2 = st.columns([3, 2])
    ecarts = ed[ed["Ecart"].abs() >= 1].sort_values("Ecart", ascending=False)
    fig = go.Figure(go.Waterfall(
        orientation="h", y=["Budget"] + ecarts["Lot"].tolist() + ["Prévision"],
        x=[bac / 1e6] + (ecarts["Ecart"] / 1e6).tolist() + [0],
        measure=["absolute"] + ["relative"] * len(ecarts) + ["total"], name="Écart",
        increasing=dict(marker_color=LEVEL_C["crit"]), decreasing=dict(marker_color=LEVEL_C["ok"]),
        totals=dict(marker_color=P["gray"]), connector=dict(line=dict(color=P["axis"]))))
    # L'axe ne part pas de zéro : sinon les deux grandes barres écrasent les écarts, qui sont l'information utile.
    niveaux = bac / 1e6 + (ecarts["Ecart"] / 1e6).cumsum()
    bas, haut = min(bac / 1e6, niveaux.min() if len(niveaux) else bac / 1e6), max(bac / 1e6, niveaux.max() if len(niveaux) else bac / 1e6)
    marge = max((haut - bas) * 0.25, 0.5)
    fig.update_layout(showlegend=False, yaxis=dict(autorange="reversed"), xaxis=dict(range=[bas - marge, haut + marge], title="M FCFA"))
    with w1:
        chart(fig, "Du budget à la prévision, lot par lot",
              "Les lots qui s'écartent du budget, du plus gros surcoût au plus petit. Rouge : surcoût. Vert : économie. L'axe est zoomé.",
              max(300, 36 * (len(ecarts) + 2) + 110), "closest")
    with w2:
        watch_card(ed, cpi)

    # --- lots : les plus en dérive d'abord ---
    section("Lots à surveiller", "Classés du plus gros écart au plus petit. Dans le budget · Vigilance : jusqu'à +5 % de dépassement prévu · Dérive : au-delà de +5 %.")
    st.markdown(lot_cards(ed.sort_values("Ecart", ascending=False)), unsafe_allow_html=True)
    with st.expander(":material/table_view: Tableau détaillé (toutes les colonnes)"):
        _stretch(st.dataframe, ed[["Statut", "Lot", "Budget", "Depense", "Consomme", "Avancement", "EAC", "Ecart", "EcartPct"]],
                 hide_index=True, column_config={
                     "Budget": st.column_config.NumberColumn("Budget", format="%d"),
                     "Depense": st.column_config.NumberColumn("Dépensé", format="%d"),
                     "Consomme": st.column_config.ProgressColumn("Budget consommé", min_value=0, max_value=100, format="%.0f%%"),
                     "Avancement": st.column_config.ProgressColumn("Avancement", min_value=0, max_value=100, format="%.0f%%"),
                     "EAC": st.column_config.NumberColumn("Prévision (EAC)", format="%d"),
                     "Ecart": st.column_config.NumberColumn("Écart (FCFA)", format="%+d"),
                     "EcartPct": st.column_config.NumberColumn("Écart %", format="%+.1f%%")})


# ---------------------------------------------------------------------
# 2. PROBABILITÉS & MONTE-CARLO
# ---------------------------------------------------------------------
def _mc_total(d, cols, n, seed):
    """Totaux simulés pour le coût ou le délai (None si les estimations a ≤ m ≤ b sont incohérentes)."""
    if not ((d[cols[0]] <= d[cols[1]]) & (d[cols[1]] <= d[cols[2]]) & (d[cols[0]] < d[cols[2]]) & (d[cols[0]] > 0)).all():
        return None
    params = tuple((r.Loi, float(getattr(r, cols[0])), float(getattr(r, cols[1])), float(getattr(r, cols[2]))) for r in d.itertuples())
    return simulate(params, n, seed).sum(axis=1)


def _cmp_card(titre, total, seuil, scale, u, p_warn):
    p = float((total > seuil).mean())
    lvl = "crit" if p >= 0.5 else ("warn" if p >= p_warn else "ok")
    verdict = {"crit": "Dépassement probable", "warn": "Risque élevé", "ok": "Risque maîtrisé"}[lvl]
    lim = min(p_warn, 0.5)
    zones = [(0, lim, LEVEL_C["ok"]), (lim, 0.5, LEVEL_C["warn"]), (0.5, 1, LEVEL_C["crit"])]
    p50, p95 = np.percentile(total, [50, 95]) / scale
    return ('<div class="hcard">'
            + gauge(p, LEVEL_C[lvl], zones, mark=p_warn, big=cnt(p * 100, 1) + "<small> %</small>",
                    sub=f"seuil d'alerte {pc(p_warn, 0)}", label=f"{titre} : probabilité de dépassement {pc(p)}")
            + f'<div class="hc-body"><div class="hc-title">{titre}</div>' + chip(lvl, verdict)
            + f'<div class="hc-note">Seuil : <b>{num(seuil / scale)} {u}</b><br>Médiane {num(p50)} · P95 {num(p95)} {u}</div></div></div>')


def tab_proba(n, seed):
    P = _pal()
    d = st.session_state["lots_cur"]
    sub1, sub2 = st.tabs([":material/query_stats: Monte-Carlo du chantier", ":material/functions: Laboratoire des lois"])
    with sub1:
        c1, c2, c3 = st.columns(3)
        mode = c1.radio("Variable étudiée", ["Coût", "Délai"], horizontal=True)
        cost = mode == "Coût"
        u, scale = ("M FCFA", 1e6) if cost else ("jours", 1.0)
        cols = ["Copt", "Cprob", "Cpess"] if cost else ["Dopt", "Dprob", "Dpess"]
        default = float(d["Budget"].sum()) if cost else 360.0
        seuil = c2.number_input(f"Seuil d'alerte ({'FCFA' if cost else 'jours'})", value=default,
                                step=1e6 if cost else 5.0, format="%.0f", key=f"seuil_{mode}_{int(default)}")
        p_warn = c3.slider("Alerte si P(dépassement) ≥ (%)", 5, 80, 25) / 100
        st.caption("Lots supposés indépendants et exécutés en séquence (délai = somme des durées). "
                   "Estimations a / m / b modifiables dans l'onglet Budget (section « Vos lots »).")

        bad = d[~((d[cols[0]] <= d[cols[1]]) & (d[cols[1]] <= d[cols[2]]) & (d[cols[0]] < d[cols[2]]) & (d[cols[0]] > 0))]
        if len(bad):
            st.error("Incohérence a ≤ m ≤ b (a < b, a > 0) pour : " + ", ".join(bad["Lot"]))
            return
        params = tuple((r.Loi, float(getattr(r, cols[0])), float(getattr(r, cols[1])), float(getattr(r, cols[2])))
                       for r in d.itertuples())
        total = simulate(params, n, seed).sum(axis=1)
        dists = [make_dist(*p) for p in params]
        mus = np.array([x.mean() for x in dists])
        sds = np.array([x.std() for x in dists])
        mu_t, sd_t = mus.sum(), math.sqrt((sds ** 2).sum())

        p_over = float((total > seuil).mean())
        se = math.sqrt(p_over * (1 - p_over) / n)
        section("Résultat de la simulation", f"{n:,} chantiers simulés au hasard pour estimer le {'coût total' if cost else 'délai total'} et le risque de dépasser le seuil.".replace(",", " "))
        what = "budgétaire" if cost else "de délai"
        level = "crit" if p_over >= 0.5 else ("warn" if p_over >= p_warn else "ok")
        verdict = {"crit": "Dépassement probable", "warn": "Risque élevé", "ok": "Risque maîtrisé"}[level]
        lim = min(p_warn, 0.5)
        zones = [(0, lim, LEVEL_C["ok"]), (lim, 0.5, LEVEL_C["warn"]), (0.5, 1, LEVEL_C["crit"])]
        st.markdown(
            '<div class="hcard">'
            + gauge(p_over, LEVEL_C[level], zones, mark=p_warn, big=cnt(p_over * 100, 1) + "<small> %</small>",
                    sub=f"repère blanc : seuil d'alerte {pc(p_warn, 0)}", label=f"Probabilité de dépassement : {pc(p_over)}")
            + '<div class="hc-body">' + _lab("Probabilité de dépasser le seuil")
            + f'<div class="hc-title">Risque {what}</div>' + chip(level, verdict)
            + f'<div class="hc-note">Seuil : <b>{num(seuil / scale, 1)} {u}</b><br>'
              f'Intervalle de confiance à 95 % : ± {pc(1.96 * se, 2)}</div></div>'
            + risk_track(np.percentile(total, [1, 50, 95, 99]) / scale, seuil / scale, u) + '</div>',
            unsafe_allow_html=True)
        kpi_grid([
            ("Moyenne E[X]", cnt(mu_t / scale, 1) + unit(u), "", "mut", ""),
            ("Écart-type σ", cnt(sd_t / scale, 2) + unit(u), "", "mut", ""),
            ("Médiane P50", cnt(np.percentile(total, 50) / scale, 1) + unit(u), "", "mut", ""),
            ("P95 (VaR)", cnt(np.percentile(total, 95) / scale, 1) + unit(u), "", "mut", ""),
        ])
        if p_over >= 0.5:
            banner("crit", f"Dépassement {what} plus probable que non : {pc(p_over)}",
                   "Le seuil est inférieur à la médiane de la distribution : renégocier ou prévoir une réserve.")
        elif p_over >= p_warn:
            banner("warn", f"Risque {what} élevé : {pc(p_over)} ≥ seuil d'alerte {pc(p_warn, 0)}",
                   f"Réserve pour P=90 % : {(np.percentile(total, 90) - seuil) / scale:,.1f} {u} au-dessus du seuil.")
        else:
            banner("ok", f"Risque {what} maîtrisé : {pc(p_over)}", f"IC à 95 % : [{pc(max(0, p_over - 1.96 * se), 2)} ; {pc(p_over + 1.96 * se, 2)}]")

        section("Où se situe le seuil ?", "Plus le seuil (ligne rouge) est à gauche, plus le dépassement est probable.")
        x = np.linspace(total.min(), total.max(), 300) / scale
        ts, ss = total / scale, seuil / scale
        g1, g2 = st.columns(2)
        counts, edges = np.histogram(ts, bins=70, density=True)
        centres, larg = (edges[:-1] + edges[1:]) / 2, np.diff(edges)
        fig = go.Figure()
        fig.add_bar(x=centres, y=counts, width=larg * .92, name="Monte-Carlo", marker_color=P["c1"], opacity=.85)
        fig.add_scatter(x=x, y=stats.norm.pdf(x, mu_t / scale, sd_t / scale), name="Normale (TCL)",
                        line=dict(width=3, color=P["c2"]))
        fig.add_vline(x=ss, line_dash="dash", line_color=LEVEL_C["crit"], annotation_text="Seuil",
                      annotation_font_color=P["text"])
        fig.update_layout(xaxis_title=u)
        with g1:
            chart(fig, "Répartition des chantiers simulés", "Plus une barre est haute, plus ce résultat est fréquent.", 380)
        srt = np.sort(ts)
        sel = np.linspace(0, n - 1, 600).astype(int)
        fig = go.Figure()
        fig.add_scatter(x=srt[sel], y=(sel + 1) / n, name="Empirique (simulation)", line=dict(width=3, color=P["c1"]))
        fig.add_scatter(x=x, y=stats.norm.cdf(x, mu_t / scale, sd_t / scale), name="Normale",
                        line=dict(width=2.5, dash="dot", color=P["c2"]))
        fig.add_vline(x=ss, line_dash="dash", line_color=LEVEL_C["crit"])
        fig.add_hline(y=1 - p_over, line_dash="dot", line_color=P["muted"], annotation_text=f"P(X ≤ seuil) = {pc(1 - p_over)}",
                      annotation_font_color=P["text"])
        fig.update_layout(xaxis_title=u, yaxis_range=[0, 1])
        with g2:
            chart(fig, "Probabilité cumulée", "Lire : « avec quelle probabilité le résultat reste-t-il sous cette valeur ? »", 380)
        st.caption(f"Approximation normale (théorème central limite) : P(dépassement) ≈ "
                   f"{pc(1 - stats.norm.cdf(seuil, mu_t, sd_t))} ; Monte-Carlo : {pc(p_over)} (N = {num(n, 0)})")

        section("Et si… ?", "Bougez le seuil pour voir le risque réagir, ou choisissez le niveau de sécurité voulu pour savoir quelle réserve prévoir.")
        lo_s, hi_s = float(np.percentile(total, 1)) / scale, float(np.percentile(total, 99)) / scale
        w1, w2 = st.columns(2)
        s_if = w1.slider(f"Et si le seuil était de ({u})", lo_s, hi_s, float(min(max(seuil / scale, lo_s), hi_s)),
                         step=max((hi_s - lo_s) / 200, 1e-6), format="%.1f", key=f"si_{mode}_{int(seuil)}_{n}_{seed}")
        conf = w2.select_slider("Niveau de sécurité voulu", [50, 70, 80, 90, 95, 99], value=90, format_func=lambda c: f"{c} %", key=f"conf_{mode}")
        p_if = float((total > s_if * scale).mean())
        a_prevoir = float(np.percentile(total, conf)) / scale
        reserve = a_prevoir - seuil / scale
        d_p = (p_if - p_over) * 100
        kpi_grid([
            ("Risque à ce seuil", cnt(p_if * 100, 1) + unit("%"),
             f"{signed(d_p)}{num(abs(d_p))} pts par rapport au seuil actuel" if abs(d_p) >= 0.05 else "identique au seuil actuel",
             "bad" if p_if >= 0.5 else ("warn" if p_if >= p_warn else "ok"), "up" if d_p > 0.05 else ("down" if d_p < -0.05 else "minus")),
            (f"À prévoir pour être sûr à {conf} %", cnt(a_prevoir, 1) + unit(u),
             f"+{num(reserve)} {u} par rapport au seuil" if reserve > 0 else "sous le seuil actuel : aucune réserve",
             "warn" if reserve > 0 else "ok", "up" if reserve > 0 else "check"),
        ])

        section("Quels lots pèsent le plus sur l'incertitude ?", "Les lots en haut sont ceux qu'il faut surveiller en priorité.")
        h1, h2 = st.columns([1, 1])
        share = (sds ** 2 / (sds ** 2).sum() * 100)
        order = np.argsort(share)[::-1]
        with h1:
            st.markdown('<div class="card"><div class="chart-title">Contribution de chaque lot à la variance</div>'
                        + hbars([(d["Lot"].to_numpy()[i], float(share[i]), f"{num(share[i])} %", "mut") for i in order])
                        + "</div>", unsafe_allow_html=True)
        tab = pd.DataFrame({"Lot": d["Lot"], "Loi": d["Loi"], "Moyenne": mus / scale, "σ": sds / scale,
                            "P10": [x.ppf(.1) / scale for x in dists], "P90": [x.ppf(.9) / scale for x in dists]})
        with h2:
            st.markdown(f"**Moments théoriques par lot ({u})**")
            _stretch(st.dataframe, tab, hide_index=True, column_config={
                c: st.column_config.NumberColumn(format="%.2f") for c in ["Moyenne", "σ", "P10", "P90"]})

        section("Coût ou délai : lequel risque le plus ?", "Les deux simulations côte à côte, avec chacune son seuil d'alerte.")
        ci1, ci2 = st.columns(2)
        s_cout = ci1.number_input("Seuil coût (FCFA)", value=float(d["Budget"].sum()), step=1e6, format="%.0f", key=f"cmp_cost_{int(d['Budget'].sum())}")
        s_delai = ci2.number_input("Seuil délai (jours)", value=360.0, step=5.0, format="%.0f", key="cmp_delay")
        k1, k2 = st.columns(2)
        for col, titre, cols_v, sc_v, u_v, seuil_v in ((k1, "Coût", ["Copt", "Cprob", "Cpess"], 1e6, "M FCFA", s_cout),
                                                         (k2, "Délai", ["Dopt", "Dprob", "Dpess"], 1.0, "jours", s_delai)):
            tot_v = _mc_total(d, cols_v, n, seed)
            with col:
                if tot_v is None:
                    st.warning(f"{titre} : les estimations a ≤ m ≤ b sont incohérentes, la simulation est impossible.")
                else:
                    st.markdown(_cmp_card(titre, tot_v, seuil_v, sc_v, u_v, p_warn), unsafe_allow_html=True)

    with sub2:
        section("Paramètres de la loi", "Choisissez une loi, puis trois estimations : a (optimiste), m (probable) et b (pessimiste).")
        e = st.columns(5)
        name = e[0].selectbox("Loi", DISTS, index=2)
        a = e[1].number_input("a (optimiste)", value=80.0)
        m = e[2].number_input("m (probable)", value=100.0)
        b = e[3].number_input("b (pessimiste)", value=150.0)
        comp = e[4].checkbox("Comparer les 4 lois", value=True)
        if not (a < b and a <= m <= b and a > 0):
            st.error("Il faut 0 < a ≤ m ≤ b avec a < b.")
            return
        w = b - a
        xs = np.linspace(a - .25 * w, b + .25 * w, 500)
        x0 = st.slider("Valeur x₀", float(xs[0]), float(xs[-1]), float(m))
        dist = make_dist(name, a, m, b)
        mean, var, skew = dist.stats("mvs")
        p0 = float(dist.cdf(x0))
        kpi_grid([
            ("Moyenne", cnt(float(mean), 2, "−" if mean < 0 else ""), "", "mut", ""),
            ("Écart-type", cnt(math.sqrt(float(var)), 2), "", "mut", ""),
            ("Asymétrie", cnt(float(skew), 2, "−" if skew < 0 else ""), "", "mut", ""),
            ("P(X ≤ x₀)", cnt(p0 * 100, 1) + unit("%"), "", "mut", ""),
            ("P(X > x₀)", cnt((1 - p0) * 100, 1) + unit("%"), "", "mut", ""),
        ])
        lab_c = dict(zip(DISTS, [P["c1"], P["c2"], P["c3"], P["c4"]]))   # une couleur par loi, toujours la même
        f1, f2 = go.Figure(), go.Figure()
        for nm in (DISTS if comp else [name]):
            dd = make_dist(nm, a, m, b)
            lw = 4 if nm == name else 2
            f1.add_scatter(x=xs, y=dd.pdf(xs), name=nm, line=dict(width=lw, color=lab_c[nm]))
            f2.add_scatter(x=xs, y=dd.cdf(xs), name=nm, line=dict(width=lw, color=lab_c[nm]))
        xt = xs[xs >= x0]
        f1.add_scatter(x=xt, y=dist.pdf(xt), fill="tozeroy", name=f"P(X > x₀) = {pc(1 - p0)}", mode="none",
                       fillcolor=_rgba(LEVEL_C["crit"], .28))
        for f in (f1, f2):
            f.add_vline(x=x0, line_dash="dash", line_color=LEVEL_C["crit"])
        f2.update_layout(yaxis_range=[0, 1])
        c1, c2 = st.columns(2)
        with c1:
            chart(f1, "Densité de probabilité", "La zone rouge est la probabilité de dépasser x₀.", 360)
        with c2:
            chart(f2, "Probabilité cumulée", "Lire : « quelle part des cas reste sous x₀ ? »", 360)


# ---------------------------------------------------------------------
# 3. POUTRE BÉTON ARMÉ & FIABILITÉ  (unités : mm, N, MPa ; q en kN/m = N/mm)
# ---------------------------------------------------------------------
def mrd(fc, fy, As, b, d):
    """Moment résistant (N·mm), bloc rectangulaire λ=0.8 : x = As·fy/(0.8·b·fc)."""
    x = As * fy / (0.8 * b * fc)
    return As * fy * (d - 0.4 * x)


def beam_calc(L, b, h, dp, fck, fyk, G, Q, auto, As_man):
    r = {"d": h - dp}
    d = r["d"]
    fcd, fyd = 0.85 * fck / 1.5, fyk / 1.15
    sw = b / 1000 * h / 1000 * 25
    Gt, Lmm = G + sw, L * 1000
    qu = 1.35 * Gt + 1.5 * Q
    Med = qu * Lmm ** 2 / 8
    mu = Med / (b * d ** 2 * fcd)
    r.update(sw=sw, Gt=Gt, qu=qu, Med=Med, mu=mu, section_ok=mu <= 0.295, fcd=fcd, fyd=fyd)
    if r["section_ok"]:
        xi = 1.25 * (1 - math.sqrt(1 - 2 * mu))
        r["As_req"] = Med / (d * (1 - 0.4 * xi) * fyd)
    else:
        r["As_req"] = float("nan")
    r["As_min"] = max(0.26 * 0.30 * fck ** (2 / 3) / fyk * b * d, 0.0013 * b * d)
    r["As_max"] = 0.04 * b * h
    r["As"] = (1.10 * r["As_req"] if r["section_ok"] else r["As_max"]) if auto else As_man
    As = r["As"]
    r["x"] = As * fyd / (0.8 * b * fcd)
    r["Mrd"] = mrd(fcd, fyd, As, b, d)
    # Flèche quasi-permanente (estimation simplifiée, section fissurée / brute, fluage φ = 2)
    Ec = 22000 * ((fck + 8) / 10) ** 0.3 / 3
    n_ = 200000 / Ec
    xcr = (-n_ * As + math.sqrt((n_ * As) ** 2 + 2 * b * n_ * As * d)) / b
    Icr = b * xcr ** 3 / 3 + n_ * As * (d - xcr) ** 2
    Ieff = 0.5 * (b * h ** 3 / 12 + Icr)
    r["fleche"] = 5 * (Gt + 0.3 * Q) * Lmm ** 4 / (384 * Ec * Ieff)
    r["fleche_lim"] = Lmm / 250
    return r


@st.cache_data(show_spinner=False)
def beam_mc(fcm, cfc, fym, cfy, cth, Gm, cG, Qm, cQ, As, b, d, L, n, seed):
    """Monte-Carlo de la marge g = R - S (kN·m). fc, fy, θ : log-normales ; G : normale ; Q : Gumbel."""
    rng = np.random.default_rng(seed)

    def logn(m, c):
        return rng.lognormal(math.log(m / math.sqrt(1 + c * c)), math.sqrt(math.log(1 + c * c)), n)

    sc = Qm * cQ * math.sqrt(6) / math.pi
    Q = stats.gumbel_r.rvs(loc=Qm - 0.5772 * sc, scale=sc, size=n, random_state=rng)
    G = rng.normal(Gm, Gm * cG, n)
    R = logn(1.0, cth) * mrd(logn(fcm, cfc), logn(fym, cfy), As, b, d)
    return ((R - (G + Q) * (L * 1000) ** 2 / 8) / 1e6).astype(np.float32)


def beam_svg(L, b, h, dp, As, x, qu, fleche, level):
    """Schéma de la poutre (élévation simplifiée) et de sa section avec les armatures, à partir des valeurs saisies."""
    col = LEVEL_C[level]
    hb = max(30.0, min(84.0, 452 * (h / 1000) / L))
    x0, x1, yt = 24.0, 476.0, 84.0
    yb = yt + hb
    xm = (x0 + x1) / 2
    fleche_trait = "fill:none;stroke:var(--mut);stroke-width:2;stroke-linecap:round;stroke-linejoin:round"
    fleches = "".join(f'<path class="bs-a" style="--i:{k};{fleche_trait}" d="M{ax:.1f} 36V{yt - 8:.1f}M{ax - 5:.1f} {yt - 15:.1f}l5 7 5-7"/>'
                      for k, ax in enumerate(np.linspace(x0 + 10, x1 - 10, 12)))
    haut = yb + 100
    el = (f'<svg class="el" viewBox="0 0 500 {haut:.0f}" role="img" aria-label="Poutre de {num(L)} m sur deux appuis, chargée uniformément">'
          f'<text class="bt b" x="{x0}" y="22">q = {num(qu, 0)} kN/m (charge ELU)</text>'
          f'<path class="bs-a" style="--i:0;{fleche_trait}" d="M{x0} 36H{x1}"/>{fleches}'
          f'<rect class="bs-d" pathLength="1" x="{x0}" y="{yt}" width="{x1 - x0}" height="{hb:.1f}" rx="4" style="fill:var(--track);stroke:{col};stroke-width:2.5"/>'
          f'<path class="bs-a" style="--i:6;{fleche_trait}" d="M{x0} {yb:.1f}l-14 22h28zM{x1} {yb:.1f}l-14 22h28zM{x0 - 22} {yb + 22:.1f}h44M{x1 - 22} {yb + 22:.1f}h44"/>'
          f'<path class="bs-a" style="--i:8;fill:none;stroke:{col};stroke-width:2;stroke-dasharray:6 5" d="M{x0} {yb:.1f}Q{xm} {yb + 20:.1f} {x1} {yb:.1f}"/>'
          f'<text class="bt" x="{xm}" y="{yb + 38:.1f}" text-anchor="middle">flèche ≈ {num(fleche)} mm (trait exagéré)</text>'
          f'<path class="bs-a" style="--i:9;{fleche_trait}" d="M{x0} {yb + 64:.1f}H{x1}M{x0} {yb + 58:.1f}v12M{x1} {yb + 58:.1f}v12"/>'
          f'<text class="bt b" x="{xm}" y="{yb + 90:.1f}" text-anchor="middle">portée L = {num(L)} m</text></svg>')
    hs, cx, sy = 150.0, 120.0, 16.0
    sc = hs / h
    ws = max(50.0, min(150.0, hs * b / h))
    sx = cx - ws / 2
    ci = max(6.0, min(14.0, dp * sc * 0.5))
    phi = 16 if As < 1500 else 20
    nb = max(2, min(8, round(As / (math.pi / 4 * phi * phi))))
    rb = max(3.5, min(8.0, phi * sc / 2))
    ybar = sy + hs - max(dp * sc, ci + rb + 1)
    xs_bar = np.linspace(sx + ci + rb + 2, sx + ws - ci - rb - 2, nb) if nb > 1 else [cx]
    barres = "".join(f'<circle class="bs-r" style="--i:{k};fill:var(--c-blue)" cx="{bx:.1f}" cy="{ybar:.1f}" r="{rb:.1f}"/>' for k, bx in enumerate(xs_bar))
    chap = "".join(f'<circle class="bs-r" style="--i:{k};fill:var(--mut)" cx="{bx:.1f}" cy="{sy + ci + 5:.1f}" r="3.5"/>'
                   for k, bx in enumerate((sx + ci + 6, sx + ws - ci - 6)))
    xh = min(x, h) * sc
    trait = "fill:none;stroke:var(--mut);stroke-width:1.6"
    sec = (f'<svg class="sc" viewBox="0 0 240 232" role="img" aria-label="Section {num(b, 0)} sur {num(h, 0)} mm avec environ {nb} barres d\'acier">'
           f'<rect x="{sx:.1f}" y="{sy}" width="{ws:.1f}" height="{0.8 * xh:.1f}" style="fill:var(--r2);opacity:.35"/>'
           f'<rect class="bs-d" pathLength="1" x="{sx:.1f}" y="{sy}" width="{ws:.1f}" height="{hs}" rx="4" style="fill:var(--track);fill-opacity:.55;stroke:{col};stroke-width:2.5"/>'
           f'<rect class="bs-d" pathLength="1" x="{sx + ci:.1f}" y="{sy + ci:.1f}" width="{ws - 2 * ci:.1f}" height="{hs - 2 * ci:.1f}" rx="4" style="{trait}"/>'
           f'{chap}{barres}'
           f'<path class="bs-a" style="--i:4;fill:none;stroke:var(--accent);stroke-width:1.6;stroke-dasharray:5 4" d="M{sx - 6:.1f} {sy + xh:.1f}H{sx + ws + 6:.1f}"/>'
           f'<path class="bs-a" style="--i:5;{trait}" d="M{sx - 12:.1f} {sy}V{sy + hs}M{sx - 17:.1f} {sy}h10M{sx - 17:.1f} {sy + hs}h10"/>'
           f'<text class="bt" transform="translate({sx - 20:.1f} {sy + hs / 2}) rotate(-90)" text-anchor="middle">h = {num(h, 0)} mm</text>'
           f'<path class="bs-a" style="--i:6;{trait}" d="M{sx:.1f} {sy + hs + 12}H{sx + ws:.1f}M{sx:.1f} {sy + hs + 7}v10M{sx + ws:.1f} {sy + hs + 7}v10"/>'
           f'<text class="bt" x="{cx}" y="{sy + hs + 34}" text-anchor="middle">b = {num(b, 0)} mm</text>'
           f'<text class="bt b" x="{cx}" y="{sy + hs + 54}" text-anchor="middle">≈ {nb} HA{phi} · As = {num(As, 0)} mm²</text></svg>')
    return ('<div class="card"><div class="chart-title">Schéma de la poutre</div>'
            '<div class="chart-sub">La section est à l\'échelle. Zone teintée : béton comprimé. Pointillé : axe neutre. '
            'Le nombre de barres est indicatif (barres HA16, ou HA20 au-delà de 1 500 mm²).</div>'
            f'<div class="bsch">{el}{sec}</div></div>')


PRESETS = {
    "Poutre conforme": dict(L=6.0, b=300, h=600, dp=50, fck=30, fyk=500, G=15.0, Q=10.0, auto="Auto (As requis +10 %)", As=1005),
    "Cas limite": dict(L=7.0, b=300, h=550, dp=50, fck=30, fyk=500, G=18.0, Q=15.0, auto="Manuel", As=1250),
    "Cas critique (sous-armée)": dict(L=7.0, b=300, h=550, dp=50, fck=30, fyk=500, G=18.0, Q=15.0, auto="Manuel", As=800),
}
AUTO = "Auto (As requis +10 %)"


def _apply_preset():
    for k, v in PRESETS[st.session_state["preset"]].items():
        st.session_state[k] = v


PRESET_INFO = {
    "Poutre conforme": ("check", "ok", "Dimensionnée avec de la marge : le cas de référence."),
    "Cas limite": ("alert", "warn", "Les armatures sont à la limite : elle tient, mais sans marge."),
    "Cas critique (sous-armée)": ("x", "crit", "Pas assez d'acier : la rupture est possible. À éviter."),
}


def _set_preset(name):
    st.session_state["preset"] = name
    _apply_preset()


def _form_head(icone, titre, indice=""):
    st.markdown(f'<span class="cc-mark"></span><div class="fh"><span class="fh-ic">{ico(icone, 16)}</span><b>{titre}</b>'
                f'<span class="fh-h">{indice}</span></div>', unsafe_allow_html=True)


def tab_beam(n, seed):
    P = _pal()
    for k, v in {**PRESETS["Poutre conforme"], "cfc": 15, "cfy": 5, "cG": 8, "cQ": 30, "cth": 8}.items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("preset", "Poutre conforme")
    section("Choisir un scénario", "Trois poutres prêtes à l'emploi pour voir le verdict changer. Vous pouvez ensuite modifier les valeurs.")
    for col, (nom, (ic, lvl, texte)) in zip(st.columns(3), PRESET_INFO.items()):
        pr = PRESETS[nom]
        rr = beam_calc(pr["L"], pr["b"], pr["h"], pr["dp"], pr["fck"], pr["fyk"], pr["G"], pr["Q"], pr["auto"] == AUTO, pr["As"])
        util_p = rr["Med"] / rr["Mrd"] * 100 if rr["Mrd"] > 0 else 999.0
        choisi = st.session_state.get("preset") == nom
        with col.container(border=True):
            st.markdown(
                f'<span class="cc-mark{" pc-on" if choisi else ""}"></span><div class="pc-head"><span class="pc-ic {lvl}">{ico(ic, 16)}</span><b>{nom}</b></div>'
                f'<div class="chart-sub">{texte}</div>'
                f'<div class="pc-facts">{num(pr["L"])} m · {pr["b"]} × {pr["h"]} mm · acier {num(rr["As"], 0)} mm²</div>'
                f'<div class="lot-line"><span>Moment résistant utilisé</span><b>{num(util_p, 0)} %</b></div>'
                f'<div class="bar"><i style="width:{min(util_p, 100):.0f}%;background:{LEVEL_C[lvl]}"></i></div>',
                unsafe_allow_html=True)
            _stretch(st.button, "Scénario choisi" if choisi else "Utiliser ce scénario", key=f"pcb_{lvl}",
                     on_click=_set_preset, args=(nom,), disabled=choisi)
    section("Données de la poutre", "Modifiez les valeurs : les résultats et le schéma se mettent à jour.")
    fa, fb = st.columns(2)
    with fa:
        with st.container(border=True):
            _form_head("ruler", "Géométrie", "m et mm")
            c = st.columns(2)
            L = c[0].number_input("Portée L (m)", 0.0, 40.0, step=0.5, key="L")
            b = c[1].number_input("Largeur b (mm)", 0, 2000, step=10, key="b")
            c = st.columns(2)
            h = c[0].number_input("Hauteur h (mm)", 0, 3000, step=10, key="h")
            dp = c[1].number_input("Distance armatures d' (mm)", 0, 500, step=5, key="dp")
    with fb:
        with st.container(border=True):
            _form_head("layers", "Matériaux", "MPa")
            c = st.columns(2)
            fck = c[0].select_slider("Béton fck (MPa)", [20, 25, 30, 35, 40, 45, 50], key="fck")
            fyk = c[1].select_slider("Acier fyk (MPa)", [400, 500], key="fyk")
            st.caption("fck : résistance du béton. fyk : résistance de l'acier.")
    fc_, fd_ = st.columns(2)
    with fc_:
        with st.container(border=True):
            _form_head("weight", "Charges", "kN/m")
            c = st.columns(2)
            G = c[0].number_input("Permanente G (kN/m, hors poids propre)", 0.0, 500.0, step=1.0, key="G")
            Q = c[1].number_input("Exploitation Q (kN/m)", 0.0, 500.0, step=1.0, key="Q")
    with fd_:
        with st.container(border=True):
            _form_head("rebar", "Armatures tendues", "mm²")
            c = st.columns([3, 2])
            auto = c[0].radio("Choix de la section d'acier", [AUTO, "Manuel"], key="auto") == AUTO
            As_in = c[1].number_input("As manuel (mm²)", 0, 50000, step=50, key="As", disabled=auto)
    schema_box = st.container()                 # le schéma s'affiche ici une fois le calcul fait
    with st.expander(":material/science: Variabilité stochastique (coefficients de variation, %)"):
        v = st.columns(5)
        cfc = v[0].slider("Béton fc", 5, 30, key="cfc") / 100
        cfy = v[1].slider("Acier fy", 2, 15, key="cfy") / 100
        cG = v[2].slider("Perm. G", 2, 20, key="cG") / 100
        cQ = v[3].slider("Expl. Q", 10, 60, key="cQ") / 100
        cth = v[4].slider("Modèle θ", 2, 20, key="cth") / 100
        st.caption("Hypothèses : fc moyen = fck + 8 MPa (×0,85), fy moyen = fyk(1 + 1,645·CV), G moyen = valeur nominale, "
                   "Q moyen = 0,6·Qk (Gumbel), θ = incertitude de modèle sur la résistance (moyenne 1). Variables indépendantes.")

    # --- Validation : données erronées ou incohérentes ---
    errs = []
    if L <= 0:
        errs.append("la portée doit être strictement positive")
    if b <= 0 or h <= 0:
        errs.append("les dimensions b et h doivent être strictement positives")
    elif dp <= 0 or dp >= h / 2:
        errs.append(f"d' = {dp} mm est incohérent avec h = {h} mm (il faut 0 < d' < h/2)")
    if not auto and As_in <= 0:
        errs.append("As manuel doit être > 0")
    if errs:
        banner("crit", "DONNÉES ERRONÉES OU INCOHÉRENTES : calcul impossible", " ; ".join(errs))
        return
    if L / (h / 1000) > 30:
        banner("warn", "Élancement aberrant", f"L/h = {L / (h / 1000):.0f} (> 30) : vérifiez les unités (L en m, h en mm).")
    if L / (h / 1000) < 4:
        banner("warn", "Poutre très courte", f"L/h = {L / (h / 1000):.1f} : la théorie des poutres (flexion) n'est plus valable (poutre-voile).")

    r = beam_calc(L, b, h, dp, fck, fyk, G, Q, auto, As_in)
    d, As, Lmm = r["d"], r["As"], L * 1000
    # --- Fiabilité de Cornell ---
    fcm, fym = 0.85 * (fck + 8), fyk * (1 + 1.645 * cfy)
    R0 = mrd(fcm, fym, As, b, d)
    dfc = (mrd(fcm * 1.01, fym, As, b, d) - R0) / 0.01
    dfy = (mrd(fcm, fym * 1.01, As, b, d) - R0) / 0.01
    sR = math.sqrt((dfc * cfc) ** 2 + (dfy * cfy) ** 2 + (R0 * cth) ** 2)
    Gm, Qm = r["Gt"], 0.6 * Q
    muS = (Gm + Qm) * Lmm ** 2 / 8
    sS = Lmm ** 2 / 8 * math.sqrt((Gm * cG) ** 2 + (Qm * cQ) ** 2)
    beta = (R0 - muS) / math.sqrt(sR ** 2 + sS ** 2)
    pf = float(stats.norm.cdf(-beta))
    g = beam_mc(fcm, cfc, fym, cfy, cth, Gm, cG, Qm, cQ, As, b, d, L, n, seed)
    pf_mc = float((g < 0).mean())

    # --- Alerte intelligente ---
    section("Verdict de sécurité", "Ce que le calcul conclut pour cette poutre.")
    crit, warn = [], []
    if not r["section_ok"]:
        crit.append(f"section insuffisante (μ = {r['mu']:.2f} > 0,295) : augmenter b ou h")
    if beta < 1.5:
        crit.append(f"indice de fiabilité β = {beta:.2f} < 1,5 (Pf ≈ {pf:.1e}) : rupture en flexion très probable")
    if muS >= R0:
        crit.append("charge moyenne ≥ résistance moyenne : la poutre s'effondre en moyenne")
    if r["fleche"] > Lmm / 150:
        crit.append(f"flèche excessive {r['fleche']:.0f} mm > L/150 = {Lmm / 150:.0f} mm")
    if r["Med"] > r["Mrd"]:
        warn.append(f"ELU non vérifié : MEd = {r['Med'] / 1e6:.0f} > MRd = {r['Mrd'] / 1e6:.0f} kN·m")
    if beta < 3.8:
        warn.append(f"β = {beta:.2f} < cible 3,8 (EN 1990, classe RC2, 50 ans)")
    if r["fleche"] > r["fleche_lim"]:
        warn.append(f"flèche {r['fleche']:.0f} mm > L/250 = {r['fleche_lim']:.0f} mm")
    if As < r["As_min"]:
        warn.append(f"As < As,min = {r['As_min']:.0f} mm² (rupture fragile)")
    if As > r["As_max"]:
        warn.append(f"As > As,max = {r['As_max']:.0f} mm² (4 % de la section)")
    if r["x"] > 0.45 * d:
        warn.append("x/d > 0,45 : section sur-armée, rupture non ductile")
    level = "crit" if crit else ("warn" if warn else "ok")
    with schema_box:
        st.markdown(beam_svg(L, b, h, dp, As, r["x"], r["qu"], r["fleche"], level), unsafe_allow_html=True)
    headline = {"crit": "Risque critique de rupture", "warn": "Non conforme ou marge réduite", "ok": "Poutre conforme"}[level]
    lo = min(0, math.floor(beta))
    span = 8 - lo

    def fr(x):
        return max(0.0, min(1.0, (x - lo) / span))
    util = r["Med"] / r["Mrd"] * 100 if r["Mrd"] > 0 else float("inf")
    util_ok = math.isfinite(util)
    st.markdown(
        '<div class="hcard">'
        + gauge(fr(beta), LEVEL_C["ok"] if beta >= 3.8 else (LEVEL_C["warn"] if beta >= 1.5 else LEVEL_C["crit"]),
                [(0, fr(1.5), LEVEL_C["crit"]), (fr(1.5), fr(3.8), LEVEL_C["warn"]), (fr(3.8), 1, LEVEL_C["ok"])],
                mark=fr(3.8), big=cnt(beta, 2, "−" if beta < 0 else ""), sub="β de Cornell · repère blanc : cible 3,8",
                label=f"Indice de fiabilité β = {beta:.2f}, cible 3,8")
        + f'<div class="hc-body">{_lab("Verdict de sécurité")}<div class="hc-title">{headline}</div>'
        + chip(level, {"ok": "Conforme", "warn": "À surveiller", "crit": "Critique"}[level])
        + (meter("Moment résistant utilisé", util, cnt(util, 0) + " %",
                 f"MEd {r['Med'] / 1e6:.0f} kN·m sur MRd {r['Mrd'] / 1e6:.0f} kN·m") if util_ok else "")
        + '</div></div>',
        unsafe_allow_html=True)
    if crit:
        banner("crit", "RISQUE CRITIQUE : RUPTURE / EFFONDREMENT DE LA POUTRE POSSIBLE", " • " + "<br> • ".join(crit + warn))
    elif warn:
        banner("warn", "Poutre non conforme ou à marge réduite", " • " + "<br> • ".join(warn))
    else:
        banner("ok", "Poutre conforme", f"β = {beta:.2f} ≥ 3,8 ; ELU, ductilité, armatures min/max et flèche vérifiés.")

    section("Résultats du calcul", "Moment sollicitant (MEd) contre moment résistant (MRd), armatures, flèche et fiabilité.")
    flk = r["fleche"] <= r["fleche_lim"]
    kpi_grid([
        ("MEd (ELU)", cnt(r["Med"] / 1e6, 0) + unit("kN·m"), "", "mut", ""),
        ("MRd", cnt(r["Mrd"] / 1e6, 0) + unit("kN·m"), f"{pc(r['Med'] / r['Mrd'], 0)} utilisé" if r["Mrd"] > 0 else "",
         "ok" if r["Med"] <= r["Mrd"] else "bad", "check" if r["Med"] <= r["Mrd"] else "up"),
        ("As requis / retenu", (cnt(r["As_req"], 0) + " / " if r["section_ok"] else "n.d. / ") + cnt(As, 0) + unit("mm²"), "", "mut", ""),
        ("Flèche", cnt(r["fleche"], 1) + unit("mm"), f"limite L/250 = {r['fleche_lim']:.0f}", "ok" if flk else "bad", "check" if flk else "up"),
        ("β de Cornell", cnt(beta, 2, "−" if beta < 0 else ""), "", "mut", ""),
        ("Pf (Cornell / MC)", f"{pf:.1e}", f"MC : {pf_mc:.1e}" if pf_mc > 0 else f"MC : < {1 / n:.0e}", "mut", ""),
    ])

    section("Graphiques", "L'indice β doit dépasser 3,8 ; la marge g doit rester à droite de zéro.")
    g1, g2 = st.columns(2)
    xs = np.linspace(0, L, 120)
    fig = go.Figure()
    fig.add_scatter(x=xs, y=r["qu"] * xs * (L - xs) / 2, fill="tozeroy", name="MEd(x) ELU",
                    line=dict(width=3, color=P["c1"]), fillcolor=_rgba(P["c1"], .25))
    fig.add_hline(y=r["Mrd"] / 1e6, line_color=LEVEL_C["crit"], line_dash="dash", annotation_text="MRd",
                  annotation_font_color=P["text"])
    # l'axe vertical doit laisser voir la ligne MRd, même quand elle est juste au-dessus de la courbe
    fig.update_layout(xaxis_title="x (m)", yaxis=dict(title="kN·m", range=[0, max(r["qu"] * L * L / 8, r["Mrd"] / 1e6) * 1.15]))
    with g1:
        chart(fig, "Moments le long de la poutre", "La courbe doit rester sous la ligne rouge (MRd), le moment que la poutre peut reprendre.", 360)

    mu_g, sd_g = (R0 - muS) / 1e6, math.sqrt(sR ** 2 + sS ** 2) / 1e6
    gx = np.linspace(min(g.min(), -1), g.max(), 300)
    counts, edges = np.histogram(g[:60000], bins=80, density=True)
    centres, larg = (edges[:-1] + edges[1:]) / 2, np.diff(edges)
    fig = go.Figure()
    fig.add_bar(x=centres, y=counts, width=larg * .92, name="Monte-Carlo", opacity=.9,
                marker_color=[LEVEL_C["crit"] if c < 0 else P["c1"] for c in centres])
    fig.add_scatter(x=gx, y=stats.norm.pdf(gx, mu_g, sd_g), name="Normale (Cornell)", line=dict(width=3, color=P["c2"]))
    fig.add_vline(x=0, line_color=LEVEL_C["crit"], line_dash="dash", annotation_text="g = 0 (rupture à gauche)",
                  annotation_font_color=P["text"])
    fig.update_layout(xaxis_title="kN·m")
    with g2:
        chart(fig, "Marge de sécurité g = R − S", "Les barres rouges (g < 0) sont les cas simulés où la poutre cède.", 360)
    st.caption("Cornell : β = (μR − μS) / √(σR² + σS²), Pf = Φ(−β). Monte-Carlo : lois non gaussiennes (log-normale, Gumbel) "
               "pour vérifier la validité de l'hypothèse normale. Modèle pédagogique simplifié, non destiné au dimensionnement réel.")


# ---------------------------------------------------------------------
# 4. DÉPLOIEMENT
# ---------------------------------------------------------------------
def _step(num_, titre, texte):
    st.markdown(f'<div class="step"><span class="step-n">{num_}</span><div><h4>{titre}</h4><p>{texte}</p></div></div>', unsafe_allow_html=True)


def tab_deploy():
    section("Comprendre les indicateurs", "Les mots utilisés dans l'application, en langage simple.")
    st.markdown('<div class="gl">' + "".join(f"<div><b>{esc(k)}</b>{esc(v)}</div>" for k, v in GLOSS.items()) + "</div>", unsafe_allow_html=True)
    section("Mettre l'application en ligne, gratuitement", "Quatre étapes avec GitHub et Streamlit Community Cloud. Il suffit d'un compte GitHub.")
    _step(1, "Préparer les fichiers", "Dans un même dossier, gardez ces trois fichiers. Le fichier de configuration donne les couleurs de base.")
    st.markdown(f'<div class="files"><span>{ico("check", 13)} app_chantier.py</span><span>{ico("check", 13)} requirements.txt</span>'
                f'<span>{ico("check", 13)} .streamlit/config.toml</span></div>', unsafe_allow_html=True)
    st.code("streamlit>=1.40\npandas\nnumpy\nplotly\nscipy", language="text")
    _step(2, "Envoyer le dossier sur GitHub", "Créez un dépôt vide nommé chantier-pro sur github.com, puis lancez ces commandes dans le dossier "
                                              "(remplacez le nom d'utilisateur).")
    st.code('git init\ngit add app_chantier.py requirements.txt .streamlit/config.toml\ngit commit -m "Chantier Pro"\n'
            "git branch -M main\ngit remote add origin https://github.com/<utilisateur>/chantier-pro.git\n"
            "git push -u origin main", language="bash")
    _step(3, "Créer l'application sur Streamlit", "Connectez-vous avec GitHub, cliquez sur « Create app », puis choisissez le dépôt chantier-pro, "
                                                  "la branche main et le fichier principal app_chantier.py. Terminez avec « Deploy ».")
    st.link_button("Ouvrir share.streamlit.io", "https://share.streamlit.io", icon=":material/open_in_new:")
    _step(4, "Mettre à jour plus tard", "Après chaque modification du code, ces trois lignes suffisent : l'application en ligne se met à jour toute seule.")
    st.code('git add .\ngit commit -m "Mise à jour"\ngit push', language="bash")
    st.info("Les modifications faites dans les tableaux ne sont conservées que pendant la session (pas de base de données). "
            "Pour les garder, modifiez les valeurs de DEMO dans le code. Les graphiques animés ont besoin d'internet pour s'afficher.")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    hero()
    # Ce sélecteur est caché (voir .st-key-theme) : les entrées « Chantier / Insights » ajoutées au menu ⋮ cliquent dessus.
    pick = st.segmented_control("Thème", list(THEMES), default=THEME_KEY, key="theme", label_visibility="collapsed",
                                format_func=lambda k: THEMES[k]["name"])
    if pick:                                    # (désélectionner garde le thème actuel)
        st.session_state["_theme"] = pick
        if st.query_params.get("theme") != pick:
            st.query_params["theme"] = pick      # le choix survit à un rechargement de la page
    _theme_bridge()
    # Les réglages sont dans une fenêtre (bouton à droite de la barre du haut), pas dans une barre latérale cachée sur téléphone.
    with st.popover(":material/tune:", help="Réglages de la simulation"):
        n = st.segmented_control("Nombre de simulations N", [20_000, 50_000, 100_000, 200_000, 300_000], default=100_000, key="n_sim",
                                 format_func=lambda v: f"{v:,}".replace(",", " ")) or 100_000
        seed = st.number_input("Graine aléatoire", 0, 10_000, 42)
        st.caption("Plus N est grand, plus le résultat est précis (et plus le calcul est long). "
                   "Même graine = résultats reproductibles (utile pour une soutenance).")
    t = st.tabs([":material/payments: Budget", ":material/monitoring: Risques", ":material/architecture: Poutre", ":material/help: Aide"])
    with t[0]:
        tab_dashboard()
    with t[1]:
        tab_proba(n, int(seed))
    with t[2]:
        tab_beam(n, int(seed))
    with t[3]:
        tab_deploy()
    st.markdown('<div class="foot">Chantier Pro · modèle pédagogique simplifié, non destiné au dimensionnement réel</div>',
                unsafe_allow_html=True)


main()
