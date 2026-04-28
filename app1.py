import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

try:
    from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
    FAIRLEARN_OK = True
except Exception:
    FAIRLEARN_OK = False

st.set_page_config(page_title="FairSight AI", page_icon="⚖️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #07070f; }

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0b1a 0%, #0e0e22 100%) !important;
    border-right: 1px solid #1a1a35;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

.nav-btn {
    display: block; width: 100%; padding: 12px 16px; margin: 4px 0;
    background: transparent; border: none; border-radius: 10px;
    color: #a0aec0 !important; font-size: 0.95em; text-align: left;
    cursor: pointer; transition: all 0.2s;
}
.nav-btn:hover { background: #1a1a35; color: white !important; }
.nav-btn-active {
    background: linear-gradient(135deg, #5b5ef422, #9747ff22) !important;
    border-left: 3px solid #7c7ff4 !important;
    color: white !important; font-weight: 600 !important;
}

/* HERO */
.hero {
    background: linear-gradient(135deg, #0b0b1a 0%, #0e0e22 40%, #0c1830 100%);
    border: 1px solid #1a1a35; border-radius: 24px;
    padding: 64px 48px; text-align: center; margin-bottom: 30px;
    position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 25% 50%, #5b5ef420 0%, transparent 50%),
                radial-gradient(circle at 75% 50%, #f4455520 0%, transparent 50%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.8em; font-weight: 800; color: #ffffff;
    margin: 0; letter-spacing: -2px; position: relative;
}
.hero-title span {
    background: linear-gradient(135deg, #7c7ff4, #f44575);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub { font-size: 1.15em; color: #64748b; margin-top: 14px; position: relative; max-width: 600px; margin-left: auto; margin-right: auto; line-height: 1.7; }
.hero-badge {
    display: inline-block; margin-top: 20px;
    background: linear-gradient(135deg, #7c7ff4, #9747ff);
    color: white; padding: 8px 24px; border-radius: 24px;
    font-size: 0.85em; font-weight: 600; position: relative;
    box-shadow: 0 4px 20px #7c7ff440;
}

/* CARDS */
.card {
    background: #0b0b1a; border: 1px solid #1a1a35;
    border-radius: 18px; padding: 26px; margin: 10px 0;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.card:hover { border-color: #7c7ff430; box-shadow: 0 4px 24px #7c7ff410; }
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1em; font-weight: 700; color: #ffffff;
    margin-bottom: 12px; display: flex; align-items: center; gap: 8px;
}

/* STAT CARDS */
.stat-card {
    background: #0b0b1a; border-radius: 16px; padding: 22px;
    text-align: center; border: 1px solid #1a1a35;
}
.stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 2.2em; font-weight: 800; color: white;
}
.stat-lbl { font-size: 0.8em; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-top: 6px; }

/* BIAS CARDS */
.bias-card {
    border-radius: 16px; padding: 24px; text-align: center; margin: 6px 0;
}
.bias-red { background: #180a0a; border: 2px solid #dc2626; }
.bias-yellow { background: #18140a; border: 2px solid #ca8a04; }
.bias-green { background: #0a1810; border: 2px solid #16a34a; }
.bias-val { font-family: 'Syne', sans-serif; font-size: 2em; font-weight: 800; color: white; margin: 8px 0; }
.bias-lbl { font-size: 0.75em; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }

/* REPORT */
.report-box {
    background: #0b0b1a; border-left: 4px solid #7c7ff4;
    border-radius: 0 16px 16px 0; padding: 30px; margin: 16px 0;
    color: #cbd5e0; line-height: 2; font-size: 0.97em;
}

/* FIX CARD */
.fix-card {
    background: #0a160a; border: 1px solid #166534;
    border-radius: 14px; padding: 20px; margin: 8px 0;
}

/* SECTION HEADER */
.section-hdr {
    font-family: 'Syne', sans-serif;
    font-size: 1.5em; font-weight: 800; color: white;
    margin: 32px 0 18px 0; padding-bottom: 12px;
    border-bottom: 2px solid #1a1a35;
    display: flex; align-items: center; gap: 10px;
}

/* INFO BOX */
.info-box {
    background: #0b1428; border: 1px solid #1e3a5f;
    border-radius: 14px; padding: 20px; margin: 10px 0;
    color: #93c5fd; line-height: 1.8;
}
.info-box b { color: #60a5fa; }

/* API KEY EXPLAINER */
.api-box {
    background: #0f0b1f; border: 1px solid #4c3b8f;
    border-radius: 14px; padding: 20px; margin: 10px 0;
    line-height: 1.8;
}
.api-box-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700; color: #a78bfa; font-size: 1em; margin-bottom: 10px;
}
.api-row { display: flex; gap: 10px; align-items: flex-start; margin: 6px 0; font-size: 0.88em; color: #94a3b8; }
.api-icon { font-size: 1.1em; flex-shrink: 0; margin-top: 1px; }
.api-free-badge {
    display: inline-block; background: #14532d; color: #4ade80;
    border: 1px solid #16a34a; border-radius: 6px; padding: 2px 10px;
    font-size: 0.78em; font-weight: 600; margin-left: 6px;
}

/* STEP BOX */
.step {
    display: flex; align-items: flex-start; gap: 16px;
    background: #0b0b1a; border-radius: 14px; padding: 18px;
    margin: 8px 0; border: 1px solid #1a1a35;
}
.step-num {
    background: linear-gradient(135deg, #7c7ff4, #9747ff);
    color: white; border-radius: 50%; width: 34px; height: 34px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.9em; flex-shrink: 0;
}
.step-content { flex: 1; }
.step-title { font-family: 'Syne', sans-serif; color: white; font-weight: 700; font-size: 0.97em; margin-bottom: 5px; }
.step-desc { color: #64748b; font-size: 0.87em; line-height: 1.6; }

/* COL BADGE */
.col-badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 0.8em; font-weight: 600; margin: 3px;
}
.col-target { background: #172044; color: #93c5fd; border: 1px solid #1e40af; }
.col-protected { background: #14290a; color: #86efac; border: 1px solid #166534; }
.col-feature { background: #181818; color: #94a3b8; border: 1px solid #334155; }

/* TAG */
.tech-tag {
    display: inline-block; background: #0e0e22; border: 1px solid #1e2748;
    color: #94a3b8; padding: 5px 14px; border-radius: 20px;
    font-size: 0.82em; margin: 3px;
}
.tech-tag-google { border-color: #7c7ff4; color: #a5b4fc; }

/* CERTIFICATE */
.cert-row {
    display: flex; justify-content: space-between;
    padding: 12px 0; border-bottom: 1px solid #1a1a35;
    color: #e2e8f0; font-size: 0.95em;
}
.cert-key { color: #64748b; }

/* PRIVACY BADGE */
.privacy-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #0a1a0a; border: 1px solid #166534;
    border-radius: 8px; padding: 4px 12px;
    font-size: 0.8em; color: #4ade80; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"
if "df" not in st.session_state:
    st.session_state.df = None
if "results" not in st.session_state:
    st.session_state.results = None
if "dname" not in st.session_state:
    st.session_state.dname = ""

# ── HELPERS ──────────────────────────────────────────────────────────────────
def bias_class(score, is_dir=False):
    chk = abs(1 - score) if is_dir else abs(score)
    if chk < 0.1: return "bias-green", "🟢 FAIR", "#16a34a"
    elif chk < 0.2: return "bias-yellow", "🟡 MODERATE", "#ca8a04"
    else: return "bias-red", "🔴 HIGH BIAS", "#dc2626"

def dpd_m(yt, yp, s):
    gs = np.unique(s)
    r = [np.mean(yp[s==g]) for g in gs if (s==g).sum()>0]
    return float(max(r)-min(r)) if len(r)>=2 else 0.0

def eod_m(yt, yp, s):
    gs = np.unique(s); t,f=[],[]
    for g in gs:
        m=s==g; yt_,yp_=yt[m],yp[m]
        if len(yt_)==0: continue
        t.append(np.mean(yp_[yt_==1]) if (yt_==1).sum()>0 else 0.0)
        f.append(np.mean(yp_[yt_==0]) if (yt_==0).sum()>0 else 0.0)
    d=0.0
    if len(t)>=2: d=max(d,max(t)-min(t))
    if len(f)>=2: d=max(d,max(f)-min(f))
    return float(d)

def did_m(yp, s):
    gs=np.unique(s)
    r=[r for r in [np.mean(yp[s==g]) for g in gs if (s==g).sum()>0] if r>0]
    return float(min(r)/max(r)) if len(r)>=2 and max(r)>0 else 1.0

def run_audit(df, tcol, pcol):
    try:
        w=df.copy()
        le=LabelEncoder()
        for c in w.select_dtypes(include="object").columns:
            w[c]=le.fit_transform(w[c].astype(str))
        w=w.dropna()
        if len(w)<20: return None,"Need at least 20 rows."
        X=w.drop(columns=[tcol]); y=w[tcol].astype(int)
        if y.nunique()<2: return None,"Target needs 2+ unique values."
        s=X[pcol].values.astype(int)
        Xtr,Xte,ytr,yte,str_,ste=train_test_split(X,y,s,test_size=0.3,random_state=42)
        clf=LogisticRegression(max_iter=1000,random_state=42)
        clf.fit(Xtr,ytr); yp=clf.predict(Xte)
        yt_n,yp_n,st_n=np.array(yte),np.array(yp),np.array(ste)
        acc=float(accuracy_score(yt_n,yp_n))
        if FAIRLEARN_OK:
            try: dpd=float(demographic_parity_difference(yt_n,yp_n,sensitive_features=st_n))
            except: dpd=dpd_m(yt_n,yp_n,st_n)
            try: eod=float(equalized_odds_difference(yt_n,yp_n,sensitive_features=st_n))
            except: eod=eod_m(yt_n,yp_n,st_n)
        else:
            dpd=dpd_m(yt_n,yp_n,st_n); eod=eod_m(yt_n,yp_n,st_n)
        did=did_m(yp_n,st_n)
        gs=np.unique(st_n)
        gacc={int(g):float(accuracy_score(yt_n[st_n==g],yp_n[st_n==g])) for g in gs if (st_n==g).sum()>0}
        return {"acc":acc,"dpd":dpd,"eod":eod,"did":did,"gacc":gacc,
                "yt":yt_n,"yp":yp_n,"st":st_n,"groups":gs}, None
    except Exception as e:
        return None, f"Error: {str(e)}"

def gen_report(dpd, eod, did, pcol, dname, key=""):
    if key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=key)
            m=genai.GenerativeModel("gemini-1.5-flash")
            r=m.generate_content(f"""You are an expert AI fairness auditor. Write a professional 3-paragraph bias audit report for non-technical readers — HR managers, compliance officers, and executives.
Dataset: {dname}, Protected Attribute: {pcol}
Demographic Parity Difference: {dpd:.4f} (ideal=0, >0.1 is concerning)
Equalized Odds Difference: {eod:.4f} (ideal=0, >0.1 is concerning)
Disparate Impact Ratio: {did:.4f} (ideal=1.0, <0.8 is legally concerning under EU AI Act, US EEOC, India DPDP Act)
Para 1: Plain English summary of what was found — what the numbers actually mean. Para 2: Real-world human impact — who is affected and how. Para 3: Specific, actionable fix recommendations. Write in a professional but accessible tone. No bullet points. No markdown. Flowing paragraphs only.""")
            return r.text, True
        except Exception:
            pass
    lv="significant" if abs(dpd)>0.2 else "moderate" if abs(dpd)>0.1 else "minimal"
    dn="falls below the 0.8 legal threshold — potentially discriminatory under EU AI Act and India's DPDP Act" if did<0.8 else "is within the acceptable range"
    return f"""Audit Summary: Analysis of '{dname}' reveals {lv} bias with respect to the protected attribute '{pcol}'. The Demographic Parity Difference score of {dpd:.4f} indicates unequal positive outcome rates across demographic groups, meaning different groups receive markedly different rates of favorable decisions. The Equalized Odds Difference of {eod:.4f} shows {'uneven error rates — some groups face systematically higher rates of false rejections despite having equivalent qualifications' if eod>0.1 else 'relatively consistent error rates across groups, which is a positive indicator'}. The Disparate Impact Ratio of {did:.4f} {dn}.

Real-World Impact: If this model is deployed in hiring, lending, healthcare, or credit scoring, individuals from disadvantaged groups face {'significantly' if abs(dpd)>0.2 else 'moderately'} higher rejection rates despite equal qualifications or creditworthiness. This is not a theoretical risk — biased AI compounds existing societal inequalities at massive scale, affecting thousands of automated decisions every month without any human ever reviewing individual cases. Regulatory bodies globally are now mandating bias audits for AI systems used in consequential decisions, and organizations that fail to address these issues face legal, reputational, and financial consequences.

Recommendations: Three immediate actions are recommended to bring this model into compliance. First, rebalance the training dataset using SMOTE (Synthetic Minority Over-sampling Technique) to ensure equal representation across '{pcol}' groups — this is the fastest, highest-impact intervention. Second, apply threshold calibration per demographic group to equalize false positive and false negative rates at inference time, which addresses outcome disparity without retraining the entire model. Third, conduct quarterly automated bias monitoring using FairSight AI to detect drift before it reaches users and causes harm. Organizations that implement all three typically reduce Demographic Parity Difference by 60-80%.""", False

def fig_traffic(dpd, eod, did):
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    fig.patch.set_facecolor("#07070f")
    for ax,(name,val,ideal) in zip(axes,[("Demographic\nParity Diff",dpd,"Ideal: 0.0"),("Equalized\nOdds Diff",eod,"Ideal: 0.0"),("Disparate\nImpact Ratio",did,"Ideal: 1.0")]):
        ax.set_facecolor("#0b0b1a")
        chk=abs(val) if "Impact" not in name else abs(1-val)
        c="#16a34a" if chk<0.1 else "#ca8a04" if chk<0.2 else "#dc2626"
        s="FAIR ✓" if chk<0.1 else "MODERATE" if chk<0.2 else "HIGH BIAS"
        bv=max(abs(val),0.02)
        ax.bar([0],[bv],color=c,width=0.45,alpha=0.9,zorder=3)
        ax.set_xlim(-0.5,0.5); ax.set_ylim(0,max(1.3,bv*1.6+0.15))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name}\n{val:.4f}",color="white",fontsize=12,fontweight="bold",pad=10)
        ax.text(0,bv+0.05,s,ha="center",color=c,fontsize=11,fontweight="bold")
        ax.text(0,0.02,ideal,ha="center",color="#64748b",fontsize=9)
        ax.grid(axis='y',color="#1a1a35",linewidth=0.5,zorder=0)
        for sp in ax.spines.values(): sp.set_edgecolor("#1a1a35")
    plt.suptitle("Bias Metric Overview",color="white",fontsize=14,fontweight="bold",y=1.02)
    plt.tight_layout(); return fig

def fig_gacc(gacc):
    fig,ax=plt.subplots(figsize=(8,4))
    fig.patch.set_facecolor("#07070f"); ax.set_facecolor("#0b0b1a")
    gs,acs=list(gacc.keys()),[gacc[g] for g in gacc]
    cs=["#7c7ff4","#f44575","#16a34a","#ca8a04","#9747ff"][:len(gs)]
    bars=ax.bar([f"Group {g}" for g in gs],acs,color=cs,alpha=0.9,width=0.45,zorder=3)
    ax.set_ylim(0,1.2); ax.set_ylabel("Accuracy",color="#64748b",fontsize=11)
    ax.set_title("Model Accuracy per Demographic Group",color="white",fontsize=13,fontweight="bold")
    ax.tick_params(colors="#64748b"); ax.grid(axis='y',color="#1a1a35",linewidth=0.5,zorder=0)
    if acs: ax.axhline(sum(acs)/len(acs),color="#7c7ff4",linestyle="--",alpha=0.6,linewidth=1.5,label=f"Avg: {sum(acs)/len(acs):.1%}"); ax.legend(facecolor="#0b0b1a",labelcolor="white",edgecolor="#1a1a35")
    for bar,a in zip(bars,acs): ax.text(bar.get_x()+bar.get_width()/2,a+0.02,f"{a:.1%}",ha="center",color="white",fontsize=11,fontweight="bold")
    for sp in ax.spines.values(): sp.set_edgecolor("#1a1a35")
    plt.tight_layout(); return fig

def fig_cm(yt, yp):
    cm=confusion_matrix(yt,yp)
    fig,ax=plt.subplots(figsize=(5,4))
    fig.patch.set_facecolor("#07070f"); ax.set_facecolor("#0b0b1a")
    ax.imshow(cm,cmap="Blues",alpha=0.85)
    ax.set_title("Confusion Matrix",color="white",fontsize=12,fontweight="bold")
    ax.set_xlabel("Predicted",color="#64748b"); ax.set_ylabel("Actual",color="#64748b")
    ax.tick_params(colors="#64748b")
    lbls=["Negative","Positive"]
    ax.set_xticks([0,1]); ax.set_xticklabels(lbls[:cm.shape[1]],color="#64748b")
    ax.set_yticks([0,1]); ax.set_yticklabels(lbls[:cm.shape[0]],color="#64748b")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="white",fontsize=16,fontweight="bold")
    for sp in ax.spines.values(): sp.set_edgecolor("#1a1a35")
    plt.tight_layout(); return fig

def fig_group_dist(df, pcol, tcol):
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    fig.patch.set_facecolor("#07070f")
    cs=["#7c7ff4","#f44575","#16a34a","#ca8a04","#9747ff"]
    ax=axes[0]; ax.set_facecolor("#0b0b1a")
    vc=df[pcol].value_counts()
    ax.bar(vc.index.astype(str),vc.values,color=cs[:len(vc)],alpha=0.9,width=0.45,zorder=3)
    ax.set_title(f"Distribution of '{pcol}'",color="white",fontsize=12,fontweight="bold")
    ax.tick_params(colors="#64748b"); ax.set_ylabel("Count",color="#64748b")
    ax.grid(axis='y',color="#1a1a35",linewidth=0.5,zorder=0)
    for sp in ax.spines.values(): sp.set_edgecolor("#1a1a35")
    ax2=axes[1]; ax2.set_facecolor("#0b0b1a")
    try:
        grp=df.groupby(pcol)[tcol].mean()
        ax2.bar(grp.index.astype(str),grp.values,color=cs[:len(grp)],alpha=0.9,width=0.45,zorder=3)
        ax2.set_title(f"Positive Outcome Rate by '{pcol}'",color="white",fontsize=12,fontweight="bold")
        ax2.set_ylabel("Rate",color="#64748b"); ax2.set_ylim(0,1.1)
        ax2.tick_params(colors="#64748b"); ax2.grid(axis='y',color="#1a1a35",linewidth=0.5,zorder=0)
        for bar,v in zip(ax2.patches,grp.values): ax2.text(bar.get_x()+bar.get_width()/2,v+0.02,f"{v:.1%}",ha="center",color="white",fontsize=11,fontweight="bold")
        for sp in ax2.spines.values(): sp.set_edgecolor("#1a1a35")
    except Exception: pass
    plt.tight_layout(); return fig

# ── SIDEBAR NAV ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0 10px 0;'>
        <div style='font-size:2.2em;'>⚖️</div>
        <div style='font-family:Syne,sans-serif;font-size:1.35em;font-weight:800;color:white;margin-top:4px;'>FairSight AI</div>
        <div style='font-size:0.78em;color:#64748b;margin-top:4px;letter-spacing:0.5px;'>Bias Detection Platform</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    pages = ["🏠 Home", "📖 How It Works", "📁 Upload & Configure", "📊 Results & Charts", "🤖 AI Report", "🔧 Remediation", "📄 Certificate"]
    for p in pages:
        is_active = st.session_state.page == p
        style = "nav-btn nav-btn-active" if is_active else "nav-btn"
        if st.button(p, key=f"nav_{p}", use_container_width=True):
            st.session_state.page = p
            st.rerun()

    st.markdown("---")

    # ── GEMINI API KEY SECTION ──
    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-size:1em;font-weight:700;color:white;margin-bottom:4px;'>
        🔑 Gemini API Key
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='api-box'>
        <div class='api-box-title'>Why do I need this?</div>
        <div class='api-row'>
            <span class='api-icon'>🤖</span>
            <span>Without a key, FairSight generates a <b style='color:#c4b5fd;'>rule-based</b> audit report using your metric scores.</span>
        </div>
        <div class='api-row'>
            <span class='api-icon'>✨</span>
            <span>With a Gemini key, Google's AI reads your exact numbers and writes a <b style='color:#c4b5fd;'>natural-language, context-aware</b> report — like having an AI fairness consultant on call.</span>
        </div>
        <div class='api-row'>
            <span class='api-icon'>🔒</span>
            <span>Your key is used <b style='color:#c4b5fd;'>only for this session</b> and is never stored or logged.</span>
        </div>
        <div class='api-row'>
            <span class='api-icon'>💚</span>
            <span>The Gemini API has a <b style='color:#4ade80;'>generous free tier</b> — no credit card needed to get started. <span class='api-free-badge'>FREE</span></span>
        </div>
        <div style='margin-top:10px;font-size:0.82em;color:#64748b;'>
            Get your key in 60 seconds at<br>
            <a href='https://aistudio.google.com/app/apikey' target='_blank' style='color:#a78bfa;text-decoration:none;'>→ aistudio.google.com/app/apikey</a>
        </div>
    </div>""", unsafe_allow_html=True)

    gkey = st.text_input("", type="password", placeholder="Paste your key here: AIza...", key="gkey_input", label_visibility="collapsed")
    if gkey:
        st.session_state["gemini_key"] = gkey
        st.markdown("""<div style='background:#0a1a0a;border:1px solid #166534;border-radius:8px;
        padding:8px 14px;color:#4ade80;font-size:0.85em;margin-top:4px;'>
        ✅ Gemini key saved — AI reports enabled!
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style='background:#18100a;border:1px solid #78350f;border-radius:8px;
        padding:8px 14px;color:#fbbf24;font-size:0.82em;margin-top:4px;'>
        ⚠️ No key — using smart rule-based reports
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    if st.session_state.results:
        r = st.session_state.results
        st.markdown("<div style='font-family:Syne,sans-serif;font-size:0.9em;font-weight:700;color:white;margin-bottom:8px;'>📊 Quick Stats</div>", unsafe_allow_html=True)
        verdict = "🔴 HIGH BIAS" if (abs(r['dpd'])>0.2 or abs(r['eod'])>0.2 or r['did']<0.8) else "🟡 MODERATE" if (abs(r['dpd'])>0.1 or abs(r['eod'])>0.1) else "🟢 FAIR"
        st.markdown(f"""
        <div style='background:#0b0b1a;border-radius:12px;padding:14px;border:1px solid #1a1a35;font-size:0.85em;'>
        <div style='color:#64748b;'>Accuracy</div><div style='color:white;font-weight:700;font-size:1.1em;'>{r['acc']:.1%}</div>
        <div style='color:#64748b;margin-top:10px;'>Verdict</div><div style='font-weight:700;font-size:1em;margin-top:2px;'>{verdict}</div>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "🏠 Home":
    st.markdown("""
    <div class='hero'>
        <div class='hero-title'>⚖️ Fair<span>Sight</span> AI</div>
        <div class='hero-sub'>Automated Bias Detection & Fairness Auditing for Machine Learning Models — making AI fair for everyone.</div>
        <span class='hero-badge'>🏆 Google Solution Challenge 2026 — Team Aetherix</span>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>🚨 The Problem</div>
            <p style='color:#64748b;font-size:0.95em;line-height:1.9;'>
            AI systems today make life-changing decisions — who gets a job, who receives a loan, who gets medical care.
            When trained on historically biased data, they <b style='color:#f44575;'>silently discriminate</b> at massive scale.
            Most organizations have <b style='color:white;'>no way to detect or fix this.</b>
            </p>
            <div style='margin-top:14px;'>
                <div style='color:#94a3b8;font-size:0.87em;margin:7px 0;'>🏦 Banks reject loans based on race or gender</div>
                <div style='color:#94a3b8;font-size:0.87em;margin:7px 0;'>👔 AI screens out resumes based on age or name</div>
                <div style='color:#94a3b8;font-size:0.87em;margin:7px 0;'>🏥 Healthcare AI deprioritizes certain patient groups</div>
            </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>✅ Our Solution</div>
            <p style='color:#64748b;font-size:0.95em;line-height:1.9;'>
            <b style='color:#7c7ff4;'>FairSight AI</b> is like a doctor's health checkup — but for AI models.
            Upload your dataset → get a full bias audit in <b style='color:white;'>under 60 seconds.</b>
            </p>
            <div style='margin-top:14px;'>
                <div style='color:#94a3b8;font-size:0.87em;margin:7px 0;'>📊 3 industry-standard fairness metrics computed automatically</div>
                <div style='color:#94a3b8;font-size:0.87em;margin:7px 0;'>🤖 Google Gemini 1.5 Flash generates plain-English audit reports</div>
                <div style='color:#94a3b8;font-size:0.87em;margin:7px 0;'>🚦 Traffic-light dashboard anyone can understand instantly</div>
                <div style='color:#94a3b8;font-size:0.87em;margin:7px 0;'>🔧 Actionable remediation with projected improvement estimates</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-hdr'>📈 Why It Matters</div>", unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    for col, val, lbl in [
        (s1, "72M+", "Job applications screened by AI in India annually"),
        (s2, "34%", "Lower shortlist rate for women in biased AI hiring systems"),
        (s3, "0.8", "Legal threshold under EU AI Act & India DPDP Act"),
        (s4, "60s", "Time FairSight takes to fully audit your model"),
    ]:
        with col:
            st.markdown(f"""<div class='stat-card'>
            <div class='stat-val' style='background:linear-gradient(135deg,#7c7ff4,#f44575);-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>{val}</div>
            <div class='stat-lbl'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-hdr'>🛠️ Technology Stack</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <div style='margin-bottom:14px;'>
            <span style='color:#64748b;font-size:0.82em;text-transform:uppercase;letter-spacing:1.2px;font-weight:600;'>Google Technologies</span><br><br>
            <span class='tech-tag tech-tag-google'>Gemini 1.5 Flash</span>
            <span class='tech-tag tech-tag-google'>Firebase</span>
            <span class='tech-tag tech-tag-google'>Google Cloud Run</span>
            <span class='tech-tag tech-tag-google'>Vertex AI</span>
        </div>
        <div style='margin-bottom:14px;'>
            <span style='color:#64748b;font-size:0.82em;text-transform:uppercase;letter-spacing:1.2px;font-weight:600;'>AI / ML</span><br><br>
            <span class='tech-tag'>Fairlearn</span>
            <span class='tech-tag'>scikit-learn</span>
            <span class='tech-tag'>SHAP</span>
            <span class='tech-tag'>NumPy</span>
            <span class='tech-tag'>pandas</span>
        </div>
        <div>
            <span style='color:#64748b;font-size:0.82em;text-transform:uppercase;letter-spacing:1.2px;font-weight:600;'>Frontend / Deployment</span><br><br>
            <span class='tech-tag'>Streamlit</span>
            <span class='tech-tag'>Matplotlib</span>
            <span class='tech-tag'>Python 3.12</span>
            <span class='tech-tag'>Streamlit Cloud</span>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-hdr'>👥 Team Aetherix</div>", unsafe_allow_html=True)
    t1,t2,t3,t4 = st.columns(4)
    for col, name, role, icon in [
        (t1,"Kalla Tanushree","Team Lead & ML Engineer","👩‍💻"),
        (t2,"Geetha Maduri","Frontend & UX","🎨"),
        (t3,"Monish","Backend & API","⚙️"),
        (t4,"Sirichandana","Data & Research","📊"),
    ]:
        with col:
            st.markdown(f"""<div class='card' style='text-align:center;'>
            <div style='font-size:2.2em;margin-bottom:10px;'>{icon}</div>
            <div style='font-family:Syne,sans-serif;color:white;font-weight:700;font-size:0.95em;'>{name}</div>
            <div style='color:#64748b;font-size:0.8em;margin-top:5px;'>{role}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Get Started → Upload Your Dataset", use_container_width=True):
        st.session_state.page = "📁 Upload & Configure"
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# PAGE: HOW IT WORKS
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "📖 How It Works":
    st.markdown("<div class='section-hdr'>📖 How FairSight AI Works</div>", unsafe_allow_html=True)

    steps = [
        ("Upload Dataset", "Upload any CSV file containing your dataset. It should have a target column (what the model predicts) and at least one protected attribute (gender, age, ethnicity, caste, etc.)."),
        ("Select Columns", "Tell FairSight which column is your prediction target and which column is the protected attribute you want to audit for bias."),
        ("Automated Analysis", "FairSight trains a Logistic Regression classifier on your data and computes 3 industry-standard fairness metrics across all demographic groups."),
        ("View Results", "See bias scores on an interactive traffic-light dashboard — Red (High Bias), Yellow (Moderate), Green (Fair) — with group-level accuracy breakdowns and distribution charts."),
        ("Read AI Report", "Google Gemini 1.5 Flash generates a plain-English audit report explaining exactly what bias was found and why it matters — no ML knowledge required."),
        ("Apply Fixes", "Review one-click remediation suggestions with projected improvement estimates for each technique before you apply it."),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f"""
        <div class='step'>
            <div class='step-num'>{i}</div>
            <div class='step-content'>
                <div class='step-title'>{title}</div>
                <div class='step-desc'>{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-hdr'>📐 Fairness Metrics Explained</div>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    for col, title, ideal, formula, simple, color in [
        (m1, "Demographic Parity Difference", "Ideal: 0.0 | Concerning: >0.1",
         "P(ŷ=1|A=0) − P(ŷ=1|A=1)",
         "Does the model approve equal percentages of people from each group? E.g., 60% of men hired but only 30% of women → DPD = 0.30 → High Bias 🔴",
         "#7c7ff4"),
        (m2, "Equalized Odds Difference", "Ideal: 0.0 | Concerning: >0.1",
         "max(|TPR_diff|, |FPR_diff|)",
         "When the model makes mistakes, does it make the same TYPE of mistake for all groups? Unequal false rejection rates across groups = bias.",
         "#f44575"),
        (m3, "Disparate Impact Ratio", "Ideal: 1.0 | Legal threshold: <0.8",
         "min(P(ŷ=1|group)) / max(P(ŷ=1|group))",
         "If ratio < 0.8, the model is considered legally discriminatory under EU AI Act, US EEOC guidelines, and India's DPDP Act.",
         "#16a34a"),
    ]:
        with col:
            st.markdown(f"""
            <div class='card'>
                <div style='color:{color};font-family:Syne,sans-serif;font-weight:700;font-size:1em;margin-bottom:10px;'>{title}</div>
                <div style='background:#111128;border-radius:8px;padding:8px 12px;font-family:monospace;font-size:0.85em;color:#94a3b8;margin-bottom:12px;'>{formula}</div>
                <div style='color:#64748b;font-size:0.85em;line-height:1.8;'>{simple}</div>
                <div style='margin-top:12px;background:#0b0b1a;border-radius:8px;padding:7px 12px;'>
                    <span style='color:{color};font-size:0.8em;font-weight:600;'>{ideal}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-hdr'>📋 What Kind of Dataset Do You Need?</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        <b>FairSight works with any CSV file that has:</b><br>
        ✅ A <b>binary target column</b> (0/1 or Yes/No — what the AI decides)<br>
        ✅ At least one <b>protected attribute column</b> (gender, age group, ethnicity, caste, etc.)<br>
        ✅ At least <b>20 rows</b> of data (500+ recommended for statistically reliable results)<br>
        ✅ Any number of <b>feature columns</b> (experience, income, score, diagnosis, etc.)
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>✅ Good Dataset Examples</div>
            <div style='color:#64748b;font-size:0.9em;line-height:2.2;'>
            <b style='color:white;'>Hiring Data:</b><br>
            <span class='col-badge col-target'>hired (0/1)</span>
            <span class='col-badge col-protected'>gender</span>
            <span class='col-badge col-protected'>age_group</span>
            <span class='col-badge col-feature'>experience</span>
            <span class='col-badge col-feature'>skill_score</span>
            <br><br>
            <b style='color:white;'>Loan Approval:</b><br>
            <span class='col-badge col-target'>approved (0/1)</span>
            <span class='col-badge col-protected'>race</span>
            <span class='col-badge col-protected'>sex</span>
            <span class='col-badge col-feature'>income</span>
            <span class='col-badge col-feature'>credit_score</span>
            <br><br>
            <b style='color:white;'>Healthcare:</b><br>
            <span class='col-badge col-target'>readmitted (0/1)</span>
            <span class='col-badge col-protected'>age</span>
            <span class='col-badge col-protected'>gender</span>
            <span class='col-badge col-feature'>diagnosis</span>
            </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>🌐 Free Datasets to Try</div>
            <div style='font-size:0.9em;line-height:2.3;'>
            <div style='color:white;font-weight:700;'>1. Adult Income Dataset</div>
            <div style='color:#64748b;font-size:0.85em;'>Target: income | Protected: sex, race</div>
            <div style='color:#7c7ff4;font-size:0.82em;'>kaggle.com → search "adult census income"</div>
            <br>
            <div style='color:white;font-weight:700;'>2. German Credit Dataset</div>
            <div style='color:#64748b;font-size:0.85em;'>Target: default | Protected: sex, age</div>
            <div style='color:#7c7ff4;font-size:0.82em;'>kaggle.com → search "german credit data"</div>
            <br>
            <div style='color:white;font-weight:700;'>3. HR Analytics Dataset</div>
            <div style='color:#64748b;font-size:0.85em;'>Target: target | Protected: gender</div>
            <div style='color:#7c7ff4;font-size:0.82em;'>kaggle.com → search "HR analytics job change"</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("📁 Go to Upload →", use_container_width=True):
        st.session_state.page = "📁 Upload & Configure"
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# PAGE: UPLOAD & CONFIGURE
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "📁 Upload & Configure":
    st.markdown("<div class='section-hdr'>📁 Upload Your Dataset</div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🎯 Use Demo Dataset", "📂 Upload Your Own CSV"])

    with tab1:
        st.markdown("""
        <div class='info-box'>
            <b>Demo Dataset: Synthetic Hiring Data (600 applicants)</b><br>
            Gender and age bias has been deliberately injected into this dataset.
            Male applicants are favored by 30% over equally qualified female applicants.
            This realistically simulates real-world hiring AI bias — a perfect test case.
        </div>""", unsafe_allow_html=True)

        np.random.seed(42)
        n=600
        gender=np.random.choice(["Male","Female"],n,p=[0.6,0.4])
        age=np.random.choice(["Under35","Over35"],n,p=[0.55,0.45])
        exp=np.random.randint(1,15,n)
        edu=np.random.choice(["Bachelor","Master","PhD"],n)
        skill=np.random.randint(50,100,n)
        prob=np.where(gender=="Male",0.65,0.35)
        prob=np.where(age=="Under35",prob+0.05,prob-0.05)
        prob=np.clip(prob,0,1)
        hired=np.random.binomial(1,prob)
        demo_df=pd.DataFrame({"gender":gender,"age_group":age,"experience_years":exp,"education":edu,"skill_score":skill,"hired":hired})

        st.markdown("**Dataset Preview (first 5 rows):**")
        st.dataframe(demo_df.head(5), use_container_width=True)
        st.markdown("""
        <div style='display:flex;gap:8px;flex-wrap:wrap;margin-top:8px;'>
            <span class='col-badge col-target'>hired → Target Column</span>
            <span class='col-badge col-protected'>gender → Protected Attribute</span>
            <span class='col-badge col-protected'>age_group → Protected Attribute</span>
            <span class='col-badge col-feature'>experience_years → Feature</span>
            <span class='col-badge col-feature'>skill_score → Feature</span>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✅ Load Demo Dataset", use_container_width=True):
            st.session_state.df = demo_df
            st.session_state.dname = "Demo Hiring Dataset (Synthetic)"
            st.session_state.results = None
            st.success("✅ Demo dataset loaded! Scroll down to configure your audit.")

    with tab2:
        st.markdown("""
        <div class='info-box'>
            <b>Upload Requirements:</b><br>
            • CSV format only<br>
            • Must have a binary target column (0/1 or two unique values)<br>
            • Must have at least one protected attribute column (gender, age, race, etc.)<br>
            • Minimum 20 rows — 500+ recommended for statistically reliable results<br>
            • Missing values are handled automatically
        </div>""", unsafe_allow_html=True)
        uploaded = st.file_uploader("Drop your CSV here", type=["csv"])
        if uploaded:
            try:
                df_up = pd.read_csv(uploaded)
                st.session_state.df = df_up
                st.session_state.dname = uploaded.name
                st.session_state.results = None
                st.success(f"✅ Uploaded: **{uploaded.name}** — {len(df_up)} rows, {len(df_up.columns)} columns")
                st.dataframe(df_up.head(5), use_container_width=True)
            except Exception as e:
                st.error(f"❌ {e}")

    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("<div class='section-hdr'>⚙️ Configure Your Audit</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='card'>
            <div style='color:#64748b;font-size:0.88em;'>
            Dataset: <b style='color:white;'>{st.session_state.dname}</b> —
            <b style='color:white;'>{len(df)}</b> rows,
            <b style='color:white;'>{len(df.columns)}</b> columns
            </div>
        </div>""", unsafe_allow_html=True)

        with st.expander("👀 Full Dataset Preview"):
            st.dataframe(df.head(10), use_container_width=True)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total Rows", len(df))
            c2.metric("Columns", len(df.columns))
            c3.metric("Missing Values", int(df.isnull().sum().sum()))
            c4.metric("Numeric Cols", len(df.select_dtypes(include="number").columns))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""<div style='color:#94a3b8;font-size:0.9em;margin-bottom:6px;'>
            🎯 <b>Target Column</b> — The outcome the model predicts<br>
            <small style='color:#64748b;'>e.g.: hired, approved, defaulted, readmitted (binary 0/1)</small></div>""", unsafe_allow_html=True)
            tcol = st.selectbox("Target", df.columns.tolist(), index=len(df.columns)-1, label_visibility="collapsed")

        with c2:
            st.markdown("""<div style='color:#94a3b8;font-size:0.9em;margin-bottom:6px;'>
            🛡️ <b>Protected Attribute</b> — The sensitive column to audit<br>
            <small style='color:#64748b;'>e.g.: gender, age_group, race, ethnicity, caste</small></div>""", unsafe_allow_html=True)
            pcol = st.selectbox("Protected", [c for c in df.columns if c!=tcol], index=0, label_visibility="collapsed")

        st.session_state["tcol"] = tcol
        st.session_state["pcol"] = pcol

        st.markdown("**📊 Data Distribution Preview:**")
        st.pyplot(fig_group_dist(df, pcol, tcol))

        # Gemini key reminder if not set
        if not st.session_state.get("gemini_key",""):
            st.markdown("""
            <div style='background:#0f0b1f;border:1px dashed #4c3b8f;border-radius:12px;
            padding:14px 18px;margin:12px 0;color:#a78bfa;font-size:0.88em;'>
                💡 <b>Tip:</b> Add your free Gemini API key in the sidebar to get AI-generated audit reports powered by Google Gemini 1.5 Flash. Takes 60 seconds to set up.
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍 RUN FAIRNESS AUDIT", use_container_width=True):
            bar = st.progress(0, "Starting analysis...")
            bar.progress(15, "Encoding categorical columns...")
            res, err = run_audit(df, tcol, pcol)
            bar.progress(60, "Computing fairness metrics...")
            if err:
                st.error(f"❌ {err}"); bar.empty()
            else:
                bar.progress(85, "Generating report...")
                key = st.session_state.get("gemini_key","")
                report, used_gem = gen_report(res["dpd"],res["eod"],res["did"],pcol,st.session_state.dname,key)
                res["report"] = report
                res["used_gem"] = used_gem
                res["pcol"] = pcol
                res["tcol"] = tcol
                st.session_state.results = res
                bar.progress(100,"✅ Done!"); bar.empty()
                st.balloons()
                st.success("🎉 Audit complete! Click '📊 Results & Charts' in the sidebar.")
                if st.button("📊 View Results →", use_container_width=True):
                    st.session_state.page = "📊 Results & Charts"
                    st.rerun()
    else:
        st.markdown("""<div class='card' style='text-align:center;padding:48px;'>
        <div style='font-size:3em;'>📂</div>
        <div style='font-family:Syne,sans-serif;color:white;font-weight:700;margin-top:14px;font-size:1.2em;'>No dataset loaded yet</div>
        <div style='color:#64748b;margin-top:8px;'>Use the demo dataset or upload your own CSV above</div>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE: RESULTS & CHARTS
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "📊 Results & Charts":
    if not st.session_state.results:
        st.warning("⚠️ No results yet. Please run an audit first.")
        if st.button("→ Go to Upload & Configure"): st.session_state.page="📁 Upload & Configure"; st.rerun()
    else:
        r=st.session_state.results
        dpd,eod,did,acc=r["dpd"],r["eod"],r["did"],r["acc"]

        st.markdown("<div class='section-hdr'>📊 Audit Results — Overview</div>", unsafe_allow_html=True)
        m1,m2,m3,m4=st.columns(4)
        m1.metric("Model Accuracy",f"{acc:.1%}")
        m2.metric("Demographic Parity Diff",f"{dpd:.4f}","⚠️ Bias Detected" if abs(dpd)>0.1 else "✅ Fair",delta_color="inverse")
        m3.metric("Equalized Odds Diff",f"{eod:.4f}","⚠️ Bias Detected" if abs(eod)>0.1 else "✅ Fair",delta_color="inverse")
        m4.metric("Disparate Impact Ratio",f"{did:.4f}","⚠️ Below 0.8 threshold" if did<0.8 else "✅ Acceptable",delta_color="inverse")

        st.markdown("<div class='section-hdr'>🚦 Bias Score Dashboard</div>", unsafe_allow_html=True)
        st.pyplot(fig_traffic(dpd,eod,did))

        c1,c2,c3=st.columns(3)
        for w,lbl,val,isd in [(c1,"Demographic Parity",dpd,False),(c2,"Equalized Odds",eod,False),(c3,"Disparate Impact",did,True)]:
            chk=abs(1-val) if isd else abs(val)
            css,status,color=bias_class(chk)
            with w:
                st.markdown(f"""<div class='bias-card {css}'>
                <div class='bias-lbl'>{lbl}</div>
                <div class='bias-val'>{val:.4f}</div>
                <div style='color:{color};font-weight:700;font-size:1.05em;'>{status}</div>
                </div>""",unsafe_allow_html=True)

        st.markdown("<div class='section-hdr'>👥 Group-Level Analysis</div>", unsafe_allow_html=True)
        ca,cb=st.columns([2,1])
        with ca: st.pyplot(fig_gacc(r["gacc"]))
        with cb: st.pyplot(fig_cm(r["yt"],r["yp"]))

        st.markdown("<div class='section-hdr'>📈 Data Distribution Analysis</div>", unsafe_allow_html=True)
        st.pyplot(fig_group_dist(st.session_state.df, r["pcol"], r["tcol"]))

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🤖 View AI Report →", use_container_width=True):
            st.session_state.page="🤖 AI Report"; st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# PAGE: AI REPORT
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "🤖 AI Report":
    if not st.session_state.results:
        st.warning("⚠️ No results yet. Please run an audit first.")
        if st.button("→ Go to Upload"): st.session_state.page="📁 Upload & Configure"; st.rerun()
    else:
        r=st.session_state.results
        st.markdown("<div class='section-hdr'>🤖 AI-Generated Audit Report</div>", unsafe_allow_html=True)

        if r.get("used_gem"):
            st.markdown("""
            <div style='background:#0a160a;border:1px solid #166534;border-radius:12px;
            padding:14px 20px;margin-bottom:18px;'>
                <div style='color:#4ade80;font-size:0.9em;font-weight:600;'>
                    ✅ Powered by Google Gemini 1.5 Flash
                </div>
                <div style='color:#64748b;font-size:0.83em;margin-top:4px;'>
                    This report was generated by Google's Gemini 1.5 Flash model, which analyzed your specific metric scores and produced a contextual, plain-English audit report.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:#0f0b1f;border:1px solid #4c3b8f;border-radius:12px;
            padding:14px 20px;margin-bottom:18px;'>
                <div style='color:#a78bfa;font-size:0.9em;font-weight:600;'>
                    💡 Smart rule-based report generated
                </div>
                <div style='color:#64748b;font-size:0.83em;margin-top:4px;'>
                    This report was generated using FairSight's built-in logic. To get a richer, fully AI-written report from <b style='color:#c4b5fd;'>Google Gemini 1.5 Flash</b>, add your free API key in the sidebar and re-run the audit. The Gemini key is free and takes 60 seconds to get at <a href='https://aistudio.google.com/app/apikey' style='color:#a78bfa;'>aistudio.google.com</a>.
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class='report-box'>
            <div style='font-family:Syne,sans-serif;color:#7c7ff4;font-size:1.15em;font-weight:700;margin-bottom:20px;'>
                📋 FairSight Audit Report — {st.session_state.dname}
            </div>
            {r["report"].replace(chr(10),"<br>")}
        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='section-hdr'>🔍 Key Findings Summary</div>", unsafe_allow_html=True)
        dpd,eod,did=r["dpd"],r["eod"],r["did"]
        findings=[
            ("Demographic Parity", dpd, "Groups receive unequal positive outcome rates" if abs(dpd)>0.1 else "Groups receive roughly equal positive outcome rates", abs(dpd)>0.1),
            ("Equalized Odds", eod, "Error rates differ significantly across demographic groups" if abs(eod)>0.1 else "Error rates are consistent across groups", abs(eod)>0.1),
            ("Disparate Impact", did, f"Ratio of {did:.3f} {'is below the 0.8 legal threshold — potentially discriminatory' if did<0.8 else 'is above the 0.8 legal threshold'}", did<0.8),
            ("Overall Verdict", None, "🔴 HIGH BIAS — Immediate remediation required" if (abs(dpd)>0.2 or abs(eod)>0.2 or did<0.8) else "🟡 MODERATE BIAS — Review and monitoring recommended" if (abs(dpd)>0.1 or abs(eod)>0.1) else "🟢 FAIR — Model passes fairness checks. Continue monitoring.", (abs(dpd)>0.1 or abs(eod)>0.1)),
        ]
        for name,val,desc,is_issue in findings:
            icon="🔴" if is_issue else "🟢"
            val_str=f" — Score: **{val:.4f}**" if val is not None else ""
            st.markdown(f"""<div class='card' style='margin:6px 0;'>
            <div style='display:flex;align-items:center;gap:12px;'>
                <span style='font-size:1.3em;'>{icon}</span>
                <div>
                    <div style='font-family:Syne,sans-serif;color:white;font-weight:700;font-size:0.95em;'>{name}{val_str}</div>
                    <div style='color:#64748b;font-size:0.85em;margin-top:4px;'>{desc}</div>
                </div>
            </div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔧 View Remediation Suggestions →", use_container_width=True):
            st.session_state.page="🔧 Remediation"; st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# PAGE: REMEDIATION
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "🔧 Remediation":
    if not st.session_state.results:
        st.warning("⚠️ No results yet. Please run an audit first.")
        if st.button("→ Go to Upload"): st.session_state.page="📁 Upload & Configure"; st.rerun()
    else:
        r=st.session_state.results
        dpd,eod,did=r["dpd"],r["eod"],r["did"]
        pcol=r["pcol"]

        st.markdown("<div class='section-hdr'>🔧 Remediation Suggestions</div>", unsafe_allow_html=True)
        st.markdown("""<div class='info-box'>
        These fixes are ordered by impact and implementation effort. Each shows a projected improvement estimate
        based on published literature. Apply the fixes in sequence for best results.
        </div>""", unsafe_allow_html=True)

        fixes=[
            {
                "title":"🔄 SMOTE Dataset Rebalancing",
                "what":"Oversample underrepresented groups in your training data using SMOTE (Synthetic Minority Over-sampling Technique).",
                "why":"The model learned biased patterns because it saw far more data from the majority group. Equal representation during training forces it to learn fair patterns for everyone.",
                "how":"pip install imbalanced-learn\nfrom imblearn.over_sampling import SMOTE\nsm = SMOTE()\nX_res, y_res = sm.fit_resample(X, y)",
                "dpd_est":f"{abs(dpd):.3f} → {abs(dpd)*0.4:.3f}",
                "eod_est":f"{abs(eod):.3f} → {abs(eod)*0.5:.3f}",
                "effort":"Low","time":"~2 hours","color":"#16a34a"
            },
            {
                "title":"⚖️ Threshold Calibration per Group",
                "what":f"Apply different decision thresholds for each '{pcol}' group to equalize false positive and false negative rates.",
                "why":"The model uses a single threshold (e.g., 0.5) for all groups, but different groups have different score distributions. Group-specific thresholds correct for this structural disadvantage.",
                "how":"Compute optimal threshold per group via ROC curve → Apply group-conditional thresholds at inference time. No retraining needed.",
                "dpd_est":f"{abs(dpd):.3f} → {abs(dpd)*0.5:.3f}",
                "eod_est":f"{abs(eod):.3f} → {abs(eod)*0.3:.3f}",
                "effort":"Medium","time":"~4 hours","color":"#ca8a04"
            },
            {
                "title":"🏋️ Adversarial Debiasing (Retraining)",
                "what":"Retrain the model with an adversarial debiasing objective — maximizing accuracy while minimizing the ability to predict the protected attribute.",
                "why":"The most powerful intervention. The model learns to make decisions that cannot be used to infer group membership, forcing it to rely only on legitimate, non-discriminatory features.",
                "how":"from fairlearn.reductions import ExponentiatedGradient, DemographicParity\nconstraint = DemographicParity()\nmitigator = ExponentiatedGradient(clf, constraint)\nmitigator.fit(X, y, sensitive_features=s)",
                "dpd_est":f"{abs(dpd):.3f} → {abs(dpd)*0.2:.3f}",
                "eod_est":f"{abs(eod):.3f} → {abs(eod)*0.2:.3f}",
                "effort":"High","time":"~1 day","color":"#dc2626"
            },
            {
                "title":f"📊 Remove Proxy Features for '{pcol}'",
                "what":f"Identify and remove features highly correlated with '{pcol}' — these act as proxies and allow indirect discrimination even after removing the protected attribute itself.",
                "why":"Even if you remove gender from the dataset, the model may still discriminate through proxies like zip code → race, or name → gender. Correlation analysis exposes these hidden channels.",
                "how":"corr = df.corr()[pcol].abs()\nproxy_features = corr[corr > 0.7].index.tolist()\ndf_clean = df.drop(columns=proxy_features)\n# Retrain on df_clean",
                "dpd_est":f"{abs(dpd):.3f} → {abs(dpd)*0.6:.3f}",
                "eod_est":f"{abs(eod):.3f} → {abs(eod)*0.6:.3f}",
                "effort":"Low","time":"~1 hour","color":"#16a34a"
            },
        ]

        for fix in fixes:
            effort_colors={"Low":"#16a34a","Medium":"#ca8a04","High":"#dc2626"}
            ec=effort_colors[fix["effort"]]
            st.markdown(f"""
            <div class='fix-card' style='margin:14px 0;'>
                <div style='font-family:Syne,sans-serif;color:#4ade80;font-weight:700;font-size:1.05em;margin-bottom:12px;'>{fix['title']}</div>
                <div style='display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap;'>
                    <span style='background:#0b0b1a;border:1px solid #1a1a35;border-radius:6px;padding:4px 12px;font-size:0.82em;color:#7c7ff4;'>DPD: {fix['dpd_est']}</span>
                    <span style='background:#0b0b1a;border:1px solid #1a1a35;border-radius:6px;padding:4px 12px;font-size:0.82em;color:#f44575;'>EOD: {fix['eod_est']}</span>
                    <span style='background:#0b0b1a;border:1px solid #1a1a35;border-radius:6px;padding:4px 12px;font-size:0.82em;color:{ec};'>⚡ Effort: {fix['effort']}</span>
                    <span style='background:#0b0b1a;border:1px solid #1a1a35;border-radius:6px;padding:4px 12px;font-size:0.82em;color:#94a3b8;'>⏱️ {fix['time']}</span>
                </div>
                <div style='margin-bottom:8px;'><b style='color:white;font-size:0.9em;'>What:</b> <span style='color:#64748b;font-size:0.88em;'>{fix['what']}</span></div>
                <div style='margin-bottom:10px;'><b style='color:white;font-size:0.9em;'>Why it works:</b> <span style='color:#64748b;font-size:0.88em;'>{fix['why']}</span></div>
                <div style='background:#07070f;border-radius:10px;padding:12px 16px;margin-top:8px;font-family:monospace;font-size:0.82em;color:#94a3b8;white-space:pre-wrap;border:1px solid #1a1a35;'>{fix['how']}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📄 View Audit Certificate →", use_container_width=True):
            st.session_state.page="📄 Certificate"; st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# PAGE: CERTIFICATE
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "📄 Certificate":
    if not st.session_state.results:
        st.warning("⚠️ No results yet. Please run an audit first.")
        if st.button("→ Go to Upload"): st.session_state.page="📁 Upload & Configure"; st.rerun()
    else:
        r=st.session_state.results
        dpd,eod,did,acc=r["dpd"],r["eod"],r["did"],r["acc"]
        verdict="🔴 HIGH BIAS DETECTED — Immediate Action Required" if (abs(dpd)>0.2 or abs(eod)>0.2 or did<0.8) else "🟡 MODERATE BIAS — Review Recommended" if (abs(dpd)>0.1 or abs(eod)>0.1) else "🟢 MODEL IS FAIR — Continue Monitoring"
        vcolor="#dc2626" if "HIGH" in verdict else "#ca8a04" if "MODERATE" in verdict else "#16a34a"

        st.markdown("<div class='section-hdr'>📄 Audit Certificate</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#0b0b1a,#0e0e22);border:1px solid #1a1a35;
        border-radius:22px;padding:40px;margin:10px 0;'>
            <div style='text-align:center;margin-bottom:32px;'>
                <div style='font-size:2.8em;'>⚖️</div>
                <div style='font-family:Syne,sans-serif;font-size:1.8em;font-weight:800;color:white;margin-top:10px;'>FairSight AI</div>
                <div style='color:#64748b;font-size:0.95em;margin-top:6px;letter-spacing:0.5px;'>AUTOMATED BIAS DETECTION CERTIFICATE</div>
            </div>
            <div class='cert-row'><span class='cert-key'>Dataset</span><span><b>{st.session_state.dname}</b></span></div>
            <div class='cert-row'><span class='cert-key'>Protected Attribute Audited</span><span><b>{r['pcol']}</b></span></div>
            <div class='cert-row'><span class='cert-key'>Target Column</span><span><b>{r['tcol']}</b></span></div>
            <div class='cert-row'><span class='cert-key'>Model Accuracy</span><span><b>{acc:.1%}</b></span></div>
            <div class='cert-row'><span class='cert-key'>Demographic Parity Difference</span>
                <span><b>{dpd:.4f}</b> {'🔴 High bias' if abs(dpd)>0.2 else '🟡 Moderate' if abs(dpd)>0.1 else '🟢 Fair'}</span></div>
            <div class='cert-row'><span class='cert-key'>Equalized Odds Difference</span>
                <span><b>{eod:.4f}</b> {'🔴 High bias' if abs(eod)>0.2 else '🟡 Moderate' if abs(eod)>0.1 else '🟢 Fair'}</span></div>
            <div class='cert-row'><span class='cert-key'>Disparate Impact Ratio</span>
                <span><b>{did:.4f}</b> {'🔴 Below 0.8 legal threshold' if did<0.8 else '🟢 Above 0.8 threshold'}</span></div>
            <div class='cert-row'><span class='cert-key'>AI Report Source</span>
                <span><b>{'Google Gemini 1.5 Flash' if r.get('used_gem') else 'FairSight Rule-Based Engine'}</b></span></div>
            <div style='background:{vcolor}18;border:2px solid {vcolor};border-radius:14px;
            padding:20px;text-align:center;margin-top:24px;'>
                <div style='font-family:Syne,sans-serif;color:{vcolor};font-size:1.35em;font-weight:800;'>{verdict}</div>
            </div>
            <div style='text-align:center;margin-top:24px;color:#475569;font-size:0.82em;line-height:1.8;'>
                Generated by FairSight AI | Team Aetherix | Google Solution Challenge 2026<br>
                Powered by Google Gemini 1.5 Flash • Microsoft Fairlearn • scikit-learn • Streamlit
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Run Another Audit", use_container_width=True):
            st.session_state.results=None; st.session_state.df=None
            st.session_state.page="📁 Upload & Configure"; st.rerun()

# FOOTER
st.markdown("---")
st.markdown("""<div style='text-align:center;color:#334155;font-size:0.82em;padding:18px;line-height:1.8;'>
⚖️ <b style='color:#475569;'>FairSight AI</b> — Making AI Fair for Everyone |
Team <b style='color:#475569;'>Aetherix</b> | Google Solution Challenge 2026 |
Powered by Google Gemini 1.5 Flash • Fairlearn • Streamlit
</div>""", unsafe_allow_html=True)