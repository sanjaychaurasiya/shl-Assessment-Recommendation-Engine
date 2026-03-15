import streamlit as st
import anthropic
import json
import uuid
import hashlib
from datetime import datetime

#  Optional provider SDKs (imported lazily) 
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

#  Page config 
st.set_page_config(
    page_title="SHL Assessment Recommendation Engine",
    page_icon="🎯",
    layout="wide",
)

#  Custom CSS 
st.markdown("""
<style>
    .main-header{font-size:2rem;font-weight:600;color:#1a1a2e;margin-bottom:.25rem}
    .sub-header{font-size:1rem;color:#555;margin-bottom:2rem}
    .card{background:#fff;border:1px solid #e0e0e0;border-radius:12px;padding:1.25rem 1.5rem;margin-bottom:1rem;box-shadow:0 1px 3px rgba(0,0,0,.06)}
    .card-name{font-size:1.1rem;font-weight:600;color:#1a1a2e;margin-bottom:.25rem}
    .card-rationale{font-size:.9rem;color:#444;line-height:1.6;margin:.5rem 0 .75rem}
    .badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:.75rem;font-weight:600}
    .badge-core{background:#fce8e8;color:#b91c1c}
    .badge-highly{background:#fef3c7;color:#92400e}
    .badge-supporting{background:#e5e7eb;color:#374151}
    .type-badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:.75rem;font-weight:500}
    .type-cognitive{background:#dbeafe;color:#1e40af}
    .type-personality{background:#ede9fe;color:#5b21b6}
    .type-behavioral{background:#d1fae5;color:#065f46}
    .type-skills{background:#fef9c3;color:#78350f}
    .type-simulation{background:#ffe4e6;color:#9f1239}
    .type-360{background:#dcfce7;color:#166534}
    .type-interview{background:#fce7f3;color:#831843}
    .type-solution{background:#f3f4f6;color:#1f2937}
    .meta-tag{display:inline-block;padding:2px 8px;border-radius:6px;font-size:.75rem;background:#f3f4f6;color:#374151;border:1px solid #e5e7eb;margin:2px}
    .summary-box{background:#f0f4ff;border-left:4px solid #4f46e5;border-radius:0 8px 8px 0;padding:1rem 1.25rem;font-size:.95rem;color:#1e1b4b;line-height:1.7;margin-bottom:1.5rem}
    .section-label{font-size:.7rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#9ca3af;margin-bottom:.75rem}
    .provider-pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:.7rem;font-weight:600}
    .provider-anthropic{background:#fef3c7;color:#92400e}
    .provider-openai{background:#d1fae5;color:#065f46}
    .provider-gemini{background:#dbeafe;color:#1e40af}
    .stButton>button{border-radius:999px!important;font-size:.8rem!important}
</style>
""", unsafe_allow_html=True)


#  Provider registry 
PROVIDERS = {
    "Anthropic (Claude)": {
        "key":         "anthropic",
        "models":      ["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-haiku-4-5-20251001"],
        "placeholder": "sk-ant-...",
        "pill_class":  "provider-anthropic",
        "label":       "Anthropic",
    },
    "OpenAI (GPT)": {
        "key":         "openai",
        "models":      ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo-instruct", "gpt-3.5-turbo-0613"],
        "placeholder": "sk-...",
        "pill_class":  "provider-openai",
        "label":       "OpenAI",
    },
    "Google (Gemini)": {
        "key":         "gemini",
        "models":      ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"],
        "placeholder": "AIza...",
        "pill_class":  "provider-gemini",
        "label":       "Gemini",
    },
}

#  System prompt (SHL catalogue) 
SYSTEM_PROMPT = """You are an expert SHL assessment consultant with deep knowledge of SHL's full product catalogue.
You recommend the most appropriate SHL assessments for hiring and talent development needs.

SHL PRODUCT CATALOGUE (comprehensive):

COGNITIVE ASSESSMENTS (SHL Verify range):
- Verify G+ (General Ability): Combined numerical + inductive reasoning. Adaptive/IRT. Best for graduate, professional, manager roles. Remote-compatible.
- Verify Numerical Reasoning: Interprets charts, tables, financial data. For roles requiring data analysis.
- Verify Verbal Reasoning: Evaluates written communication comprehension.
- Verify Inductive Reasoning: Abstract pattern recognition, learning agility. For tech, analytical, graduate roles.
- Verify Deductive Reasoning: Logical deduction and rule application.
- Verify Mechanical Comprehension: Physical/mechanical principles. For engineering, trades, manufacturing.
- Verify Checking: Accuracy/attention to detail. For clerical, data entry, admin.

PERSONALITY ASSESSMENTS:
- OPQ32: Flagship 32-dimension personality tool. Relationships, thinking style, feelings & emotions. Gold standard for manager+ roles.
- MQ (Motivational Questionnaire): Measures 18 motivation dimensions. For retention, culture fit, engagement. Often paired with OPQ.
- OPQ32r: Shorter OPQ. Good for volume hiring of professional roles.

BEHAVIORAL ASSESSMENTS:
- SJT (Situational Judgment Tests): Scenario-based judgment and decision-making. Role-specific and customisable.
- Manager SJT: Leadership situations and people management decisions.
- Customer Service SJT: Service orientation and handling customer interactions.

JOB-FOCUSED ASSESSMENTS (JFA) — Pre-packaged multi-tool solutions:
- Account Manager Solution: Cognitive + personality + competency. B2B sales.
- Branch Manager Solution: Banking branch management.
- Contact Center Agent Solution: Behavioral + simulation. Volume-friendly.
- Cashier Solution: Behavioral + ability. Retail front-line.
- Graduate Solution: OPQ + Verify G+ + SJT. Best-practice graduate hiring.
- Administrative Professional Solution: Ability + knowledge + personality.
- IT Professional Solution: Technical + reasoning + personality.
- Supervisor/Manager Solution: Behavioral + ability + personality.

SKILLS & SIMULATIONS:
- Coding Simulations: Live coding IDE in 40+ languages. AI-powered scoring.
- Call Center Simulations: Realistic customer interaction scenarios, email-handling, multi-tasking.
- Business Skills Assessments: MS Office, typing, data entry.
- Language Evaluation: AI-powered spoken and written language proficiency.
- Technical Skills Knowledge Tests: 1200+ topics — accounting, IT, healthcare, finance, HR, legal, engineering.

VIDEO INTERVIEWS:
- Smart Interview On-Demand: AI-powered async video, early-stage screening.
- Smart Interview Live: Enhanced live video with structured guides and real-time AI analysis.

360 FEEDBACK:
- SHL 360: Multi-rater feedback. Competency-based (UCF). Mobile-optimised.

ASSESSMENT & DEVELOPMENT CENTERS:
- Virtual Assessment & Development Center (VADC): Exercises, role plays, in-tray simulations, competency interviews.
- In-Tray / e-Tray Exercises: Prioritisation exercises simulating managerial workload.
- Presentation Exercises: Business case presentations assessed by trained assessors.

SPECIAL-PURPOSE:
- Global Skills Development Report: OPQ + Verify G+ combined development profile.
- Apprentice 8.0 JFA: Apprentice and vocational entry-level roles.
- AMCAT: High-volume hiring in emerging markets.

Recommend 3–6 SHL assessments. For EACH provide:
1. name — exact SHL product name
2. type — one of: Cognitive Assessment, Personality Assessment, Behavioral Assessment, Skills Assessment, Simulation, 360 Feedback, Video Interview, Job-Focused Solution
3. priority — one of: Core, Highly recommended, Supporting
4. rationale — 2–3 sentences
5. measures — list of 3–5 short label strings
6. stage — e.g. "Early screening", "Final shortlist", "Development"
7. duration — approximate completion time

Respond ONLY in valid JSON. No markdown, no preamble. Format:
{
  "summary": "...",
  "recommendations": [
    {
      "name": "...",
      "type": "...",
      "priority": "...",
      "rationale": "...",
      "measures": ["...", "...", "..."],
      "stage": "...",
      "duration": "..."
    }
  ]
}"""

#  Badge maps 
TYPE_CLASS = {
    "Cognitive Assessment":  "type-cognitive",
    "Personality Assessment":"type-personality",
    "Behavioral Assessment": "type-behavioral",
    "Skills Assessment":     "type-skills",
    "Simulation":            "type-simulation",
    "360 Feedback":          "type-360",
    "Video Interview":       "type-interview",
    "Job-Focused Solution":  "type-solution",
}
PRIORITY_CLASS = {
    "Core":               "badge-core",
    "Highly recommended": "badge-highly",
    "Supporting":         "badge-supporting",
}

#  Quick-fill templates 
TEMPLATES = {
    "Software Engineer":   dict(job_title="Software Engineer",       job_level="Professional / Individual contributor", industry="Technology",           use_case="External hiring / Talent acquisition", priorities="Coding ability, logical reasoning, problem-solving and collaboration in agile teams."),
    "Sales Manager":       dict(job_title="Sales Manager",           job_level="Manager",                               industry="Banking / Finance",     use_case="External hiring / Talent acquisition", priorities="Negotiation, resilience, customer orientation and ability to lead a sales team."),
    "Call Center Agent":   dict(job_title="Customer Service Agent",  job_level="Entry-level",                           industry="Telecommunications",    use_case="Volume / High-volume hiring",          priorities="Communication skills, customer focus, attention to detail, handling difficult conversations."),
    "Senior Leader":       dict(job_title="VP of Operations",        job_level="Senior manager / Director",             industry="Professional Services", use_case="Leadership development",               priorities="Leadership style, strategic thinking, executive presence, C-suite readiness."),
    "Graduate Hire":       dict(job_title="Graduate Analyst",        job_level="Graduate",                              industry="Banking / Finance",     use_case="Graduate / Early careers hiring",      priorities="Cognitive potential, personality fit, motivation, fairness and inclusivity at scale."),
    "HiPo Identification": dict(job_title="Mid-level Manager",       job_level="Manager",                               industry="Technology",            use_case="High potential identification",        priorities="Learning agility, ambition, leadership competencies for senior role readiness."),
}


#  Session state helpers 
def _init_state():
    defaults = {
        "user_id":       str(uuid.uuid4())[:8],
        "user_name":     "",
        "provider_name": "Anthropic (Claude)",
        "model_name":    PROVIDERS["Anthropic (Claude)"]["models"][0],
        "api_key":       "",
        "job_title":     "",
        "job_level":     "",
        "industry":      "",
        "use_case":      "",
        "priorities":    "",
        "result":        None,
        "error":         None,
        "history":       [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def _save_to_history(result: dict):
    entry = {
        "id":        str(uuid.uuid4())[:6],
        "ts":        datetime.now().strftime("%d %b %Y %H:%M"),
        "job_title": st.session_state.job_title,
        "job_level": st.session_state.job_level,
        "industry":  st.session_state.industry,
        "use_case":  st.session_state.use_case,
        "provider":  PROVIDERS[st.session_state.provider_name]["label"],
        "model":     st.session_state.model_name,
        "result":    result,
    }
    st.session_state.history.insert(0, entry)
    st.session_state.history = st.session_state.history[:20]

def _load_history_entry(entry: dict):
    st.session_state.job_title = entry["job_title"]
    st.session_state.job_level = entry["job_level"]
    st.session_state.industry  = entry["industry"]
    st.session_state.use_case  = entry["use_case"]
    st.session_state.result    = entry["result"]
    st.session_state.error     = None

def apply_template(name: str):
    for k, v in TEMPLATES[name].items():
        st.session_state[k] = v


#  LLM dispatcher 
def call_llm(provider_key: str, model: str, api_key: str, user_msg: str) -> dict:
    if provider_key == "anthropic":
        client   = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model, max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = response.content[0].text.strip()

    elif provider_key == "openai":
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed — run: pip install openai")
        client   = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model, max_tokens=1500,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
        )
        raw = response.choices[0].message.content.strip()

    elif provider_key == "gemini":
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package not installed — run: pip install google-generativeai")
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=SYSTEM_PROMPT,
        )
        response = gemini_model.generate_content(user_msg)
        raw = response.text.strip()

    else:
        raise ValueError(f"Unknown provider: {provider_key}")

    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


#  Card renderer 
def render_card(rec: dict) -> str:
    type_cls = TYPE_CLASS.get(rec.get("type", ""), "type-solution")
    pri_cls  = PRIORITY_CLASS.get(rec.get("priority", ""), "badge-supporting")
    chips    = " ".join(f'<span class="meta-tag">{m}</span>' for m in rec.get("measures", []))
    stage    = f'<span class="meta-tag">📍 {rec["stage"]}</span>'    if rec.get("stage")    else ""
    duration = f'<span class="meta-tag">⏱ {rec["duration"]}</span>' if rec.get("duration") else ""
    return f"""
<div class="card">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;flex-wrap:wrap;">
    <span class="type-badge {type_cls}">{rec.get('type','')}</span>
    <span class="badge {pri_cls}">{rec.get('priority','')}</span>
  </div>
  <div class="card-name">{rec.get('name','')}</div>
  <div class="card-rationale">{rec.get('rationale','')}</div>
  <div>{chips} {stage} {duration}</div>
</div>"""


# 
# UI
# 
_init_state()

st.markdown('<div class="main-header">🎯 SHL Assessment Recommendation Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Describe your hiring or development need and get science-backed assessment recommendations powered by your choice of AI provider.</div>', unsafe_allow_html=True)

#  Sidebar 
with st.sidebar:

    # Session identity
    st.header("👤 Session")
    name_val = st.text_input("Your name (optional)", value=st.session_state.user_name, placeholder="e.g. Alex")
    st.session_state.user_name = name_val
    greeting = f"Hi, **{name_val}**! " if name_val else ""
    st.caption(f"{greeting}Session ID: `{st.session_state.user_id}`")
    if st.button("🔄 Reset session", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.divider()

    # Provider selection
    st.header("🤖 AI Provider")
    provider_names = list(PROVIDERS.keys())
    prev_provider  = st.session_state.provider_name

    selected_provider = st.selectbox(
        "LLM provider", provider_names,
        index=provider_names.index(prev_provider) if prev_provider in provider_names else 0,
    )
    # Reset model + key when provider changes
    if selected_provider != prev_provider:
        st.session_state.provider_name = selected_provider
        st.session_state.model_name    = PROVIDERS[selected_provider]["models"][0]
        st.session_state.api_key       = ""

    pinfo = PROVIDERS[selected_provider]

    # Model selector
    model_opts = pinfo["models"]
    selected_model = st.selectbox(
        "Model", model_opts,
        index=model_opts.index(st.session_state.model_name) if st.session_state.model_name in model_opts else 0,
    )
    st.session_state.model_name = selected_model

    # API key
    api_key_val = st.text_input(
        f"{pinfo['label']} API key",
        type="password",
        value=st.session_state.api_key,
        placeholder=pinfo["placeholder"],
    )
    st.session_state.api_key = api_key_val

    if pinfo["key"] == "openai" and not OPENAI_AVAILABLE:
        st.warning("Install: `pip install openai`", icon="⚠️")
    if pinfo["key"] == "gemini" and not GEMINI_AVAILABLE:
        st.warning("Install: `pip install google-generativeai`", icon="⚠️")

    st.caption("Key used only for this session — never stored.")
    st.divider()

    # Session history
    st.header("🕑 Session history")
    if not st.session_state.history:
        st.caption("No searches yet.")
    else:
        for i, entry in enumerate(st.session_state.history):
            # look up pill class by label
            pill_cls = next(
                (v["pill_class"] for v in PROVIDERS.values() if v["label"] == entry.get("provider", "")),
                "provider-anthropic",
            )
            title = entry["job_title"] or entry["use_case"] or "Unnamed"
            c1, c2 = st.columns([5, 2])
            c1.markdown(f"**{title}**")
            c1.markdown(
                f'<span style="font-size:.72rem;color:#94a3b8">{entry["ts"]} · '
                f'<span class="provider-pill {pill_cls}">{entry["provider"]}</span></span>',
                unsafe_allow_html=True,
            )
            if c2.button("Load", key=f"hist_{i}_{entry['id']}"):
                _load_history_entry(entry)
                st.rerun()

        if st.button("🗑 Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    st.divider()
    st.header("🏷️ Priority legend")
    st.markdown("- 🔴 **Core** — essential\n- 🟡 **Highly recommended** — strong addition\n- ⚪ **Supporting** — contextual / optional")


#  Quick-fill templates 
st.markdown('<div class="section-label">Quick-fill templates</div>', unsafe_allow_html=True)
t_cols = st.columns(len(TEMPLATES))
for col, name in zip(t_cols, TEMPLATES):
    with col:
        if st.button(name, key=f"tpl_{name}", use_container_width=True):
            apply_template(name)

st.divider()

#  Input form 
col1, col2 = st.columns(2)

with col1:
    st.session_state.job_title = st.text_input(
        "Job title / role", value=st.session_state.job_title,
        placeholder="e.g. Sales Manager, Software Engineer",
    )
    industry_opts = ["", "Banking / Finance", "Technology", "Retail", "Healthcare",
                     "Manufacturing", "Telecommunications", "Insurance", "Oil & Gas",
                     "Hospitality", "Professional Services", "Public Sector", "Other"]
    st.session_state.industry = st.selectbox(
        "Industry", industry_opts,
        index=industry_opts.index(st.session_state.industry) if st.session_state.industry in industry_opts else 0,
    )

with col2:
    level_opts = ["", "Entry-level", "Graduate", "Professional / Individual contributor",
                  "Supervisor / Front-line manager", "Manager", "Senior manager / Director",
                  "Executive / C-suite", "General population / Volume hiring"]
    st.session_state.job_level = st.selectbox(
        "Job level", level_opts,
        index=level_opts.index(st.session_state.job_level) if st.session_state.job_level in level_opts else 0,
    )
    use_case_opts = ["", "External hiring / Talent acquisition", "Volume / High-volume hiring",
                     "Graduate / Early careers hiring", "Leadership development",
                     "High potential identification", "Succession planning",
                     "Talent mobility / Internal moves", "Skills gap analysis",
                     "360 feedback / Development"]
    st.session_state.use_case = st.selectbox(
        "Use case", use_case_opts,
        index=use_case_opts.index(st.session_state.use_case) if st.session_state.use_case in use_case_opts else 0,
    )

st.session_state.priorities = st.text_area(
    "What do you most need to evaluate? (optional)",
    value=st.session_state.priorities,
    placeholder="e.g. Strong analytical thinking, customer orientation, ability to manage pressure…",
    height=100,
)

# Active provider indicator
pinfo_cur = PROVIDERS[st.session_state.provider_name]
st.markdown(
    f'Using <span class="provider-pill {pinfo_cur["pill_class"]}">{pinfo_cur["label"]}</span> · '
    f'<code>{st.session_state.model_name}</code>',
    unsafe_allow_html=True,
)

run = st.button("🔍 Get assessment recommendations", type="primary", use_container_width=True)

#  Inference 
if run:
    if not st.session_state.api_key:
        st.error(f"Please enter your {pinfo_cur['label']} API key in the sidebar.")
    elif not (st.session_state.job_title or st.session_state.job_level or st.session_state.use_case):
        st.warning("Please fill in at least the job title, level, or use case.")
    else:
        user_msg = (
            f"Job title: {st.session_state.job_title or 'Not specified'}\n"
            f"Job level: {st.session_state.job_level or 'Not specified'}\n"
            f"Industry:  {st.session_state.industry  or 'Not specified'}\n"
            f"Use case:  {st.session_state.use_case  or 'Not specified'}\n"
            f"Additional priorities: {st.session_state.priorities or 'None provided'}\n\n"
            "Please recommend the most appropriate SHL assessments for this need."
        )
        with st.spinner(f"Calling {pinfo_cur['label']} ({st.session_state.model_name})…"):
            try:
                result = call_llm(
                    provider_key=pinfo_cur["key"],
                    model=st.session_state.model_name,
                    api_key=st.session_state.api_key,
                    user_msg=user_msg,
                )
                st.session_state.result = result
                st.session_state.error  = None
                _save_to_history(result)
            except json.JSONDecodeError as e:
                st.session_state.error  = f"Could not parse AI response as JSON: {e}"
                st.session_state.result = None
            except ImportError as e:
                st.session_state.error  = str(e)
                st.session_state.result = None
            except Exception as e:
                st.session_state.error  = str(e)
                st.session_state.result = None

#  Error 
if st.session_state.error:
    st.error(f"Error: {st.session_state.error}")

#  Results 
if st.session_state.result:
    result = st.session_state.result
    recs   = result.get("recommendations", [])
    label  = " · ".join(filter(None, [
        st.session_state.job_title,
        st.session_state.job_level,
        st.session_state.industry,
    ])) or "Assessment plan"

    st.divider()
    st.markdown(f'<div class="section-label">Assessment plan — {label}</div>', unsafe_allow_html=True)

    if result.get("summary"):
        st.markdown(f'<div class="summary-box">{result["summary"]}</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Assessments recommended",  len(recs))
    m2.metric("Core assessments",         sum(1 for r in recs if r.get("priority") == "Core"))
    m3.metric("Assessment types covered", len({r.get("type") for r in recs}))
    st.markdown("---")

    for section_title, pred in [
        ("#### Core assessments",           lambda r: r.get("priority") == "Core"),
        ("#### Additional recommendations", lambda r: r.get("priority") != "Core"),
    ]:
        subset = [r for r in recs if pred(r)]
        if subset:
            st.markdown(section_title)
            for rec in subset:
                st.markdown(render_card(rec), unsafe_allow_html=True)

    st.divider()
    st.download_button(
        label="⬇️ Download recommendation plan (JSON)",
        data=json.dumps(result, indent=2),
        file_name=f"shl_plan_{label.replace(' · ','_').replace(' ','_').lower()}.json",
        mime="application/json",
        use_container_width=True,
    )
