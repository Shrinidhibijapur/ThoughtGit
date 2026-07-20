import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

# Add project root to path to ensure core imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.thought_store import ThoughtStore
from core.embedder import EmbeddingEngine
from core.semantic_diff import SemanticDiffEngine
from core.branch import BranchManager
from core.time_machine import TimeMachine
from core.duplicate_detector import DuplicateDetector
from core.decision_log import DecisionLogger
from core.dead_ideas import DeadIdeasTracker
from core.learning_velocity import LearningVelocityEngine
from core.forgetting_curve import ForgettingCurveTracker
from core.ai_mentor import AIMentor
from core.semantic_commits import SemanticCommitLogger
from core.memory_health import MemoryHealthEngine

# Set page configuration with premium dark look
st.set_page_config(
    page_title="ThoughtGit | Memory Hub",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling injection with Google Fonts
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
        
        /* Global Font and Colors Overrides */
        html, body, [class*="css"], .stApp {
            font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
            background-color: #080b11 !important;
            color: #e2e8f0 !important;
        }
        
        /* Main glowing title */
        .main-header-container {
            padding: 24px 20px;
            background: linear-gradient(135deg, rgba(22, 28, 45, 0.6) 0%, rgba(13, 17, 28, 0.6) 100%);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            margin-bottom: 24px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(12px);
        }
        
        .main-header {
            font-size: 42px;
            font-weight: 800;
            background: linear-gradient(135deg, #00f2fe 0%, #4facfe 50%, #8b5cf6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -1px;
            margin-bottom: 4px;
        }
        
        .main-subtitle {
            font-size: 14px;
            color: #94a3b8;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* Glassmorphic card design */
        .glass-card {
            background: rgba(17, 24, 39, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(16px);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .glass-card:hover {
            border-color: rgba(0, 242, 254, 0.3);
            box-shadow: 0 8px 32px 0 rgba(0, 242, 254, 0.05);
            transform: translateY(-2px);
        }
        
        .card-title {
            font-size: 13px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #94a3b8;
            margin-bottom: 8px;
        }
        
        .card-value {
            font-size: 36px;
            font-weight: 800;
            color: #ffffff;
            line-height: 1.1;
        }
        
        .card-value-glow {
            color: #00f2fe;
            text-shadow: 0 0 12px rgba(0, 242, 254, 0.4);
        }
        
        .card-desc {
            font-size: 12px;
            color: #64748b;
            margin-top: 6px;
            font-weight: 500;
        }

        /* Spaced repetition review list custom styles */
        .review-item {
            background: rgba(239, 68, 68, 0.05);
            border: 1px solid rgba(239, 68, 68, 0.15);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .review-topic {
            font-weight: 700;
            font-size: 14px;
            color: #f87171;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .review-meta {
            font-size: 12px;
            color: #94a3b8;
            margin-top: 4px;
        }

        /* Timeline evolution items */
        .timeline-card {
            border-left: 3px solid #8b5cf6;
            padding-left: 20px;
            margin-left: 10px;
            margin-bottom: 24px;
            position: relative;
        }
        
        .timeline-card::before {
            content: '';
            position: absolute;
            left: -7px;
            top: 0;
            width: 11px;
            height: 11px;
            background: #8b5cf6;
            border: 2px solid #080b11;
            border-radius: 50%;
            box-shadow: 0 0 8px #8b5cf6;
        }

        /* Drift event badge */
        .drift-badge {
            display: inline-block;
            font-size: 10px;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 6px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .drift-reinforced { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
        .drift-deepened { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
        .drift-refined { background: rgba(139, 92, 246, 0.15); color: #a78bfa; border: 1px solid rgba(139, 92, 246, 0.3); }
        .drift-changed_direction { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
        .drift-major_shift { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }

        /* Streamlit widget buttons styling overrides */
        .stButton>button {
            border-radius: 10px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            background-color: rgba(255, 255, 255, 0.03) !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            padding: 8px 20px !important;
            transition: all 0.2s ease !important;
        }
        
        .stButton>button:hover {
            border-color: #00f2fe !important;
            background: linear-gradient(135deg, rgba(0, 242, 254, 0.1) 0%, rgba(79, 172, 254, 0.1) 100%) !important;
            box-shadow: 0 0 10px rgba(0, 242, 254, 0.2) !important;
        }
        
        /* Interactive Glowing Action Buttons */
        .glow-btn>div>button {
            background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%) !important;
            color: #080b11 !important;
            border: none !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 15px rgba(0, 242, 254, 0.35) !important;
        }
        
        .glow-btn>div>button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 20px rgba(0, 242, 254, 0.5) !important;
        }

        /* Clean inputs */
        .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>div {
            background-color: rgba(17, 24, 39, 0.8) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 10px !important;
            color: #ffffff !important;
        }
        
        .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
            border-color: #00f2fe !important;
            box-shadow: 0 0 8px rgba(0, 242, 254, 0.2) !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize engines
@st.cache_resource
def get_engines():
    store = ThoughtStore()
    engine = EmbeddingEngine()
    diff_engine = SemanticDiffEngine(store)
    
    branch_manager = BranchManager()
    time_machine = TimeMachine(store)
    duplicate_detector = DuplicateDetector(store)
    decision_logger = DecisionLogger(engine)
    dead_ideas_tracker = DeadIdeasTracker(engine)
    velocity_engine = LearningVelocityEngine(store, diff_engine)
    forgetting_tracker = ForgettingCurveTracker()
    ai_mentor = AIMentor(store)
    commit_logger = SemanticCommitLogger()
    health_engine = MemoryHealthEngine(store, forgetting_tracker)
    
    return {
        "store": store,
        "engine": engine,
        "diff_engine": diff_engine,
        "branch_manager": branch_manager,
        "time_machine": time_machine,
        "duplicate_detector": duplicate_detector,
        "decision_logger": decision_logger,
        "dead_ideas_tracker": dead_ideas_tracker,
        "velocity_engine": velocity_engine,
        "forgetting_tracker": forgetting_tracker,
        "ai_mentor": ai_mentor,
        "commit_logger": commit_logger,
        "health_engine": health_engine
    }

egs = get_engines()

# Header layout with status badge
status_online = "Local Ollama Connected" if egs["ai_mentor"]._check_ollama_status() else "Mock Ingestion Mode"
status_color = "#10b981" if egs["ai_mentor"]._check_ollama_status() else "#fbbf24"

st.markdown(
    f"""
    <div class="main-header-container">
        <div class="main-header">🧠 THOUGHTGIT CONSOLE</div>
        <div class="main-subtitle">
            <span>Version Control for Human Thinking</span>
            <span style="color: #64748b;">|</span>
            <span style="color: {status_color}; font-weight: 700; font-size: 12px; display: inline-flex; align-items: center; gap: 4px;">
                ● {status_online}
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar Navigation with custom radio styling
st.sidebar.markdown(
    """
    <div style="padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 20px;">
        <span style="font-weight:800; font-size:18px; color:#ffffff; letter-spacing: -0.5px;">Dashboard Menu</span>
    </div>
    """,
    unsafe_allow_html=True
)
menu = st.sidebar.radio(
    "Console Navigation",
    [
        "Memory Health",
        "Timeline Explorer",
        "Concept Diff Viewer",
        "AI Developer Mentor",
        "Decision Log",
        "Dead Ideas Tracker",
        "Branches Manager",
        "Semantic Commits"
    ],
    label_visibility="collapsed"
)

active_branch = egs["branch_manager"].get_active_branch()
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div style="background:rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); padding: 12px; border-radius: 12px;">
        <div style="font-size:11px; color:#94a3b8; font-weight:600;">ACTIVE VAULT BRANCH</div>
        <div style="font-size:15px; font-weight:800; color:#00f2fe; margin-top:2px;">🌿 {active_branch.upper()}</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------
# 1. Page: Memory Health
# ----------------------------------------------------
if menu == "Memory Health":
    st.markdown("### 📈 Overall Memory Health")
    
    # Calculate health report
    report = egs["health_engine"].calculate_health_report(branch=active_branch)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="card-title">Memory Health Score</div>
                <div class="card-value card-value-glow">{report['health_score']}/100</div>
                <div class="card-desc" style="text-transform: capitalize; color: #34d399;">{report['interpretation'].split(':')[0]} status</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="card-title">Recent Activity (30d)</div>
                <div class="card-value">{report['metrics']['activity']['recent_chunks_count']} <span style="font-size: 16px; color: #64748b;">chunks</span></div>
                <div class="card-desc">Score: {report['metrics']['activity']['score']}/35.0</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="card-title">Topic Diversity</div>
                <div class="card-value">{report['metrics']['diversity']['unique_topics_count']} <span style="font-size: 16px; color: #64748b;">topics</span></div>
                <div class="card-desc">Score: {report['metrics']['diversity']['score']}/35.0</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="card-title">Memory Retention</div>
                <div class="card-value">{int(report['metrics']['spacing']['retained_topics_ratio']*100)}%</div>
                <div class="card-desc">Score: {report['metrics']['spacing']['score']}/30.0</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Activity and Source graphs
    gcol1, gcol2 = st.columns([2, 1])
    
    with gcol1:
        # Load activity bar chart
        colls = egs["store"].list_collections(branch=active_branch)
        if colls:
            months = []
            counts = []
            for name in colls:
                coll = egs["store"].client.get_collection(name=name)
                months.append(name.split("_")[-2] + "-" + name.split("_")[-1])
                counts.append(coll.count())
            df_act = pd.DataFrame({"Month": months, "Thoughts (Chunks)": counts})
            fig = px.bar(df_act, x="Month", y="Thoughts (Chunks)", title="Writing Ingestion Volume by Month", template="plotly_dark")
            fig.update_traces(marker_color='#4facfe', marker_line_color='#00f2fe', marker_line_width=1, opacity=0.85)
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No collections found in this branch.")
            
    with gcol2:
        # Source breakdown
        if colls:
            vscode_cnt = 0
            obsidian_cnt = 0
            mcp_cnt = 0
            for name in colls:
                coll = egs["store"].client.get_collection(name=name)
                res = coll.get()
                if res and res["metadatas"]:
                    for m in res["metadatas"]:
                        src = m.get("source", "cli")
                        if src == "vscode": vscode_cnt += 1
                        elif src == "obsidian": obsidian_cnt += 1
                        else: mcp_cnt += 1
            total = vscode_cnt + obsidian_cnt + mcp_cnt
            if total > 0:
                df_src = pd.DataFrame({
                    "Source": ["VS Code", "Obsidian", "CLI/MCP"],
                    "Count": [vscode_cnt, obsidian_cnt, mcp_cnt]
                })
                fig_pie = px.pie(df_src, values="Count", names="Source", title="Ingestion Source Distribution", hole=0.5, template="plotly_dark")
                fig_pie.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=40, b=0, l=0, r=0))
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No thoughts records retrieved.")
        else:
            st.info("No collections found.")

    # Spaced Repetition Forgotten Concepts review pane
    st.markdown("### ⏳ Spaced Repetition review list")
    reviews = egs["forgetting_tracker"].get_review_schedule()
    if reviews:
        for r in reviews:
            r_col1, r_col2 = st.columns([4, 1])
            with r_col1:
                st.markdown(
                    f"""
                    <div class="review-item">
                        <div>
                            <div class="review-topic">💡 {r['topic']}</div>
                            <div class="review-meta">
                                Memory strength: <strong>{int(r['retention_strength']*100)}%</strong> | 
                                Days since last access: <strong>{r['days_since_access']:.1f} days</strong> |
                                Stability: {r['stability_days']:.1f} days
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with r_col2:
                # Add centering style
                st.markdown("<div style='padding-top: 20px;'></div>", unsafe_allow_html=True)
                if st.button("Mark Review", key=r['topic']):
                    egs["forgetting_tracker"].record_access(r["topic"])
                    st.success(f"Reviewed! Stability increased.")
                    st.rerun()
    else:
        st.markdown(
            """
            <div class="glass-card" style="border-left: 4px solid #10b981; padding: 20px;">
                <span style="font-weight: 700; color: #10b981;">✓ ALL TOPICS MEMORY IS STRONG</span>
                <p style="font-size: 13px; color: #94a3b8; margin: 4px 0 0 0;">Spaced repetition algorithm will trigger review alerts as memory decays over time.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ----------------------------------------------------
# 2. Page: Timeline Explorer
# ----------------------------------------------------
elif menu == "Timeline Explorer":
    st.markdown("### ⏳ Concept Evolution Timeline & PCA Topology")
    
    topic = st.text_input("Search a topic concept to trace its evolution:", value="RAG")
    
    if topic:
        query_vector = egs["engine"].embed(topic)
        analysis = egs["diff_engine"].analyze_drift(topic, query_vector, branch=active_branch)
        
        if not analysis["snapshots"]:
            st.warning(f"No memories found about '{topic}' in branch '{active_branch}'.")
        else:
            # Display learning velocity
            vel = egs["velocity_engine"].calculate_velocity(topic, query_vector, branch=active_branch)
            
            vcol1, vcol2, vcol3 = st.columns(3)
            with vcol1:
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <div class="card-title">Writing Velocity</div>
                        <div class="card-value">{vel['volume_velocity']:.1f} <span style="font-size:16px; color:#64748b;">chunks/mo</span></div>
                        <div class="card-desc" style="text-transform: capitalize;">Trend: {vel['volume_trend']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with vcol2:
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <div class="card-title">Conceptual Velocity</div>
                        <div class="card-value">{vel['conceptual_velocity']:.3f} <span style="font-size:16px; color:#64748b;">drift/mo</span></div>
                        <div class="card-desc">Centroid movement speed</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with vcol3:
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <div class="card-title">Learning Phase</div>
                        <div class="card-value" style="color:#8b5cf6; font-size:20px;">{vel['status'].replace('_', ' ').upper()}</div>
                        <div class="card-desc">Ingestion vs evolution status</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Interactive Plotly scatter plot using PCA dimensionality reduction!
            # Renders the 2D cluster map of all thoughts colored by month
            st.markdown("#### 🗺️ 2D Semantic Cluster Topology (PCA Projection)")
            
            # Fetch all thoughts to project
            raw_data = egs["store"].get_all_chunks_for_diff(query_vector, threshold=0.3, branch=active_branch)
            if raw_data and len(raw_data) >= 3:
                embeddings = [r["embedding"] for r in raw_data]
                texts = [r["text"] for r in raw_data]
                periods = [r["collection"].split("_")[-2] + "-" + r["collection"].split("_")[-1] for r in raw_data]
                
                # Apply simple PCA reduction
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                reduced = pca.fit_transform(embeddings)
                
                df_proj = pd.DataFrame({
                    "PCA Dimension 1": reduced[:, 0],
                    "PCA Dimension 2": reduced[:, 1],
                    "Period": periods,
                    "Text": [t[:80] + "..." for t in texts]
                })
                
                fig_proj = px.scatter(
                    df_proj,
                    x="PCA Dimension 1",
                    y="PCA Dimension 2",
                    color="Period",
                    hover_data=["Text"],
                    title="Semantic distance mapping of thoughts across time slices",
                    template="plotly_dark"
                )
                fig_proj.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=1, color='white')))
                fig_proj.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_proj, use_container_width=True)
            else:
                st.info("Ingest more notes about this topic to visualize the 2D concept topology.")

            # Timeline rendering
            st.markdown("#### 📅 Chronological Timeline Snapshots")
            snapshots_map = {s["time_label"]: s for s in analysis["snapshots"]}
            
            if analysis["drift_events"]:
                for e in analysis["drift_events"]:
                    snap = snapshots_map.get(e["to_period"], {})
                    sample = snap["sample_texts"][0] if snap.get("sample_texts") else "Ingested chunk"
                    
                    st.markdown(
                        f"""
                        <div class="timeline-card">
                            <span style="font-size: 11px; color:#94a3b8; font-weight:700; background:rgba(255,255,255,0.03); padding:4px 8px; border-radius:6px;">{e['to_period']}</span>
                            <div style="font-size: 15px; font-weight: 500; margin: 10px 0; color: #ffffff;">"{sample}"</div>
                            <div style="margin-top: 8px;">
                                <span class="drift-badge drift-{e['drift_type']}">{e['drift_type']}</span>
                                <span style="font-size: 12px; color:#94a3b8; margin-left: 12px;">Centroid drift: <strong>{e['distance']:.3f}</strong></span>
                            </div>
                            <p style="font-size: 12px; color: #94a3b8; margin-top: 6px; line-height:1.4;">{e['summary']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                # Just snapshots
                for s in analysis["snapshots"]:
                    st.markdown(
                        f"""
                        <div class="timeline-card">
                            <span style="font-size: 11px; color:#94a3b8; font-weight:700; background:rgba(255,255,255,0.03); padding:4px 8px; border-radius:6px;">{s['time_label']}</span>
                            <div style="font-size: 15px; margin: 10px 0; color:#ffffff;">"{s['sample_texts'][0]}"</div>
                            <div style="font-size: 12px; color:#94a3b8;">Total Chunks: <strong>{s['chunks_count']}</strong></div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

# ----------------------------------------------------
# 3. Page: Concept Diff Viewer
# ----------------------------------------------------
elif menu == "Concept Diff Viewer":
    st.markdown("### ⚖️ Side-by-Side Concept Comparison")
    topic = st.text_input("Enter a topic (e.g. 'RAG')", value="RAG")
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        date_a = st.date_input("Compare Date A", value=datetime.utcnow() - timedelta(days=60))
    with col_d2:
        date_b = st.date_input("Compare Date B", value=datetime.utcnow())
        
    if topic:
        query_vector = egs["engine"].embed(topic)
        comparison = egs["time_machine"].compare_understanding(
            topic=topic,
            query_embedding=query_vector,
            date_a=datetime.combine(date_a, datetime.min.time()),
            date_b=datetime.combine(date_b, datetime.min.time()),
            branch=active_branch
        )
        
        st.markdown("#### Side-by-Side Snapshots")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""
                <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); padding:16px; border-radius:12px; height:100%;">
                    <div style="font-size:12px; color:#94a3b8; font-weight:700;">📅 AS OF {comparison['date_a']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
            if comparison["learnings_at_date_a"]:
                for l in comparison["learnings_at_date_a"]:
                    st.markdown(f"- {l}")
            else:
                st.info("No snapshots found as of this date.")
                
        with c2:
            st.markdown(
                f"""
                <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); padding:16px; border-radius:12px; height:100%;">
                    <div style="font-size:12px; color:#00f2fe; font-weight:700;">📅 AS OF {comparison['date_b']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
            if comparison["learnings_at_date_b"]:
                for l in comparison["learnings_at_date_b"]:
                    st.markdown(f"- {l}")
            else:
                st.info("No snapshots found as of this date.")
                
        st.markdown("#### 🌟 New Learnings Acquired in Interval")
        if comparison["new_learnings_since_a"]:
            for item in comparison["new_learnings_since_a"]:
                st.markdown(
                    f"""
                    <div class="glass-card" style="padding:14px; margin-bottom:10px; border-left:3px solid #00f2fe;">
                        <span style="font-size:11px; color:#00f2fe; font-weight:700;">{item['timestamp'][:10]}</span>
                        <div style="font-size:13px; color:#ffffff; margin-top:4px;">{item['text']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("No new conceptual learnings identified in this interval.")

# ----------------------------------------------------
# 4. Page: AI Developer Mentor
# ----------------------------------------------------
elif menu == "AI Developer Mentor":
    st.markdown("### 🎓 Interactive AI Developer Mentor Console")
    
    st.markdown(
        """
        Write down your current coding task or problem context. 
        The AI Mentor will query your entire history (including rejected designs and decisions) 
        and provide a glowing connection card with recommendations.
        """
    )
    
    current_context = st.text_area(
        "Current Writing/Coding Context:", 
        placeholder="Example: I am implementing vector similarity checks in Python using cosine distance. I need it to be zero-dependency."
    )
    
    # Glowing Action Button
    st.markdown('<div class="glow-btn">', unsafe_allow_html=True)
    get_mentor = st.button("Request Mentor Insights")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if get_mentor and current_context.strip():
        context_vector = egs["engine"].embed(current_context)
        
        with st.spinner("Analyzing memory connections..."):
            advice = egs["ai_mentor"].get_mentor_suggestion(
                current_context=current_context,
                query_embedding=context_vector,
                branch=active_branch
            )
            
        st.markdown(
            f"""<div class="glass-card" style="border: 1px solid rgba(0, 242, 254, 0.25); box-shadow: 0 0 20px rgba(0, 242, 254, 0.08);">
<div style="font-size:11px; color:#00f2fe; font-weight:700; letter-spacing:1px; margin-bottom:12px;">🎓 PROACTIVE MENTOR INSIGHT</div>
<h4 style="margin:0 0 6px 0; color:#ffffff; font-size:16px;">💡 INSIGHT</h4>
<p style="font-size:13.5px; color:#e2e8f0; margin-bottom:16px; line-height:1.4;">{advice['insight']}</p>
<h4 style="margin:0 0 6px 0; color:#ffffff; font-size:16px;">📝 REASONING</h4>
<p style="font-size:13.5px; color:#94a3b8; margin-bottom:16px; line-height:1.4;">{advice['reason']}</p>
<h4 style="margin:0 0 6px 0; color:#ffffff; font-size:16px;">📂 PAST REFERENCE</h4>
<p style="font-size:13.5px; color:#94a3b8; margin-bottom:16px;">{advice['past_reference']}</p>
<h4 style="margin:0 0 6px 0; color:#34d399; font-size:16px;">🚀 SUGGESTED ACTION</h4>
<p style="font-size:14px; color:#34d399; font-weight:600; line-height:1.4; margin-bottom:0;">{advice['action']}</p>
</div>""",
            unsafe_allow_html=True
        )

# ----------------------------------------------------
# 5. Page: Decision Log
# ----------------------------------------------------
elif menu == "Decision Log":
    st.markdown("### ⚖️ Architectural Decision Log")
    
    tab_log, tab_search = st.tabs(["Log New Decision", "Search Past Decisions"])
    
    with tab_log:
        st.markdown("#### Register Architectural Choice")
        with st.form("decision_form"):
            title = st.text_input("Decision Title", placeholder="e.g. Chose SQLite over Redis")
            chosen = st.text_input("Selected Option", placeholder="e.g. SQLite")
            alts_str = st.text_input("Rejected Alternatives", placeholder="e.g. Redis, PostgreSQL")
            reasoning = st.text_area("Reasoning trade-offs & details")
            assumptions = st.text_area("Underlying assumptions")
            tags_str = st.text_input("Tags (comma separated)", placeholder="e.g. database, caching")
            
            st.markdown('<div class="glow-btn">', unsafe_allow_html=True)
            submitted = st.form_submit_button("Log Decision")
            st.markdown('</div>', unsafe_allow_html=True)
            
            if submitted and title:
                alts = [a.strip() for a in alts_str.split(",") if a.strip()]
                tags = [t.strip() for t in tags_str.split(",") if t.strip()]
                dec_id = egs["decision_logger"].log_decision(title, chosen, alts, reasoning, assumptions, tags)
                st.success(f"Decision registered successfully! ID: {dec_id}")
                
    with tab_search:
        st.markdown("#### Semantic Decision Search")
        search_query = st.text_input("Search query (e.g. 'caching databases')", placeholder="Type search context...")
        
        if search_query:
            results = egs["decision_logger"].search_decisions(search_query)
            if results:
                for r in results:
                    st.markdown(
                        f"""
                        <div class="glass-card" style="border-left: 4px solid #00f2fe;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <h4 style="margin:0; color:#00f2fe;">{r['title']}</h4>
                                <span style="font-size:10px; background:rgba(0,242,254,0.1); padding:2px 8px; border-radius:4px; color:#00f2fe;">{int(r['similarity']*100)}% Match</span>
                            </div>
                            <div style="margin: 8px 0; font-size:12.5px;">
                                <strong>Chosen:</strong> {r['chosen']} | 
                                <span style="color:#64748b;">Alternatives: {', '.join(r['alternatives'])}</span>
                            </div>
                            <div style="font-size:13px; color:#e2e8f0; line-height:1.4;">
                                <strong>Reasoning:</strong> {r['reasoning']}
                            </div>
                            <div style="font-size:12px; color:#94a3b8; margin-top:6px;">
                                <strong>Assumptions:</strong> {r['assumptions']}
                            </div>
                            <div style="font-size:10px; color:#64748b; margin-top:10px;">
                                Tags: {', '.join(r['tags'])} | Logged: {r['created_at'][:10]}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    outcome = r.get("outcome")
                    if outcome:
                        st.markdown(
                            f"""
                            <div style="background:rgba(52,211,153,0.06); border:1px solid rgba(52,211,153,0.15); padding:10px 14px; border-radius:10px; margin-top:-10px; margin-bottom:15px; font-size:12px;">
                                <strong style="color:#34d399;">Outcome Log:</strong> {outcome}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        with st.expander("Update Decision Outcome"):
                            with st.form(f"out_form_{r['id']}"):
                                out_text = st.text_input("Enter outcome notes", key=f"inp_{r['id']}")
                                out_sub = st.form_submit_button("Save Outcome")
                                if out_sub and out_text:
                                    egs["decision_logger"].update_outcome(r["id"], out_text)
                                    st.success("Outcome logged!")
                                    st.rerun()
            else:
                st.info("No matching decisions found.")
        else:
            decisions = egs["decision_logger"].list_decisions()
            for r in decisions:
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <h4 style="margin:0; color:#4facfe;">{r['title']}</h4>
                        <div style="margin: 6px 0; font-size:12.5px;"><strong>Chosen:</strong> {r['chosen']} | Logged: {r['created_at'][:10]}</div>
                        <div style="font-size:13px; color:#94a3b8;">{r['reasoning']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ----------------------------------------------------
# 6. Page: Dead Ideas Tracker
# ----------------------------------------------------
elif menu == "Dead Ideas Tracker":
    st.header("🪦 Dead Ideas Graveyard")
    
    tab_bury, tab_graveyard, tab_resurrect = st.tabs(["Bury New Idea", "Graveyard View", "Check Resurrection"])
    
    with tab_bury:
        st.markdown("#### Archive Abandoned Idea")
        with st.form("bury_form"):
            title = st.text_input("Idea Title", placeholder="e.g. Voice-to-code compiler")
            desc = st.text_area("Description")
            reason = st.text_area("Why was it abandoned?")
            triggers_str = st.text_input("Resurrection triggers (comma separated)", placeholder="e.g. Speech latency drops below 50ms")
            
            sub_bury = st.form_submit_button("Bury Idea")
            if sub_bury and title:
                triggers = [t.strip() for t in triggers_str.split(",") if t.strip()]
                idea_id = egs["dead_ideas_tracker"].bury_idea(title, desc, reason, triggers)
                st.success(f"Idea successfully archived in the Graveyard. ID: {idea_id}")
                
    with tab_graveyard:
        st.markdown("#### Buried Ideas Graveyard")
        graveyard = egs["dead_ideas_tracker"].list_graveyard()
        if graveyard:
            for item in graveyard:
                st.markdown(
                    f"""
                    <div class="glass-card" style="border-left: 4px solid #f87171;">
                        <h4 style="margin:0; color:#f87171;">💀 {item['title']}</h4>
                        <div style="font-size:13px; margin: 8px 0; color:#e2e8f0;">{item['description']}</div>
                        <div style="font-size:12px; color:#f87171;"><strong>Why Abandoned:</strong> {item['reason_abandoned']}</div>
                        <div style="font-size:11px; color:#94a3b8; margin-top:8px;">
                            Resurrection Triggers: {', '.join(item['resurrection_triggers'])} | Buried: {item['created_at'][:10]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if st.button(f"Resurrect Idea: '{item['title']}'", key=f"res_{item['id']}"):
                    egs["dead_ideas_tracker"].resurrect_idea(item["id"])
                    st.success(f"Resurrected '{item['title']}'!")
                    st.rerun()
        else:
            st.info("The graveyard is empty! All your ideas are currently active.")
            
    with tab_resurrect:
        st.markdown("#### Auto-Resurrection Scanner")
        context_query = st.text_area("Enter your current coding project context to check resurrection triggers:")
        
        if context_query:
            context_vector = egs["engine"].embed(context_query)
            candidates = egs["dead_ideas_tracker"].check_resurrections(context_vector, similarity_threshold=0.70)
            if candidates:
                st.warning("⚠️ Semantically Connected Buried Ideas Found! Consider resurrecting:")
                for c in candidates:
                    st.markdown(
                        f"""
                        <div class="glass-card" style="border:1px solid #fbbf24; box-shadow:0 0 15px rgba(251,191,36,0.1);">
                            <h4 style="margin:0; color:#fbbf24;">💡 {c['title']} ({int(c['similarity']*100)}% Match)</h4>
                            <div style="font-size:13px; margin: 6px 0; color:#e2e8f0;">{c['description']}</div>
                            <div style="font-size:12px; color:#f87171; margin: 4px 0;"><strong>Why Abandoned:</strong> {c['reason_abandoned']}</div>
                            <div style="font-size:11px; color:#94a3b8;">Triggers: {', '.join(c['resurrection_triggers'])}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.success("No relevant buried ideas matching your current context. Keep coding!")

# ----------------------------------------------------
# 6. Page: Branches Manager
# ----------------------------------------------------
elif menu == "Branches Manager":
    st.header("🌿 Thought Branching Manager")
    
    col_b1, col_b2 = st.columns(2)
    
    with col_b1:
        st.markdown("#### Create & Switch Branches")
        
        # Switch Active Branch
        branches = egs["branch_manager"].list_branches()
        sel_branch = st.selectbox("Switch Active Branch namespace", branches, index=branches.index(active_branch))
        
        if sel_branch != active_branch:
            egs["branch_manager"].switch_branch(sel_branch)
            st.success(f"Switched active branch to '{sel_branch}'!")
            st.rerun()
            
        # Create Branch
        new_b_name = st.text_input("Create New Branch Name (alphanumeric)")
        if st.button("Create Branch"):
            if new_b_name:
                try:
                    egs["branch_manager"].create_branch(new_b_name)
                    st.success(f"Branch '{new_b_name}' created successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
                    
    with col_b2:
        st.markdown("#### Merge Branches")
        src_branch = st.selectbox("Source branch to copy from", branches)
        dest_branch = st.selectbox("Destination branch to merge into", branches, index=branches.index(active_branch))
        
        if st.button("Merge Branches"):
            if src_branch == dest_branch:
                st.error("Cannot merge a branch into itself.")
            else:
                try:
                    egs["branch_manager"].merge_branch(src_branch, dest_branch)
                    st.success(f"Merged branch '{src_branch}' into '{dest_branch}' successfully!")
                except Exception as e:
                    st.error(str(e))

# ----------------------------------------------------
# 7. Page: Semantic Commits
# ----------------------------------------------------
elif menu == "Semantic Commits":
    st.header("🔀 Semantic Commits Log")
    st.markdown("Chronological natural language commit logs generated during significant conceptual changes.")
    
    topic_filter = st.text_input("Filter by topic (leave blank to list all)")
    
    history = egs["commit_logger"].get_commit_history(topic_filter if topic_filter else None)
    
    if history:
        for c in history:
            st.markdown(
                f"""
                <div class="glass-card" style="border-left: 3px solid #4facfe;">
                    <div style="display:flex; justify-content:space-between; font-size:11px; color:#4facfe; font-weight:700;">
                        <span>TOPIC: {c['topic'].upper()}</span>
                        <span>{c['timestamp'][:19].replace('T', ' ')}</span>
                    </div>
                    <div style="font-size:14.5px; font-weight:700; margin-top:8px; color:#ffffff;">
                        🚀 {c['message']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("No semantic commits recorded yet.")
