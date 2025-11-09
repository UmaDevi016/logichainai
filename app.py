# -*- coding: utf-8 -*-
"""
🚚 LogiChain AI — Final Submission Version
- Loads data files first (before Streamlit)
- Fails gracefully if missing
- Streamlit UI runs only after successful load
"""

import os, sys, pandas as pd, numpy as np, json, time, threading, requests, folium
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st
from streamlit_folium import st_folium
import networkx as nx
import matplotlib.pyplot as plt

# ----------- Paths & Checks -----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
REQUIRED_FILES = [
    ("delivery_points.csv", True),
    ("shipments.csv", False),
    ("live_feed.jsonl", False),
    ("gps_traces.csv", False),
]

# check existence and readability
missing = []
for fname, required in REQUIRED_FILES:
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath):
        if required:
            missing.append(fname)
        continue
    try:
        with open(fpath, "rb") as f:
            f.read(100)
    except Exception:
        missing.append(fname)

if missing:
    st.set_page_config(page_title="LogiChain AI — Missing Data", layout="centered")
    st.title("🚫 Missing or Unreadable Data Files")
    st.error(
        f"The following required data file(s) are missing or unreadable:\n\n"
        + "\n".join([f"- {m}" for m in missing])
        + "\n\nPlease copy them into the `/data` folder before running."
    )
    st.stop()

# ----------- Load data before UI -----------
try:
    delivery_points = pd.read_csv(
        os.path.join(DATA_DIR, "delivery_points.csv"), dtype={"id": str}, encoding="utf-8"
    )
    st.session_state["delivery_points"] = delivery_points
except Exception as e:
    st.set_page_config(page_title="Data Load Error", layout="centered")
    st.error(f"Failed to read delivery_points.csv: {e}")
    st.stop()

shipments_df = None
if os.path.exists(os.path.join(DATA_DIR, "shipments.csv")):
    try:
        shipments_df = pd.read_csv(os.path.join(DATA_DIR, "shipments.csv"), encoding="utf-8")
        st.session_state["shipments_df"] = shipments_df
    except Exception as e:
        st.warning(f"Could not read shipments.csv: {e}")

LIVE_PATH = os.path.join(DATA_DIR, "live_feed.jsonl")
GPS_PATH = os.path.join(DATA_DIR, "gps_traces.csv")

# ----------- Streamlit App Config -----------
st.set_page_config(page_title="LogiChain AI — Demo", layout="wide", page_icon="🚚")

st.title("🚚 LogiChain AI — Real-Time Route Optimization")
st.caption("Auto-loads demo data from /data folder. Built in Python + Streamlit.")

# ----------- Utility Functions -----------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def greedy_tsp(df):
    pts = df.to_dict("records")
    ids = [p["id"] for p in pts]
    dist = np.zeros((len(ids), len(ids)))
    for i, p1 in enumerate(pts):
        for j, p2 in enumerate(pts):
            if i == j: dist[i,j] = np.inf
            else: dist[i,j] = haversine(p1["lat"],p1["lon"],p2["lat"],p2["lon"])
    route, total = [0], 0.0
    while len(route) < len(ids):
        last = route[-1]
        next_idx = np.argmin(dist[last])
        total += dist[last,next_idx]
        dist[:,last] = np.inf
        route.append(next_idx)
    route.append(0)
    route_ids = [ids[i] for i in route]
    return route_ids, total

# ----------- Route Optimization Section -----------
st.header("📍 Route Optimization")
st.dataframe(st.session_state["delivery_points"], use_container_width=True)
if st.button("⚙️ Compute Optimized Route"):
    route, total = greedy_tsp(st.session_state["delivery_points"])
    st.success(f"Optimized Route: {' → '.join(route)}")
    st.info(f"Total Distance: {total:.2f} km")
    G = nx.Graph()
    pos = {r["id"]:(r["lon"],r["lat"]) for _,r in st.session_state["delivery_points"].iterrows()}
    G.add_nodes_from(pos.keys())
    G.add_edges_from(list(zip(route[:-1],route[1:])))
    fig,ax=plt.subplots(figsize=(6,4))
    nx.draw(G,pos,with_labels=True,node_color="#dbeafe",edge_color="#ef4444",node_size=600,ax=ax)
    st.pyplot(fig)
    m=folium.Map(location=[st.session_state["delivery_points"]["lat"].mean(),st.session_state["delivery_points"]["lon"].mean()],zoom_start=13)
    for _,r in st.session_state["delivery_points"].iterrows():
        folium.Marker([r["lat"],r["lon"]],popup=r["id"]).add_to(m)
    coords=[(st.session_state["delivery_points"].loc[st.session_state["delivery_points"]["id"]==i,"lat"].values[0],
             st.session_state["delivery_points"].loc[st.session_state["delivery_points"]["id"]==i,"lon"].values[0]) for i in route]
    folium.PolyLine(coords,color="red").add_to(m)
    st_folium(m,width=900,height=500)

# ----------- Shipments Table -----------
st.header("📦 Shipments Overview")
if shipments_df is not None:
    st.dataframe(shipments_df.head(10), use_container_width=True)
else:
    st.info("No shipments.csv found. Using route data only.")

# ----------- Live Feed Section -----------
st.header("🔴 Live Feed (Simulated)")
if os.path.exists(LIVE_PATH):
    if "live_data" not in st.session_state:
        st.session_state["live_data"] = pd.DataFrame()
    if st.button("Start Live Feed"):
        df = []
        with open(LIVE_PATH,"r",encoding="utf-8") as f:
            for line in f:
                df.append(json.loads(line))
        df = pd.DataFrame(df)
        st.session_state["live_data"] = df
        st.success("Live feed loaded.")
    if not st.session_state["live_data"].empty:
        m = folium.Map(location=[st.session_state["live_data"]["lat"].mean(), st.session_state["live_data"]["lon"].mean()], zoom_start=13)
        for _, r in st.session_state["live_data"].iterrows():
            folium.CircleMarker([r["lat"], r["lon"]], radius=5, popup=f"{r['order_id']} - {r['status']}", color="#0B3D91", fill=True).add_to(m)
        st_folium(m, width=900, height=480)
else:
    st.info("No live_feed.jsonl found. Place it in /data to enable live tracking.")

st.success("✅ All data loaded successfully. Ready for submission!")
st.caption("Built by Uma Devi — Hackathon MVP")
