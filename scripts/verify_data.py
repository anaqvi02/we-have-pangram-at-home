import streamlit as st
import pandas as pd
from pathlib import Path
import random
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import Config

st.set_page_config(page_title="Pangram Data Verifier", layout="wide")

st.title("🔍 Pangram Data Verifier")

# Sidebar
st.sidebar.header("Dataset Selection")
dataset_type = st.sidebar.radio("Corpus Type", ["AI Corpus", "Human Corpus"])

if dataset_type == "AI Corpus":
    data_dir = Config.AI_DATASET_PATH
    color = "red"
else:
    data_dir = Config.HUMAN_DATASET_PATH
    color = "green"

st.sidebar.markdown(f"**Path:** `{data_dir}`")

try:
    # List files
    files = sorted(list(data_dir.glob("*.parquet")))
    if not files:
        st.error(f"No parquet files found in {data_dir}!")
        st.stop()
    
    file_names = [f.name for f in files]
    selected_file = st.sidebar.selectbox("Select File", file_names)
    
    # Load Data
    file_path = data_dir / selected_file
    
    @st.cache_data
    def load_parquet(path):
        return pd.read_parquet(path)
    
    df = load_parquet(file_path)
    
    # Overview Stats
    st.header(f"📊 Stats: {selected_file}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", len(df))
    
    # Calc avg length
    df['length'] = df['text'].apply(lambda x: len(str(x).split()))
    avg_len = int(df['length'].mean())
    col2.metric("Avg Word Count", avg_len)
    
    unique_sources = df['source'].nunique() if 'source' in df.columns else "N/A"
    col3.metric("Unique Sources", unique_sources)
    
    # Source Distribution
    if 'source' in df.columns:
        st.subheader("Source Distribution")
        source_counts = df['source'].value_counts()
        st.bar_chart(source_counts, color="#FF4B4B" if dataset_type == "AI Corpus" else "#4CAF50")
    
    # Sample Viewer
    st.subheader("👁️ Random Sample Inspector")
    if st.button("Shuffle New Sample"):
        sample = df.sample(1).iloc[0]
        st.markdown(f"**Source:** `{sample.get('source', 'Unknown')}` | **Label:** `{sample.get('label', 'Unknown')}` | **Length:** `{len(str(sample['text']).split())}` words")
        st.text_area("Raw Text", sample['text'], height=300)
        
    # Data Table
    st.subheader("🗂️ Data Preview (First 50)")
    st.dataframe(df.head(50), use_container_width=True)

except Exception as e:
    st.error(f"Error loading data: {e}")
