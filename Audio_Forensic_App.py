import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Set page layout
st.set_page_config(page_title="Audio Forensics - Voice Match", layout="centered")

st.title("🔍 Audio Forensics - Voice Matcher")

# Upload files
suspect_file = st.file_uploader("Upload Suspect Audio (.wav)", type=["wav"])
test_file = st.file_uploader("Upload Test Audio (.wav)", type=["wav"])

def extract_mfcc(audio, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T

def compare_mfcc_dtw(mfcc1, mfcc2):
    distance, path = fastdtw(mfcc1, mfcc2, dist=euclidean)
    return distance

def show_mfcc(mfcc, sr, title):
    fig, ax = plt.subplots()
    librosa.display.specshow(mfcc.T, x_axis='time', sr=sr)
    ax.set_title(title)
    st.pyplot(fig)

if suspect_file and test_file:
    st.success("Both files uploaded successfully!")

    # Load audio files
    y1, sr1 = librosa.load(suspect_file, sr=16000)
    y2, sr2 = librosa.load(test_file, sr=16000)

    # Extract MFCC features
    mfcc1 = extract_mfcc(y1, sr1)
    mfcc2 = extract_mfcc(y2, sr2)

    # Show MFCC plots
    st.subheader("🎵 Suspect MFCC")
    show_mfcc(mfcc1, sr1, "Suspect MFCC")

    st.subheader("🎵 Test Sample MFCC")
    show_mfcc(mfcc2, sr2, "Test Sample MFCC")

    # Compare using DTW
    dtw_distance = compare_mfcc_dtw(mfcc1, mfcc2)
    st.subheader("📊 Voice Comparison Result")
    st.write(f"DTW Distance: `{dtw_distance:.2f}`")

    # Decision Threshold
    threshold = 200  # You can tune this based on your dataset
    if dtw_distance < threshold:
        st.success("✅ Match Found: Voices are likely from the same speaker.")
    else:
        st.error("❌ No Match: Voices are likely from different speakers.")

else:
    st.info("Please upload both audio files to continue.")

