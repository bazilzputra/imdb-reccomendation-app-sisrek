import streamlit as st
import pandas as pd
from surprise import Dataset, Reader
from surprise import KNN

@st.cache
def load_data(uploaded_file):
    # Membaca dataset
    df = pd.read_csv(uploaded_file)
    # Rename kolom sesuai kebutuhan Surprise
    df_surprise = df.rename(columns={
        "Age Rating": "user_id",  # Age Rating sebagai user_id
        "Title": "item_id",       # Title sebagai item_id
        "IMDb Rating": "rating"   # IMDb Rating sebagai rating
    })
    return df_surprise

# Fungsi untuk melatih model
@st.cache
def train_model(df_surprise):
    # Konversi data ke format Surprise
    reader = Reader(rating_scale=(df_surprise["rating"].min(), df_surprise["rating"].max()))
    data = Dataset.load_from_df(df_surprise[["user_id", "item_id", "rating"]], reader)

    # Split data menjadi train dan test set
    trainset, testset = train_test_split(data, test_size=0.25)

    # Konfigurasi Collaborative Filtering
    sim_options = {
        "name": "cosine",  # Similarity Cosine
        "user_based": True,  # Berbasis pengguna (user-based)
    }
    algo = KNNBasic(sim_options=sim_options)

    # Melatih model
    algo.fit(trainset)

    # Evaluasi model
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)

    return algo, rmse

# Fungsi untuk membuat prediksi
def make_prediction(algo, user_id, item_id):
    prediction = algo.predict(uid=user_id, iid=item_id)
    return prediction.est

# Streamlit Interface
st.title("IMDb Movie Recommendation App")
st.write("Aplikasi ini memberikan rekomendasi film berdasarkan **Age Rating** dan **Judul Film** menggunakan Collaborative Filtering.")

# Upload file
uploaded_file = st.file_uploader("Upload dataset IMDb (.csv)", type="csv")

if uploaded_file:
    # Memuat dataset
    df_surprise = load_data(uploaded_file)
    st.write("Dataset berhasil dimuat!")
    st.dataframe(df_surprise.head())

    # Melatih model
    algo, rmse = train_model(df_surprise)
    st.write(f"Model berhasil dilatih dengan RMSE: {rmse:.4f}")

    # Input untuk prediksi
    st.header("Prediksi Rating")
    user_id = st.text_input("Masukkan Age Rating (contoh: PG-13):")
    item_id = st.text_input("Masukkan Judul Film (contoh: The Dark Knight):")

    if st.button("Prediksi"):
        if user_id and item_id:
            pred_rating = make_prediction(algo, user_id, item_id)
            st.write(f"Prediksi Rating untuk Age Rating '{user_id}' pada Judul Film '{item_id}': {pred_rating:.2f}")
        else:
            st.write("Harap masukkan Age Rating dan Judul Film terlebih dahulu.")
