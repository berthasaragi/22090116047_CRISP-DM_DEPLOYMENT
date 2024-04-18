import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import pickle 


# Load data
df_file = pd.read_csv('data_cleaned.csv')

filtered_df_numeric = df_file.select_dtypes(include=['float64', 'int64'])


# Load KMeans model
with open('kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

# Load Hierarchical Clustering model
with open('hierarchical_model.pkl', 'rb') as f:
    hierarchical = pickle.load(f)

st.set_page_config(
    page_title="Performa Renang Dashboard",
    page_icon="🏊🏻‍♀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dashboard Main Panel
with st.container():
    st.markdown('# Analisis Performa Atlet Renang dalam Olimpiade melalui histori hasil perlombaan')
   
with st.sidebar:
    st.title('Dashboard')

    st.subheader('Select Dashboard Section')
    section_option = st.selectbox('', ('Home', 'Distribution', 'Composition', 'Relationship', 'Clustering'))

# Visualisasi di layar utama
# Distribusi Jumlah Volume Renang per Penerbit
if section_option == 'Home':
    # Menampilkan gambar di dashboard
    st.image('https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEj8pKpl_00aQ-qUni61oPBNKypgbHz4WsoXoCBTof_T1R4MYZJSGC2zesIb6cLtO5EvPcdnLcYnaemxD1K8PXHzdXpdnAGozWN0Aa9lw5X9ogx810jprXZmtGgFo6nE_VTDKfiJi2NYeS4/s1600/Renang.jpg', caption='Renang Best ', use_column_width=True)
    st.write("""Performa atlet renang dalam kompetisi Olimpiade menjadi titik fokus yang menarik minat banyak orang. Atlet-atlet ini menampilkan keterampilan dan ketangguhan luar biasa di dalam air, dan hasil perlombaan mereka menjadi subjek analisis yang menarik bagi para pengamat olahraga.

Melalui analisis historis hasil perlombaan, kita dapat mengidentifikasi pola-pola dan tren-tren yang berkembang seiring waktu. Apakah ada perubahan dalam strategi balapan yang menghasilkan peningkatan waktu? Apakah ada korelasi antara kecepatan lomba dan faktor-faktor seperti jenis kolam renang atau cuaca pada saat perlombaan?

Dengan memperhatikan data perlombaan dari Olimpiade sebelumnya, kita dapat mendapatkan wawasan yang berharga tentang faktor-faktor yang memengaruhi performa atlet renang. Analisis ini tidak hanya membantu dalam memahami dinamika lomba renang tingkat tertinggi, tetapi juga dapat membantu pelatih dan atlet untuk mengembangkan strategi yang lebih baik untuk sukses di masa depan.""")
    


elif section_option == 'Distribution':
    st.header('Distribusi Rank Berdasarkan Gaya Renang')
    fig, ax = plt.subplots()
    sns.histplot(data=df_file, x='Stroke', hue='Rank', kde=True, multiple="stack", palette='pastel')
    ax.set_title('Distribusi Rank Berdasarkan Gaya Renang')
    ax.set_xlabel("Gaya Renang")
    ax.set_ylabel("rank")
    st.pyplot(fig)

    st.write("""
    Interpretasi:
             
    Grafik tersebut menyajikan distribusi peringkat perenang dalam lima gaya renang yang berbeda: Gaya Punggung, Gaya Dada, Kupu-Kupu, Gaya Bebas, dan Gaya Medley Perorangan. Distribusi peringkat tampaknya condong ke peringkat bawah untuk semua gaya, menunjukkan bahwa ada lebih banyak perenang dengan peringkat lebih rendah dibandingkan dengan peringkat lebih tinggi. Hal ini menunjukkan bahwa tingkat persaingan secara keseluruhan relatif tinggi, dengan sejumlah besar perenang bersaing untuk mendapatkan posisi teratas.
    
    Insight:
             
    Lanskap Kompetitif: Grafik memberikan gambaran lanskap kompetitif untuk setiap gaya renang pada titik waktu tertentu.
    Performa Khusus Gaya: Ada perbedaan mencolok dalam distribusi peringkat di berbagai gaya. Gaya punggung tampaknya merupakan gaya yang cukup kompetitif dengan tingkat performa yang beragam, sedangkan gaya kupu-kupu tampaknya merupakan gaya yang sangat kompetitif dengan sejumlah besar perenang yang berkumpul di peringkat tengah.
    Perenang Peringkat Teratas: Meskipun perenang peringkat teratas tertentu tidak dapat diidentifikasi dari grafik, tren distribusi menunjukkan bahwa kemungkinan ada beberapa perenang di setiap gaya yang secara konsisten mencapai peringkat tinggi.
        
    Actionable Insight:
             
    Pelatihan yang Ditargetkan: Berdasarkan kekuatan dan kelemahan individu yang diidentifikasi dari distribusi peringkat, pelatih dan atlet dapat menyesuaikan program pelatihan untuk fokus pada peningkatan bidang tertentu.
    Perencanaan Kompetisi Strategis: Atlet dapat menyusun strategi jadwal kompetisi mereka berdasarkan kekuatan mereka dan lanskap kompetitif dari berbagai gaya.
    Pemantauan Kinerja: Melacak dan menganalisis distribusi peringkat secara teratur dapat membantu menilai kemajuan dan mengidentifikasi area yang perlu ditingkatkan.
    Analisis Data Tambahan: Mengakses data tambahan, seperti usia, gender, dan metode pelatihan, dapat memberikan pemahaman yang lebih komprehensif tentang lanskap persaingan dan menginformasikan pelatihan lebih lanjut serta pengambilan keputusan strategi. """)

# Perbandingan Jumlah Volume Renang dengan , Komposisi Demografis Renang, dan Hubungan antara Jumlah Volume dan  per Demografis
if section_option == 'Composition':
    st.header('Composition')
    df = df_file.select_dtypes(include=['int', 'float64'])
    price_category_composition = df.groupby('Distance').mean()
    plt.figure(figsize=(10, 6))
    sns.heatmap(price_category_composition.T, annot=True,fmt='g' , cmap='YlGnBu')
    plt.title('Komposisi untuk setiap kategori Jarak')
    plt.xlabel('Kategori Jarak')
    plt.ylabel('Fitur')
    st.pyplot(plt)

    st.write("""
Interpretasi:
             
Tahun: Komposisi fitur "Tahun" menunjukkan distribusi pada tahun-tahun yang berbeda, dengan sedikit peningkatan dalam jumlah catatan menjelang tahun-tahun terakhir. Jarak kategori yang relatif dekat menunjukkan bahwa rentang waktu data relatif sempit.
Peringkat: Komposisi fitur "Peringkat" menunjukkan distribusi yang condong ke peringkat yang lebih rendah, yang menunjukkan jumlah catatan yang lebih besar dengan peringkat yang lebih rendah. Jarak kategori relatif dekat, menunjukkan peningkatan peringkat secara bertahap.
Hasil: Komposisi fitur “Hasil” menunjukkan dominasi kategori “Menang”, disusul “Kalah” dan “Seri”. Jarak kategori yang relatif berjauhan, menunjukkan hasil yang berbeda

             
Insight:
             
Distribusi Tahun: Data tampaknya lebih baru, dengan sedikit peningkatan catatan pada tahun-tahun berikutnya.
Distribusi Peringkat: Mayoritas catatan memiliki peringkat yang lebih rendah, menunjukkan lingkungan yang kompetitif dengan berbagai tingkat kinerja.
Distribusi Hasil: Data menunjukkan proporsi kemenangan yang lebih tinggi dibandingkan kekalahan dan seri.
             

Actionable Insight:
             
Analisis Berbasis Tahun: Pertimbangkan untuk menganalisis tren dan pola dari waktu ke waktu dengan mengelompokkan data berdasarkan fitur "Tahun".
Analisis Berbasis Peringkat: Selidiki hubungan antara peringkat dan fitur lainnya, seperti "Hasil" atau metrik kinerja tertentu.
Analisis Berbasis Hasil: Jelajahi faktor-faktor yang berkontribusi terhadap hasil yang berbeda ("Menang", "Kalah", atau "Seri") dan identifikasi area potensial untuk perbaikan
   """)


if section_option == 'Relationship':
     # Display correlation matrix in the "Relationship" section
    st.markdown('### Correlation Matrix between Numeric Features')
    corr = filtered_df_numeric.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix between Numeric Features')
    st.pyplot(plt)
    st.write("""
    Interpretasi:
                
    Korelasi Positif Kuat antara "Tahun" dan "Jarak":
    Koefisien korelasi antara “Tahun” dan “Jarak” adalah 0,8, menunjukkan korelasi positif yang kuat. Hal ini menunjukkan bahwa seiring bertambahnya tahun, jarak cenderung meningkat pula. Hal ini dapat disebabkan oleh faktor-faktor seperti peningkatan metode pelatihan, peralatan, atau kursus perlombaan seiring berjalannya waktu. Misalnya, kemajuan dalam sepatu lari atau teknik latihan memungkinkan atlet menempuh jarak yang lebih jauh dengan lebih efisien.

    Korelasi Negatif Kuat antara "Peringkat" dan "Hasil":
    Koefisien korelasi antara “Peringkat” dan “Hasil” adalah -0,98, menunjukkan korelasi negatif yang kuat. Ini berarti bahwa ketika peringkat menurun (yaitu kinerja yang lebih baik), kemungkinan hasil “Menang” meningkat. Hal ini diharapkan terjadi dalam suasana kompetitif di mana peringkat yang lebih tinggi menunjukkan kinerja yang lebih baik dan peluang menang yang lebih besar.

    Korelasi Positif Sedang antara "Jarak" dan "Hasil":
    Koefisien korelasi antara “Jarak” dan “Hasil” adalah 0,05, menunjukkan korelasi positif sedang. Hal ini menunjukkan bahwa jarak yang lebih jauh mungkin dikaitkan dengan kemungkinan hasil "Menang" yang lebih tinggi. Hal ini mungkin disebabkan oleh faktor-faktor seperti ketahanan atau pengalaman. Dalam perlombaan yang lebih panjang, atlet dengan daya tahan dan pengalaman yang unggul mungkin memiliki keuntungan, sehingga meningkatkan peluang mereka untuk menang.

    Korelasi Lemah antara "Tahun" dan "Peringkat":
    Koefisien korelasi antara “Tahun” dan “Peringkat” adalah -0,06, menunjukkan korelasi negatif yang sangat lemah. Hal ini menunjukkan bahwa tidak terdapat hubungan yang bermakna antara tahun dengan pangkat atlet. Hal ini dapat berarti bahwa variasi kinerja atau tingkat kompetisi dari tahun ke tahun tidak mempunyai dampak besar terhadap peringkat atlet secara keseluruhan.

    Korelasi Lemah antara "Tahun" dan "Hasil":
    Koefisien korelasi antara “Tahun” dan “Hasil” adalah -0,01, menunjukkan korelasi negatif yang sangat lemah. Hal ini menunjukkan bahwa tidak ada hubungan yang signifikan antara tahun dan hasil kompetisi. Hal ini dapat berarti bahwa variasi faktor yang mempengaruhi hasil dari tahun ke tahun, seperti metode pelatihan atau format kompetisi, tidak memiliki dampak besar terhadap hasil secara keseluruhan.

    Korelasi Lemah antara "Peringkat" dan "Jarak":
    Koefisien korelasi antara "Peringkat" dan "Jarak" adalah -0,00, menunjukkan korelasi negatif yang sangat lemah. Hal ini menunjukkan bahwa tidak ada hubungan yang signifikan antara pangkat atlet dengan jarak pertandingan. Hal ini dapat berarti bahwa tingkat performa tidak secara konsisten dipengaruhi oleh jarak balapan di berbagai kompetisi.

    Korelasi Lemah antara "Jarak" dan "Hasil":
    Koefisien korelasi antara "Jarak" dan "Hasil" adalah -0,00, menunjukkan korelasi negatif yang sangat lemah. Hal ini menunjukkan bahwa tidak ada hubungan yang signifikan antara jarak kompetisi dan hasil. Hal ini dapat berarti bahwa jarak perlombaan tidak memiliki dampak yang konsisten terhadap kemungkinan menang atau kalah di berbagai kompetisi.

             
    Insight:
    -Berdasarkan korelasi yang teridentifikasi, kita dapat memperoleh wawasan berharga tentang hubungan antara fitur-fitur:
    Tahun dan Jarak: Korelasi positif yang kuat menunjukkan bahwa tren jarak dari tahun ke tahun mungkin dipengaruhi oleh faktor-faktor seperti kemajuan teknologi atau metodologi pelatihan.
    Peringkat dan Hasil: Korelasi negatif yang kuat menyoroti hubungan yang diharapkan antara tingkat kinerja dan hasil dalam lingkungan kompetitif.
    Jarak dan Hasil: Korelasi positif sedang menunjukkan bahwa jarak yang lebih jauh mungkin menguntungkan atlet dengan daya tahan atau pengalaman yang lebih baik.
    Korelasi Lemah: Lemahnya korelasi antara "Tahun" dan "Peringkat", "Tahun" dan "Hasil", "Peringkat" dan "Jarak", serta "Jarak" dan "Hasil" menunjukkan bahwa faktor-faktor ini tidak mempunyai pengaruh yang signifikan atau konsisten. berdampak pada fitur individu.

                
    Actionable Insight:
    Mempertimbangkan wawasan yang diperoleh dari korelasi tersebut, berikut beberapa langkah yang dapat diambil:

    Analisis Tren Jarak Tahunan: Selidiki data historis untuk memahami tren spesifik jarak dari waktu ke waktu dan mengidentifikasi faktor-faktor potensial yang mendorong perubahan ini.

    Jelajahi Hubungan Kinerja-Hasil: Selidiki faktor-faktor yang berkontribusi terhadap korelasi kuat antara peringkat dan hasil untuk mengidentifikasi strategi untuk meningkatkan kinerja dan mencapai hasil yang lebih baik.

    Menilai Keunggulan Berbasis Jarak: Mengevaluasi kinerja atlet di berbagai kategori jarak untuk menentukan apakah ada rentang jarak tertentu di mana atlet tertentu unggul.

    Memperhitungkan Korelasi yang Lemah: Saat menganalisis atau memodelkan data kinerja, pertimbangkan lemahnya korelasi antara fitur-fitur tertentu untuk menghindari penyederhanaan yang berlebihan atau salah tafsir atas hubungan tersebut.
        """)



# Clustering Analysis
if section_option == 'Clustering':
    st.subheader('Clustering Analysis of Renang Sales')
    st.write("For clustering analysis, we'll focus on the numerical features 'Result' and 'Year'.")

    # Selecting numerical features for clustering
    clustering_data = df_file[['Year', 'Result']]

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)

    # Selecting number of clusters with slider
    num_clusters = st.slider("Select number of clusters (2-8):", min_value=2, max_value=8, value=4, step=1)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(scaled_data)
    kmeans_cluster_labels = kmeans.labels_

    # Perform Hierarchical clustering
    hierarchical = AgglomerativeClustering(n_clusters=num_clusters)
    hierarchical_cluster_labels = hierarchical.fit_predict(scaled_data)

    # Visualizing the clusters
    plt.figure(figsize=(16, 6))

    # Plot KMeans clustering
    plt.subplot(1, 2, 1)
    plt.scatter(clustering_data['Year'], clustering_data['Result'],
                c=kmeans_cluster_labels, cmap='viridis', s=50)
    plt.title(f'KMeans Clustering (Number of Clusters: {num_clusters})')
    plt.xlabel('Year')
    plt.ylabel('Result')
    plt.grid(True)

    # Plot Hierarchical clustering
    plt.subplot(1, 2, 2)
    plt.scatter(clustering_data['Year'], clustering_data['Result'],
                c=hierarchical_cluster_labels, cmap='viridis', s=50)
    plt.title(f'Hierarchical Clustering (Number of Clusters: {num_clusters})')
    plt.xlabel('Year')
    plt.ylabel('Result')
    plt.grid(True)

    st.pyplot(plt)

    # Interpretation of clusters
    st.write(f"*Number of Clusters: {num_clusters}*")
    st.write("""
Hasil analisis clustering menghasilkan tampilan visual scatter plot yang mengelompokkan data Renang berdasarkan tahun perlombaan dan hasil pertandingan dalam satuan detik. Dalam scatter plot ini, kita dapat melihat titik-titik data yang merepresentasikan setiap kinerja, dengan sumbu-x menunjukkan tahun perlombaan dan sumbu-y menunjukkan hasil pertandingan.

Dengan menerapkan dua metode clustering berbeda, yakni KMeans dan Hierarchical Clustering, data dipilah menjadi beberapa kelompok. Setiap kelompok diberi warna yang berbeda pada scatter plot, memudahkan kita untuk melihat pola-pola dan perbedaan antara kluster yang dihasilkan oleh kedua pendekatan clustering tersebut.    """)