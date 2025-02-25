#!/usr/bin/env python
# coding: utf-8

# # **Hotel Booking Demand**

# ### **Contents**
# 
# 1. Business Problem Understanding
# 2. Data Understanding
# 3. Data Preprocessing
# 4. Modeling
# 5. Conclusion
# 6. Recommendation
# 
# ****

# ## **1. Business Problem Understanding**
# 
# ### **Context**
# 
# Kebutuhan masyarakat akan akomodasi yang nyaman saat bepergian membuat industri perhotelan terus berkembang. Dua jenis hotel yang paling sering dibandingkan berdasarkan lokasi dan target pasarnya adalah Resort Hotel dan City Hotel.
# - **Resort Hotel** biasanya terletak di daerah wisata dan mengutamakan pengalaman menginap yang santai serta fasilitas rekreasi.
# - **City Hotel** berada di pusat kota dan sering digunakan oleh wisatawan bisnis atau tamu yang membutuhkan akses mudah ke transportasi dan pusat bisnis.
# 
# Namun, tren perjalanan saat ini menunjukkan bahwa banyak orang menginginkan kombinasi antara bisnis dan rekreasi [bleisure travel](https://blog.hotelogix.com/business-leisure-travel-trends/), yang membuat city hotel perlu menyesuaikan fasilitasi mereka agar lebih menarik bagi wisatawan.
# Di tengah upaya menarik pelanggan, salah satu tantangan terbesar yang dihadapi city hotel yaitu **pembatalan reservasi yang cukup tinggi (42,72%)**. Hal ini berdampak langsung pada pendapatan dan efisiensi operasional hotel.
# 
# Pembatalan reservasi yang tinggi menyebabkan:
# - **Kerugian finansial**, terutama jika kamar yang dibatalkan tidak dapat terisi kembali.
# - **Ketidakseimbangan dalam alokasi sumber daya**, seperti pengelolaan staf dan persediaan logistik.
# - **Dampak operasional pada layanan non-kamar**, seperti katering, layanan kebersihan, dan aktivitas wisata.
# 
# ### **Business Problem**
# - City Hotel menghadapi persaingan ketat dalam menarik pelanggan, terutama dengan munculnya tren **bleisure travel**, di mana wisatawan menggabungkan perjalanan bisnis dengan rekreasi. Untuk tetap kompetitif, City Hotel perlu menyesuaikan fasilitas dan layanan mereka agar lebih relevan dengan kebutuhan pelanggan masa kini.
# - Namun, salah satu tantangan terbesar dalam mencapai tujuan ini adalah **tingkat pembatalan reservasi yang tinggi**, yang berdampak langsung pada pendapatan dan efisiensi operasional. Jika hotel tidak memahami faktor utama yang menyebabkan pembatalan, mereka akan kesulitan dalam menyusun strategi untuk menarik dan mempertahankan pelanggan, termasuk bleisure traveler.
# 
# **Stakeholder yang terlibat:**
# 1. **Manajemen Hotel**
#     - Bertanggung jawab atas strategi keseluruhan hotel, termasuk kebijakan harga, kebijakan pembatalan, dan pengelolaan sumber daya.
#     - Memerlukan wawasan dari analisis data untuk mengambil keputusan yang dapat meningkatkan tingkat okupansi dan mengurangi dampak finansial dari pembatalan.
# 2. **Customer Service (CS) & Reservasi**
#     - Berinteraksi langsung dengan tamu, menangani pemesanan dan pembatalan reservasi.
#     - Memerlukan informasi mengenai alasan umum pembatalan untuk meningkatkan pelayanan dan memberikan solusi yang lebih baik kepada pelanggan.
# 3. **Tim Pemasaran**
#     - Bertanggung jawab atas strategi promosi dan branding hotel, termasuk menarik segmen bleisure travelers.
#     - Memerlukan insight dari analisis pembatalan untuk menyesuaikan kampanye pemasaran dan mengurangi kemungkinan pelanggan membatalkan reservasi.
# 
# 
# ### **Goals**
# Tujuan dari prediksi pembatalan reservasi adalah untuk membantu hotel mengurangi dampak negatif pembatalan dan meningkatkan efisiensi operasional melalui pendekatan berbasis data. Secara spesifik, tujuan utama dari analisis ini adalah:
# - **Membangun model prediksi** yang dapat mengidentifikasi pelanggan yang berpotensi membatalkan reservasi sebelum tanggal check-in.
# - **Menganalisis faktor-faktor utama** yang berkontribusi terhadap pembatalan reservasi, seperti metode pembayaran, durasi menginap, musim perjalanan, dan harga.
# - **Memberikan rekomendasi strategis** untuk mengurangi pembatalan, penyesuaian kebijakan pembatalan, atau strategi pemasaran yang lebih efektif.
# - **Meningkatkan akurasi dalam pengelolaan reservasi** untuk meminimalkan kamar kosong dan mengoptimalkan pendapatan hotel.
# Dengan model prediksi ini, hotel dapat mengoptimalkan strategi bisnisnya, meningkatkan efisiensi operasional, serta mengurangi risiko pembatalan secara signifikan.
# 
# ### **Analytic Approach**
# Jadi yang akan kita lakukan adalah **menganalisa data untuk menemukan pola yang membedakan pelanggan yang membatalkan reservasi dan yang tidak**. Kemudian, kita akan membangun **model klasifikasi** yang akan membantu **City Hotel memprediksi probabilitas seorang tamu akan membatalkan reservasi atau tidak**. Dengan model ini, hotel dapat mengambil langkah-langkah preventif untuk mengurangi pembatalan, seperti menyesuaikan kebijakan harga, memberikan insentif bagi pelanggan tertentu, atau meningkatkan strategi pemasaran yang lebih efektif.
# 
# ### **Metric Evaluation**
# Untuk mengurangi pembatalan hotel, hasil akhir model dibagi menjadi dua kategori berikut:
# 
# Target:
# 
#     0: Tidak cancel
#     1: Cancel
# 
# | Aktual / Prediksi | Negatif (Tidak Cancel) | Positif (Cancel) |
# | --- | --- | --- |
# | Negatif (Tidak Cancel) | Aktual tidak cancel, Prediksi tidak cancel (TN)  | Aktual tidak cancel, Prediksi cancel (FP) |
# | Positif (Cancel) | Aktual cancel, Prediksi tidak cancel (FN) | Aktual Cancel, Prediksi Cancel (TP) |
# 
# **False Negative: Kamarnya jadi kosong tidak tersewa**
# - Model gagal mendeteksi bahwa tamu akan membatalkan reservasi, **sehingga hotel tidak bisa mengantisipasi pembatalan ini**.
# - Akibatnya, kamar tersebut tidak dapat dipesan oleh pelanggan lain, sehingga **hotel kehilangan pendapatan**.
# 
# **False Positif: Kamar diberikan ke tamu lain, padahal pelanggan asli tetap datang**
# - Model salah memprediksi bahwa tamu akan membatalkan, padahal sebenarnya mereka tetap datang.
# - Jika hotel menerapkan strategi **overbooking**, ini bisa menyebabkan **kekurangan kamar**, dan tamu yang datang malah tidak mendapatkan kamar yang dipesan.
# 

# ***

# ## **2. Data Understanding**
# 
# Source : https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand/data
# 
# Dataset ini terdiri dari 32 fitur dan 119390 baris data, dataset ini berisi informasi mengenai hotel dan resort yang berada di negara Portugal yang mencakup pemesanan hotel antara tanggal 1 Juli 2015 sampai 31 Agustus 2017. Setiap baris data merepresentasikan informasi transaksi pemesanan hotel yang diperoleh dari database Property Management System (PMS) hotel.
# 
# **Attributes Information**
# 
# | **No** | **Attribute** | **Data Type** | **Description** |
# | --- | --- | --- | --- |
# | 1 | hotel | Object | City Hotel atau Resort Hotel |
# | 2 | is_canceled | Integer | Mengidentifikasikan pembatalkan pemesanan (0 - Tidak, 1 - Ya) |
# | 3 | lead_time | Integer | Jumlah hari antara tanggal pemesanan dan tanggal kedatangan |
# | 4 | arrival_date_year | Integer | Tahun kedatangan |
# | 5 | arrival_date_month | Object | Bulan kedatangan |
# | 6 | arrival_date_week_number | Integer | Minggu kedatangan (dalam tahun) |
# | 7 | arrival_date_day_of_month | Integer | Hari kedatangan |
# | 8 | stays_in_weekend_nights | Integer | Jumlah malam akhir pekan yang dipesan untuk menginap (sabtu dan minggu) |
# | 9 | stays_in_week_nights | Integer | Jumlah malam yang dipesan untuk menginap (senin sampai jumat) |
# | 10 | adults | Integer | Jumlah orang dewasa |
# | 11 | children | Float | Jumlah anak |
# | 12 | babies | Integer | Jumlah bayi |
# | 13 | meal | Object | Jenis makanan yang dipesan, terdiri dari: </br> 1. BB : Bed & Breakfast </br> 2. FB : Full board (breakfast, lunch and dinner) </br> 3. HB : Half board (breakfast and one other meal – usually dinner) |
# | 14 | country | Object | Negara asal pelanggan |
# | 15 | market_segment | Object | Segmen pasar, terdiri dari: </br> 1. Aviation : Penerbangan </br> 2. Complementary : Promosi silang </br> 3. Corporate : Perusahaan </br> 4. Direct : Langsung </br> 5. Groups : Kelompok </br> 6. Offline TA/TO : Offline travel agent/tour operators </br> 7. Online TA : Online travel agent |
# | 16 | distribution_channel | Object | Saluran distribusi pemesanan, terdiri dari: </br> 1. Corporate : Perusahaan </br> 2. Direct : Langsung </br> 3. GDS : Global Distribution System </br> 4. TA/TO : Travel agent/tour operators |
# | 17 | is_repeated_guest | Integer | Mengidentifikasikan pemesanan berulang (0 - Tidak, 1 - Ya) |
# | 18 | previous_cancellations | Integer | Jumlah pemesanan sebelumnya yang dibatalkan sebelum pemesanan saat ini |
# | 19 | previous_bookings_not_canceled | Integer | Jumlah pemesanan sebelumnya yang tidak dibatalkan sebelum pemesanan saat ini |
# | 20 | reserved_room_type | Object | Kode tipe kamar yang dipesan |
# | 21 | assigned_room_type | Object | Kode untuk tipe kamar yang ditetapkan untuk pemesanan. Terkadang tipe kamar yang ditetapkan berbeda dari tipe kamar yang dipesan karena alasan operasional hotel (misalnya pemesanan berlebih) atau permintaan pelanggan. Kode diberikan sebagai ganti penunjukan karena alasan anonimitas |
# | 22 | booking_changes | Integer | Jumlah perubahan pesanan kamar yang dilakukan sejak pemsanan dimasukkan ke PMS hingga saat check-in atau pembatalan |
# | 23 | deposit_type | Object | Jenis deposit pelanggan untuk menjamin pemesanan, terdiri dari: </br> 1. No Deposit : Tanpa deposit </br> 2. Non Refund : Deposit dilakukan dengan nilai total biaya menginap </br> 3. Deposit dilakukan dengan nilai di bawah total biaya menginap |
# | 24 | agent | Float | ID agen perjalanan yang melakukan pemesanan |
# | 25 | company | Float | ID perusahaan yang melakukan pemesanan atau yang bertanggung jawab untuk membayar pemesanan |
# | 26 | days_in_waiting_list | Integer | Jumlah hari pemesanan berada yang dalam daftar tunggu sebelum dikonfirmasi ke pelanggan |
# | 27 | customer_type | Object | Jenis pelanggan, terdiri dari: </br> 1. Contract : Ketika pemesanan memiliki peruntukan atau jenis kontrak lain yang terkait </br> 2. Group : Kelompok </br> 3. Transient : Ketika pemesanan bukan bagian dari suatu kelompok atau kontrak, dan tidak dikaitkan dengan pemesanan sementara lainnya </br> 4. Transient-Party : Ketika pemesanan bersifat sementara, tetapi dikaitkan dengan setidaknya pemesanan sementara lainnya|
# | 28 | adr | Float | Tarif Harian Rata-rata yang didefinisikan dengan membagi jumlah semua transaksi penginapan dengan jumlah total malam menginap |
# | 29 | required_car_parking_spaces | Integer | Jumlah tempat parkir mobil yang dibutuhkan oleh pelanggan |
# | 30 | total_of_special_requests | Integer | Jumlah permintaan khusus yang dibuat oleh pelanggan (misalnya tempat tidur kembar atau tinggi lantai) |
# | 31 | reservation_status | Object | Status terakhir reservasi, terdiri dari: </br> 1. Canceled : Pemesanan dibatalkan </br> 2. Check-Out : Check-out </br> 3. No-Show : Pelanggan tidak check-in dan tidak memberi tahu pihak hotel alasannya |
# | 32 | reservation_status_date | Object | Tanggal saat status terakhir ditetapkan. Variabel ini dapat digunakan bersama dengan ReservationStatus untuk mengetahui kapan pemesanan dibatalkan atau kapan pelanggan check-out dari hotel |

# In[1809]:


# Import library yang dibutuhkan untuk eksplorasi dataset
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import missingno
import streamlit as st

# Feature Engineering
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import category_encoders as ce
from scipy.stats import chi2_contingency

# Model Selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.metrics import roc_curve, roc_auc_score

# Imbalance Dataset
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import warnings
warnings.filterwarnings('ignore')

# Ignore Warning
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Set max columns
pd.set_option('display.max_columns', None)

# Saving Model
import pickle


# In[1810]:

st.title("Final Project Gamma")

# Load dataset
data = pd.read_csv('Dataset/hotel_bookings.csv')
st.write(data.head(), data.tail())


# In[1811]:


cancel_percentage = (pd.crosstab(data['is_canceled'], data['hotel'], normalize='columns') * 100).round(2)

cancel_summary = pd.concat([
    data['is_canceled'].value_counts().rename('Total'),
    pd.crosstab(data['is_canceled'], data['hotel']),
    cancel_percentage.rename(columns=lambda x: f"{x} (%)")
], axis=1)
cancel_summary.rename(index={0: "Not Canceled", 1: "Canceled"})


# Pada project ini, kita **hanya memilih data City Hotel** yang akan dianalisa dan dipakai untuk membangun model machine learning, yang dimana jika dilihat pada tabel diatas bahwa persentase pembatalan pemesanan City Hotel cukup besar yaitu di angka **41.73%**. Jadi kita memfokuskan hanya menggunakan data City Hotel untuk project kali ini

# In[1812]:


df = data[data['hotel'] == 'City Hotel']


# ***

# ## **3. Data Cleaning**

# Pada tahap ini, kita akan melakukan cleaning pada data yang nantinya data yang sudah dibersihkan akan kita gunakan untuk proses analisis selanjutnya. Beberapa hal yang perlu dilakukan adalah:
# - Melakukan treatment terhadap missing value, data duplikat dan outliers
# - Penambahan fitur baru dari hasil penggabungan dengan fitur lain dan melakukan Binning untuk fitur tersebut
# 
# Untuk proses data cleaning, kita akan menggunakan dataframe hasil duplikasi dari dataframe yang sebelumnya digunakan.

# In[1813]:


df_clean = df.copy()
 
st.write(df_clean.describe(), df_clean.describe(include='O'))


# In[1814]:


# data unik di tiap kolom
# listItem = []
# for col in df_clean.columns :
#     listItem.append( [col, df_clean[col].nunique(), df_clean[col].unique()])

# tabel1Desc = pd.DataFrame(columns=['Column Name', 'Number of Unique', 'Unique Sample'],
#                      data=listItem)
# tabel1Desc


# #### **Handling Missing Value**

# In[1815]:


df_clean.isnull().sum()


# In[1816]:


missingno.bar(df_clean,color="navy", sort="ascending", figsize=(14,7), fontsize=12);


# **Undefined Value**

# In[1817]:


undefined = df_clean.apply(lambda col: (col.astype(str) == 'Undefined').sum())
undefined[undefined > 0]


# Terdapat 4 fitur yang memiliki missing value dan 2 fitur yang memiliki undefined value, antara lain:
# - **company & agent** : Walaupun terdapat banyak missing value dari kedua fitur ini, kita tidak perlu memikirkan lebih lanjut karena nantinya kedua fitur ini akan kita hapus dikarenakan kedua fitur ini hanya berisi id unik dan tidak terpakai untuk analisa dan pengembangan model prediktif
# - **country & children** : Karena missing value pada kedua fitur ini tergolong kecil, diputuskan untuk menghapus baris yang terkandung missing value pada kedua fitur ini
# - **market_segment & distribution_channel** : Sama hal nya seperti **country & children**, baris pada kedua fitur ini akan di hapus karena jumlahnya kecil

# In[1818]:


df_clean.drop(columns=['agent','company'], inplace=True)
df_clean.dropna(axis=0, inplace=True)
df_clean.drop(df_clean[(df_clean['market_segment'] == 'Undefined') | (df_clean['distribution_channel'] == 'Undefined')].index, inplace=True)
print(f'Jumlah baris setelah baris yang terkandung "missing value" dan "undefined value" dihapus : {df_clean.shape[0]}')


# #### **Handling Duplicate**

# In[1819]:


duplicate = df_clean.duplicated()
print(f'Terdapat {duplicate.sum()} ({round(duplicate.sum() / df_clean.shape[0] * 100, 2)}%) baris duplikat dari total {df_clean.shape[0]} data')


# Keberadaan data duplikat dapat berdampak negatif pada model machine learning, karena memberikan bobot berlebih pada data yang sama, sehingga meningkatkan risiko **overfitting**. Hal ini membuat model lebih sulit untuk digeneralisasi dan kurang efektif dalam menangani data baru. Dengan menghapus data duplikat, ukuran dataset menjadi lebih representatif, meningkatkan akurasi model, dan memperbesar peluang model membuat prediksi yang lebih baik pada data yang belum pernah ditemui sebelumnya. Oleh karena itu, sebesar **32.67%** baris duplikat akan dihapus.
# 
# Sumber: https://medium.com/@anishnama20/how-duplicate-entries-in-data-set-leads-to-ovetfitting-2e3376e309c5 

# In[1820]:


df_clean.drop_duplicates(inplace=True)
print(f'Jumlah baris setelah baris duplikat dihapus : {df_clean.shape[0]}')


# #### **Handling Outliers**

# In[1821]:


num_cols = [
    'lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults',
    'children', 'babies', 'previous_cancellations', 'previous_bookings_not_canceled', 'booking_changes',
    'days_in_waiting_list', 'adr', 'required_car_parking_spaces', 'total_of_special_requests'
]

fig, axs = plt.subplots(5, 3, figsize=(20, 18))
axs = axs.flatten()

for i, data in enumerate(num_cols):
    sns.boxplot(data=df_clean[data], ax=axs[i], color="navy")
    axs[i].set_title(data)

for j in range(i+1, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.show()


# In[1822]:


def CheckOutliers(df_clean, col):
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR
    
    outliers = df_clean[(df_clean[col] < lower_fence) | (df_clean[col] > upper_fence)]
    
    results_df = pd.DataFrame({
        "Q1": [Q1],
        "Q3": [Q3],
        "IQR": [IQR],
        "Lower Fence": [lower_fence],
        "Upper Fence": [upper_fence],
        "Min Value" : [min(df_clean[col])],
        "Max Value" : [max(df_clean[col])],
        "Outliers Sum" : [len(outliers)],
        "Outliers Percentage": [round((len(outliers) / len(df_clean)) * 100, 2)]
    })

    return results_df

def PopulateOutliers(df_clean):
    numeric_cols = df_clean.select_dtypes(include='number').columns
    
    all_results = pd.DataFrame()
    
    for col in numeric_cols:
        result_df = CheckOutliers(df_clean, col)
        result_df.insert(0, 'Column', col)
        all_results = pd.concat([all_results, result_df], ignore_index=True)
    
    return all_results

PopulateOutliers(df_clean)


# Berdasarkan boxplot dan table diatas, dapat disimpulkan:
# 
# 1. **lead_time (3.18%)**
#     - Sebagian besar data berada dalam rentang normal (0–274), tetapi ada outlier dengan nilai di atas 274 yang akan dianalisis lebih lanjut
#     - Artinya, sebagian kecil pelanggan yang melakukan pemesanan sangat jauh dari tanggal menginap
# 2. **stays_in_weekend_nights (0.15%)**
#     - Outliers pada fitur sangat kecil, yang menandakan sebagian kecil pelanggan memesan kamar untuk malam akhir pekan dalam jumlah yang lebih lama dari kebanyakan pelanggan lain
# 3. **stays_in_week_nights (1.28%)**
#     - Ada sebagian kecil pelanggan melakukan pemesanan dengan jumlah malam yang jauh lebih tinggi dari biasanya. Ini bisa mencerminkan pelanggan yang menginap dalam waktu lama (long-stay guests)
# 4. **adults (28.89%)**
#     - Lebih serperempat data pemesesanan untuk orang dewasa pada fitur ini dianggap sebagai outlier
#     - walaupun secara persentase cukup besar, outlier pada fitur ini tergolong normal karena nilai maksimum adalah 4 orang dewasa. Untuk ukuran kamar hotel jumlah ini cukup masuk akal
# 5. **children (9.24%)**
#     - Jumlah maksimum anak-anak yang ada pada dataset ini juga masih masuk akal (3 orang anak) karena bisa saja pelanggan yang memesan untuk sebuah keluarga dengan banyak anak
# 6. **babies (0.69%)**
#     - Walaupun outlier pada fitur ini tergolong kecil, perlu di analisa kembali karena nilai maksimumnya adalah 10 bayi
# 7. **previous_cancellations (2.21%)**
#     - Beberapa pelanggan memiliki riwayat pembatalan yang sangat tinggi, nilainya sampai 21 kali pembatalan
# 8. **previous_bookings_not_canceled (2.87%)**
#     - Beberapa pelanggan memiliki banyak riwayat pemesanan sebelumnya yang tidak dibatalkan, ini menandakan bahwa sebagian kecil adalah pelanggan loyal
# 9. **booking_changes (16.56%)**
#     - Persentase outlier pada fitur ini cukup besar, menunjukkan ada beberapa tamu yang sering mengubah pemesanannya berkali-kali
# 10. **days_in_waiting_list (1.34%)**
#     - Outlier pada fitur ini menunjukkan adanya sebagian kecil pelanggan yang bersedia untuk menunggu lebih lama untuk mendapatkan kamar
#     - Fitur ini akan kita hapus karena hampir keseluruhan datanya (98.66%) berisi value 0, yang dimana data pada fitur ini tidak bervariasi dan tidak relevan untuk membangun model prediktif kita
# 11. **adr (4.66%)**
#     - Outlier pada fitur ini menunjukkan adanya sebagian data tarif harian rata-rata berada di bawah dan di atas rentang normal, ini akan dianalisis lebih lanjut
# 12. **required_car_parking_spaces (3.55%)**
#     - Walaupun outlier pada fitur ini tergolong kecil, akan tetapi jika diperhatikan beberapa pelanggan memerlukan tempat parkir lebih dari semestinya (3 tempat parkir). Ini perlu di analisa kembali apakah data ini normal atau tidak
# 13. **total_of_special_requests (3.14%)**
#     - Beberapa pelanggan memiliki banyak permintaan khusus yang jauh di atas rata-rata dibandingkan dengan pelanggan lain
# 
# Outliers pada sebagian besar fitur diatas tidak kita analisa lebih lanjut, karena nilai-nilai diluar rentang batas atas dirasa cukup masuk akal. Akan tetapi, tedapat 4 fitur yang menjadi perhatian khusus, yaitu **leadtime**, **babies**, **adr** dan **required_car_parking_spaces** yang dimana kedua fitur ini akan kita analisa lebih lanjut.    

# #### **Handling Anomaly Data**

# **1. Cek apakah ada anomali data dengan kondisi tidak cancel tetapi status reservasinya adalah Check-Out dan juga sebaliknya**

# In[1823]:


print(df_clean[(df_clean['is_canceled'] == 1) & (df_clean['reservation_status'] == 'Check-Out')].shape[0])
print(df_clean[(df_clean['is_canceled'] == 0) & ((df_clean['reservation_status'] == 'Canceled') | (df_clean['reservation_status'] == 'No-Show'))].shape[0])


# **2. lead_time**

# In[1824]:


df_clean[df_clean['lead_time'] > 355].shape[0]


# Lead time adalah jumlah hari antara tanggal pemesanan dan tanggal kedatangan, terdapat sebagian kecil pelanggan menunggu di waktu yang cukup lama (lebih dari 1 tahun). Pada fitur ini kita akan membatasi outlier yang nilainya lebih dari 1 tahun (365 hari) karena rasanya terlalu lama menunggu dalam jangka waktu lebih dari satu tahun. Ini akan menjadi limitasi project kita.

# In[1825]:


df_clean = df_clean[df_clean['lead_time'] < 366]
df_clean.shape[0]


# **3. adults**

# In[1826]:


df_clean[df_clean['adults'] < 1].shape[0]


# Berdasarkan pengecekan diatas terdapat 370 baris data yang mengindikasikan pemesanan tanpa orang dewasa, ini bisa terjadi karena terdapat anak-anak/babies yang menginap tanpa didampingi orang tua dan ada juga baris data yang tidak ada nilai diantara 3 fitur (adults, children dan babies), yang mengindikasikan tidak ada yang menginap. Data ini akan kita hapus

# In[1827]:


df_clean = df_clean[df_clean['adults'] > 0]
df_clean.shape[0]


# **4. babies**

# In[1828]:


df_clean[df_clean['babies'] > 1][['adults', 'children', 'babies']].reset_index(drop=True)


# Dari diatas terlihat cukup normal, tidak ada yang perlu di handle karena sudah terhapus pada proses pengecekkan data adults.

# **5. adr**

# In[1829]:


df_clean.sort_values(by='adr', ascending=False).head(5)


# Nilai Average Daily Rate pada Baris paling pertama menunjukkan nilai yang tidak normal. Dari sekian banyak data hanya 1 baris ini yang memiliki nilai adr yang sangat signifikan dibandinkan dengan lainnya. Kemungkinan ini disebabkan oleh kesalahan input jadi akan kita hapus 1 baris ini

# In[1830]:


df_clean[df_clean['adr'] < 1].shape[0]


# In[1831]:


df_adr = df_clean[df_clean['adr'] < 1]
df_adr['market_segment'].value_counts()


# Dari hasil pengecekan diatas terdapat nilai 0 pada fitur adr(tarif harian rata-rata). Jika di periksa lebih lanjut hampir sebagian besar berasal dari Complementary dimana pemesanan gratis atau diberikan sebagai kompensasi, misalnya untuk tamu VIP atau bagian dari promosi. [source](https://www.traveloka.com/id-id/explore/destination/macam-macam-status-kamar-hotel-acc/428326)
# 
# Meskipun penjelasan tersebut logis, secara operasional nilai adr = 0 tidak mencerminkan tarif harian aktual dan tidak valid untuk analisis. Oleh karena itu, nilai adr = 0 akan dihapus dari dataset untuk menjaga integritas analisis dan akurasi model.

# In[1832]:


df_clean = df_clean[(df_clean['adr'] < 5400) & (df_clean['adr'] >= 1)]
df_clean.shape[0]


# **6. required_car_parking_spaces**

# In[1833]:


df_clean[df_clean['required_car_parking_spaces'] > 1][['adults', 'children', 'babies', 'required_car_parking_spaces']].reset_index(drop=True)


# Umumnya sebuah keluarga kecil hanya memerlukan 1 mobil untuk menampung keluarganya (asumsi 4 orang untuk satu mobil kecil). Data diatas bisa dianggap normal jika pelanggan memesan tempat parkir lebih untuk tamu lain yang akan datang belakangan (keluarga atau rekan bisnis). Akan tetapi, terdapat data yang harus dihapus karena pelanggan membutuhkan tempat parkir yang melebihi jumlah orang yang akan menginap di hotel

# In[1834]:


df_clean = df_clean[df_clean['required_car_parking_spaces'] <= 2]
print(f'Jumlah baris setelah dataset di cleaning : {df_clean.shape[0]} baris')


# #### **Penambahan fitur baru, Binning dan Recategorize**

# Penambahan fitur baru **length_of_stay** yang didapat dari hasil penggabungan dua fitur (**stays_in_weekend_nights** dan **stays_in_week_nights**), fitur ini menginformasikan berapa lama waktu yang dipesan untuk menginap. Fitur ini akan dilakukan juga proses binning untuk membuat klasifikasi berdasarkan lama waktu yang dipesan untuk menginap.
# 
# Berdasarkan [roof264.com](https://roof264.com/classification-of-hotels-by-length-of-guest-stay), berikut klasifikasi berdasarkan lama menginap di hotel :
# 1. Temporary Hotel (≤ 1 hari)
# 2. Commercial Hotel (2–7 hari)
# 3. Extended Stay Hotel (8–14 hari)
# 4. Semi-residential Hotel (15–29 hari)
# 5. Residence/Apartment Hotel (≥ 30 hari)

# In[1835]:


df_clean['length_of_stay'] = df_clean["stays_in_weekend_nights"] + df_clean["stays_in_week_nights"]

def categorize_stay(nights):
    if nights <= 1:
        return "Temporary Hotel"
    elif nights <= 7:
        return "Commercial Hotel"
    elif nights <= 14:
        return "Extended Stay Hotel"
    elif nights <= 29:
        return "Semi-residential Hotel"
    else:
        return "Residence/Apartment Hotel"

df_clean['stay_category'] = df_clean['length_of_stay'].apply(categorize_stay)

df_clean['stay_category'].value_counts()


# Melakukan penyesuaian room type berdasarkan [kualitas kamar](https://sanhotelseries.com/hotel-room-classification-a-complete-guide/) (**Standard**, **Superior**, **Deluxe**, dan **Suite**) yang dihasilkan dari fitur **assigned_room_type** dengan memperhatikan rata-rata dari fitur **adr**

# In[1836]:


df_clean.groupby('assigned_room_type')['adr'].mean().sort_values()


# In[1837]:


def replace_room_type(df_clean):
    mapping = {
        'A': 'Standard',
        'B': 'Standard',
        'C': 'Superior',
        'K': 'Superior',
        'D': 'Deluxe',
        'E': 'Deluxe',
        'F': 'Suite',
        'G': 'Suite'
    }

    df_clean['assigned_room_type'] = df_clean['assigned_room_type'].replace(mapping)
    df_clean['reserved_room_type'] = df_clean['reserved_room_type'].replace(mapping)

    return df_clean

df_clean = replace_room_type(df_clean)


# **Pengecekan Distribusi Data**

# In[1838]:


# Memilih kolom numerik, kecuali yang ingin dikecualikan
exclude_cols = ['is_canceled', 'arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month', 'is_repeated_guest']
numeric_cols = df_clean.select_dtypes(include='number').drop(columns=exclude_cols, errors='ignore')

# Membuat subplots dengan ukuran 6x3
fig, axes = plt.subplots(5, 3, figsize=(20, 15))
axes = axes.flatten()  # Mengubah axes menjadi array 1D untuk iterasi yang lebih mudah

# Memplot histogram untuk setiap kolom numerik yang tersisa
for i, column in enumerate(numeric_cols):
    sns.histplot(numeric_cols[column], kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {column}')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Count')

    
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Menyesuaikan tata letak
plt.tight_layout()
plt.show()


# Berdasarkan histogram diatas, terlihat jelas bahwa semua variabel tidak terdistribusi normal. Untuk analisis korelasi menggunakan Spearman sebagai metode korelasi dapat memberikan wawasan yang lebih akurat tentang hubungan antar variabel.

# #### **Clean Dataset**

# In[1839]:


st.write(df_clean.head(), df_clean.tail())
st.write(df_clean.info())


# In[1840]:


## simpan dataset untuk dashboard tableau
# df_clean.to_excel('hotel_bookings_clean.xlsx', index=False)


# ## **4. Data Analysis**

# In[1841]:


df_clean['is_canceled'].value_counts()


# In[1842]:


plt.figure(figsize=(6, 6))
plt.pie(df_clean['is_canceled'].value_counts(), labels=['Not Canceled', 'Canceled'], autopct='%1.2f%%', colors=['royalblue', 'red'])
plt.title("Persentase Pembatalan")
plt.show()


# - Persentase dari tamu yang tidak membatalkan reservasi adalah 69.72% sedangkan Persentase untuk tamu yang membatalkan reservasi adalah 30.28%.
# - Jika di lihat dari Persentase pembatalan yang berada di angka 30.28%, pembatalan di hotel ini terbilang cukup tinggi. Hal ini menjadi masalah karena secara global, rata-rata tingkat pembatalan pemesanan hotel sekitar 20%.
# [Source](https://webrezpro-com.translate.goog/hospitality-by-the-numbers-40-stats-you-should-know/?_x_tr_sl=en&_x_tr_tl=id&_x_tr_hl=id&_x_tr_pto=sge#:~:text=Globally%2C%20the%20average%20hotel%20booking,compared%20to%2025%25%20in%202021.)

# In[1843]:


df_no_cancel = df_clean[df_clean['is_canceled'] == 0]
df_cancel = df_clean[df_clean['is_canceled'] == 1]


# #### **lead_time**

# In[1844]:


plt.figure(figsize=(10,5))
sns.kdeplot(df_clean[df_clean['is_canceled'] == 1]['lead_time'], label='Canceled', shade=True, color='red')
sns.kdeplot(df_clean[df_clean['is_canceled'] == 0]['lead_time'], label='Not Canceled', shade=True, color='green')
plt.xlabel('Lead Time (Days)')
plt.ylabel('Density')
plt.title('Lead Time Distribution for Canceled vs Not Canceled Bookings')
plt.legend()
plt.show()


# **Insight**
# 
# Jika dilihat dari KDE Plot diatas, dapat disimpulkan bahwa :
# - Mayoritas pemesanan memiliki lead time pendek (dibawah 50 hari)
# - kurva merah lebih landai dibandingkan kurva hijau, artinya pemesanan yang dibatalkan tersebar dalam berbagai lead time. tetapi lebih dominan pada lead time panjang
# - kurva hijau lebih terkonsenstrasi pada lead time pendek, artinya pemesanan yang tidak dibatalkan cenderung dilakukan dalam waktu dekat sebelum tanggal menginap
# - pada satu titik ketika melewati lead time 25 hari, cancel lebih tinggi dibandingkan dengan not canceled. kemudian seterusnya cancel selalu lebih tinggi dibandingkan not cancelednya.
# 
# **Recommendation**
# 
# Untuk mengurangi pembatalan oleh tamu yang memiliki lead time diatas 25 hari, kita bisa merekomendasikan pihak hotel untuk membuat penawaran khusus bagi para tamu yang memiliki lead time diatas 25 hari. Seperti memberi voucher potongan harga untuk upgrade kamar atau juga bisa voucher makan bagi tamu yang melakukan reservasi dengan lead time diatas 25 hari dengan beberapa ketentuan tambahan.
# 
# **Action**
# 
# Voucher hanya diberikan untuk tamu dengan lead time lebih dari 25 hari, voucher yang di sediakan :
# - Voucher diskon upgrade kamar sebesar 12% s/d €14.10 dengan minimal kamar yang di pesan adalah kamar Superior
# - Khusus untuk tamu dari kamar Suite karena tidak bisa upgrade kamar maka bisa memilih 1 dari 2 voucher yang disediakan :
#     1. Voucher makan sebesar 10% bisa pilih lunch atau dinner (tidak termasuk minuman/wine)
#     2. Voucher spa 12% untuk 1x pemakaian

# #### **arrival_date_month**

# In[1845]:


# Definisikan urutan bulan agar ditampilkan secara kronologis
month_order = ["January", "February", "March", "April", "May", "June", 
               "July", "August", "September", "October", "November", "December"]

# Agregasi jumlah reservasi per bulan per tahun
monthly_counts = df_clean.groupby(['arrival_date_year', 'arrival_date_month']).size().reset_index(name='reservations')
monthly_cancel_counts = df_cancel.groupby(['arrival_date_year', 'arrival_date_month']).size().reset_index(name='reservations')

# Ubah 'arrival_date_month' menjadi kategori dengan urutan yang ditentukan
monthly_counts['arrival_date_month'] = pd.Categorical(monthly_counts['arrival_date_month'], categories=month_order, ordered=True)
monthly_cancel_counts['arrival_date_month'] = pd.Categorical(monthly_cancel_counts['arrival_date_month'], categories=month_order, ordered=True)

# Urutkan data berdasarkan tahun dan bulan
monthly_counts = monthly_counts.sort_values(['arrival_date_year', 'arrival_date_month'])
monthly_cancel_counts = monthly_cancel_counts.sort_values(['arrival_date_year', 'arrival_date_month'])

# Hitung pertumbuhan persentase (month-on-month growth) untuk total reservasi dan reservasi yang dibatalkan
monthly_counts['growth_pct'] = monthly_counts.groupby('arrival_date_year')['reservations'].pct_change() * 100
monthly_cancel_counts['growth_pct'] = monthly_cancel_counts.groupby('arrival_date_year')['reservations'].pct_change() * 100

# Buat subplot untuk masing-masing tahun
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
years = [2015, 2016, 2017]

for ax, year in zip(axes, years):
    # Filter data untuk tahun tertentu
    df_cancel_year = monthly_cancel_counts[monthly_cancel_counts['arrival_date_year'] == year]
    
    # Plot pertumbuhan persentase reservasi yang dibatalkan (merah)
    sns.lineplot(x='arrival_date_month', y='growth_pct', data=df_cancel_year, marker='o', ax=ax, color='red', label='Canceled Growth %')
    
    ax.set_title(f"Pertumbuhan % Reservasi yang Dibatalkan Bulanan di {year}")
    ax.set_xlabel("Bulan")
    ax.set_ylabel("Growth (%)")
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    
    # Tambahkan anotasi persentase di atas setiap titik untuk garis pembatalan
    for line in ax.get_lines():
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        for x, y in zip(xdata, ydata):
            if not np.isnan(y):
                ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=9)

plt.tight_layout()
plt.show()


# **Insight**
# 
# - Setiap tahun menunjukkan fluktuasi cukup tajam dari satu bulan ke bulan berikutnya (misalnya, ada lonjakan signifikan hingga di atas 50%, lalu turun ke kisaran belasan persen). Ini menandakan bahwa pembatalan reservasi oleh tamu tidak selalu stabil dan dipengaruhi oleh berbagai faktor (musiman, promosi, kondisi ekonomi, dll.)[[1]](https://myportugalholiday.com/).
# 
# **Recommendation**
# 
# Sehingga hotel perlu menerapkan strategi yang tidak hanya memaksimalkan pemesanan saat puncak musim liburan maupun tidak, tetapi juga mengurangi risiko pembatalan. seperti : 
# 1. Ketika sedang high season, harga per-kamar dinaikkan dengan menyesuaikan tingkat permintaan dan Ketersediaan Kamar. Sehingga pihak hotel bisa memberikan diskon cukup besar agar tamu tertarik untuk menginap. 
# 2. Ketika sedang low season, karena sepi maka cukup banyak pilihan hotel yang tersedia sehingga tamu bisa memilih alternatif hotel lain yang lebih menguntungkan bagi tamu [[2]](https://myportugalholiday.com/portugal-guides/portugal-in-february.html). Sehingga kamu menyarankan pihak hotel untuk membuat strategi Penawaran insentif seperti pemberian voucher(voucher diskon kamar, voucher makan atau juga voucher layanan fasilitas tertentu.)
# 
# **Action**
# 
# Hal yang harus di lakukan pihak hotel saat High Season :
# 1. Menaikkan harga kamar sebesar 30% dari harga asli kamar, hal ini di lakukan untuk bisa memberikan diskon besar pada tamu agar tamu merasa tertarik karena melihat perbedaan harga sebelum diskon dan sesudah diskon.
# 2. Memberikan Diskon sebesar 20% dari harga kamar yang sudah dinaikkan kepada para tamu, namun diskon ini diikuti dengan beberapa ketentuan, antara lain :
#     - Tamu memesan langsung melalui website resmi hotel maupun datang langsung ke hotel.
#     - Lama menginap tamu minimal 2 malam.
# 3. Memberikan Diskon sebesar 10% dari harga kamar yang sudah dinaikkan kepada para tamu khusus pemesanan melalui TA/TO, namun diskon ini diikuti dengan beberapa ketentuan, antara lain :
#     - Lama menginap tamu minimal 2 malam.
#     - Pemesanan dilakukan H-7 dari tanggal Check-In.
# 
# Hal yang harus di lakukan pihak hotel saat Low Season : 
# 1. Harga kamar tidak akan dinaikkan seperti saat High Season.
# 2. Buat promosi terbatas selama 1-2 minggu dengan memberikan diskon 20% dari harga asli kamar, namun dengan beberapa ketentuan, antara lain : 
#     - Tamu memesan langsung melalui website resmi hotel maupun datang langsung ke hotel.
#     - Lama menginap tamu minimal 2 malam.

# #### **distribution_channel**

# In[1846]:


import math

channel_data = df_clean.groupby(['distribution_channel', 'is_canceled']).size().reset_index(name='count')

channel_pivot = channel_data.pivot(index='distribution_channel', columns='is_canceled', values='count').fillna(0)
channel_pivot.columns = ['Not Canceled', 'Canceled']

num_channels = len(channel_pivot.index)

cols = 3
rows = math.ceil(num_channels / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
axes = axes.flatten()

for i, channel in enumerate(channel_pivot.index):
    values = channel_pivot.loc[channel]
    axes[i].pie(values, labels=values.index, autopct='%1.1f%%', startangle=90, colors=['blue', 'red'])
    axes[i].set_title(f"{channel}")
    
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Perbandingan Reservasi: Dibatalkan vs. Tidak Dibatalkan per Distribution Channel", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()


# **Insight**
# 
# Jika di lihat dari pie chart diatas, bisa kita ketahui bahwa Distribution Channel yang memiliki pembatalan reservasi paling banyak adalah TA/TO (Travel agent/tour operators) yaitu 33.1%. Kemudian GDS memiliki pembatalan yang cukup tinggi kedua yaitu 20.3%, dimana GDS merupakan platform teknologi dengan Travel Agent sebagai penggunanya.
# 
# **Recommendation**
# 
# - Dikarenakan kita tidak mengetahui kebijakan apa yang digunakan TA/TO serta GDS dalam pembatalan reservasi, maka kita perlu Menegosiasikan kebijakan pembatalan yang lebih ketat dengan TA/TO serta Menerapkan deposit atau pembayaran di muka untuk mengurangi pembatalan.
# - Pihak hotel juga dapat meminta pihak TA/TO untuk mengumpulkan umpan balik dari tamu yang membatalkan reservasi melalui TA/TO untuk mengidentifikasi area yang perlu perbaikan.
# 
# **Action**
# 
# Hal yang di berlakukan untuk TA/TO serta GDS adalah :
# 1. Biaya Deposit minimal 60% untuk setiap kamar
# 2. Jika tamu membatalkan reservasi h-4 maka deposit hanya akan dikembalikan 70%
# 3. Jika tamu membatalkan reservasi h-3 maka deposit hanya akan dikembalikan 40%
# 4. Jika tamu membatalkan reservasi h-2 maka deposit akan hangus dan tidak bisa dikembalikan
# 5. Pihak TA/TO wajib meminta umpan balik dari tamu yang membatalkan reservasi

# #### **is_repeated_guest**

# In[1847]:


repeat_data = df_clean.groupby(['is_repeated_guest', 'is_canceled']).size().reset_index(name='count')

repeat_pivot = repeat_data.pivot(index='is_repeated_guest', columns='is_canceled', values='count').fillna(0)
repeat_pivot.columns = ['Not Canceled', 'Canceled']

repeat_pivot.index = repeat_pivot.index.map({0: "Tamu Baru", 1: "Tamu Lama"})

num_repeat = len(repeat_pivot.index)

cols = 2
rows = math.ceil(num_repeat / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
axes = axes.flatten()

for i, repeat in enumerate(repeat_pivot.index):
    values = repeat_pivot.loc[repeat]
    axes[i].pie(values, labels=values.index, autopct='%1.1f%%', startangle=90, colors=['blue', 'red'])
    axes[i].set_title(f"{repeat}")
    
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Perbandingan Reservasi: Dibatalkan vs. Tidak Dibatalkan per Repeated Guest", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()


# **Insight**
# 
# Persentase tamu baru yang membatalkan reservasi adalah 30.8%, angka ini cukup tinggi dibandingkan tamu lama yang membatalkan reservasi yaitu 12.6%. Hal ini mungkin di sebabkan karena tamu baru belum begitu mengetahui fasilitas maupun pelayanan seperti apa yang diberikan oleh hotel, sedangkan tamu yang sudah pernah menginap pasti sudah mengetahui dengan baik fasilitas dan pelayanan apa yang di dapat sehingga pembatalan reservasi dari tamu yang sudah pernah menginap lebih sedikit.
# 
# **Recommendation**
# 
# maka dari itu kami merekomendasikan pihak hotel untuk melakukan strategi agar mengurangi pembatalan reservasi dari para tamu baru, seperti :
# - Pastikan informasi mengenai fasilitas, layanan, dan keunggulan hotel tersaji dengan jelas pada platform promosi hotel.
# - Tampilkan testimoni dari tamu-tamu yang pernah menginap sehingga tamu baru merasa lebih percaya dan yakin untuk melanjutkan reservasi mereka.
# 
# **Action**
# 
# Hal yang harus di lakukan pihak hotel :
# 1. Tim pemasaran harus memastikan kelengkapan informasi mengenai fasilitas, layanan, dan keunggulan hotel di website, email konfirmasi, dan materi pemasaran.
# 2. Pada website hotel/media sosial untuk promosi hotel perlu di tampilkan testimoni dari tamu-tamu yang pernah menginap seperti ulasan tamu, foto, atau video yang menggambarkan pengalaman menginap yang positif.
# 3. Berikan syarat pembatalan reservasi, dimana tamu wajib memberikan review atau alasan pembatalan.

# #### **deposit_type**

# In[1848]:


deposit_type_counts = df_cancel['deposit_type'].value_counts()
deposit_type_counts


# In[1849]:


plt.figure(figsize=(6, 6))
plt.pie(deposit_type_counts, labels=deposit_type_counts.index, autopct='%1.2f%%', startangle=140)
plt.title('Pemilihan Deposit Type yang paling banyak melakukan pembatalan reservasi')
plt.show()


# **Insight**
# 
# - Tamu yang tidak memiliki deposit cenderung membatalkan reservasi dibandingkan tamu yang memilih Deposit Non Refund dan juga Refundable. hal itu di karenakan mereka tidak perlu memikirkan kerugian jika membatalkan reservasi karena tidak melakukan deposit, sehingga para tamu bisa dengan lebih mudah membatalkan reservasi jika memang menemukan pilihan hotel lain yang lebih baik ataupun lebih murah.
# - Sedangkan tamu yang sudah memiliki deposit sebelumnya akan cenderung lebih memikirkan lagi jika ingin melakukan pembatalan reservasi. 
# 
# **Recommendation**
# 
# Karena Deposit membantu hotel memastikan pemesanan lebih stabil dan mengurangi kemungkinan adanya kamar kosong akibat pembatalan mendadak. Kami memberikan rekomendasi pada pihak hotel untuk mengubah kebijakan perihal deposit.
# 
# **Action**
# 
# Hal yang perlu diberlakukan oleh pihak hotel :
# 1. Pilihan untuk tidak melakukan deposit di tiadakan
# 2. Pihak hotel memberlakukan agar setiap tamu yang memesan kamar wajib memberikan deposit minimal 50%
# 3. Jika tamu membatalkan reservasi h-4 maka deposit hanya akan dikembalikan 60%
# 4. Jika tamu membatalkan reservasi h-3 maka deposit hanya akan dikembalikan 30%
# 5. Jika tamu membatalkan reservasi h-2 maka deposit akan hangus dan tidak bisa dikembalikan

# **Pola pembatalan di City Hotel menunjukkan bahwa:**
# 
# - Reservasi dengan jangka waktu lead time yang panjang dan tanpa deposit memiliki risiko pembatalan yang tinggi.
# - Volume pembatalan meningkat secara proporsional dengan lonjakan reservasi di musim liburan, terutama melalui channel TA/TO.
# - Tamu baru secara signifikan lebih rentan membatalkan reservasi dibandingkan dengan tamu lama, yang menunjukkan perlunya strategi khusus untuk meningkatkan kepercayaan dan komitmen pemesanan di segmen tamu baru.

# ## **5. Data Preprocessing & Feature Engineering**

# In[1850]:


df_model = df_clean.copy()


# Setelah sebelumnya sudah dilakukan treatment terhadap missing value, duplikat value dan outliers. pada tahap ini kita akan mempersiapkan dataset untuk dipakai membangun model prediktif

# **Pengecekan Korelasi Numerik**

# In[1851]:


plt.figure(figsize=(15, 12))
corr = df_model.corr(numeric_only=True, method='spearman')
matriks = np.triu(corr)
sns.heatmap(corr, annot=True, fmt='.2f', mask=matriks, cmap='cividis', square=True, linewidths=.5)
plt.title('Correlation Matrix', size=15, weight='bold')
plt.show()


# Berdasarkan pengecekan korelasi diatas, bisa di simpulkan :
# - lead_time memiliki korelasi yang paling tinggi dengan is_canceled dibandingkan fitur lain. Ini berarti semakin lama jeda dari tanggal pemesanan ke tanggal check-in, maka tamu semakin berpotensi untuk membatalkan pemesanan meskipun korelasinya tergolong korelasi lemah (0.21%)
# - babies memiliki korelasi yang paling kecil dengan is_canceled dibandingkan fitur lain, bahkan bisa dibilang hampir tidak ada korelasi (0.02%)
# - previous_booking_not_canceled memiliki korelasi yang sangat kuat dengan is_repeat_guest (0.87%), hal ini sangat wajar karena tamu yang pernah menginap sebelumnya dan tidak membatalkan reservasi cenderung menjadi tamu berulang (repeat guest)

# **Pengecekan Chi Square untuk fitur-fitur kategorikal**

# In[1852]:


categorical_features = ['meal', 'market_segment', 'distribution_channel', 
                        'reserved_room_type', 'assigned_room_type', 
                        'deposit_type', 'customer_type', 'stay_category']

chi_square_results = []

for feature in categorical_features:
    contingency_table = pd.crosstab(df_model[feature], df_model['is_canceled']) 
    chi2, p, _, _ = chi2_contingency(contingency_table)
    
    significance = "Signifikan" if p < 0.05 else "Tidak Signifikan"
    
    chi_square_results.append({'Feature': feature, 'Chi': chi2, 'P-Value': p, 'Significance': significance})

chi_square_df = pd.DataFrame(chi_square_results).sort_values(by='P-Value')
chi_square_df


# Berdasarkan hasil pengencekan chi-square diatas, semua fitur kategorik memiliki pengaruh yang signifikan terhadap fitur **is_canceled** yang ditandai dengan kecilnya p-value (tidak ada p-value yang nilainya melebihi 0.05). Diputuskan untuk memakai semua fitur kategorik diatas untuk pengembangan model prediktif

# #### **Drop Columns**
# 
# Pada tahap ini, kita akan menghapus fitur-fitur yang tidak relevan untuk pengembangan model prediktif, dan juga akan menghapus fitur-fitur dengan memperhatikan korelasi untuk melakukan feature selection. fitur fitur yang akan kita hapus diantaranya :
# - fitur **hotel** tidak diperlukan karena hanya berisi 1 data unik saja
# - fitur **arrival_date_year, arrival_date_month, arrival_date_week_number dan arrival_date_day_of_month** hanya berisi informasi mengenai waktu kedatangan tamu, ini tidak diperlukan untuk model
# - fitur **country** tidak diperlukan, karena kita tidak menganalisa berdasarkan negara asal tamu
# - fitur **adr** tidak diperlukan untuk model prediksi pembatalan jika digunakan untuk masa depan, kita tidak bisa mengetahui nilai adr sebelum tamu benar-benar menginap atau sistem menentukan harga final
# - fitur **reservation_status dan reservation_status_date** tidak diperlukan, karena fitur ini langsung menunjukkan apakah pemesanan dibatalkan atau tidak, sehingga tidak boleh digunakan dalam model prediksi
# - fitur **babies** tidak dipakai karena memiliki korelasi yang kecil
# - fitur **days_in_waiting_list** tidak dipakai karena memiliki korelasi yang kecil
# - fitur **children** tidak dipakai karena memiliki korelasi yang kecil
# - fitur **stays_in_weekend_nights** tidak dipakai karena memiliki korelasi yang kecil

# Pemilihan fitur-fitur ini ditentukan juga dari hasil benchmarking terbaik dari file **FinalProjectGamma-TestModelBenchmark.ipynb**

# In[1853]:


df_model.drop(['hotel','arrival_date_year','arrival_date_month','arrival_date_week_number','arrival_date_day_of_month','country','adr','reservation_status','reservation_status_date','babies','days_in_waiting_list','children','stays_in_weekend_nights'], axis=1, inplace=True)


# #### **Encoding**

# Pada tahap ini, kita akan melakukan encoding untuk fitur-fitur kategorikal yang ada pada dataset ini, yang akan kita lakukan adalah :
# - Melakukan One Hot Encoding untuk fitur-fitur kategorikal nominal, diantaranya : `meal`, `market_segment`, `distribution_channel`, `deposit_type`, `customer_type` dan `stay_category`. Teknik ini digunakan karena fitur-fitur tersebut memiliki jumlah kategori yang sedikit dan tidak memiliki urutan yang bermakna.
# - Melakukan Ordinal Encoding untuk fitur `reserved_room_type` dan `assigned_room_type`. Awalnya, kedua fitur ini hanya berisi huruf tanpa makna urutan yang jelas. Namun, setelah dilakukan pemetaan berdasarkan tarif rata-rata (ADR), kini kedua fitur ini memiliki urutan yang jelas berdasakan tipe kamar, sehingga cocok untuk Ordinal Encoding.

# In[1854]:


# Ordinal mapping kolom reserved_room_type & assigned_room_type
ordinal_mapping = [
    {'col':'reserved_room_type',
    'mapping':{
        'Standard' : 0,
        'Superior': 1, 
        'Deluxe': 2, 
        'Suite' : 3
    }},
    {'col':'assigned_room_type',
    'mapping':{
        'Standard' : 0,
        'Superior': 1, 
        'Deluxe': 2, 
        'Suite' : 3
    }}
]


# In[1855]:


transformer = ColumnTransformer([
    ('onehot', OneHotEncoder(drop='first'), ['meal','market_segment', 'distribution_channel', 'deposit_type', 'customer_type', 'stay_category']),
    ('ordinal', ce.OrdinalEncoder(mapping= ordinal_mapping), ['reserved_room_type', 'assigned_room_type'])
], remainder='passthrough')


# #### **Features & Target**

# In[1856]:


# Memisahkan data independen variabel dengan target
x = df_model.drop(columns=['is_canceled'])
y = df_model['is_canceled']


# #### **Splitting**

# In[1857]:


# Splitting data training dan test dengan proporsi 80:20
x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,test_size=0.2,random_state=5)


# In[1858]:


# Melihat preview hasil encoding 
testing = pd.DataFrame(transformer.fit_transform(x_train), columns=transformer.get_feature_names_out())
testing


# ## **6. Modelling**

# #### **Choose a Benchmark Model**

# In[1859]:


logreg = LogisticRegression()
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
xgb = XGBClassifier()
lgbm = lgb.LGBMClassifier()

models = [logreg,knn,dt,rf,xgb,lgbm]
score=[]
rata=[]
std=[]

for i in models:
    skfold=StratifiedKFold(n_splits=5)
    estimator=Pipeline([
        ('preprocess',transformer),
        ('model',i)])
    model_cv=cross_val_score(estimator,x_train,y_train,cv=skfold,scoring='roc_auc')
    score.append(model_cv)
    rata.append(model_cv.mean())
    std.append(model_cv.std())
    
pd.DataFrame({'model':['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'XGBoost', 'LightGBM'],'mean roc_auc':rata,'sdev':std}).set_index('model').sort_values(by='mean roc_auc',ascending=False)


# Akan dilakukan prediksi pada test set dengan 2 benchmark model terbaik, yaitu LightGBM dan XGBoost

# #### **Predict to Test Set with the Benchmark Model**

# In[1860]:


models = [logreg,knn,dt,rf,xgb,lgbm]
score_roc_auc = []

def y_pred_func(i):
    estimator=Pipeline([
        ('preprocess',transformer),
        ('model',i)])
    x_train,x_test
    
    estimator.fit(x_train,y_train)
    return(estimator,estimator.predict(x_test),x_test)

for i,j in zip(models, ['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'XGBoost','LightGBM']):
    estimator,y_pred,x_test = y_pred_func(i)
    y_predict_proba = estimator.predict_proba(x_test)[:,1]
    score_roc_auc.append(roc_auc_score(y_test,y_predict_proba))
    print(j,'\n', classification_report(y_test,y_pred))
    
pd.DataFrame({'model':['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'XGBoost','LightGBM'],
             'roc_auc score':score_roc_auc}).set_index('model').sort_values(by='roc_auc score',ascending=False)


# Terlihat bahwa model XGBoost dan LightGBM memiliki hasil yang baik pada test data

# ### **Resampling**

# Untuk kedua model ini (XGBoost dan LightGBM) akan dicoba untuk dilakukan resampling, apakah kita bisa mendapatkan hasil yang lebih baik lagi. 
# Beberapa hal yang perlu diperhatikan dalam benchmark model ini antara lain:
# 
# - Karena dataset ini tidak seimbang (imbalance), maka perlu dilakukan proses resampling. Metode yang digunakan untuk resampling adalah **RandomOverSampling** yang bertujuan untuk menambahkan data pada kelas yang minoritas dan **RandomUnderSampling** yang bertujuan untuk mengurangi data pada kelas yang mayoritas
# - Dilakukan stratified K-Fold untuk menjaga distribusi kelas target agar tetap konsisten saat melakukan pembagian data untuk cross-validation.

# #### **Test Oversampling with K-Fold Cross Validation**

# In[1861]:


def calc_train_error(X_train, y_train, model):
    """Menghitung metrik evaluasi untuk data train."""
    predictions = model.predict(X_train)
    predictProba = model.predict_proba(X_train)
    
    accuracy = accuracy_score(y_train, predictions)
    f1 = f1_score(y_train, predictions, average='macro')
    roc_auc = roc_auc_score(y_train, predictProba[:, 1])
    recall = recall_score(y_train, predictions)
    precision = precision_score(y_train, predictions)
    report = classification_report(y_train, predictions)
    
    return { 
        'report': report, 
        'f1': f1, 
        'roc': roc_auc, 
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision
    }

def calc_validation_error(X_test, y_test, model):
    """Menghitung metrik evaluasi untuk data test."""
    predictions = model.predict(X_test)
    predictProba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='macro')
    roc_auc = roc_auc_score(y_test, predictProba[:, 1])
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    
    return { 
        'report': report, 
        'f1': f1, 
        'roc': roc_auc, 
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision
    }

def calc_metrics(X_train, y_train, X_test, y_test, model):
    """Melatih model dan menghitung metrik evaluasi train dan validation."""
    model.fit(X_train, y_train)
    
    train_error = calc_train_error(X_train, y_train, model)
    validation_error = calc_validation_error(X_test, y_test, model)
    
    return train_error, validation_error

def evaluate_model(model, x_train, y_train, transformer):
    """Evaluasi model menggunakan k-fold cross-validation dan oversampling."""
    
    K = 10
    kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

    data = x_train
    target = y_train    

    # List untuk menyimpan hasil evaluasi
    train_errors_without_oversampling = []
    validation_errors_without_oversampling = []
    train_errors_with_oversampling = []
    validation_errors_with_oversampling = []

    for train_index, val_index in kf.split(data, target):
        
        # Membagi data menjadi training dan validation
        X_train, X_val = data.iloc[train_index], data.iloc[val_index]
        Y_train, Y_val = target.iloc[train_index], target.iloc[val_index]

        # Oversampling
        ros = RandomOverSampler()
        X_ros, Y_ros = ros.fit_resample(X_train, Y_train)

        # Membuat pipeline model
        estimator = Pipeline([
            ('preprocess', transformer),
            ('model', model)
        ])

        # Menghitung error tanpa oversampling
        train_error_without_oversampling, val_error_without_oversampling = calc_metrics(X_train, Y_train, X_val, Y_val, estimator)
        train_errors_without_oversampling.append(train_error_without_oversampling)
        validation_errors_without_oversampling.append(val_error_without_oversampling)

        # Menghitung error dengan oversampling
        train_error_with_oversampling, val_error_with_oversampling = calc_metrics(X_ros, Y_ros, X_val, Y_val, estimator)
        train_errors_with_oversampling.append(train_error_with_oversampling)
        validation_errors_with_oversampling.append(val_error_with_oversampling)

    # Evaluasi tanpa oversampling
    listItem = []
    for tr, val in zip(train_errors_without_oversampling, validation_errors_without_oversampling):
        listItem.append([tr['accuracy'], val['accuracy'], tr['roc'], val['roc'], tr['f1'], val['f1'],
                         tr['recall'], val['recall'], tr['precision'], val['precision']])
    
    listItem.append(list(np.mean(listItem, axis=0)))
    
    dfEvaluate_without_oversampling = pd.DataFrame(listItem, 
        columns=['Train Accuracy', 'Test Accuracy', 'Train ROC AUC', 'Test ROC AUC', 
                 'Train F1 Score', 'Test F1 Score', 'Train Recall', 'Test Recall', 
                 'Train Precision', 'Test Precision'])

    listIndex = list(dfEvaluate_without_oversampling.index)
    listIndex[-1] = 'Average'
    dfEvaluate_without_oversampling.index = listIndex

    # Evaluasi dengan oversampling
    listItem = []
    for tr, val in zip(train_errors_with_oversampling, validation_errors_with_oversampling):
        listItem.append([tr['accuracy'], val['accuracy'], tr['roc'], val['roc'], tr['f1'], val['f1'],
                         tr['recall'], val['recall'], tr['precision'], val['precision']])
    
    listItem.append(list(np.mean(listItem, axis=0)))
    
    dfEvaluate_with_oversampling = pd.DataFrame(listItem, 
        columns=['Train Accuracy', 'Test Accuracy', 'Train ROC AUC', 'Test ROC AUC', 
                 'Train F1 Score', 'Test F1 Score', 'Train Recall', 'Test Recall', 
                 'Train Precision', 'Test Precision'])

    listIndex = list(dfEvaluate_with_oversampling.index)
    listIndex[-1] = 'Average'
    dfEvaluate_with_oversampling.index = listIndex

    return dfEvaluate_without_oversampling, dfEvaluate_with_oversampling


# ##### **1. XGBoost**

# In[1862]:


result_without_oversampling, result_with_oversampling = evaluate_model(XGBClassifier(), x_train, y_train, transformer)

print("Hasil Tanpa Oversampling:")
st.write(result_without_oversampling)

print("\nHasil Dengan Oversampling:")
st.write(result_with_oversampling)


# ##### **2. LightGBM**

# In[1863]:


result_without_oversampling_lgbm, result_with_oversampling_lgbm = evaluate_model(lgb.LGBMClassifier(), x_train, y_train, transformer)

print("Hasil Tanpa Oversampling:")
st.write(result_without_oversampling_lgbm)

print("\nHasil Dengan Oversampling:")
st.write(result_with_oversampling_lgbm)


# #### **Test Undersampling with K-Fold Cross Validation**

# In[1864]:


# Fungsi untuk menghitung metrik pada training data
def calc_train_error(X_train, y_train, model):
    predictions = model.predict(X_train)
    predictProba = model.predict_proba(X_train)
    
    accuracy = accuracy_score(y_train, predictions)
    f1 = f1_score(y_train, predictions, average='macro')
    roc_auc = roc_auc_score(y_train, predictProba[:,1])
    recall = recall_score(y_train, predictions)
    precision = precision_score(y_train, predictions)
    report = classification_report(y_train, predictions)
    
    return { 
        'report': report, 
        'f1': f1, 
        'roc': roc_auc, 
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision
    }

# Fungsi untuk menghitung metrik pada validation data
def calc_validation_error(X_test, y_test, model):
    predictions = model.predict(X_test)
    predictProba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='macro')
    roc_auc = roc_auc_score(y_test, predictProba[:,1])
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    
    return { 
        'report': report, 
        'f1': f1, 
        'roc': roc_auc, 
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision
    }

# Fungsi untuk menjalankan model dengan undersampling
def calc_metrics(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    train_error = calc_train_error(X_train, y_train, model)
    validation_error = calc_validation_error(X_test, y_test, model)
    return train_error, validation_error

# Set jumlah fold untuk Stratified KFold
K = 10
kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

# Data yang digunakan untuk training
data = x_train
target = y_train	

# Fungsi untuk menjalankan evaluasi dengan undersampling pada model tertentu
def evaluate_model_with_undersampling(model, data, target, transformer):
    train_errors_without_undersampling = []
    validation_errors_without_undersampling = []

    train_errors_with_undersampling = []
    validation_errors_with_undersampling = []

    for train_index, val_index in kf.split(data, target):
        
        # Split data
        X_train, X_val = data.iloc[train_index], data.iloc[val_index]
        Y_train, Y_val = target.iloc[train_index], target.iloc[val_index]

        # Random Undersampling
        rus = RandomUnderSampler(random_state=5)
        X_rus, Y_rus = rus.fit_resample(X_train, Y_train)

        # Model pipeline
        estimator = Pipeline([
            ('preprocess', transformer),
            ('model', model)
        ])

        # Tanpa Undersampling
        train_error_without_undersampling, val_error_without_undersampling = calc_metrics(X_train, Y_train, X_val, Y_val, estimator)
        train_errors_without_undersampling.append(train_error_without_undersampling)
        validation_errors_without_undersampling.append(val_error_without_undersampling)

        # Dengan Undersampling
        train_error_with_undersampling, val_error_with_undersampling = calc_metrics(X_rus, Y_rus, X_val, Y_val, estimator)
        train_errors_with_undersampling.append(train_error_with_undersampling)
        validation_errors_with_undersampling.append(val_error_with_undersampling)

    # Evaluasi hasil tanpa undersampling
    listItem = []
    for tr, val in zip(train_errors_without_undersampling, validation_errors_without_undersampling):
        listItem.append([tr['accuracy'], val['accuracy'], tr['roc'], val['roc'], tr['f1'], val['f1'],
                         tr['recall'], val['recall'], tr['precision'], val['precision']])
    listItem.append(list(np.mean(listItem, axis=0)))

    dfEvaluate_without = pd.DataFrame(listItem, 
                                      columns=['Train Accuracy', 'Test Accuracy', 'Train ROC AUC', 'Test ROC AUC', 
                                               'Train F1 Score', 'Test F1 Score', 'Train Recall', 'Test Recall', 
                                               'Train Precision', 'Test Precision'])
    dfEvaluate_without.index = list(dfEvaluate_without.index)
    dfEvaluate_without.rename(index={dfEvaluate_without.index[-1]: 'Average'}, inplace=True)

    # Evaluasi hasil dengan undersampling
    listItem = []
    for tr, val in zip(train_errors_with_undersampling, validation_errors_with_undersampling):
        listItem.append([tr['accuracy'], val['accuracy'], tr['roc'], val['roc'], tr['f1'], val['f1'],
                         tr['recall'], val['recall'], tr['precision'], val['precision']])
    listItem.append(list(np.mean(listItem, axis=0)))

    dfEvaluate_with = pd.DataFrame(listItem, 
                                   columns=['Train Accuracy', 'Test Accuracy', 'Train ROC AUC', 'Test ROC AUC', 
                                            'Train F1 Score', 'Test F1 Score', 'Train Recall', 'Test Recall', 
                                            'Train Precision', 'Test Precision'])
    dfEvaluate_with.index = list(dfEvaluate_with.index)
    dfEvaluate_with.rename(index={dfEvaluate_with.index[-1]: 'Average'}, inplace=True)


    return dfEvaluate_without, dfEvaluate_with


# ##### **1. XGBoost**

# In[1865]:


result_without_undersampling, result_with_undersampling = evaluate_model_with_undersampling(XGBClassifier(), x_train, y_train, transformer)

print("Hasil Tanpa Oversampling:")
display(result_without_undersampling)

print("\nHasil Dengan Oversampling:")
display(result_with_undersampling)


# ##### **2. LightGBM**

# In[1866]:


result_without_undersampling_lgbm, result_with_undersampling_lgbm = evaluate_model_with_undersampling(lgb.LGBMClassifier(), x_train, y_train, transformer)

print("Hasil Tanpa Oversampling:")
st.write(result_without_undersampling_lgbm)

print("\nHasil Dengan Oversampling:")
st.write(result_with_undersampling_lgbm)


# ##### **Comparison of Resampling with K-Fold Cross Validation**

# In[1867]:


def format_results(df):
    df_percentage = df.copy()
    for col in df.columns:
        df_percentage[col] = df[col] * 100  # Ubah ke persentase
    df_percentage = df_percentage.round(2)  # Batasi dua angka di belakang koma
    return df_percentage

resampling_results = pd.DataFrame({
    "XGBoost Tanpa Resampling": result_without_oversampling.loc['Average'],
    "XGBoost Dengan Oversampling": result_with_oversampling.loc['Average'],
    "XGBoost Dengan Undersampling": result_with_undersampling.loc['Average'],
    "LightGBM Tanpa Resampling": result_without_oversampling_lgbm.loc['Average'],
    "LightGBM Dengan Oversampling": result_with_oversampling_lgbm.loc['Average'],
    "LightGBM Dengan Undersampling": result_with_undersampling_lgbm.loc['Average']
})

format_results(resampling_results.T)


# Dari hasil evaluasi metrik, terlihat bahwa Recall untuk kelas positif (Cancel) meningkat setelah dilakukan oversampling dibandingkan dengan model tanpa resampling. Namun, Precision untuk kelas positif mengalami penurunan.
# 
# Ini masuk akal karena oversampling meningkatkan jumlah sampel di kelas positif sehingga model lebih baik dalam mengenali kasus pembatalan (False Negative berkurang, Recall meningkat).
# Namun, sebagai konsekuensinya, model juga lebih sering salah mengklasifikasikan tamu yang sebenarnya tidak akan membatalkan reservasi sebagai pembatalan (False Positive meningkat, Precision menurun).
# 
# Dampak terhadap kasus pembatalan hotel:
# - Jika kita memprioritaskan False Negative (Recall lebih penting daripada Precision), model dengan undersampling lebih disarankan, karena dapat mengurangi risiko kamar kosong yang tidak tersewa akibat kesalahan model dalam mendeteksi pembatalan.
# - Jika kita lebih memprioritaskan False Positive (Precision lebih penting daripada Recall), model tanpa resampling lebih disarankan, karena lebih akurat dalam memprediksi tamu yang benar-benar akan membatalkan reservasi.
# 
# Karena dalam kasus ini False Negative lebih diprioritaskan, maka model dengan undersampling adalah pilihan yang lebih tepat (Untuk selanjutnya dilakukan Hyperparameter Tuning). Dengan meningkatkan Recall, model akan lebih baik dalam mengantisipasi pembatalan, sehingga hotel dapat mengambil langkah preventif seperti menawarkan diskon, menghubungi tamu untuk konfirmasi ulang, atau menyesuaikan strategi pemasaran agar kamar tetap terisi.

# ### **Hyperparameter Tuning**

# Pada tahap ini kita akan melakukan Hyperparameter Tuning untuk model XGBoost dan LightGBM untuk mendapatkan hasil yang lebih baik lagi. Untuk efisiensi waktu, pada tuning kali ini kita akan menggunakan method RandomizedSearchCV yang dimana metode ini memilih kombinasi secara acak dalam jumlah iterasi tertentu (n_iter)

# XGBoost
# 1. https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html
# 2. https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
# 
# LightGBM
# 1. https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
# 2. https://www.restack.io/p/hyperparameter-tuning-answer-lightgbm-random-search-cat-ai
# 

# ##### **1. XGBoost**

# In[1868]:


xgb = XGBClassifier()
rus = RandomUnderSampler(random_state=5)

estimator_xgb=Pipeline([
    ('undersampling',rus),
    ('preprocess',transformer),
    ('model',xgb)
])

hyperparam_space_xgb = {
    'model__n_estimators': [100, 200, 300],  # Jumlah pohon keputusan
    'model__max_depth': [3, 5, 7, 9],  # Kedalaman maksimum pohon
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],  # Ukuran langkah pembelajaran
    'model__subsample': [0.6, 0.8, 1.0],  # Proporsi sampel yang digunakan untuk setiap pohon
    'model__colsample_bytree': [0.6, 0.8, 1.0],  # Proporsi fitur yang dipilih untuk setiap pohon
    'model__gamma': [0, 0.1, 0.2, 0.3],  # Minimal reduction loss agar node baru dibuat
    'model__min_child_weight': [1, 3, 5],  # Minimal jumlah sampel untuk membuat node baru
}

random_xgb = RandomizedSearchCV(
    estimator_xgb,
    param_distributions=hyperparam_space_xgb, 
    scoring='roc_auc', 
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=5),
    n_jobs=-1, 
    n_iter=100,
    verbose=2
 )

random_xgb.fit(x_train,y_train)


# In[1869]:


print(random_xgb.best_score_)
print(random_xgb.best_params_)


# Hyperparameter terbaik XGBoost dari hasil RandomSearch => `subsample` = 0.8, `n_estimators` = 200, `min_child_weight` = 3, `max_depth` = 7, `learning_rate` = 0.05, `gamma` = 0.3, `colsample_bytree` = 1.0

# In[1870]:


best_model_xgb = random_xgb.best_estimator_
best_model_xgb.fit(x_train, y_train)


# In[1871]:


estimator_xgb=Pipeline([
    ('undersampling',rus),
    ('preprocess',transformer),
    ('model',xgb)
])
estimator_xgb.fit(x_train, y_train)


# In[1872]:


y_pred_default_xgb = estimator_xgb.predict(x_test)
y_pred_proba_default_xgb = estimator_xgb.predict_proba(x_test)
y_pred_tuned_xgb = best_model_xgb.predict(x_test)
y_pred_proba_tuned_xgb = best_model_xgb.predict_proba(x_test)

roc_auc_default_xgb = roc_auc_score(y_test, y_pred_proba_default_xgb[:,1])
roc_auc_tuned_xgb = roc_auc_score(y_test, y_pred_proba_tuned_xgb[:,1])

print('ROC AUC Score Default XGB : ', roc_auc_default_xgb)
print('ROC AUC Score Tuned XGB : ', roc_auc_tuned_xgb)


# ##### **2. LightGBM**

# In[1873]:


lgbm = lgb.LGBMClassifier()
rus_lgbm = RandomUnderSampler(random_state=5)

estimator=Pipeline([
    ('undersampling',rus_lgbm),
    ('preprocess',transformer),
    ('model',lgbm)
])

hyperparam_space = [{
    'model__max_bin': [255, 300, 350],  # Jumlah bin untuk histogram splitting
    'model__num_leaves': [31, 50, 70, 100],  # Meningkatkan kompleksitas decision tree
    'model__min_data_in_leaf': [20, 30, 50],  # Minimum data dalam leaf node untuk menghindari overfitting
    'model__num_iterations':[100,75, 125, 150],
    'model__learning_rate': [0.1, 0.05, 0.01], 
    'model__random_state': [5]
}]

random_lgbm = RandomizedSearchCV(
    estimator,
    param_distributions=hyperparam_space, 
    scoring='roc_auc', 
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=5),
    n_jobs=-1, 
    n_iter=100,
    verbose=2
 )

random_lgbm.fit(x_train,y_train)


# In[1874]:


print(random_lgbm.best_score_)
print(random_lgbm.best_params_)


# Hyperparameter terbaik LGBM dari hasil RandomSearch => `num_leaves` = 50, `num_iterations` = 75, `min_data_in_leaf` = 20, `max_bin` = 300, `learning_rate` = 0.1

# In[1875]:


best_model_lgbm = random_lgbm.best_estimator_
best_model_lgbm.fit(x_train, y_train)


# In[1876]:


estimator_lgbm=Pipeline([
    ('undersampling',rus_lgbm),
    ('preprocess',transformer),
    ('model',lgbm)
])
estimator_lgbm.fit(x_train, y_train)


# In[1877]:


y_pred_default_lgbm = estimator_lgbm.predict(x_test)
y_pred_proba_default_lgbm = estimator_lgbm.predict_proba(x_test)
y_pred_tuned_lgbm = best_model_lgbm.predict(x_test)
y_pred_proba_tuned_lgbm = best_model_lgbm.predict_proba(x_test)

roc_auc_default_lgbm = roc_auc_score(y_test, y_pred_proba_default_lgbm[:,1])
roc_auc_tuned_lgbm = roc_auc_score(y_test, y_pred_proba_tuned_lgbm[:,1])

print('ROC AUC Score Default LGBM : ', roc_auc_default_lgbm)
print('ROC AUC Score Tuned LGBM : ', roc_auc_tuned_lgbm)


# #### **Performance Comparison**
# 
# Perbandingan performa model sebelum dan sesudah dilakukan hyperparameter tuning.

# In[1878]:


data = {
    "Model": ["XGBoost", "LightGBM"],
    "Default ROC AUC": [roc_auc_default_xgb, roc_auc_default_lgbm],
    "Tuned ROC AUC": [roc_auc_tuned_xgb, roc_auc_tuned_lgbm]
}

df_roc_auc = pd.DataFrame(data)
df_roc_auc["Peningkatan (%)"] = (df_roc_auc["Tuned ROC AUC"] - df_roc_auc["Default ROC AUC"]) * 100
df_roc_auc


# Berdasarkan hasil diatas, Model XGBoost mengalami peningkatan sebesar 0.25% dibandingkan dengan model sebelum dilakukan tuning dan untuk Model LightGBM mengalami peningkatan sebesar 0.02%. Walaupun peningkatan pada kedua model ini tergolong kecil, ini menunjukkan bahwa tuning memberikan kontribusi positif terhadap performa model

# In[1879]:


report_default_xgb = classification_report(y_test, y_pred_default_xgb)
report_tuned_xgb = classification_report(y_test, y_pred_tuned_xgb)
report_default_lgbm = classification_report(y_test, y_pred_default_lgbm)
report_tuned_lgbm = classification_report(y_test, y_pred_tuned_lgbm)

print('Classification Report Default XGBoost : \n', report_default_xgb)
print('Classification Report Tuned XGBoost : \n', report_tuned_xgb)
print('Classification Report Default LightGBM : \n', report_default_lgbm)
print('Classification Report Tuned LightGBM : \n', report_tuned_lgbm)


# Berdasarkan hasil perbandingan Default dan Tuned diatas, dapat disimpulkan :
# 1. Perbandingan Default vs Tuned XGBoost :
#     - Setelah tuning, recall untuk class positif meningkat dari 0.71 menjadi 0.72, menunjukkan sedikit kenaikan dalam mendeteksi class positif (cancel)
#     - Precision untuk class positif meningkat dari 0.56 menjadi 0.57, artinya prediksi class positif lebih tepat setelah tuning
#     - F1-score untuk class positif tetap di 0.63, menunjukkan keseimbangan precision dan recall tidak berubah signifikan
#     - Akurasi naik dari 0.74 menjadi 0.75, menunjukkan model yang sedikit lebih baik setelah tuning
# 2. Perbandingan Default vs Tuned LightGBM :
#     - Recall untuk class positif naik dari 0.70 menjadi 0.72, menunjukkan sedikit kenaikan dalam mendeteksi class positif (cancel)
#     - Precision untuk class positif naik dari 0.56 menjadi 0.57, artinya prediksi class positif lebih tepat setelah tuning
#     - F1-score untuk class positif tetap di 0.63, menunjukkan keseimbangan precision dan recall tidak berubah signifikan
#     - Akurasi tetap di 0.75, menunjukkan model tidak ada perubahan setelah tuning
# 

# #### **Model Interpretation**

# In[1880]:


coef1 = pd.Series(best_model_xgb['model'].feature_importances_, transformer.get_feature_names_out()).sort_values(ascending = False).head(10)
coef1.plot(kind='barh', title='Feature Importances')
plt.show()


# Berdasarkan visualisasi Feature Importance dari model XGBoost, berikut adalah beberapa poin penting:
# - `deposit_type_Non Refund` adalah fitur paling penting dalam prediksi model, dengan kontribusi terbesar dibandingkan fitur lainnya. Ini menunjukkan bahwa metode deposit (non-refundable) memiliki dampak besar terhadap target yang diprediksi (kemungkinan pembatalan pemesanan)
# - `market_segment_Online TA` dan `required_car_parking_spaces` memiliki pengaruh signifikan, yang berarti segmen pasar pemesanan online melalui travel agent dan jumlah permintaan tempat parkir dapat menjadi faktor utama dalam pembatalan
# -  `remainder__previous_cancellations` memiliki kontribusi cukup tinggi, jumlah pembatalan sebelumnya dapat menjadi indikasi kebiasaan tamu dalam membatalkan pemesanan

# In[1883]:


# Model Akhir
best_model = random_xgb.best_estimator_
final_model = best_model.fit(x_train, y_train)


# In[1884]:


# save model
best_model = random_xgb.best_estimator_
final_model = best_model.fit(x_train, y_train)
pickle.dump(final_model, open('Model_final.sav', 'wb'))


# Dalam penerapan model, ada beberapa langkah penting yang perlu diperhatikan, yaitu sebelum menggunakan model, data harus melalui beberapa proses yang telah ditentukan, antara lain:
# 
# 1. Menghapus duplikat data.
# 2. Handling missing value.
# 3. Handling outlier dengan menghapus data anomaly pada kolom tertentu.
# 4. Melakukan Encoding untuk fitur kategorikal
# 5. Menggunakan model untuk memprediksi apakah tamu berpotensi cancel atau tidak.

# ## **7. Conclusion**

# | Aktual / Prediksi | Prediction No Canceled (Book) | Prediction Canceled |
# | --- | --- | --- |
# | Actual No Canceled (Book) | 0,76  | 0,24 |
# | Actual canceled | 0,28 | 0,72 |
# 
# 
# -	Model yang digunakan adalah XGBoost dengan Undersampling. Dari model yang digunakan, didpatkan kesimpulan:
# 1.	Model dapat mengidentifikasi pelanggan yang no-canceled dengan benar sebanyak 76% (TN Rate).
# 2.	Model dapat mengidentifikasi pelanggan yang akan canceled dengan benar sebanyak 72% (TP Rate).
# 3.	Sebanyak 24% pelanggan yang seharusnya no-canceled, malah diprediksi akan canceled (FP rate).
# 4.	Sebanyak 28% pelanggan yang seharusnya canceled, malah diprediksi tidak canceled (FN Rate).
# -	Model Limitation: Model ini menggunakan batasan lead time maksimum selama 1 tahun (365 hari), yang berarti pemesanan hanya dapat dilakukan dengan rentang waktu maksimal 365 hari dari tanggal check-in yang diinginkan.
# 
# 
# 

# ## **Business Calculation**
# 
# Bila seandainya biaya untuk reservasi kamar standard sebesar 100,42 Euro (Berdasarkan ADR kamar tipe standar yang paling murah) dan andaikan jumlah reservasi yang hotel dapatkan untuk suatu kurun waktu sebanyak 200 (dimana andaikan 100 Cancelled dan 100 tidak canceled), maka hitungannya kurang lebih seperti ini:
# 
# **1.Tanpa Model (Semua Reservasi Diproses Seperti Biasa)**
# 
# Jika tidak ada model prediksi, maka semua reservasi akan diproses tanpa mempertimbangkan kemungkinan pembatalan.
# -	Pendapatan Potensial (Jika semua menginap)
# 200 x 100,42 = 20.084 euro
# -	Kerugian Akibat Pembatalan
# 100 x 100,42 = 10.042 euro
# -	Pendapatan Aktual setelah Pembatalan
# 20.084 – 10.042 = 10.042 euro
# Tanpa model, kerugian akibat pembatalan mencapai 10.042 euro 
# 
# **2.Dengan Model Prediksi Pembatalan**
# -	Recall canceled 72%, maka model dapat mengidentifikasi 72% dari 100 reservasi yang akan dibatalkan (72 reservasi).
# -	Recall untuk non-canceled 76%, maka Model dapat mengidentifikasi 76% dari 100 reservasi yang benar-benar tidak akan dibatalkan (76 reservasi).
# -	False positive (reservasi yang sebenarnya tidak dibatalkan tetapi diprediksi akan dibatalkan) = 24% (24 reservasi).
# -	False negative (reservasi yang sebenarnya dibatalkan tetapi diprediksi tidak akan dibatalkan) = 28% (28 reservasi).
# 
# **3.Prediksi model**
# 
# **1.Reservasi yang diprediksi akan dibatalkan (Positif model)**
# 
# 72% x 100 = 72 Reservasi
# -	Apabila hotel bisa menawarkan promosi (misal diskon atau fleksibilitas re-schedule) untuk mengurangi pembatalan dan berhasil mempertahankan 30% pelanggan, maka: 30% x 72 = 22 reservasi tetap menginap
# -	Pendapatan tambahan dari pelanggan yang tetap meinginap: 22 x 100,42 = 2.209,24 euro
# 
# **2.Reservasi yang salah diprediksi (Actual tidak canceled, prediksi canceled) (False positive)**
# 
# 24% x 100 = 24 reservasi
# 
# **3.Reservasi yang salah diprediksi (Actual Canceled, prediksi tidak canceled) (False negative)**
# 
# 29% x 100 = 29 reservasi
# -	Pendapatan yang hilang akibat pembatalan ini: 28 x 100,42 = 2.811,76 euro
# 
# **4.Pendapatan setelah menggunakan model**
# -	Pendapatan dari reservasi non-canceled tetap: 100 x 100,42 = 10,042 euro
# -	Pendapatan tambahan dari reservasi yang berhasil di selamatkan: 2.209,24 euro
# -	Total pendapatan setelah strategi mitigasi: 10.042 + 2.209,24 = 12.219,282 euro
# -	Kerugian akibat false negatives yang tidak dapat diselamatkan: 2.811,76 euro
# 
# 
# **Kesimpulan:**
# -	Tanpa model, pendapatan 10.042 euro, dengan kerugian 10.042 euro akibat pembatalan.
# -	Dengan model, pendapatan meningkat menjadi 12.219,282 euro, karena model memungkinkan hotel mempertahankan 22 reservasi yang seharusnya dibatalkan.
# -	Penghematan sebesar 2.209,24 euro dengan strategi mitigasi berdasarkan prediksi model.
# -	Model masih memiliki false negatives, yang menyebabkan kerugian 2.811,76 euro.
# 
# 
# 

# ## **6. Recommendations**

# ### **Rekomendasi Model**
# 
# Untuk dapat meningkatkan recall, dapat dilakukan:
# -	**Kumpulkan lebih banyak data**, Khususnya dalam kasus ketidakseimbangan target, sebaiknya jumlah data pemesanan yang dibatalkan lebih seimbang dengan data pemesanan yang tidak dibatalkan, jika memungkinkan.
# -	**Penyesuaian threshold prediksi (misalnya menurunkan dari 0.5 ke 0.4).**
# -	**Menggunakan model balancing lain seperti focal loss atau cost-sensitive learning.**
# -	**Melakukan hyperparameter tuning lebih lanjut untuk meningkatkan recall.**
# 
# ### **Rekomendasi Bisnis**
# - Membuat kebijakan untuk pemberian voucher pada tamu dengan lead time lebih dari 25 hari.
# - Membuat kebijakan baru untuk harga kamar saat high season dan low season.
# - Pembuatan kesepakatan baru terkait deposit dengan TA/TO serta GDS. 
# - Memastikan kelengkapan informasi mengenai fasilitas, layanan, dan keunggulan hotel di website.
# - Menetapkan syarat pembatalan untuk reservasi, dimana tamu wajib memberikan review atau alasan pembatalan.
# - Menetapkan kebijakan baru untuk biaya deposit saat memesan langsung pada pihak hotel (Melalui Website resmi hotel maupun datang secara langsung ke hotel).
# - Menggunakan model machine learning yang telah dibuat sebagai solusi untuk menentukan strategi pemasaran yang tepat sasaran. 
# 
