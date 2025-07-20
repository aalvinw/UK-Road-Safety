`
#  Laporan Proyek Machine Learning - Agustinus Alvin

## 1. Domain Proyek

Kecelakaan lalu lintas merupakan masalah penting di Inggris yang berdampak serius terhadap keselamatan publik dan biaya ekonomi. Mengetahui faktor-faktor penyebab kecelakaan serta memprediksi tingkat keparahannya dapat membantu pemerintah dan otoritas transportasi dalam menyusun kebijakan keselamatan yang lebih efektif.

Melalui pemodelan machine learning, proyek ini berfokus pada prediksi tingkat keparahan kecelakaan berdasarkan berbagai informasi terkait insiden, seperti kondisi cuaca, waktu, lokasi, serta jumlah korban dan kendaraan yang terlibat.

**Sumber Data:**

* UK Department for Transport
* Dataset via Kaggle: UK Road Safety Dataset 
* Link Dataset: 'https://www.kaggle.com/datasets/devansodariya/road-accident-united-kingdom-uk-dataset'

---

## 2. Business Understanding

### 2.1 Rumusan Masalah

1. Faktor apa saja yang berkaitan erat dengan tingkat keparahan kecelakaan lalu lintas?
2. Bisakah kita membangun model klasifikasi yang mampu memprediksi `Accident_Severity` secara akurat?

### 2.2 Tujuan Proyek

1. Mengembangkan model klasifikasi untuk memprediksi tingkat keparahan kecelakaan lalu lintas.
2. Mengevaluasi dan membandingkan performa beberapa model machine learning.

### 2.3 Rencana Solusi

* Melakukan eksplorasi dan pembersihan data.
* Mengonversi data kategorikal menjadi numerik dan melakukan standardisasi fitur numerik.
* Membangun model klasifikasi dan mengevaluasinya menggunakan metrik yang relevan.
* Menentukan fitur mana yang paling berpengaruh terhadap prediksi keparahan kecelakaan.

---

## 3. Data Understanding

### 3.1 Sumber Dataset

Dataset ini berasal dari UK Department for Transport, yang tersedia secara publik di Kaggle dengan nama **UK Road Safety Dataset**. Dataset asli terdiri dari **1.504.150 baris** dan **33 kolom** sebelum dilakukan pembersihan dan penyaringan. Karena ukuran data yang sangat besar, dilakukan proses sampling dan seleksi fitur untuk efisiensi proses pemodelan.

Link dataset: [UK Road Safety Dataset on Kaggle](https://www.kaggle.com/datasets/devansodariya/road-accident-united-kingdom-uk-dataset)

### 3.2 Fitur-Fitur Penting yang Digunakan

| No | Nama Fitur                                            | Tipe Data | Deskripsi                                                                                                         |
| -- | ----------------------------------------------------- | --------- | ----------------------------------------------------------------------------------------------------------------- |
| 1  | **Unnamed: 0**                                        | `int64`   | Indeks baris yang muncul saat ekspor CSV. Tidak mengandung informasi penting, bisa dihapus.                       |
| 2  | **Accident\_Index**                                   | `object`  | ID unik untuk setiap kecelakaan. Digunakan sebagai identifier.                                                    |
| 3  | **Location\_Easting\_OSGR**                           | `float64` | Koordinat horizontal (timur) berdasarkan sistem grid nasional Inggris (OSGR).                                     |
| 4  | **Location\_Northing\_OSGR**                          | `float64` | Koordinat vertikal (utara) berdasarkan sistem grid nasional Inggris (OSGR).                                       |
| 5  | **Longitude**                                         | `float64` | Koordinat bujur (longitude) lokasi kecelakaan dalam format desimal.                                               |
| 6  | **Latitude**                                          | `float64` | Koordinat lintang (latitude) lokasi kecelakaan dalam format desimal.                                              |
| 7  | **Police\_Force**                                     | `int64`   | Kode numerik unit kepolisian yang menangani kecelakaan.                                                           |
| 8  | **Accident\_Severity**                                | `int64`   | Tingkat keparahan kecelakaan: 1 = Fatal, 2 = Serious, 3 = Slight. Ini adalah target untuk model klasifikasi.      |
| 9  | **Number\_of\_Vehicles**                              | `int64`   | Jumlah kendaraan yang terlibat dalam kecelakaan.                                                                  |
| 10 | **Number\_of\_Casualties**                            | `int64`   | Jumlah korban (baik luka maupun meninggal) akibat kecelakaan.                                                     |
| 11 | **Date**                                              | `object`  | Tanggal kecelakaan terjadi. Format: dd/mm/yyyy. Bisa diekstrak menjadi fitur waktu lainnya (tahun, bulan).        |
| 12 | **Day\_of\_Week**                                     | `int64`   | Hari saat kecelakaan terjadi (1 = Minggu, 2 = Senin, ..., 7 = Sabtu).                                             |
| 13 | **Time**                                              | `object`  | Waktu kejadian dalam format HH\:MM. Beberapa nilai mungkin hilang.                                                |
| 14 | **Local\_Authority\_(District)**                      | `int64`   | Kode distrik pemerintahan lokal tempat kecelakaan terjadi.                                                        |
| 15 | **Local\_Authority\_(Highway)**                       | `object`  | Nama otoritas jalan lokal yang bertanggung jawab atas wilayah kecelakaan.                                         |
| 16 | **1st\_Road\_Class**                                  | `int64`   | Kategori jalan utama (contoh: 1 = Motorway, 2 = A Road, 3 = B Road, dll.).                                        |
| 17 | **1st\_Road\_Number**                                 | `int64`   | Nomor dari jalan utama tempat kecelakaan terjadi.                                                                 |
| 18 | **Road\_Type**                                        | `object`  | Tipe jalan: single carriageway, dual carriageway, roundabout, one-way street, dll.                                |
| 19 | **Speed\_limit**                                      | `int64`   | Batas kecepatan pada lokasi kecelakaan (dalam mil per jam).                                                       |
| 20 | **Junction\_Control**                                 | `object`  | Kontrol yang ada di persimpangan: traffic signal, give way, uncontrolled, dll. Banyak nilai null.                 |
| 21 | **2nd\_Road\_Class**                                  | `int64`   | Kategori jalan sekunder (jika ada persimpangan).                                                                  |
| 22 | **2nd\_Road\_Number**                                 | `int64`   | Nomor dari jalan sekunder di persimpangan.                                                                        |
| 23 | **Pedestrian\_Crossing-Human\_Control**               | `object`  | Apakah ada kontrol manusia seperti lampu lalu lintas pejalan kaki atau petugas di lokasi.                         |
| 24 | **Pedestrian\_Crossing-Physical\_Facilities**         | `object`  | Fasilitas fisik penyeberangan seperti zebra cross, underpass, overpass, dll.                                      |
| 25 | **Light\_Conditions**                                 | `object`  | Kondisi pencahayaan saat kecelakaan (siang, malam dengan/ tanpa lampu jalan, dll).                                |
| 26 | **Weather\_Conditions**                               | `object`  | Kondisi cuaca saat kejadian: clear, hujan ringan, hujan deras, salju, kabut, dll.                                 |
| 27 | **Road\_Surface\_Conditions**                         | `object`  | Kondisi permukaan jalan: kering, basah, bersalju, dll.                                                            |
| 28 | **Special\_Conditions\_at\_Site**                     | `object`  | Kondisi khusus yang mungkin memengaruhi kejadian: lampu rusak, permukaan jalan rusak, dll. Banyak missing values. |
| 29 | **Carriageway\_Hazards**                              | `object`  | Bahaya di jalan seperti kendaraan mogok, tumpahan oli, barang di jalan, dll. Banyak missing values.               |
| 30 | **Urban\_or\_Rural\_Area**                            | `int64`   | Jenis area: 1 = Urban, 2 = Rural, 3 = Tak diketahui.                                                              |
| 31 | **Did\_Police\_Officer\_Attend\_Scene\_of\_Accident** | `object`  | Apakah petugas polisi hadir di lokasi: Yes, No, Unknown.                                                          |
| 32 | **LSOA\_of\_Accident\_Location**                      | `object`  | Kode wilayah statistik terkecil (Lower Super Output Area). Banyak nilai kosong.                                   |
| 33 | **Year**                                              | `int64`   | Tahun terjadinya kecelakaan.                                                                                      |
                         |

### 3.3 Informasi Missing Values

Berikut ini kolom yang memiliki missing values:
- `Weather_Conditions`, `Light_Conditions`, `Road_Surface_Conditions`, dll memiliki <5% missing values.
- Kolom dengan missing values >90% (seperti `Special_Conditions_at_Site`, `Carriageway_Hazards`) dihapus dari dataset.

### 3.4 Struktur Data

| Tipe Data | Jumlah Kolom | Contoh Kolom                     |
| --------- | ------------ | -------------------------------- |
| int64     | 14 kolom     | `Accident_Severity`, `Speed_limit` |
| float64   | 4 kolom      | `Latitude`, `Longitude`          |
| object    | 15 kolom     | `Time`, `Weather_Conditions`     |

### 3.5 Distribusi Target (Accident_Severity)

- 1 (Fatal): 1.1%
- 2 (Serious): 11.9%
- 3 (Slight): 87.0%

Distribusi ini sangat tidak seimbang. Hal ini penting diperhatikan dalam proses pelatihan model.
.
---

## 4. Data Preparation

### 4.1 Langkah-Langkah yang Dilakukan:

1. **Sampling Data**

   * Karena ukuran dataset asli sangat besar, dilakukan pengambilan sampel sebanyak **5.000 baris secara acak** untuk efisiensi pemrosesan dan pelatihan model.

2. **Menghapus Kolom Tidak Relevan atau Banyak Missing Values**

   * Kolom seperti `Location_Easting_OSGR`, `Location_Northing_OSGR`, `Special_Conditions_at_Site`, dan `Carriageway_Hazards` dihapus karena:

     * Nilainya mayoritas kosong (>90%)
     * Tidak berkontribusi langsung pada prediksi

3. **Imputasi Missing Values**

   * **Fitur numerik**: Diimputasi menggunakan **median**.
   * **Fitur kategorikal**: Diimputasi menggunakan **konstanta ‘Unknown’**, bukan modus, sesuai kode `SimpleImputer(strategy='constant', fill_value='Unknown')`.

4. **Ekstraksi Informasi Waktu**

   * Kolom **`Date`** diolah untuk mengekstrak fitur **`Month`**, bukan `Hour`. Tidak ada transformasi dari kolom `Time` dalam notebook.

5. **Encoding**

   * **Target (`Accident_Severity`)** diubah menjadi label numerik menggunakan mapping:
     `Slight (3)` → 0, `Serious (2)` → 1, `Fatal (1)` → 2
   * **Fitur kategorikal** lainnya dilakukan **One-Hot Encoding** dengan penanganan untuk kategori tak dikenal (`handle_unknown='ignore'`).

6. **Standardisasi Fitur Numerik**

   * Fitur seperti `Speed_limit`, `Number_of_Casualties`, dan `Number_of_Vehicles` distandardisasi menggunakan `StandardScaler`.

7. **Split Data**

   * Dataset dibagi menjadi data latih (70%) dan data uji (30%) menggunakan `train_test_split`.
   * Stratifikasi (`stratify=y`) digunakan untuk menjaga proporsi kelas target tetap seimbang.
   * `random_state=42` digunakan untuk memastikan hasil dapat direplikasi.

---

### 4.2 Pembagian Data

* Dataset dipisahkan menjadi **fitur (`X`) dan target (`y`)**.
* Data kemudian dibagi menjadi training dan test set dengan rasio **70:30**.
* Stratifikasi (`stratify=y`) diterapkan untuk mempertahankan distribusi kelas target.
* Parameter `random_state=42` digunakan untuk konsistensi pembagian data.

---

## 5. Modeling

### 5.1 Klasifikasi Keparahan Kecelakaan

#### Tujuan:

Bagian ini bertujuan membandingkan performa beberapa model klasifikasi untuk memprediksi tingkat keparahan kecelakaan lalu lintas (`Accident_Severity`). Pemilihan model mencakup pendekatan linear, non-linear, dan ensemble learning agar dapat dibandingkan secara menyeluruh dan adil.

#### Model yang Digunakan:

1. **Logistic Regression**

   * **Cara Kerja**: Model ini menggunakan fungsi sigmoid untuk menghitung probabilitas suatu kelas. Untuk klasifikasi multiclass, pendekatan yang digunakan adalah One-vs-Rest.
   * **Parameter**:

     * `max_iter=1000`: Meningkatkan jumlah iterasi agar konvergen.
     * `class_weight='balanced'`: Menangani ketidakseimbangan kelas.
     * `random_state=42`: Reproduksibilitas hasil.

2. **Support Vector Machine (SVM)**

   * **Cara Kerja**: SVM mencari hyperplane terbaik yang memisahkan kelas-kelas dalam ruang berdimensi tinggi. Untuk data non-linear, kernel dapat digunakan.
   * **Parameter**:

     * `probability=True`: Mengizinkan prediksi probabilitas (diperlukan untuk ROC/AUC).
     * `class_weight='balanced'`: Menyeimbangkan bobot kelas minoritas.
     * `random_state=42`: Untuk konsistensi hasil.

3. **Random Forest Classifier**

   * **Cara Kerja**: Merupakan ensemble dari banyak pohon keputusan (decision tree) dengan teknik bagging, dan hasil akhir ditentukan berdasarkan voting mayoritas.
   * **Parameter**:

     * `class_weight='balanced'`: Mengatasi imbalance.
     * `random_state=42`: Untuk hasil yang reprodusibel.

4. **Gradient Boosting Classifier**

   * **Cara Kerja**: Model dibangun secara berurutan dengan memperbaiki kesalahan model sebelumnya menggunakan teknik boosting berbasis pohon.
   * **Parameter**:

     * `random_state=42`: Untuk hasil yang konsisten.

5. **XGBoost Classifier**

   * **Cara Kerja**: Optimasi lanjutan dari gradient boosting dengan penambahan regularisasi dan efisiensi komputasi tinggi.
   * **Parameter**:

     * `use_label_encoder=False`: Untuk kompatibilitas dengan Scikit-Learn.
     * `eval_metric='mlogloss'`: Metrik evaluasi untuk multiclass.
     * `random_state=42`: Untuk hasil yang stabil.

6. **LightGBM Classifier**

   * **Cara Kerja**: Model boosting berbasis histogram yang efisien dalam kecepatan dan memori.
   * **Parameter**:

     * `class_weight='balanced'`: Menyesuaikan bobot tiap kelas.
     * `random_state=42`: Konsistensi hasil.

#### Alasan Menggunakan Berbagai Model:

* **Perbandingan performa**: Untuk mengetahui model mana yang memberikan hasil terbaik pada dataset ini.
* **Robustness**: Beberapa model seperti tree-based (Random Forest, XGBoost) mampu menangani fitur kategorikal dan outlier.
* **Efisiensi**: Model seperti LightGBM dan XGBoost dirancang untuk skala besar dan efisiensi tinggi.
* **Interpretabilitas**: Logistic Regression mudah diinterpretasi dan cocok sebagai baseline.


---


## 6. Evaluation

### 6.1 Metrik Evaluasi

* **Accuracy**: Persentase prediksi yang benar dibanding total data.
* **Precision, Recall, F1-score (macro average)**: Digunakan karena dataset memiliki distribusi kelas yang tidak seimbang (imbalanced). Macro average memberi bobot yang sama ke setiap kelas.
* **Confusion Matrix**: Membantu mengidentifikasi kesalahan klasifikasi antar kelas.

### 6.2 Performa Model

| Model               | Accuracy | F1-macro |
| ------------------- | -------- | -------- |
| Gradient Boosting   | 0.840    | 0.30     |
| XGBoost             | 0.823    | 0.33     |
| Random Forest       | 0.725    | 0.35     |
| LightGBM            | 0.605    | 0.33     |
| SVM                 | 0.549    | 0.34     |
| Logistic Regression | 0.525    | 0.32     |

### 6.3 Analisis Hasil

* **Gradient Boosting** memiliki **akurasi tertinggi** (84%), namun **F1-macro hanya 0.30**, menandakan performa terhadap kelas minoritas masih buruk.
* **XGBoost dan Random Forest** menunjukkan keseimbangan yang lebih baik antara akurasi dan F1-macro.
* **F1-macro score di semua model relatif rendah**, mengindikasikan bahwa prediksi terhadap kelas `Serious` dan `Fatal` belum optimal.
* Kelas `Fatal` bahkan tidak dikenali oleh sebagian besar model (precision, recall, dan F1 = 0), yang disebabkan oleh jumlah data yang sangat kecil untuk kelas ini.
* Model linear (Logistic Regression dan SVM) cenderung memiliki akurasi dan F1 yang lebih rendah dibanding model ensemble seperti Random Forest dan Gradient Boosting.

### 6.4 Rekomendasi

* **Tambahkan lebih banyak data untuk kelas `Fatal`** guna meningkatkan kemampuan model dalam mengenali pola pada kelas langka.
* Coba terapkan teknik penyeimbangan seperti **SMOTE (Synthetic Minority Over-sampling Technique)** atau **class weight balancing** untuk mengatasi ketimpangan distribusi kelas.
* Di iterasi selanjutnya, evaluasi model juga dapat dilengkapi dengan **weighted F1-score** agar lebih merepresentasikan performa model pada distribusi asli.
* Eksplorasi lebih lanjut model berbasis boosting (seperti **CatBoost**) yang lebih sensitif terhadap ketidakseimbangan kelas bisa dipertimbangkan.

---
