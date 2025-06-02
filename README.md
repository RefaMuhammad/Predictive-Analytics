
# Laporan Proyek Machine Learning - Prediksi Harga Rumah di Amerika

## Domain Proyek

Pasar properti merupakan salah satu sektor ekonomi yang paling dinamis dan krusial, terutama di negara-negara maju seperti Amerika Serikat. Harga rumah menjadi indikator penting dalam menilai stabilitas ekonomi, kekuatan daya beli masyarakat, dan potensi investasi. Namun, penilaian harga rumah tidaklah sederhana karena melibatkan banyak faktor—mulai dari karakteristik fisik rumah, kondisi lingkungan sekitar, hingga kondisi pasar saat itu.

Tradisionalnya, penilaian rumah dilakukan oleh penilai profesional berdasarkan inspeksi fisik dan data historis. Proses ini tidak hanya mahal dan memakan waktu, tetapi juga mengandung unsur subjektivitas. Oleh karena itu, pendekatan berbasis *machine learning* menawarkan solusi yang menjanjikan: prediksi harga rumah secara otomatis, cepat, dan konsisten berdasarkan data historis dan atribut rumah.

Proyek ini bertujuan membangun model regresi untuk memprediksi harga jual rumah di kota Ames, Iowa, menggunakan dataset "Ames Housing". Dataset ini populer dalam riset akademik dan dianggap sebagai versi perbaikan dari dataset klasik Boston Housing. Dengan lebih dari 80 fitur, dataset ini menawarkan kompleksitas yang cukup untuk pengujian berbagai model machine learning.

Referensi pendukung untuk domain ini termasuk:
- De Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project. *Journal of Statistics Education*, 19(3). https://doi.org/10.1080/10691898.2011.11889627
- A. Kumar, S. Aggarwal and N. Kumar, "Predicting House Prices Using Machine Learning Algorithms: A Review," *2020 International Conference on Computational Performance Evaluation (ComPE)*, 2020.

## Business Understanding

### Problem Statements

1. Bagaimana cara memprediksi harga rumah secara akurat berdasarkan fitur struktural, kondisi rumah, dan lingkungan sekitar?
2. Algoritma machine learning mana yang memberikan hasil terbaik untuk regresi harga rumah?
3. Fitur-fitur mana yang paling signifikan dalam menentukan harga rumah?

### Goals

1. Mengembangkan model prediksi harga rumah dengan akurasi tinggi.
2. Membandingkan performa beberapa model regresi.
3. Melakukan interpretasi fitur penting yang memengaruhi harga rumah.

### Solution Statements

- Menerapkan dan membandingkan 3 model:
  - Linear Regression (baseline)
  - Random Forest Regressor
  - Gradient Boosting Regressor (XGBoost)
- Melakukan preprocessing: imputasi missing value, encoding kategorikal, scaling numerik, dan log-transformasi target.
- Melakukan tuning hyperparameter pada model ensemble.
- Mengukur performa menggunakan RMSE dan R².

## Data Understanding

Dataset: [Ames Housing Dataset - OpenML ID 43926](https://www.openml.org/d/43926)  
Jumlah entri: 1460  
Jumlah fitur: 81 (80 fitur input + 1 target/output)

### Penjelasan Setiap Fitur dalam Dataset:

_(Hanya 10 fitur pertama ditampilkan di sini. Semua 81 fitur akan dilanjutkan dalam file lengkap.)_

1. **Id**: Nomor urut unik untuk setiap entri rumah.
2. **MSSubClass**: Tipe bangunan yang dikodekan (misal: 20 = 1-story 1946 & newer).
3. **MSZoning**: Kategori zona lahan (misal: RL = Residential Low Density).
4. **LotFrontage**: Panjang frontage properti yang menghadap jalan (dalam kaki).
5. **LotArea**: Luas total lahan (dalam kaki persegi).
6. **Street**: Tipe akses jalan (misal: Pave = jalan beraspal).
7. **Alley**: Tipe akses gang belakang rumah (banyak nilai kosong).
8. **LotShape**: Bentuk lahan (Reg = Regular, IR1/2/3 = Irregular).
9. **LandContour**: Kontur lahan (Flat, HLS = Hillside, Bnk = Banked).
10. **Utilities**: Ketersediaan utilitas publik.

_(Fitur 11–80 akan dilanjutkan di bawah bagian ini dalam file lengkap)_

Target variabel: **SalePrice** – Harga jual rumah dalam dolar.

## Data Preparation

- **Imputasi Missing Value**:
  - Numerik: Median (contoh: `LotFrontage`, `MasVnrArea`)
  - Kategorikal: Modus (contoh: `GarageType`, `Electrical`)
- **Drop kolom dengan >50% missing**: `PoolQC`, `Fence`, `MiscFeature`, dll.
- **Encoding**:
  - OneHot untuk fitur nominal.
  - Ordinal encoding untuk fitur kategorikal bertingkat (misalnya `KitchenQual`, `ExterQual`).
- **Scaling**: StandardScaler pada fitur numerik.
- **Transformasi Target**: Log-transformasi pada `SalePrice` untuk mengurangi skewness.

## Modeling

### 1. Linear Regression
- Digunakan sebagai baseline.
- Cepat, tetapi kurang fleksibel terhadap data non-linear.

### 2. Random Forest Regressor
- Ensemble dari decision tree.
- Lebih robust terhadap overfitting dibanding linear regression.
- Tuning parameter: `n_estimators`, `max_depth`

### 3. Gradient Boosting Regressor (XGBoost)
- Lebih presisi, mampu menangani hubungan non-linear dan outlier.
- Tuning parameter: `learning_rate`, `max_depth`, `n_estimators`

## Evaluation

### Metrik:
- **RMSE (Root Mean Squared Error)** – untuk menghitung error prediksi dalam satuan log-dollar.
- **R² Score** – untuk mengukur variansi yang dijelaskan oleh model.

### Hasil:

| Model                    | RMSE     | R² Score |
|--------------------------|----------|----------|
| Linear Regression        | 0.181    | 0.89     |
| Random Forest Regressor | 0.145    | 0.93     |
| Gradient Boosting        | **0.138** | **0.94** |

### Kesimpulan:
Gradient Boosting memberikan hasil terbaik dalam memprediksi harga rumah. Model ini menangani variabel kompleks dan data non-linear dengan baik.

---

> Laporan ini ditulis untuk mendokumentasikan proses pengembangan model machine learning pada dataset Ames Housing. Model akhir dapat digunakan dalam sistem estimasi harga properti berbasis AI.
