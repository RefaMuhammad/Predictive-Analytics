# Laporan Proyek Machine Learning - Refa Muhammad

## Domain Proyek

Pasar properti merupakan salah satu sektor ekonomi yang sangat vital dalam perekonomian global. Stabilitas dan pertumbuhan harga properti mencerminkan kondisi sosial ekonomi suatu negara, termasuk daya beli masyarakat, perkembangan kawasan pemukiman, serta arah kebijakan fiskal dan moneter. Di Amerika Serikat, harga rumah menjadi indikator penting yang digunakan dalam berbagai keputusan investasi dan perencanaan pembangunan kota [4].

Proses penilaian harga rumah selama ini dilakukan secara manual oleh profesional properti melalui inspeksi langsung, pembandingan data historis, dan pendekatan penilaian subjektif. Namun, pendekatan ini memiliki banyak keterbatasan, terutama dalam hal efisiensi waktu, biaya operasional, dan konsistensi hasil penilaian. Penelitian modern menyarankan bahwa metode berbasis machine learning dapat digunakan untuk membangun model otomatis yang tidak hanya efisien tetapi juga mampu mendeteksi pola kompleks dalam data properti [2].

Model machine learning memungkinkan sistem untuk mempelajari hubungan antara berbagai atribut rumah — seperti ukuran, kualitas bangunan, lokasi, dan usia — dengan harga jual aktualnya. Hal ini memberikan potensi besar dalam otomasi proses valuasi aset, khususnya untuk platform digital, lembaga keuangan, maupun konsultan properti [5].

Dataset yang digunakan dalam proyek ini adalah **Ames Housing**, yang disusun oleh Dean De Cock dan disediakan secara publik untuk keperluan akademik dan pengujian algoritma prediktif. Dataset ini dinilai lebih baik daripada dataset Boston Housing yang telah banyak digunakan sebelumnya karena menyediakan lebih dari 80 fitur yang mencakup dimensi struktural, fungsional, dan lingkungan rumah [1].

Selain dari struktur dan kelengkapan fitur, dataset Ames Housing juga mendukung eksperimen model prediktif lanjutan karena kualitas datanya yang baik, proporsi missing value yang dapat ditangani, dan dokumentasi yang jelas. Oleh karena itu, dataset ini sangat cocok digunakan sebagai studi kasus penerapan regresi dalam machine learning, termasuk untuk model seperti regresi linier, Random Forest, dan Gradient Boosting [3].

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
11. **LotConfig**: Konfigurasi bentuk lot rumah (Inside, Corner, CulDSac, FR2, FR3).
12. **LandSlope**: Kemiringan lahan (Gtl = Gentle, Mod = Moderate, Sev = Severe).
13. **Neighborhood**: Lingkungan sekitar tempat tinggal.
14. **Condition1**: Kondisi rumah relatif terhadap jalan utama atau rel kereta.
15. **Condition2**: Kondisi kedua jika ada (misalnya rumah di antara dua rel kereta).
16. **BldgType**: Jenis bangunan (1Fam, 2fmCon, Duplex, Twnhs, TwnhsE).
17. **HouseStyle**: Gaya rumah (1Story, 2Story, 1.5Fin, SLvl, SFoyer, dll).
18. **OverallQual**: Kualitas material dan penyelesaian secara keseluruhan (skala 1–10).
19. **OverallCond**: Kondisi keseluruhan rumah (skala 1–10).
20. **YearBuilt**: Tahun rumah dibangun.
21. **YearRemodAdd**: Tahun renovasi terakhir atau tambahan struktur.
22. **RoofStyle**: Gaya atap (Flat, Gable, Hip, etc.).
23. **RoofMatl**: Material atap (CompShg, Metal, WdShngl, dll).
24. **Exterior1st**: Material eksterior utama.
25. **Exterior2nd**: Material eksterior sekunder (jika ada).
26. **MasVnrType**: Jenis veneer batu (None, BrkFace, Stone, dll).
27. **MasVnrArea**: Luas veneer batu dalam kaki persegi.
28. **ExterQual**: Kualitas eksterior (Ex, Gd, TA, Fa, Po).
29. **ExterCond**: Kondisi eksterior (Ex, Gd, TA, Fa, Po).
30. **Foundation**: Jenis fondasi rumah.
31. **BsmtQual**: Kualitas basement.
32. **BsmtCond**: Kondisi basement.
33. **BsmtExposure**: Tingkat eksposur basement terhadap luar.
34. **BsmtFinType1**: Jenis penyelesaian utama di basement.
35. **BsmtFinSF1**: Luas penyelesaian utama basement.
36. **BsmtFinType2**: Jenis penyelesaian sekunder di basement.
37. **BsmtFinSF2**: Luas penyelesaian sekunder basement.
38. **BsmtUnfSF**: Luas basement yang tidak selesai.
39. **TotalBsmtSF**: Total luas basement.
40. **Heating**: Jenis pemanas (GasA, Wall, etc.).
41. **HeatingQC**: Kualitas dan kondisi sistem pemanas.
42. **CentralAir**: Apakah ada AC sentral (Y/N).
43. **Electrical**: Sistem kelistrikan (SBrkr, FuseA, FuseF, etc.).
44. **1stFlrSF**: Luas lantai pertama.
45. **2ndFlrSF**: Luas lantai kedua.
46. **LowQualFinSF**: Luas area dengan kualitas rendah.
47. **GrLivArea**: Luas area tinggal di atas tanah.
48. **BsmtFullBath**: Jumlah kamar mandi penuh di basement.
49. **BsmtHalfBath**: Jumlah kamar mandi setengah di basement.
50. **FullBath**: Jumlah kamar mandi penuh di atas tanah.
51. **HalfBath**: Jumlah kamar mandi setengah di atas tanah.
52. **BedroomAbvGr**: Jumlah kamar tidur di atas tanah.
53. **KitchenAbvGr**: Jumlah dapur di atas tanah.
54. **KitchenQual**: Kualitas dapur (Ex, Gd, TA, Fa, Po).
55. **TotRmsAbvGrd**: Total ruangan di atas tanah (tidak termasuk kamar mandi).
56. **Functional**: Fungsionalitas rumah (Typ, Min1, Maj1, etc.).
57. **Fireplaces**: Jumlah perapian.
58. **FireplaceQu**: Kualitas perapian.
59. **GarageType**: Tipe garasi (Attached, Detached, BuiltIn, etc.).
60. **GarageYrBlt**: Tahun pembangunan garasi.
61. **GarageFinish**: Penyelesaian interior garasi.
62. **GarageCars**: Kapasitas mobil di garasi.
63. **GarageArea**: Luas garasi.
64. **GarageQual**: Kualitas garasi.
65. **GarageCond**: Kondisi garasi.
66. **PavedDrive**: Apakah jalan masuk dipaving (Y, P, N).
67. **WoodDeckSF**: Luas dek kayu.
68. **OpenPorchSF**: Luas beranda terbuka.
69. **EnclosedPorch**: Luas beranda tertutup.
70. **3SsnPorch**: Luas beranda 3 musim.
71. **ScreenPorch**: Luas beranda dengan kelambu.
72. **PoolArea**: Luas kolam renang.
73. **PoolQC**: Kualitas kolam.
74. **Fence**: Kualitas pagar.
75. **MiscFeature**: Fitur tambahan seperti lift atau rumah penyimpanan.
76. **MiscVal**: Nilai dari fitur tambahan.
77. **MoSold**: Bulan penjualan rumah.
78. **YrSold**: Tahun penjualan rumah.
79. **SaleType**: Tipe penjualan (WD, New, COD, dll).
80. **SaleCondition**: Kondisi penjualan (Normal, Abnorml, Partial, dll).

Target variabel: **SalePrice** – Harga jual rumah dalam dolar.

### Exploratory Data Analysis (EDA)

Beberapa langkah eksplorasi data dilakukan untuk memahami distribusi dan karakteristik dataset sebelum modeling:

- **Distribusi SalePrice**:
  Distribusi harga rumah (SalePrice) bersifat skewed kanan. Untuk mengatasi hal ini, dilakukan log-transformasi terhadap SalePrice agar distribusi menjadi lebih mendekati normal. Ini penting untuk meningkatkan performa model linear regression.

- **Korelasi antar fitur**:
  Korelasi Pearson dihitung antara semua fitur numerik dan target `SalePrice`. Fitur-fitur dengan korelasi tinggi meliputi:

  - `OverallQual`: 0.79
  - `GrLivArea`: 0.71
  - `GarageCars`: 0.64
  - `TotalBsmtSF`: 0.61
  - `1stFlrSF`: 0.61

- **Missing Value Analysis**:
  Visualisasi missing value dilakukan menggunakan heatmap. Fitur-fitur dengan missing value tinggi seperti `PoolQC`, `Fence`, `MiscFeature` di-drop karena lebih dari 50% datanya kosong. Fitur lain diimputasi dengan metode yang sesuai.

## Data Preparation

Tahapan data preparation dilakukan secara menyeluruh melalui langkah-langkah berikut:

### 1. Handling Missing Values

Dilakukan pemeriksaan terhadap nilai yang hilang (`NaN`) pada seluruh kolom. Strategi penanganannya dibedakan berdasarkan tipe dan proporsi nilai hilang:

- **Penghapusan Kolom**:

  - Kolom dengan persentase missing >90% dihapus karena dianggap tidak informatif, yaitu: `PoolQC`, `MiscFeature`, dan `Alley`.

- **Imputasi dengan Nilai Khusus ('None')**:

  - Untuk fitur kategorikal yang menunjukkan keberadaan suatu kondisi seperti garasi, perapian, atau basement, nilai yang hilang diisi dengan `'None'`, antara lain:  
    `FireplaceQu`, `Fence`, `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond`, `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2`, `MasVnrType`.

- **Imputasi dengan Nilai Statistik**:
  - `LotFrontage`: Diisi dengan **median** berdasarkan `Neighborhood` agar lebih kontekstual.
  - `MasVnrArea`: Diisi dengan **0**, menandakan tidak adanya veneer.
  - `GarageYrBlt`: Diisi dengan **0**, menandakan tidak ada garasi.
  - `Electrical`: Diisi dengan **modus** (nilai paling sering muncul).

### 2. Outlier Detection and Removal

Outlier pada fitur numerik dideteksi menggunakan metode **IQR (Interquartile Range)**. Fitur dengan outlier signifikan seperti `EnclosedPorch`, `BsmtFinSF2`, dan `ScreenPorch` dibersihkan dari nilai ekstrem yang berada di luar batas bawah (Q1 - 1.5×IQR) dan batas atas (Q3 + 1.5×IQR). Tujuannya adalah meningkatkan performa model dan menjaga distribusi data.

### 3. Feature Encoding

Untuk mengonversi fitur kategorikal ke dalam format numerik yang dapat diproses oleh algoritma machine learning, dilakukan beberapa teknik encoding:

- **Penghapusan Fitur Kategorikal dengan Satu Nilai Unik**:

  Beberapa fitur kategorikal dalam dataset memiliki hanya satu nilai (kategori) yang muncul di hampir seluruh entri. Fitur semacam ini tidak memberikan informasi yang berguna bagi model karena tidak memiliki variasi. Oleh karena itu, fitur-fitur berikut dihapus dari dataset:
    - Street
    - Utilities
    - Condition2
    - RoofMatl
    - BsmtFinType2

  Penghapusan dilakukan sebelum proses encoding untuk menghindari pemborosan dimensi dan mencegah noise yang tidak relevan dalam proses pelatihan model. Keputusan ini didasarkan pada analisis distribusi frekuensi kategori dari masing-masing fitur.

- **Ordinal Encoding**:

  - Diterapkan pada fitur kategorikal yang memiliki urutan atau skala tingkat kualitas. Fitur-fitur berikut dikodekan berdasarkan urutan logis atau kualitas yang telah ditentukan: `ExterQual`, `ExterCond`, `BsmtQual`, `BsmtCond`, `BsmtExposure`, `HeatingQC`, `KitchenQual`, `FireplaceQu`, `GarageQual`, `GarageCond`, `GarageFinish`, dan `PavedDrive`.

- **Frequency Encoding**:

  - Diterapkan pada fitur `Neighborhood`, dengan mengganti tiap nilai kategori dengan frekuensi kemunculannya. Ini menghindari ledakan dimensi akibat One-Hot Encoding.

- **Label Encoding**:

  - Diterapkan pada beberapa fitur kategorikal lain seperti: `MSZoning`, `LotShape`, `LandContour`, `LandSlope`, `Condition1`, `RoofStyle`, `Heating`, dan `BsmtFinType1`. Encoding ini memberikan label numerik unik untuk tiap kategori.

- **One-Hot Encoding**:
  - Digunakan pada fitur nominal yang memiliki jumlah kategori terbatas dan tidak memiliki urutan logis. Fitur-fitur yang diberikan one-hot encoding adalah: `LotConfig`, `HouseStyle`, `Exterior1st`, `Exterior2nd`, `SaleType`, `SaleCondition`, `BldgType`, `Foundation`, `GarageType`, `Fence`, `MasVnrType`, `CentralAir`, `Electrical`, `Functional`, dan `GarageFinish`.

### 4. Data Splitting

Data dibagi menjadi:

- **80% untuk pelatihan (training set)**
- **20% untuk pengujian (test set)**

Pembagian dilakukan secara acak, dan target `SalePrice` ditransformasi menggunakan logaritma natural (`log(SalePrice)`) untuk menstabilkan varian dan mengatasi skewness distribusi harga rumah.

---

## Modeling

### Model 1: Linear Regression

#### Cara Kerja:

Linear Regression bekerja dengan mencari hubungan linier antara variabel input (fitur) dan output (target). Model ini meminimalkan jumlah kuadrat kesalahan (mean squared error) untuk mendapatkan garis terbaik yang merepresentasikan data.

#### Parameter:

- `fit_intercept=True`: menyertakan bias (intersep) dalam model.
- Semua parameter lainnya menggunakan nilai default.

#### Kelebihan:

- Mudah diimplementasikan dan cepat dilatih.
- Hasil model mudah untuk diinterpretasikan oleh pengguna non-teknis.

#### Kekurangan:

− Tidak cocok untuk hubungan non-linear antar variabel.  
− Sensitif terhadap outlier dan multikolinearitas.

---

### Model 2: Random Forest Regressor

#### Cara Kerja:

Random Forest adalah algoritma ensemble yang membangun banyak pohon keputusan dan menggabungkan hasilnya (rata-rata) untuk membuat prediksi regresi. Setiap pohon dibangun dari subset data dan subset fitur secara acak (_bagging_), sehingga meningkatkan generalisasi dan mengurangi overfitting.

#### Parameter:

- `n_estimators=100`: jumlah pohon dalam hutan.
- `random_state=42`: untuk memastikan hasil replikasi yang konsisten.
- Parameter lainnya menggunakan nilai default.

#### Kelebihan:

- Mampu menangkap hubungan non-linear antar fitur.
- Relatif tahan terhadap overfitting dan outlier.

#### Kekurangan:

− Interpretasi model lebih kompleks daripada linear regression.  
− Proses pelatihan dan prediksi memerlukan waktu dan memori lebih besar.

---

### Model 3: Gradient Boosting Regressor

#### Cara Kerja:

Gradient Boosting membangun model secara bertahap. Setiap model baru dibuat untuk memperbaiki kesalahan model sebelumnya menggunakan pendekatan _gradient descent_. Model ini efektif dalam menangani fitur kompleks dan interaksi non-linear.

#### Parameter:

- `n_estimators=100`: jumlah boosting rounds.
- `random_state=42`: untuk hasil yang konsisten.
- Parameter lainnya menggunakan nilai default.

#### Kelebihan:

- Performa prediksi sangat baik pada data tabular.
- Mampu menangani fitur yang kompleks dan interaksi antar fitur.

#### Kekurangan:

− Butuh tuning parameter yang teliti untuk performa optimal.  
− Proses training lebih lama dan lebih kompleks dibanding model lain.

---

## Evaluation

### Metrik Evaluasi:

- **Root Mean Squared Error (RMSE)**: Mengukur rata-rata kesalahan prediksi dalam satuan asli (USD).
- **R² Score**: Proporsi variansi target yang dapat dijelaskan oleh fitur input.

| Model                     | RMSE (USD) | R² Score | Fitur Digunakan |
| ------------------------- | ---------- | -------- | --------------- |
| Linear Regression         | 16,859.43  | 0.89     | Semua fitur     |
| Linear Regression + Lasso | 16,737.38  | 0.90     | Top fitur       |
| Random Forest Regressor   | 15,482.64  | 0.91     | Semua fitur     |
| Random Forest + FS        | 15,276.63  | 0.91     | Top fitur       |
| Gradient Boosting         | 14,088.34  | 0.93     | Semua fitur     |
| Gradient Boosting + FS    | 14,000.01  | 0.93     | Top 20 fitur    |

> _Catatan: Target `SalePrice` ditransformasikan ke skala logaritmik selama pelatihan model untuk menstabilkan distribusi. Namun, nilai RMSE dan R² di atas dihitung setelah prediksi dikembalikan ke skala asli menggunakan `np.expm1()`._

---

### Analisis dan Keterkaitan dengan Business Understanding:

Model Gradient Boosting menunjukkan performa terbaik dengan **RMSE terendah (~$14.000)** dan **R² tertinggi (0.93)**. Ini menunjukkan bahwa model mampu menjelaskan sekitar **93% variasi harga rumah**, dengan rata-rata kesalahan prediksi dalam kisaran yang dapat diterima secara bisnis.

#### Kesesuaian dengan Problem Statements:

- **Problem 1: Akurasi Prediksi** — Model sangat akurat dan realistis.
- **Problem 2: Model Terbaik** — Gradient Boosting unggul dalam performa.
- **Problem 3: Fitur Signifikan** — Fitur seperti `OverallQual`, `GrLivArea`, dan `GarageCars` terbukti dominan.

---

## Kesimpulan

Proyek ini berhasil menunjukkan bagaimana metode machine learning dapat digunakan untuk memprediksi harga rumah berdasarkan fitur-fitur struktural dan lingkungan yang tersedia.

### Poin-Poin Utama:

- **Linear Regression** digunakan sebagai baseline, namun kurang cocok untuk pola data yang kompleks.
- **Random Forest Regressor** meningkatkan akurasi dan mampu menangani interaksi non-linear antar fitur.
- **Gradient Boosting Regressor** menghasilkan performa terbaik dengan **RMSE sekitar $14.000** dan **R² sebesar 0.93**, menjadikannya model akhir.
- Feature importance dari Gradient Boosting menunjukkan bahwa `OverallQual`, `GrLivArea`, dan `GarageCars` adalah fitur yang paling berpengaruh terhadap harga rumah.

### Relevansi Bisnis:

- Estimasi harga rumah menjadi **otomatis, cepat, dan konsisten**, membantu agen properti, penjual, maupun pembeli.
- Interpretasi fitur penting dapat digunakan oleh pengembang atau pemilik rumah untuk **meningkatkan nilai jual properti**.

### Pengembangan Lanjutan:

- Integrasi data spasial seperti GPS atau lokasi kota.
- Penggunaan data historis harga properti untuk tren jangka panjang.
- Eksperimen dengan teknik ensemble lain seperti **stacking** untuk meningkatkan akurasi.

Secara keseluruhan, proyek ini berhasil memenuhi tujuan teknis dan bisnis, dan berpotensi besar untuk diimplementasikan dalam aplikasi valuasi properti digital.

## Referensi

[1] De Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project. _Journal of Statistics Education_, 19(3). https://doi.org/10.1080/10691898.2011.11889627

[2] A. Kumar, S. Aggarwal and N. Kumar, "Predicting House Prices Using Machine Learning Algorithms: A Review," _2020 International Conference on Computational Performance Evaluation (ComPE)_, 2020.

[3] Barr, J., Ellis, E. A., Kassab, A., Redfearn, C. L., Srinivasan, N. N., & Voris, K. B. (2017). Home Price Index: A Machine Learning Methodology. International Journal of Semantic Computing, 11(1), 111–133. https://doi.org/10.1142/S1793351X17500015

[4] Park, B., & Bae, J. K. (2015). Using machine learning algorithms for housing price prediction: The case of Fairfax County, Virginia housing data. Expert Systems with Applications, 42(6), 2928–2934. https://doi.org/10.1016/j.eswa.2014.11.040

[5] Yazdani, M. (2021). Machine Learning, Deep Learning, and Hedonic Methods for Real Estate Price Prediction. arXiv preprint arXiv:2110.07151. https://arxiv.org/abs/2110.07151
