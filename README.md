
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
