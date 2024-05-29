### Comparing EfficientNet, Hybrid CNN EncoderTransformer, and ConvNext in Skin Cancer Classification 🧩

# LATAR BELAKANG
Kanker kulit merupakan salah satu dari lima jenis kanker yang paling banyak kasusnya di seluruh dunia (World Health Organization, 2022). Deteksi dini kanker kulit sangat penting untuk meningkatkan peluang kesembuhan dan mengurangi risiko komplikasi lebih lanjut. Namun, diagnosis kanker kulit secara manual oleh dokter ahli dapat menjadi proses yang menantang dan memakan waktu. Dengan kemajuan teknologi AI dalam beberapa tahun terakhir, pendekatan berbasis Convolutional Neural Network (CNN) telah menjadi metode yang populer untuk klasifikasi dan deteksi kanker kulit dari gambar (Esteva et al., 2017). Namun, model CNN tradisional memiliki beberapa keterbatasan, seperti sulitnya menangkap informasi kontekstual dan hubungan spasial yang kompleks dalam suatu gambar. Untuk menghadapi masalah ini, disini peneliti mencoba membandingkan kinerja pendekatan klasifikasi kanker kulit dengan model-model SOTA seperti EfficientNet, Hybrid CNN EncoderTransformer, dan ConvNext.
# TUJUAN PENELITIAN
- Memvalidasi kinerja model dengan menggunakan dataset gambar dermatoskopi yang representatif
- Menginspirasi penelitian lanjutan untuk mengembangkan metode yang lebih efektif dan efisien dalam bidang deteksi kanker kulit berbasis AI
- Menyediakan alat bantu diagnostik yang dapat membantu tenaga medis dalam mendeteksi kanker kulit secara akurat dan cepat
- Menyediakan solusi berbasis teknologi yang dapat diakses oleh berbagai fasilitas kesehatan, termasuk yang berada di daerah dengan akses terbatas ke spesialis kulit
# BATASAN MASALAH
- Penelitian ini dibatasi hanya menggunakan open data dari MNIST HAM10000
- Penelitian ini hanya memfokuskan pada proses klasifikasi gambar dermatoskopi untuk mengidentifikasi jenis-jenis kanker kulit, belum masuk di tahap lanjutan seperti deteksi dan segmentasi objek
- Tidak ada analisis tambahan terkait interpretabilitas model atau studi kasus spesifik secara klinis.
# METODE PENELITIAN
## Data Loading
Memuat dataset MNIST HAM10000 ke dalam environment. Dengan melibatkan pembacaan dataset dalam format yang sesuai serta menggabungkan directory pada image 1 dan 2 guna mempersiapkan data untuk pengolahan selanjutnya.
## Konfigurasi dan Inisialisasi
Selanjutnya, melakukan konfigurasi awal, termasuk ukuran gambar, transformasi, dan metode augmentasi yang akan diterapkan. Inisialisasi ini dilakukan untuk mempersiapkan data gambar yang telah ditransformasi sebelum dimasukkan ke dalam model.
## Pembuatan Custom Dataset & Dataloader
Dataset khusus dibangun untuk memuat data gambar dan label terkait. Hal ini melibatkan proses pembuatan kelas custom dataset yang mengatur cara data dimuat dan diproses, serta pembuatan dataLoader untuk memudahkan penggunaan data dalam proses pelatihan dan validasi.
## Modeling
### Arsitektur EfficientNetB0
<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=12_1OTnFHoQd3bqi6fRS8fo4LFnXvQseI" alt="EffNetB0">
</p>
Jadi input layer disini berdimensi HxWxC -> 224 pixel x 224 pixel x 3 channel RGB, berikutnya akan diteruskan ke layer konvolusi pertama dengan ukuran kernel 3x3 stride 2 yang menjadikan tensor berukuruan 112x112x32, berikutnya masuk ke layer MBCONV1 dengan ukuran kernel 3x3 stride 1 menjadikan output tensor 112x112x16 dst dengan skema yang sama. Disini peneliti langsung melakukan pretrained pada data ImageNet lalu melakukan unfreeze di final layernya untuk menyesuaikan output kelas yang ada.
<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1YcJlsJPsMCm2aKQE-vkm1FV6Lk3TT5ie" alt="mbconv">
</p>
Sedikit penjelasan tentang layer MBConv, jadi MBConv adalah `Mobile Inverted Bottleneck Convolution` dimana network dirancang untuk dapat dijalankan di resource yang terbatas karena parameter dan komputasi yang lebih efisien. Inverted Bottleneck disini menggunakan prinsip bottleneck seperti apa yang digunakan di arsitektur ResNet, dimana `ruang filter akan dikurangi sebelum ekspansi`. Inverted disini berari urutan operasinya setalah kita ekspansi dimensi kemudian melakukan pengurangan dimensi. 

### Arsitektur Hybrid CNN EncoderTransformer
<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1EjdHF6RMhzWroYfrRtqo_3y7erIUpJXf" alt="mbconv">
</p>
Arsitektur ini dibangun secara custom, dimulai dari memasukan input gambar 224x224x3 berikutnya input diubah(reshape) menjadi bentuk 1 dimensi. Lalu masuk kedalam layer convolutional berdimensi 1 untuk melakukan ekstraksi fitur lebih lanjut. Untuk mengurangi dimensi output dari layer convolutional 1 dimensi, peneliti menggunakan Adaptive Max Pooling berikutnya tensor diflatten menjadi bentuk 2 Dimensi agar dapat diproses oleh layer fully connected, berikutnya tensor di unsqueeze agar dapat diterima oleh layer transformer. Setelah selesai di proses oleh layer transformer kembalikan ukuran dengan skema squeeze agar dapat diproses oleh layer fully connected terakhir untuk case klasfikasi. 
Detail per layer:<br>

```
input (batch_size,3,224,224)
reshape(batch_size,3,224x224)
CNN Block(batch_size, cnn_out_channels, 224x224)
Pooling (batch_size,cnn_out_channels,1)
Flattening(batch_size, cnn_out_channels)
Fully Connected(batch_size,transformer_hidden_dim)
Unsqueeze(batch_size,1,transformer_hidden_dim)
Transformer Encoder(batch_size,1,transformer_hidden_dim)
Squeeze(batch_size,transformer_hidden_dim)
Output Classifier(batch_size, output_dim)
```

### Arsitektur ConvNext
<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1-DmrmIx9B2z53X1GJsh8PFqF7Mb3JCHf" alt="mbconv">
</p>
Untuk arsitekur dari ConvNext dimulai dari Stem layer yang menggunakan conv2D dengan kernel size 4 dan stride 4 jadi resolusi dari gambar yang semulanya 224x224 menjadi 56x56, lalu layer dinormalisasi. Masuk ke stage 1 channel akan diubah yang semulanya 64 menjadi 128. Stage 2 mengubah 128 menjadi  256 channels. Stage 3 mengubah dari 256 menjadi 512 channels. Stage 4 akan mengubah 512 menjadi 1024 channels. Terakhir akan diteruskan ke GAP dan layer linear fully connected yang akan menghasilkan output sebanyak class yang sudah didefine sebelumnya. 
Untuk penjelasan arsitektur yang ada didalam ConvNext block, seperti berikut. <br>
<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1ScRATbX5_XwVjqMGtCj0lm1RrOY3HCNc" alt="mbconv">
</p>
<p align="center">
  []
</p>
Input tensor akan masuk kedalam Depthwise Conv2D ebrikutnya diproses melewati Layer Norm, tensor yang telah dinormalisasi akan diteruskan ke Pointwise Conv2D, berikutnya Tensor akan diberikan fungsi aktivasi GELU, dan diteruskan lagi ke Pointwise Conv2D kedua. Tensor akan diubah lagi dimensinya lalu diskalasi dengan Scale Layer yang diteruskan ke drop path untuk melakukan regularization, terkahir tensor akan ditambahkan dengan input asli dengan skema skip connection, hasil inilah yang nantinya akan diteruskan menjadi output dari ConvNext Block.

# HASIL & ANALISIS



| Model         | Class | Precision | Recall | F1-Score | Support | Model        | Class | Precision | Recall | F1-Score | Support |
|---------------|-------|-----------|--------|----------|---------|--------------|-------|-----------|--------|----------|---------|
|   | akiec | 0.66      | 0.66   | 0.66     | 65      |  | akiec | 0.00      | 0.00   | 0.00     | 65      |
|   | bcc   | 0.84      | 0.83   | 0.84     | 103     |  | bcc   | 0.00      | 0.00   | 0.00     | 103     |
|   | bkl   | 0.80      | 0.67   | 0.73     | 220     |  | bkl   | 0.00      | 0.00   | 0.00     | 220     |
|**ConvNeXt** | df    | 0.58      | 0.48   | 0.52     | 23      | **CNN+Transformer** | df    | 0.00      | 0.00   | 0.00     | 23      |
|   | mel   | 0.83      | 0.41   | 0.55     | 223     |  | mel   | 0.20      | 0.00   | 0.01     | 223     |
|   | nv    | 0.88      | 0.99   | 0.93     | 1341    |  | nv    | 0.67      | 1.00   | 0.80     | 1341    |
|   | vasc  | 0.96      | 0.82   | 0.88     | 28      |  | vasc  | 0.00      | 0.00   | 0.00     | 28      |
|  | **accuracy** | **0.86** | | | **2003** |  | **accuracy** | **0.67** | | | **2003** |
|  | **macro avg** | **0.79** | **0.69** | **0.73** | **2003** |  | **macro avg** | **0.12** | **0.14** | **0.12** | **2003** |
|  | **weighted avg** | **0.86** | **0.86** | **0.85** | **2003** |  | **weighted avg** | **0.47** | **0.67** | **0.54** | **2003** |

| Model         | Class | Precision | Recall | F1-Score | Support |
|---------------|-------|-----------|--------|----------|---------|
|    | akiec | 0.50           | 0.45   | 0.47    | 65      |
|       | bcc   | 0.69      | 0.82   | 0.75     | 103     |
|       | bkl   | 0.68      | 0.67   | 0.73     | 220     |
| **EffNetB0**      | df    | 0.00      | 0.00   | 0.00     | 23      |
|       | mel   | 0.73      | 0.32   | 0.45     | 223     |
|       | nv    | 0.85      | 0.98   | 0.91     | 1341    |
|       | vasc  | 0.70      | 0.68   | 0.69     | 28      |
|  | **accuracy** | **0.81** | | | **2003** |
|  | **macro avg** | **0.59** | **0.53** | **0.55** | **2003** |
|  | **weighted avg** | **0.79** | **0.81** | **0.78** | **2003** |

Dari hasil classification report tsb, model dari ConvNext meraih performa paling baik dengan nilai acc 86 diikuti oleh model EfficientNet dan Hybrid CNN+Transformers diposisi terendah. Di semua model nilai tertinggi ada dikelas nv hal ini disebabkan juga oleh persebaran data yang ada dimana data kelas nv sangat dominan dibanding yang lainnya. Pada model Hybrid CNN+Transformers kelas yang berhasil diprediksi bahkan sangat sedikit hanya ada kelas nv dan mel, hal ini terjadi karena saya disini hanya menggunakan layer CNN yang tidak mendalam dan transformer blocknya juga selain itu model ini tidak seperti yang lainnya yang telah dipretrained jadi score 67 bukan menjadi sebuah kejutan. disisi lain EfficientNet juga masih terdapat kelas yang tidak bisa diprediksi sama sekali yaitu df karena datanya memang sedikit. Untuk ConvNext model ini sudah sangat baik dalam memprediksi sebagian besar kelas tetapi juga masih kesulitan dalam mempredict kelas kelas minoritas seperti df dan mel yang nilai accnya masih dibawah 60. Untuk perbandingan antar waktu trainingnya sebagai berikut.
|Model|iteration/s|
|---|---|
|EffNetB0| 1.7|
|Hybrid CNN+Transformers|2.01|
|ConvNeXt|2.13|

Dari hasil waktu training tsb, peneliti menemukan model ConvNeXt tidak hanya memberikan akurasi yang lebih tinggi tetapi juga memiliki kecepatan iterasi yang jauh lebih baik dibandingkan dengan EfficientNetB0 dan Hybrid CNN+Transformers. Kecepatan iterasi ConvNeXt sebesar 2.13 iteration/s menunjukkan efisiensi dalam proses training model. Kesimpulannya, ConvNeXt merupakan model yang paling efektif dalam problem ini, mengingat performa akurasinya yang tinggi serta waktu training yang cepat.

# REFERENCES

