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

# HASIL & ANALISIS

# KESIMPULAN

# REFERENCES

