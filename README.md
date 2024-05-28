# Triple-Threat-in-Oncology
### Comparing EfficientNet, Hybrid CNN EncoderTransformer, and ConvNext in Skin Cancer Classification

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

### Arsitektur Hybrid CNN EncoderTransformer

### Arsitektur ConvNext
# HASIL & ANALISIS

# KESIMPULAN

# REFERENCES

