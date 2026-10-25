# Mission-06C — KNN: Machine Condition Classification on ESP32

> **Mata kuliah:** Machine Learning · **Platform:** ESP32 + MicroPython · **Jenis:** klasifikasi biner · **Perangkat tambahan:** tidak ada  
> **Estimasi waktu:** 80–100 menit · **Prasyarat:** konsep train/test split, KNN, normalisasi, accuracy, confusion matrix, menjalankan MicroPython di Thonny.

## 1. Tujuan pembelajaran

Setelah praktikum, mahasiswa mampu:

1. Menjelaskan mengapa KNN dapat berjalan di ESP32 tanpa Scikit-learn dan tanpa fase pelatihan parameter eksplisit.
2. Mengimplementasikan perhitungan jarak, pemilihan **K tetangga terdekat**, dan *majority voting* dengan MicroPython.
3. Menguji pengaruh nilai **K** dan skala fitur pada hasil klasifikasi.
4. Mengevaluasi model dengan **test set terpisah**, accuracy, dan confusion matrix.
5. Mengukur perkiraan *inference latency* dan mengidentifikasi keterbatasan memori perangkat.

## 2. Studi kasus

Bayangkan sebuah mesin industri dipantau berdasarkan dua fitur:

- **Temperature** ($x_1$): temperatur mesin dalam derajat Celsius (°C).
- **Vibration** ($x_2$): tingkat getaran mesin dalam mm/s.

Target klasifikasi:

- **0 = NORMAL**
- **1 = WARNING**

**Penting:** Semua data adalah **sintetik** untuk pembelajaran. Tidak ada sensor yang dihubungkan ke ESP32. Label bukan standar diagnosis kondisi mesin nyata; jangan gunakan model ini untuk keselamatan mesin.

**Alur kerja:** `Data sintetik → penyimpanan di ESP32 → scaling → KNN → prediksi → evaluasi`.

## 3. Alat dan persiapan

- ESP32 yang telah di-*flash* dengan firmware MicroPython.
- Kabel USB data dan komputer dengan Thonny IDE.
- Tidak perlu breadboard, sensor, LED eksternal, koneksi internet, atau paket ML.

Di Thonny, pilih interpreter **MicroPython (ESP32)** dan port serial ESP32. Buat file `main.py` **di perangkat ESP32**. Program akan mencetak hasil di Shell Thonny. Jika sudah ada `main.py` dari praktikum blink, salin/ubah namanya terlebih dahulu agar tidak berjalan bersamaan.

## 4. Ringkasan teori

KNN melakukan klasifikasi dengan mencari **K contoh training yang paling dekat** dengan input baru dan memilih label yang paling banyak muncul di antara tetangga tersebut.

Dengan dua fitur, jarak Euclidean adalah:

$$
d = \sqrt{(x_1-a_1)^2 + (x_2-a_2)^2}
$$

Untuk pengurutan tetangga, akar tidak diperlukan sehingga kita menghitung **jarak kuadrat**:

$$
d^2 = (x_1-a_1)^2 + (x_2-a_2)^2
$$

Keterangan:

- $x_1$, $x_2$: fitur input yang akan diklasifikasikan.
- $a_1$, $a_2$: fitur satu sampel di data training.
- $d$: jarak Euclidean antara kedua sampel.
- $K$: jumlah tetangga yang ikut *voting*.

Karena temperatur dan getaran memiliki rentang angka berbeda, kita gunakan **min–max scaling berdasarkan data training**:

$$
x' = \frac{x-x_{\min}}{x_{\max}-x_{\min}}
$$

> Gunakan rentang minimum dan maksimum **hanya dari training set**. Data test dan input baru menggunakan parameter scaling yang sama. Jangan menyesuaikan scaler menggunakan test set. Nilai data baru yang berada di luar rentang training dapat menghasilkan nilai scaling di luar $[0,1]$; hal ini tidak otomatis salah.

**Catatan konsep:** KNN menyimpan sampel referensi saat disebut “training”. Beban utamanya terjadi ketika **inference**, karena harus menghitung jarak ke sampel referensi.

## 5. Dataset

Modul menyediakan **36 sampel training** dan **12 sampel test** dalam kode. Label dibuat dengan aturan simulasi yang memiliki area batas kelas sehingga prediksi tidak selalu 100% benar. Data test **tidak** digunakan sebagai tetangga saat inference.

Format setiap baris:

```python
(temperature, vibration, label)
```

Contoh: `(68.0, 6.1, 1)` berarti temperatur 68°C, getaran 6,1 mm/s, label WARNING.

## 6. Program utama MicroPython

Simpan kode ini sebagai `main.py` pada ESP32, lalu klik **Run**.

```python
# Mission-06D - KNN pada ESP32 (MicroPython)
import time
import gc

# temperature (C), vibration (mm/s), label (0=NORMAL; 1=WARNING)
TRAIN = [
    (29.0, 1.0, 0),
    (42.4, 2.7, 0),
    (55.8, 4.4, 0),
    (68.0, 6.1, 1),
    (31.4, 7.8, 0),
    (44.8, 1.2, 0),
    (57.0, 2.9, 0),
    (70.4, 4.6, 1),
    (33.8, 6.3, 0),
    (46.0, 8.0, 1),
    (59.4, 1.4, 0),
    (72.8, 3.1, 1),
    (35.0, 4.8, 0),
    (48.4, 6.5, 1),
    (61.8, 8.2, 1),
    (74.0, 1.6, 0),
    (37.4, 3.3, 0),
    (50.8, 5.0, 0),
    (63.0, 6.7, 1),
    (76.4, 8.4, 1),
    (39.8, 1.8, 0),
    (52.0, 3.5, 0),
    (65.4, 5.2, 1),
    (78.8, 6.9, 1),
    (41.0, 8.6, 1),
    (54.4, 2.0, 0),
    (67.8, 3.7, 1),
    (30.0, 5.4, 0),
    (43.4, 7.1, 0),
    (56.8, 8.8, 1),
    (69.0, 2.2, 0),
    (32.4, 3.9, 0),
    (45.8, 5.6, 0),
    (58.0, 7.3, 1),
    (71.4, 9.0, 1),
    (34.8, 2.4, 0),
]
TEST = [
    (32, 1.6, 0),
    (39, 3.1, 0),
    (52, 3.0, 0),
    (60, 4.5, 1),
    (47, 6.5, 1),
    (70, 7.5, 1),
    (78, 8.0, 1),
    (35, 7.9, 0),
    (73, 2.5, 1),
    (57, 5.5, 1),
    (46, 4.4, 0),
    (65, 6.2, 1),
]
LABELS = ("NORMAL", "WARNING")

# Parameter scaling dihitung HANYA dari data training.
MIN_T = min(r[0] for r in TRAIN)
MAX_T = max(r[0] for r in TRAIN)
MIN_V = min(r[1] for r in TRAIN)
MAX_V = max(r[1] for r in TRAIN)
RANGE_T = MAX_T - MIN_T
RANGE_V = MAX_V - MIN_V


def predict_knn(temp, vibration, k=3, scaled=True):
    if not isinstance(k, int) or k < 1 or k > len(TRAIN) or k % 2 == 0:
        raise ValueError("Gunakan K ganjil: 1 sampai jumlah data training")
    distances = []
    for idx, (t, v, label) in enumerate(TRAIN):
        dt = temp - t
        dv = vibration - v
        if scaled:
            dt = dt / RANGE_T
            dv = dv / RANGE_V
        distance_sq = dt * dt + dv * dv
        distances.append((distance_sq, idx, label))

    # Tie jarak dipecahkan secara deterministik memakai urutan data.
    distances.sort(key=lambda r: (r[0], r[1]))
    neighbors = distances[:k]
    warning_votes = sum(r[2] for r in neighbors)
    prediction = 1 if warning_votes > k // 2 else 0
    return prediction, neighbors


def evaluate(k=3, scaled=True, verbose=False):
    correct = 0
    tp = tn = fp = fn = 0
    for temp, vib, actual in TEST:
        predicted, _ = predict_knn(temp, vib, k, scaled)
        correct += predicted == actual
        if predicted == 1 and actual == 1:
            tp += 1
        elif predicted == 0 and actual == 0:
            tn += 1
        elif predicted == 1 and actual == 0:
            fp += 1
        else:
            fn += 1
        if verbose:
            print("T=%.1f, V=%.1f | actual=%s predicted=%s" % (
                temp, vib, LABELS[actual], LABELS[predicted]))
    accuracy = 100 * correct / len(TEST)
    print("K=%d, scaled=%s | accuracy=%.2f%% (%d/%d) | TP=%d TN=%d FP=%d FN=%d" % (
        k, scaled, accuracy, correct, len(TEST), tp, tn, fp, fn))
    return accuracy


def demo():
    print("TRAIN =", len(TRAIN), "TEST =", len(TEST))
    print("Rentang training: T=%.1f..%.1f; V=%.1f..%.1f" % (
        MIN_T, MAX_T, MIN_V, MAX_V))
    for k in (1, 3, 5, 7):
        evaluate(k)
    print("\nPengaruh scaling untuk K=3")
    evaluate(3, scaled=False)
    evaluate(3, scaled=True)


def interactive():
    print("Ketik q untuk keluar. Input sintetik, BUKAN pembacaan sensor.")
    while True:
        raw = input("Temperature [C] atau q: ").strip()
        if raw.lower() == "q":
            return
        try:
            temp = float(raw)
            vib = float(input("Vibration [mm/s]: ").strip())
            if not (20 <= temp <= 100 and 0 <= vib <= 12):
                print("Input di luar rentang simulasi (T:20..100; V:0..12)")
                continue
            k = int(input("K ganjil [1/3/5/7]: ").strip())
            pred, neighbors = predict_knn(temp, vib, k)
            print("Prediction:", LABELS[pred])
            print("Nearest neighbors (jarak kuadrat, indeks, kelas):")
            for d, idx, label in neighbors:
                print("  %.5f | %d | %s" % (d, idx, LABELS[label]))
        except (ValueError, TypeError) as e:
            print("Input tidak valid:", e)


def benchmark(repetitions=100):
    # Menghindari waktu input/print dalam timing; tanpa dataset generator.
    samples = [(39, 3.1), (57, 5.5), (70, 7.5), (47, 6.5)]
    gc.collect()
    start_free = gc.mem_free()
    for temp, vib in samples:  # warm-up
        predict_knn(temp, vib, 3)
    elapsed = []
    for j in range(repetitions):
        temp, vib = samples[j % len(samples)]
        start = time.ticks_us()
        predict_knn(temp, vib, 3)
        elapsed.append(time.ticks_diff(time.ticks_us(), start))
    elapsed.sort()
    print("Inference median:", elapsed[len(elapsed) // 2], "us")
    print("Free memory before/after:", start_free, gc.mem_free(), "bytes")
    print("Note: these are approximate MicroPython-level measurements")


if __name__ == "__main__":
    demo()
    # Untuk eksperimen lain, panggil interactive() atau benchmark()
```

HASIL EXPERIMENT A - KNN MACHINE CONDITION CLASSIFICATION

TRAIN = 36 TEST = 12
Rentang training: T=29.0..78.8; V=1.0..9.0

K=1, scaled=True | accuracy=83.33% (10/12) | TP=5 TN=5 FP=0 FN=2
K=3, scaled=True | accuracy=75.00% (9/12) | TP=4 TN=5 FP=0 FN=3
K=5, scaled=True | accuracy=83.33% (10/12) | TP=5 TN=5 FP=0 FN=2
K=7, scaled=True | accuracy=75.00% (9/12) | TP=4 TN=5 FP=0 FN=3

Pengaruh scaling untuk K=3
K=3, scaled=False | accuracy=83.33% (10/12) | TP=5 TN=5 FP=0 FN=2
K=3, scaled=True | accuracy=75.00% (9/12) | TP=4 TN=5 FP=0 FN=3



### Apa yang dilakukan program?

1. `TRAIN` disimpan sebagai data referensi KNN; `TEST` disimpan terpisah untuk evaluasi.
2. `predict_knn()` menghitung jarak kuadrat ke setiap data training, mengurutkan jarak, dan melakukan voting.
3. `evaluate()` membandingkan prediksi dengan label test sehingga menghasilkan accuracy dan confusion matrix.
4. `demo()` menjalankan eksperimen awal secara otomatis. `interactive()` dan `benchmark()` dipanggil saat dibutuhkan.

## 7. Eksperimen A — KNN berjalan di ESP32 (15 menit)

Jalankan `main.py`. Amati keluaran `TRAIN`, `TEST`, dan accuracy untuk K = 1, 3, 5, 7.

**Target pemeriksaan dengan dataset dan kode yang tidak diubah:**

| K | Jumlah benar | Accuracy |
|---:|---:|---:|
| 1 | 10 dari 12 | 83,33% |
| 3 | 9 dari 12 | 75,00% |
| 5 | 10 dari 12 | 83,33% |
| 7 | 9 dari 12 | 75,00% |

Angka di atas merupakan hasil acuan pada Python untuk dataset ini; output MicroPython seharusnya sesuai selama logikanya sama. **K lebih besar tidak otomatis memberi akurasi lebih tinggi.**


HASIL EXPERIMENT A - KNN MACHINE CONDITION CLASSIFICATION

TRAIN = 36 TEST = 12
Rentang training: T=29.0..78.8; V=1.0..9.0

K=1, scaled=True | accuracy=83.33% (10/12) | TP=5 TN=5 FP=0 FN=2
K=3, scaled=True | accuracy=75.00% (9/12) | TP=4 TN=5 FP=0 FN=3
K=5, scaled=True | accuracy=83.33% (10/12) | TP=5 TN=5 FP=0 FN=2
K=7, scaled=True | accuracy=75.00% (9/12) | TP=4 TN=5 FP=0 FN=3

Pengaruh scaling untuk K=3
K=3, scaled=False | accuracy=83.33% (10/12) | TP=5 TN=5 FP=0 FN=2
K=3, scaled=True | accuracy=75.00% (9/12) | TP=4 TN=5 FP=0 FN=3


## 8. Eksperimen B — Input interaktif melalui Thonny (15 menit)

Di bagian akhir program, ubah:

```python
if __name__ == "__main__":
    demo()
```

menjadi:

```python
if __name__ == "__main__":
    interactive()
```

Jalankan, lalu masukkan temperatur, getaran, dan K melalui Shell Thonny. Coba input berikut:

| Temperatur (°C) | Getaran (mm/s) | K |
|---:|---:|---:|
| 32 | 1.6 | 3 |
| 70 | 7.5 | 3 |
| 57 | 5.5 | 3 |

Perhatikan tiga tetangga terdekat. Tuliskan alasan prediksi berdasarkan **voting**, bukan asumsi ambang temperatur atau getaran.

Hasil_Eksperimen_B.txt.

Ketik q untuk keluar. Input sintetik, BUKAN pembacaan sensor.
Temperature [C] atau q: 32
Vibration [mm/s]: 1.6
K ganjil [1/3/5/7]: 3
Prediction: NORMAL
Nearest neighbors (jarak kuadrat, indeks, kelas):
  0.00925 | 0 | NORMAL
  0.01316 | 35 | NORMAL
  0.02516 | 20 | NORMAL
Temperature [C] atau q: 70
Vibration [mm/s]: 7.5
K ganjil [1/3/5/7]: 3
Prediction: WARNING
Nearest neighbors (jarak kuadrat, indeks, kelas):
  0.02917 | 19 | WARNING
  0.02976 | 18 | WARNING
  0.03224 | 3 | WARNING
Temperature [C] atau q: 57
Vibration [mm/s]: 5.5
K ganjil [1/3/5/7]: 3
Prediction: NORMAL
Nearest neighbors (jarak kuadrat, indeks, kelas):
  0.01941 | 17 | NORMAL
  0.01949 | 2 | NORMAL
  0.02986 | 22 | WARNING
Temperature [C] atau q:

## Analisis Voting KNN

**Percobaan 1 — Prediction: NORMAL**
Tiga tetangga terdekat memiliki kelas NORMAL. Hasil voting adalah 3 NORMAL dan 0 WARNING, sehingga prediksi akhirnya NORMAL.

**Percobaan 2 — Prediction: WARNING**
Tiga tetangga terdekat memiliki kelas WARNING. Hasil voting adalah 0 NORMAL dan 3 WARNING, sehingga prediksi akhirnya WARNING.

**Percobaan 3 — Prediction: NORMAL**
Tiga tetangga terdekat terdiri dari 2 NORMAL dan 1 WARNING. Karena NORMAL memperoleh suara terbanyak, prediksi akhirnya NORMAL.

**Kesimpulan:**
Dengan K = 3, KNN menentukan kelas berdasarkan voting mayoritas dari tiga tetangga terdekat. Prediksi NORMAL atau WARNING ditentukan oleh kelas tetangga yang memperoleh suara terbanyak, bukan berdasarkan asumsi ambang temperatur atau getaran.

## 9. Eksperimen C — Pengaruh K dan normalisasi (20 menit)

Pertama, panggil `demo()` lagi dan amati output `scaled=False` dibandingkan `scaled=True`. Kedua, untuk satu input yang sama, uji K = 1, 3, 5, dan 7 melalui `interactive()`.

**Pertanyaan investigasi:** Mengapa fitur temperatur bisa terlalu dominan saat jarak dihitung tanpa scaling? Apakah seluruh input selalu berubah kelas sesudah scaling?

Untuk memperdalam, catat minimal satu contoh yang prediksinya berubah ketika K atau status scaling diubah. Jika tidak menemukan perubahan pada input awal, coba nilai yang lebih dekat dengan batas kelas seperti temperatur 45–65°C dan getaran 4–7 mm/s.

Hasil_Eksperimen_C.txt

Ketik q untuk keluar. Input sintetik, BUKAN pembacaan sensor.
Temperature [C] atau q: 57
Vibration [mm/s]: 5.5
K ganjil [1/3/5/7]: 1
Prediction: NORMAL
Nearest neighbors (jarak kuadrat, indeks, kelas):
  0.01941 | 17 | NORMAL
Temperature [C] atau q: 57
Vibration [mm/s]: 5.5
K ganjil [1/3/5/7]: 3
Prediction: NORMAL
Nearest neighbors (jarak kuadrat, indeks, kelas):
  0.01941 | 17 | NORMAL
  0.01949 | 2 | NORMAL
  0.02986 | 22 | WARNING
Temperature [C] atau q: 57
Vibration [mm/s]: 5.5
K ganjil [1/3/5/7]: 5
Prediction: WARNING
Nearest neighbors (jarak kuadrat, indeks, kelas):
  0.01941 | 17 | NORMAL
  0.01949 | 2 | NORMAL
  0.02986 | 22 | WARNING
  0.03702 | 18 | WARNING
  0.04545 | 13 | WARNING
Temperature [C] atau q: 57
Vibration [mm/s]: 5.5
K ganjil [1/3/5/7]: 7
Prediction: WARNING
Nearest neighbors (jarak kuadrat, indeks, kelas):
  0.01941 | 17 | NORMAL
  0.01949 | 2 | NORMAL
  0.02986 | 22 | WARNING
  0.03702 | 18 | WARNING
  0.04545 | 13 | WARNING
  0.05074 | 32 | NORMAL
  0.05103 | 33 | WARNING
Temperature [C] atau q: 

# HASIL EKSPERIMEN C — PENGARUH K DAN NORMALISASI

## A. Pengaruh Scaling

Untuk K = 3, diperoleh hasil:
- Scaling = False: Accuracy = 83,33% (10/12)
- Scaling = True: Accuracy = 75,00% (9/12)

**Analisis:**
Pada dataset sintetik ini, penggunaan scaling menurunkan accuracy dari 83,33% menjadi 75,00%. Scaling mengubah kontribusi relatif temperatur dan getaran dalam perhitungan jarak, sehingga tetangga terdekat yang dipilih KNN dapat berubah. Hasil ini menunjukkan bahwa scaling tidak selalu meningkatkan accuracy untuk setiap dataset.

## B. Pengaruh Nilai K

Input yang digunakan sama pada setiap percobaan:
- Temperature = 57 °C
- Vibration = 5.5 mm/s
- Scaling = True

| K | Voting tetangga | Prediction |
|---:|---|---|
| 1 | 1 NORMAL, 0 WARNING | NORMAL |
| 3 | 2 NORMAL, 1 WARNING | NORMAL |
| 5 | 2 NORMAL, 3 WARNING | WARNING |
| 7 | 3 NORMAL, 4 WARNING | WARNING |

**Analisis berdasarkan voting:**

1. **K = 1:** Satu tetangga terdekat berkelas NORMAL. Hasil voting adalah NORMAL, sehingga prediksi NORMAL.
2. **K = 3:** Dua tetangga berkelas NORMAL dan satu WARNING. NORMAL menang dengan voting 2–1, sehingga prediksi NORMAL.
3. **K = 5:** Dua tetangga berkelas NORMAL dan tiga WARNING. WARNING menang dengan voting 3–2, sehingga prediksi WARNING.
4. **K = 7:** Tiga tetangga berkelas NORMAL dan empat WARNING. WARNING menang dengan voting 4–3, sehingga prediksi WARNING.

## Kesimpulan

Eksperimen C menunjukkan bahwa perubahan nilai K dapat mengubah hasil klasifikasi meskipun input temperatur dan getaran tetap sama. Pada input 57 °C dan 5.5 mm/s, prediksi berubah dari NORMAL saat K = 1 dan K = 3 menjadi WARNING saat K = 5 dan K = 7. Perubahan ini terjadi karena jumlah dan komposisi tetangga yang ikut voting berbeda. Scaling juga dapat memengaruhi pemilihan tetangga dan accuracy, tetapi tidak selalu meningkatkan hasil klasifikasi.

## 10. Eksperimen D — Evaluasi pada data yang belum pernah dilihat (15 menit)

Jalankan:

```python
evaluate(k=3, scaled=True, verbose=True)
```

Perhatikan confusion matrix biner yang menggunakan **WARNING sebagai kelas positif**:

| | Prediksi WARNING (1) | Prediksi NORMAL (0) |
|---|---|---|
| Aktual WARNING (1) | TP | FN |
| Aktual NORMAL (0) | FP | TN |

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

Bandingkan hasil accuracy dengan angka acuan pada Eksperimen A. Jelaskan mengapa hasil evaluasi tidak boleh hanya dihitung pada `TRAIN`.

Hasil_Eksperimen_C.txt

ini hasilnya 
T=32.0, V=1.6 | actual=NORMAL predicted=NORMAL
T=39.0, V=3.1 | actual=NORMAL predicted=NORMAL
T=52.0, V=3.0 | actual=NORMAL predicted=NORMAL
T=60.0, V=4.5 | actual=WARNING predicted=WARNING
T=47.0, V=6.5 | actual=WARNING predicted=NORMAL
T=70.0, V=7.5 | actual=WARNING predicted=WARNING
T=78.0, V=8.0 | actual=WARNING predicted=WARNING
T=35.0, V=7.9 | actual=NORMAL predicted=NORMAL
T=73.0, V=2.5 | actual=WARNING predicted=NORMAL
T=57.0, V=5.5 | actual=WARNING predicted=NORMAL
T=46.0, V=4.4 | actual=NORMAL predicted=NORMAL
T=65.0, V=6.2 | actual=WARNING predicted=WARNING
K=3, scaled=True | accuracy=75.00% (9/12) | TP=4 TN=5 FP=0 FN=3
>>>

### Perbandingan Accuracy dengan Eksperimen A

Pada Eksperimen A, model KNN dengan parameter **K=3 dan scaling=True** memperoleh accuracy sebesar **75% (9 dari 12 data test)**. Hasil ini sama dengan angka acuan Eksperimen A, sehingga hasil evaluasi sesuai dengan acuan yang diberikan.

Accuracy sebesar 75% menunjukkan bahwa model berhasil mengklasifikasikan 9 data dengan benar, sedangkan 3 data lainnya salah diklasifikasikan.

### Mengapa Evaluasi Tidak Boleh Hanya Menggunakan Data TRAIN?

Evaluasi tidak boleh hanya menggunakan data `TRAIN` karena data tersebut sudah digunakan oleh model selama proses pembelajaran. Akibatnya, hasil pengujian pada data training dapat terlihat lebih baik daripada kemampuan model saat menghadapi data baru.

Data `TEST` digunakan untuk mengukur kemampuan model dalam mengklasifikasikan data yang belum pernah digunakan saat proses training. Dengan demikian, hasil evaluasi pada data test memberikan gambaran yang lebih objektif mengenai kemampuan model dalam melakukan prediksi.

**Kesimpulan:** Accuracy model KNN dengan K=3 dan scaling=True adalah 75%, sesuai dengan angka acuan Eksperimen A. Penggunaan data test penting untuk menilai kemampuan generalisasi model, bukan hanya kemampuannya mengenali data training.

## 11. Eksperimen E — Waktu inference pada ESP32 (15 menit)

Panggil `benchmark(100)` dari Shell Thonny atau ubah pemanggilan pada bagian `__main__`. Fungsi ini:

- Menjalankan pemanasan ringan sebelum pencatatan waktu.
- Mengukur 100 kali waktu eksekusi `predict_knn()` dengan `time.ticks_us()`.
- Melaporkan **median** waktu inference dan memori bebas (`gc.mem_free()`).

**Penting tentang metrik:** median waktu adalah perkiraan dari pengukuran MicroPython, bukan latency real-time deterministik. Memori bebas sebelum/sesudah benchmark **bukan** ukuran persis total konsumsi RAM model; alokasi sementara dan *garbage collection* dapat memengaruhinya. Untuk membandingkan ukuran dataset, gunakan prosedur yang sama dan simpan jumlah titik, latency median, serta memori bebas secara konsisten.
hasil Run

MPY: soft reboot
Inference median: 5120 us
Free memory before/after: 219120 206320 bytes
Note: these are approximate MicroPython-level measurements
>>> 

### Tantangan lanjutan (opsional)

Salin beberapa baris `TRAIN` untuk membuat dataset referensi lebih besar, misalnya 72 atau 144 sampel. Ulangi benchmark dengan input dan jumlah repetisi yang sama. Ingat, duplikasi data dapat memengaruhi voting dan **bukan** penambahan informasi baru sehingga jangan menafsirkan peningkatan accuracy sebagai akibat data lebih bermutu.
hasil Run

MPY: soft reboot

=== DATASET 36 SAMPEL ===
Inference median: 5113 us
Free memory before/after: 218880 202368 bytes
Note: these are approximate MicroPython-level measurements

=== DATASET 72 SAMPEL ===
Inference median: 12004 us
Free memory before/after: 218688 180096 bytes
Note: these are approximate MicroPython-level measurements

=== DATASET 144 SAMPEL ===
Inference median: 29448 us
Free memory before/after: 218400 185152 bytes
Note: these are approximate MicroPython-level measurements
>>> 

### Eksperimen E — Pengaruh Ukuran Dataset terhadap Performa KNN pada ESP32

#### Tujuan
Membandingkan waktu inference KNN pada ESP32 ketika jumlah sampel referensi `TRAIN` diperbesar menjadi 36, 72, dan 144 sampel dengan jumlah pengulangan benchmark yang sama.

#### Prosedur
1. Menjalankan benchmark pada dataset awal sebanyak 36 sampel.
2. Menggandakan sampel `TRAIN` menjadi 72 sampel dan mengulangi benchmark.
3. Menggandakan kembali sampel menjadi 144 sampel dan mengulangi benchmark.
4. Menggunakan 100 pengulangan benchmark pada setiap ukuran dataset.
5. Mencatat median waktu inference serta memori bebas sebelum dan sesudah benchmark.

#### Hasil Pengujian

| Jumlah sampel | Median inference | Memori bebas sebelum | Memori bebas sesudah |
|---:|---:|---:|---:|
| 36 | 5.113 µs | 218.880 bytes | 202.368 bytes |
| 72 | 12.004 µs | 218.688 bytes | 180.096 bytes |
| 144 | 29.448 µs | 218.400 bytes | 185.152 bytes |

#### Analisis
Hasil pengujian menunjukkan bahwa median waktu inference meningkat seiring dengan bertambahnya jumlah sampel referensi. Pada 36 sampel, median inference adalah 5.113 µs. Ketika jumlah sampel menjadi 72, median meningkat menjadi 12.004 µs. Pada 144 sampel, median mencapai 29.448 µs.

Dibandingkan dengan dataset awal, waktu inference pada 72 sampel sekitar 2,35 kali lebih lama, sedangkan pada 144 sampel sekitar 5,76 kali lebih lama. Hasil ini menunjukkan bahwa penambahan jumlah referensi meningkatkan beban komputasi prediksi KNN pada pengujian ini.

Memori bebas sebelum benchmark tercatat sebesar 218.880 bytes, 218.688 bytes, dan 218.400 bytes untuk masing-masing ukuran dataset. Setelah benchmark, nilainya menjadi 202.368 bytes, 180.096 bytes, dan 185.152 bytes. Nilai memori bebas sesudah benchmark tidak menurun secara konsisten, sehingga perubahan tersebut tidak dapat dianggap sebagai ukuran langsung memori yang digunakan dataset. Pengelolaan memori MicroPython dan garbage collection dapat memengaruhi hasil pengukuran.

Sampel tambahan dibuat dengan menduplikasi data yang sudah ada. Oleh karena itu, jumlah sampel yang lebih besar tidak berarti dataset memiliki informasi baru atau kualitas data yang lebih baik. Duplikasi juga dapat memengaruhi voting tetangga terdekat dan hasil klasifikasi KNN.

#### Kesimpulan
Pada pengujian ESP32 dengan 100 pengulangan, median inference meningkat dari 5.113 µs untuk 36 sampel menjadi 12.004 µs untuk 72 sampel dan 29.448 µs untuk 144 sampel. Hal ini menunjukkan bahwa ukuran dataset referensi memengaruhi waktu komputasi KNN. Namun, karena dataset diperbesar melalui duplikasi, hasil ini tidak membuktikan adanya peningkatan kualitas informasi maupun akurasi model.

## 12. Lembar hasil mahasiswa

| Variasi | K | Scaling | Accuracy (%) | Median inference (µs) | Catatan |
|---|---:|---|---:|---:|---|
| A | 1 | Ya | | | |
| B | 3 | Ya | | | |
| C | 5 | Ya | | | |
| D | 7 | Ya | | | |
| E | 3 | Tidak | | | |

## 13. Pertanyaan analisis

1. Mengapa KNN tidak memerlukan proses pembaruan koefisien seperti regresi linear, tetapi tetap membutuhkan data training?
2. Mengapa normalisasi dapat mengubah prediksi KNN? Jelaskan menggunakan skala temperatur dan getaran.
3. Apakah K yang lebih besar selalu menghasilkan accuracy lebih tinggi? Gunakan hasil percobaan.
4. Apa perbedaan waktu inference dan waktu input melalui terminal serial? Mengapa *timing* tidak dilakukan saat `input()` atau `print()`?
5. Jika data training diperbesar 10 kali, apa yang kemungkinan terjadi pada penggunaan memori dan waktu inference di ESP32? Apa yang perlu diukur untuk membuktikannya?
jawaban

## Pertanyaan Analisis

### 1. Mengapa KNN tidak memerlukan proses pembaruan koefisien seperti regresi linear, tetapi tetap membutuhkan data training?

KNN termasuk algoritma *instance-based learning* atau *lazy learning*. KNN tidak membentuk persamaan model dengan koefisien yang harus diperbarui seperti regresi linear. Sebaliknya, KNN menyimpan data training sebagai referensi untuk proses prediksi. Ketika data baru masuk, KNN menghitung jarak antara data tersebut dan data training, memilih K tetangga terdekat, lalu menentukan kelas berdasarkan voting mayoritas.

Dengan demikian, KNN tetap membutuhkan data training karena data tersebut menjadi dasar untuk menentukan hasil klasifikasi.

### 2. Mengapa normalisasi dapat mengubah prediksi KNN? Jelaskan menggunakan skala temperatur dan getaran.

Normalisasi dapat mengubah prediksi KNN karena algoritma ini menentukan tetangga berdasarkan jarak antardata. Dalam dataset praktikum, temperatur memiliki rentang sekitar 29–78,8, sedangkan getaran memiliki rentang sekitar 1–9. Tanpa normalisasi, selisih temperatur dapat lebih mendominasi perhitungan jarak dibandingkan selisih getaran.

Dengan normalisasi Min-Max, kedua fitur diubah ke rentang yang sebanding, biasanya 0 sampai 1. Akibatnya, kontribusi relatif temperatur dan getaran terhadap jarak berubah. Urutan tetangga terdekat dapat berbeda sehingga hasil voting dan prediksi kelas juga dapat berubah.

Hasil eksperimen menunjukkan bahwa untuk K=3, accuracy tanpa scaling adalah 83,33%, sedangkan dengan scaling adalah 75%. Hal ini menunjukkan bahwa normalisasi tidak selalu meningkatkan accuracy pada setiap dataset.

### 3. Apakah K yang lebih besar selalu menghasilkan accuracy lebih tinggi? Gunakan hasil percobaan.

Tidak. Nilai K yang lebih besar tidak selalu menghasilkan accuracy lebih tinggi karena prediksi KNN bergantung pada komposisi kelas dari sejumlah tetangga terdekat. Nilai K yang kecil lebih sensitif terhadap data di sekitar sampel, sedangkan nilai K yang lebih besar dapat melibatkan lebih banyak tetangga dari kelas lain.

Pada eksperimen A dengan scaling, hasil accuracy adalah:
- K=1: 83,33% (10 dari 12 data benar).
- K=3: 75,00% (9 dari 12 data benar).
- K=5: 83,33% (10 dari 12 data benar).
- K=7: 75,00% (9 dari 12 data benar).

Hasil tersebut membuktikan bahwa peningkatan K tidak selalu meningkatkan accuracy. Pemilihan K sebaiknya didasarkan pada evaluasi menggunakan data test, bukan hanya berdasarkan besar kecilnya nilai K.

### 4. Apa perbedaan waktu inference dan waktu input melalui terminal serial? Mengapa timing tidak dilakukan saat `input()` atau `print()`?

Waktu inference adalah waktu yang diperlukan algoritma untuk memproses data masukan dan menghasilkan prediksi. Dalam praktikum ini, pengukuran dilakukan pada proses prediksi KNN.

Sementara itu, waktu input melalui terminal serial mencakup waktu pengguna mengetik, mengirim data, dan menunggu respons. Waktu `print()` juga mencakup proses mengirim serta menampilkan teks di terminal.

Jika `input()` dan `print()` dimasukkan ke dalam pengukuran, hasil timing akan dipengaruhi oleh kecepatan pengguna, komunikasi serial, dan tampilan terminal. Oleh karena itu, timing dilakukan hanya pada bagian inference agar perkiraan waktu komputasi algoritma dapat diukur dengan lebih terfokus.

### 5. Jika data training diperbesar 10 kali, apa yang kemungkinan terjadi pada penggunaan memori dan waktu inference di ESP32? Apa yang perlu diukur untuk membuktikannya?

Jika jumlah data training diperbesar 10 kali, penggunaan memori kemungkinan meningkat karena KNN perlu menyimpan lebih banyak sampel referensi. Waktu inference juga kemungkinan meningkat karena algoritma harus menghitung jarak dari data masukan ke lebih banyak sampel dan menentukan tetangga terdekat.

Hasil benchmark ESP32 mendukung perkiraan tersebut. Pada 36 sampel, median inference adalah 5.113 µs; pada 72 sampel menjadi 12.004 µs; dan pada 144 sampel menjadi 29.448 µs. Artinya, waktu inference meningkat ketika jumlah sampel referensi diperbesar pada pengujian ini.

Untuk membuktikannya pada dataset yang diperbesar 10 kali, perlu dilakukan pengukuran dengan prosedur yang sama, antara lain:
1. Jumlah sampel training.
2. Median waktu inference dengan jumlah pengulangan yang sama.
3. Memori bebas sebelum dan sesudah benchmark.
4. Konsistensi kondisi perangkat dan prosedur pengujian.
5. Kemampuan program berjalan tanpa kehabisan memori.

Penambahan sampel melalui duplikasi tidak menambah informasi baru. Karena itu, pengujian ini terutama menunjukkan pengaruh ukuran dataset terhadap kebutuhan komputasi dan memori, bukan membuktikan peningkatan kualitas model atau accuracy.

## 14. Kesimpulan dan tugas

**Kesimpulan:** KNN dapat ditulis dengan MicroPython dan dijalankan seluruhnya pada ESP32 tanpa sensor atau library machine learning. Parameter preprocessing dan pemisahan data training/test tetap penting meskipun perangkat sangat sederhana.

**Pengumpulan:** serahkan `main.py`, foto/screenshot output Shell, tabel eksperimen, dan jawaban pertanyaan analisis. Jangan menyerahkan accuracy training sebagai pengganti accuracy test.

---

**Catatan instruktur:** seluruh hasil acuan diuji pada Python standar dengan algoritma identik; performa waktu/memori perlu diukur pada board mahasiswa. Tidak ada hardware sensor maupun koneksi cloud yang terlibat.
