# Verilerin kaynakları ve lisansları

## Udacity CarND-Advanced-Lane-Lines (MIT)

`kolay_video` ve `zor_video` kurulumlarının bütün toplanan verileri bu depodan alınmıştır:
<https://github.com/udacity/CarND-Advanced-Lane-Lines>. İki video ve satranç tahtası fotoğrafları aynı kameradan,
1280x720.

| Bu projede | Kaynak dosya |
|---|---|
| `ortam_verisi/videolar/kolay_video/kolay_video_iyi_isik.mp4` | `project_video.mp4` |
| `ortam_verisi/videolar/zor_video/zor_video_golge_zemin.mp4` | `challenge_video.mp4` |
| `hazir_veri/kalibrasyon_fotograflari/<kurulum>/satranc_01–15.jpg` | `camera_cal/calibration*.jpg` (20 fotoğraftan köşeleri bulunan ve 1280x720 olan 15'i; tahta 9x6 iç köşe) |
| `ortam_verisi/test_kareleri/kolay_video/duz_yol_1.jpg`, `duz_yol_2.jpg` | `test_images/straight_lines1.jpg`, `straight_lines2.jpg` |
| `ortam_verisi/test_kareleri/kolay_video/duz_yol_3.jpg` | `project_video.mp4`, 18.5. saniye |
| `ortam_verisi/test_kareleri/kolay_video/` deneme kareleri (6) | `test_images/test1–6.jpg` |
| `ortam_verisi/test_kareleri/zor_video/` (9 kare) | `challenge_video.mp4` kareleri (adlarında saniye ya da içerik) |

Yol ABD otoyolu: şerit genişliği 3.66 m, kesik çizgi boyu 3.05 m.

Lisans metni (kaynak deponun `LICENSE` dosyası):

```
MIT License

Copyright (c) 2016-2018 Udacity, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## YOLO11 modelleri (Ultralytics, AGPL-3.0)

`hazir_veri/yolo_modeli/yolo11n.pt` Ultralytics'in hazır modelidir
(<https://github.com/ultralytics/assets/releases/tag/v8.4.0>). Model ve `ultralytics` kütüphanesi AGPL-3.0
lisanslıdır: <https://github.com/ultralytics/ultralytics/blob/main/LICENSE>. Diğer boyutlar (s, m, l, x) depoda
yoktur; nasıl indirileceği kök klasördeki `README.md` dosyasında.
