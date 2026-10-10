# Veri

Yarışma verisi bu depoda **yer almaz**. Veri, TEKNOFEST Sağlıkta Yapay Zekâ
yarışması kapsamında takımlara sağlanan anonimleştirilmiş bir settir ve yeniden
dağıtım hakkı bize ait değildir.

Kodu çalıştırmak için dört CSV dosyasını bu klasöre koyun:

| Dosya | Panel | Satır |
|---|---|---:|
| `YARISMA_TRAIN_MASTER.csv` | MASTER | 2.931 |
| `YARISMA_TRAIN_KANSER.csv` | KANSER | 388 |
| `YARISMA_TRAIN_PAH.csv` | PAH | 372 |
| `YARISMA_TRAIN_CFTR.csv` | CFTR | 111 |

Veri başka bir klasördeyse yolu ortam değişkeniyle verebilirsiniz:

```bash
export AEGIS_DATA_DIR=/veri/klasoru      # Windows PowerShell: $env:AEGIS_DATA_DIR = "C:\veri"
```

## Beklenen yapı

Her dosyada 353 sütun vardır:

| Sütun | Açıklama |
|---|---|
| `Variant_ID` | Varyant kimliği. Aynı varyant birden fazla panelde bulunabilir. |
| `Label` | Hedef: 1 = patojenik, 0 = benign |
| `AL_1` … `AL_334` | Popülasyon / alel istatistikleri (sayısal) |
| `EK_1` … `EK_9` | In-silico tahmin skorları (sayısal) |
| `CAT_1` … `CAT_6` | Kategorik genomik bağlam |
| `AA_1`, `AA_2` | Referans ve değişen amino asit |

Sütun adları yarışma tarafından anonimleştirilmiştir; aile adları (AL, EK, CAT,
AA) dışında hangi sütunun hangi kaynağa karşılık geldiği bilinmemektedir.
