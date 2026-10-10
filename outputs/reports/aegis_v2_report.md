# AEGIS v2 - sonuç özeti

Bu dosya `python -m aegis report` ile üretilir; elle düzenlemeyin.

## Yapılandırma

| Alan | Değer |
|---|---|
| Model | Panel başına ayarlanmış XGBoost |
| Özellikler | Mühendislik özellikleri + ACMG benzeri kanıt skorları |
| Doğrulama | StratifiedGroupKFold (5 kat), grup = Variant_ID |
| Variant_ID sızıntısı | 0 |

## Dört panelin ortalaması

`all_positive_f1`, her varyanta patojenik diyen modelin F1 değeridir; karşılaştırma tabanıdır.

| strategy                                 |     f1 |   precision |   recall |   specificity |   balanced_accuracy |    mcc |   pr_auc |   roc_auc |   all_positive_f1 |
|:-----------------------------------------|-------:|------------:|---------:|--------------:|--------------------:|-------:|---------:|----------:|------------------:|
| Sabit eşik (0,5)                         | 0.9163 |      0.8734 |   0.9655 |        0.4873 |              0.7264 | 0.5426 |   0.9426 |    0.8528 |            0.8670 |
| İç içe CV eşiği                          | 0.9082 |      0.8532 |   0.9723 |        0.3904 |              0.6814 | 0.4614 |   0.9426 |    0.8528 |            0.8670 |
| Doğrulama katında seçilen eşik (iyimser) | 0.9250 |      0.8802 |   0.9759 |        0.5214 |              0.7487 | 0.5935 |   0.9426 |    0.8528 |            0.8670 |

## Panel bazında

| panel   | strategy      |   threshold |     f1 |   precision |   recall |   specificity |    mcc |   pr_auc |   roc_auc |   all_positive_f1 |
|:--------|:--------------|------------:|-------:|------------:|---------:|--------------:|-------:|---------:|----------:|------------------:|
| MASTER  | fixed_0.5     |      0.5000 | 0.8907 |      0.8383 |   0.9502 |        0.4961 | 0.5271 |   0.9239 |    0.8463 |            0.8461 |
| MASTER  | nested        |      0.5120 | 0.8906 |      0.8406 |   0.9470 |        0.5063 | 0.5291 |   0.9239 |    0.8463 |            0.8461 |
| MASTER  | per_fold_best |      0.5060 | 0.8940 |      0.8417 |   0.9535 |        0.5064 | 0.5437 |   0.9239 |    0.8463 |            0.8461 |
| KANSER  | fixed_0.5     |      0.5000 | 0.9198 |      0.9006 |   0.9403 |        0.7667 | 0.7293 |   0.9322 |    0.8979 |            0.8171 |
| KANSER  | nested        |      0.3440 | 0.9088 |      0.8691 |   0.9551 |        0.6667 | 0.6769 |   0.9322 |    0.8979 |            0.8171 |
| KANSER  | per_fold_best |      0.3220 | 0.9300 |      0.8970 |   0.9664 |        0.7500 | 0.7621 |   0.9322 |    0.8979 |            0.8171 |
| PAH     | fixed_0.5     |      0.5000 | 0.9278 |      0.8705 |   0.9935 |        0.2564 | 0.4183 |   0.9371 |    0.7713 |            0.9091 |
| PAH     | nested        |      0.4260 | 0.9149 |      0.8530 |   0.9871 |        0.1487 | 0.2393 |   0.9371 |    0.7713 |            0.9091 |
| PAH     | per_fold_best |      0.5360 | 0.9328 |      0.8881 |   0.9839 |        0.3692 | 0.4987 |   0.9371 |    0.7713 |            0.9091 |
| CFTR    | fixed_0.5     |      0.5000 | 0.9267 |      0.8842 |   0.9778 |        0.4300 | 0.4959 |   0.9773 |    0.8956 |            0.8956 |
| CFTR    | nested        |      0.3780 | 0.9187 |      0.8501 |   1.0000 |        0.2400 | 0.4002 |   0.9773 |    0.8956 |            0.8956 |
| CFTR    | per_fold_best |      0.3520 | 0.9433 |      0.8940 |   1.0000 |        0.4600 | 0.5694 |   0.9773 |    0.8956 |            0.8956 |

## Karmaşıklık matrisi özeti (5 katın toplamı)

| panel   | strategy      |   TP |   TN |   FP |   FN |   sensitivity |   specificity |   precision |    npv |
|:--------|:--------------|-----:|-----:|-----:|-----:|--------------:|--------------:|------------:|-------:|
| MASTER  | fixed_0.5     | 2042 |  388 |  394 |  107 |        0.9502 |        0.4962 |      0.8383 | 0.7838 |
| MASTER  | nested        | 2035 |  396 |  386 |  114 |        0.9470 |        0.5064 |      0.8406 | 0.7765 |
| MASTER  | per_fold_best | 2049 |  396 |  386 |  100 |        0.9535 |        0.5064 |      0.8415 | 0.7984 |
| KANSER  | fixed_0.5     |  252 |   92 |   28 |   16 |        0.9403 |        0.7667 |      0.9000 | 0.8519 |
| KANSER  | nested        |  256 |   80 |   40 |   12 |        0.9552 |        0.6667 |      0.8649 | 0.8696 |
| KANSER  | per_fold_best |  259 |   90 |   30 |    9 |        0.9664 |        0.7500 |      0.8962 | 0.9091 |
| PAH     | fixed_0.5     |  308 |   16 |   46 |    2 |        0.9935 |        0.2581 |      0.8701 | 0.8889 |
| PAH     | nested        |  306 |    9 |   53 |    4 |        0.9871 |        0.1452 |      0.8524 | 0.6923 |
| PAH     | per_fold_best |  305 |   23 |   39 |    5 |        0.9839 |        0.3710 |      0.8866 | 0.8214 |
| CFTR    | fixed_0.5     |   88 |    9 |   12 |    2 |        0.9778 |        0.4286 |      0.8800 | 0.8182 |
| CFTR    | nested        |   90 |    5 |   16 |    0 |        1.0000 |        0.2381 |      0.8491 | 1.0000 |
| CFTR    | per_fold_best |   90 |   10 |   11 |    0 |        1.0000 |        0.4762 |      0.8911 | 1.0000 |

## Özellik ailelerinin önemi

| panel   | family                    |   total_importance |   mean_importance |   importance_share |
|:--------|:--------------------------|-------------------:|------------------:|-------------------:|
| CFTR    | AA / Protein Impact       |             0.0497 |            0.0020 |             0.0099 |
| CFTR    | ACMG-like Evidence        |             0.7789 |            0.0223 |             0.1558 |
| CFTR    | AEGIS Combined Evidence   |             0.0464 |            0.0010 |             0.0093 |
| CFTR    | AL / Population Frequency |             3.5528 |            0.0021 |             0.7106 |
| CFTR    | CAT / Genomic Context     |             0.0571 |            0.0003 |             0.0114 |
| CFTR    | EK / In-silico Evidence   |             0.5126 |            0.0068 |             0.1025 |
| CFTR    | Missingness Evidence      |             0.0024 |            0.0001 |             0.0005 |
| KANSER  | AA / Protein Impact       |             0.0499 |            0.0011 |             0.0100 |
| KANSER  | ACMG-like Evidence        |             0.2608 |            0.0075 |             0.0522 |
| KANSER  | AEGIS Combined Evidence   |             0.4838 |            0.0138 |             0.0968 |
| KANSER  | AL / Population Frequency |             3.8484 |            0.0023 |             0.7697 |
| KANSER  | CAT / Genomic Context     |             0.0313 |            0.0002 |             0.0063 |
| KANSER  | EK / In-silico Evidence   |             0.2279 |            0.0030 |             0.0456 |
| KANSER  | Missingness Evidence      |             0.0979 |            0.0039 |             0.0196 |
| MASTER  | AA / Protein Impact       |             0.0821 |            0.0018 |             0.0164 |
| MASTER  | ACMG-like Evidence        |             0.1886 |            0.0047 |             0.0377 |
| MASTER  | AEGIS Combined Evidence   |             0.0928 |            0.0186 |             0.0186 |
| MASTER  | AL / Population Frequency |             3.9146 |            0.0023 |             0.7829 |
| MASTER  | CAT / Genomic Context     |             0.3065 |            0.0015 |             0.0613 |
| MASTER  | EK / In-silico Evidence   |             0.3126 |            0.0042 |             0.0625 |
| MASTER  | Missingness Evidence      |             0.1027 |            0.0041 |             0.0205 |
| PAH     | AA / Protein Impact       |             0.0833 |            0.0021 |             0.0167 |
| PAH     | ACMG-like Evidence        |             0.2088 |            0.0060 |             0.0418 |
| PAH     | AEGIS Combined Evidence   |             0.2636 |            0.0048 |             0.0527 |
| PAH     | AL / Population Frequency |             3.8585 |            0.0023 |             0.7717 |
| PAH     | CAT / Genomic Context     |             0.0055 |            0.0000 |             0.0011 |
| PAH     | EK / In-silico Evidence   |             0.5134 |            0.0068 |             0.1027 |
| PAH     | Missingness Evidence      |             0.0669 |            0.0027 |             0.0134 |

## Panel başına en önemli 20 özellik

| panel   | feature                               | family                    |   importance |
|:--------|:--------------------------------------|:--------------------------|-------------:|
| CFTR    | ACMG_PP3_insilico_max                 | ACMG-like Evidence        |       0.0830 |
| CFTR    | log_AL_298                            | AL / Population Frequency |       0.0536 |
| CFTR    | log_AL_7                              | AL / Population Frequency |       0.0457 |
| CFTR    | log_AL_302                            | AL / Population Frequency |       0.0376 |
| CFTR    | EK_2                                  | EK / In-silico Evidence   |       0.0376 |
| CFTR    | log_AL_58                             | AL / Population Frequency |       0.0306 |
| CFTR    | log_AL_296                            | AL / Population Frequency |       0.0296 |
| CFTR    | ACMG_COMBINED_PM2_PP3                 | ACMG-like Evidence        |       0.0293 |
| CFTR    | log_AL_8                              | AL / Population Frequency |       0.0289 |
| CFTR    | ACMG_PP3_insilico_mean                | ACMG-like Evidence        |       0.0257 |
| CFTR    | log_AL_327                            | AL / Population Frequency |       0.0244 |
| CFTR    | log_AL_311                            | AL / Population Frequency |       0.0234 |
| CFTR    | log_AL_14                             | AL / Population Frequency |       0.0222 |
| CFTR    | log_AL_11                             | AL / Population Frequency |       0.0168 |
| CFTR    | log_AL_73                             | AL / Population Frequency |       0.0161 |
| CFTR    | log_AL_319                            | AL / Population Frequency |       0.0159 |
| CFTR    | log_AL_118                            | AL / Population Frequency |       0.0146 |
| CFTR    | EK_1                                  | EK / In-silico Evidence   |       0.0137 |
| CFTR    | log_AL_315                            | AL / Population Frequency |       0.0129 |
| CFTR    | EK_6                                  | EK / In-silico Evidence   |       0.0127 |
| KANSER  | log_AL_16                             | AL / Population Frequency |       0.0786 |
| KANSER  | AEGIS_KANSER_evidence_strength        | AEGIS Combined Evidence   |       0.0386 |
| KANSER  | log_AL_19                             | AL / Population Frequency |       0.0311 |
| KANSER  | log_AL_215                            | AL / Population Frequency |       0.0295 |
| KANSER  | AEGIS_KANSER_conservation_mean        | AEGIS Combined Evidence   |       0.0274 |
| KANSER  | log_AL_23                             | AL / Population Frequency |       0.0267 |
| KANSER  | log_AL_18                             | AL / Population Frequency |       0.0251 |
| KANSER  | ACMG_PP3_insilico_mean                | ACMG-like Evidence        |       0.0218 |
| KANSER  | log_AL_13                             | AL / Population Frequency |       0.0169 |
| KANSER  | log_AL_184                            | AL / Population Frequency |       0.0156 |
| KANSER  | AEGIS_KANSER_conservation_max         | AEGIS Combined Evidence   |       0.0149 |
| KANSER  | log_AL_22                             | AL / Population Frequency |       0.0141 |
| KANSER  | ACMG_PM2_population_rare_mean         | ACMG-like Evidence        |       0.0136 |
| KANSER  | log_AL_315                            | AL / Population Frequency |       0.0130 |
| KANSER  | MISS_AL_ratio                         | Missingness Evidence      |       0.0113 |
| KANSER  | log_AL_175                            | AL / Population Frequency |       0.0108 |
| KANSER  | log_AL_224                            | AL / Population Frequency |       0.0100 |
| KANSER  | AEGIS_KANSER_missing_evidence_density | AEGIS Combined Evidence   |       0.0100 |
| KANSER  | ACMG_PP3_insilico_max                 | ACMG-like Evidence        |       0.0094 |
| KANSER  | MISS_total_ratio                      | Missingness Evidence      |       0.0082 |
| MASTER  | log_AL_1                              | AL / Population Frequency |       0.0254 |
| MASTER  | log_AL_16                             | AL / Population Frequency |       0.0215 |
| MASTER  | CAT_1_missing                         | CAT / Genomic Context     |       0.0202 |
| MASTER  | EK_7                                  | EK / In-silico Evidence   |       0.0200 |
| MASTER  | AEGIS_MASTER_evidence_strength        | AEGIS Combined Evidence   |       0.0186 |
| MASTER  | ACMG_PP3_insilico_max                 | ACMG-like Evidence        |       0.0152 |
| MASTER  | log_AL_6                              | AL / Population Frequency |       0.0121 |
| MASTER  | log_AL_215                            | AL / Population Frequency |       0.0115 |
| MASTER  | log_AL_3                              | AL / Population Frequency |       0.0113 |
| MASTER  | log_AL_287                            | AL / Population Frequency |       0.0089 |
| MASTER  | log_AL_327                            | AL / Population Frequency |       0.0084 |
| MASTER  | log_AL_4                              | AL / Population Frequency |       0.0075 |
| MASTER  | log_AL_26                             | AL / Population Frequency |       0.0070 |
| MASTER  | MISS_total_ratio                      | Missingness Evidence      |       0.0066 |
| MASTER  | MISS_AL_ratio                         | Missingness Evidence      |       0.0066 |
| MASTER  | log_AL_7                              | AL / Population Frequency |       0.0065 |
| MASTER  | AL_328                                | AL / Population Frequency |       0.0061 |
| MASTER  | log_AL_9                              | AL / Population Frequency |       0.0059 |
| MASTER  | log_AL_11                             | AL / Population Frequency |       0.0057 |
| MASTER  | ACMG_PP3_insilico_mean                | ACMG-like Evidence        |       0.0054 |
| PAH     | EK_7                                  | EK / In-silico Evidence   |       0.0198 |
| PAH     | log_AL_323                            | AL / Population Frequency |       0.0193 |
| PAH     | log_AL_12                             | AL / Population Frequency |       0.0184 |
| PAH     | log_AL_49                             | AL / Population Frequency |       0.0165 |
| PAH     | log_AL_7                              | AL / Population Frequency |       0.0162 |
| PAH     | log_AL_26                             | AL / Population Frequency |       0.0157 |
| PAH     | log_AL_300                            | AL / Population Frequency |       0.0142 |
| PAH     | log_AL_13                             | AL / Population Frequency |       0.0140 |
| PAH     | log_AL_15                             | AL / Population Frequency |       0.0134 |
| PAH     | log_AL_331                            | AL / Population Frequency |       0.0133 |
| PAH     | ACMG_PP3_insilico_mean                | ACMG-like Evidence        |       0.0126 |
| PAH     | log_AL_319                            | AL / Population Frequency |       0.0124 |
| PAH     | log_AL_327                            | AL / Population Frequency |       0.0120 |
| PAH     | AEGIS_PAH_evidence_strength           | AEGIS Combined Evidence   |       0.0119 |
| PAH     | log_AL_306                            | AL / Population Frequency |       0.0118 |
| PAH     | AL_328                                | AL / Population Frequency |       0.0117 |
| PAH     | log_AL_311                            | AL / Population Frequency |       0.0116 |
| PAH     | log_AL_145                            | AL / Population Frequency |       0.0113 |
| PAH     | ACMG_PP3_insilico_max                 | ACMG-like Evidence        |       0.0113 |
| PAH     | log_AL_8                              | AL / Population Frequency |       0.0109 |
