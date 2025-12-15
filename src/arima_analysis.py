# src/arima_analysis.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

# ARIMA başlangıç parametrelerine dair uyarıları bastır
# (model yine çalışır, sadece konsolu kirletmemesi için)
warnings.filterwarnings(
    "ignore",
    message="Non-stationary starting autoregressive parameters"
)

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Veri temizleme, interpolasyon ve outlier kontrolü yapan yardımcı fonksiyon
from data_preprocessing import load_prepare_and_check


# ======================================================
# Ayarlar
# ======================================================

# Veri dosyası yolu
CSV_PATH = "data/renew_energy_consumption.csv"

# Analiz edilecek ülke (ISO country code)
COUNTRY_CODE = "DEU"

# Analize dahil edilecek ilk yıl
START_YEAR = 1990

# ARIMA(p, d, q) parametreleri
# p: AR derecesi
# d: fark alma derecesi
# q: MA derecesi
P, D, Q = 1, 1, 1

# Tahmin aralığı
FUTURE_START = 2022
FUTURE_END = 2040

# Backtest ayarları
DO_BACKTEST = True
TEST_SIZE = 0.3  # son %30 yıl test seti (zaman serisi olduğu için shuffle yok)


# ======================================================
# Veri hazırlama
# ======================================================
# 2021 sonrası için interpolasyon yapılmaz
df_clean, summary, flagged = load_prepare_and_check(
    csv_path=CSV_PATH,
    country_code=COUNTRY_CODE,
    start_year=START_YEAR,
    do_interpolate=True,
    interpolate_upto_year=2020,  # sadece ölçüm yılları doldurulur
)

# Year verisini kesin olarak integer'a çevir
df_clean["Year"] = pd.to_numeric(
    df_clean["Year"], errors="coerce"
).astype(int)

# Value boş olanları temizle ve tekrar sırala
df_clean = (
    df_clean
    .dropna(subset=["Value"])
    .sort_values("Year")
    .reset_index(drop=True)
)

# Zaman serisini oluştur
# PeriodIndex kullanımı ARIMA için daha uygundur
tmp = df_clean[["Year", "Value"]].copy()
tmp.index = pd.PeriodIndex(tmp["Year"], freq="Y")
series = tmp["Value"]

# Kullanılan gözlem aralığını yazdır
last_obs_year = int(df_clean["Year"].max())
first_obs_year = int(df_clean["Year"].min())
print(
    f"\nARIMA analizi için kullanılan yıl aralığı: "
    f"{first_obs_year}–{last_obs_year}"
)


# ======================================================
# ARIMA modelinin kurulması ve eğitilmesi
# ======================================================
model = ARIMA(series, order=(P, D, Q))
model_fit = model.fit()


# ======================================================
# ARIMA model özeti
# ======================================================
params = model_fit.params
pvalues = model_fit.pvalues

print("\n--- ARIMA MODEL ÖZETİ ---")
print(f"Model: ARIMA({P},{D},{Q})")
print(f"Gözlem sayısı: {model_fit.nobs}")

# Model hata varyansı (σ²)
print(f"Hata varyansı (sigma²): {params['sigma2']:.3f}")


# ======================================================
# Backtest (out-of-sample)
# Son %30 yıl test kümesi olarak ayrılır
# ======================================================
if DO_BACKTEST:
    n_total = len(df_clean)
    n_test = int(np.ceil(n_total * TEST_SIZE))

    # Veri yetersizliği kontrolü
    if n_total < 5 or n_test < 1 or (n_total - n_test) < 3:
        print(
            "\n[Uyarı] Backtest için yeterli veri yok. "
            "TEST_SIZE değerini düşür veya daha fazla yıl ekle."
        )
    else:
        # Train / test ayrımı (zaman sırası korunur)
        df_train = df_clean.iloc[:-n_test].copy()
        df_test = df_clean.iloc[-n_test:].copy()

        BACKTEST_START = int(df_test["Year"].iloc[0])
        BACKTEST_END = int(df_test["Year"].iloc[-1])

        print("\n--- Split Bilgisi (Backtest) ---")
        print(
            "Eğitim seti yıl aralığı:",
            int(df_train["Year"].min()),
            "-",
            int(df_train["Year"].max())
        )
        print(
            "Test seti yıl aralığı:",
            int(df_test["Year"].min()),
            "-",
            int(df_test["Year"].max())
        )
        print(
            f"Test oranı (yaklaşık): "
            f"{len(df_test)}/{n_total} = {len(df_test)/n_total:.2f}"
        )

        # Eğitim seti zaman serisi
        tmp_tr = df_train[["Year", "Value"]].copy()
        tmp_tr.index = pd.PeriodIndex(tmp_tr["Year"], freq="Y")
        series_tr = tmp_tr["Value"]

        # Gerçek test değerleri
        y_true = df_test["Value"].to_numpy()

        # Eğitim verisiyle ARIMA'yı yeniden eğit
        arima_bt = ARIMA(series_tr, order=(P, D, Q)).fit()

        # Test uzunluğu kadar ileri tahmin
        steps_bt = len(df_test)
        y_pred_bt = np.asarray(
            arima_bt.forecast(steps=steps_bt)
        )

        # Yüzdelik değer olduğu için sayılar 0–100 arası
        y_pred_bt = np.clip(y_pred_bt, 0, 100)

        # Backtest hata metrikleri
        rmse_bt = np.sqrt(
            mean_squared_error(y_true, y_pred_bt)
        )
        mae_bt = mean_absolute_error(y_true, y_pred_bt)

        print(
            f"\nARIMA out-of-sample RMSE "
            f"(Backtest {BACKTEST_START}–{BACKTEST_END}): "
            f"{rmse_bt:.3f}"
        )
        print(
            f"ARIMA out-of-sample MAE  "
            f"(Backtest {BACKTEST_START}–{BACKTEST_END}): "
            f"{mae_bt:.3f}"
        )

        # Yıl bazında gerçek vs tahmin çıktısı
        print("\n--- Backtest: Gerçek vs Tahmin ---")
        for yr, gt, pr in zip(
            df_test["Year"].to_numpy(),
            y_true,
            y_pred_bt
        ):
            print(
                f"Yıl: {int(yr)} | "
                f"Gerçek: {float(gt):.2f} | "
                f"ARIMA: {float(pr):.2f}"
            )

        # -----------------------------
        # Backtest grafiği
        # -----------------------------
        plt.figure()

        # Tüm gerçek değerler (train + test)
        years_all_bt = np.concatenate(
            [df_train["Year"].to_numpy(),
             df_test["Year"].to_numpy()]
        )
        values_all_bt = np.concatenate(
            [df_train["Value"].to_numpy(),
             y_true]
        )
        plt.plot(
            years_all_bt,
            values_all_bt,
            marker="o",
            label="Gerçek (Train+Test)"
        )

        # Tahmin çizgisi (son train noktasıyla başlatılır)
        last_train_year = int(df_train["Year"].iloc[-1])
        last_train_val = float(df_train["Value"].iloc[-1])

        years_pred_bt = np.concatenate(
            [[last_train_year],
             df_test["Year"].to_numpy()]
        )
        values_pred_bt = np.concatenate(
            [[last_train_val],
             y_pred_bt]
        )

        plt.plot(
            years_pred_bt,
            values_pred_bt,
            linestyle="--",
            marker="o",
            label=f"ARIMA({P},{D},{Q}) – Test Tahmin"
        )

        # Train / test ayrım çizgisi
        plt.axvline(
            x=last_train_year,
            linestyle=":",
            color="gray",
            label="Train/Test Ayrımı"
        )

        plt.xlabel("Yıl")
        plt.ylabel("Yenilenebilir enerji oranı (%)")
        plt.title(
            f"Almanya – ARIMA Backtest "
            f"(test_size={TEST_SIZE})"
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ======================================================
# Eğitim içi performans (in-sample)
# ======================================================
# Modelin kendi eğitim verisi üzerindeki uyumu
fitted = model_fit.fittedvalues

rmse_in = np.sqrt(
    mean_squared_error(
        series.iloc[1:],
        fitted.iloc[1:]
    )
)
mae_in = mean_absolute_error(
    series.iloc[1:],
    fitted.iloc[1:]
)

print(f"\nARIMA in-sample RMSE: {rmse_in:.3f}")
print(f"ARIMA in-sample MAE : {mae_in:.3f}")


# ======================================================
# Gelecek tahmini (2022–2040)
# ======================================================
steps = FUTURE_END - FUTURE_START + 1

forecast = model_fit.forecast(steps=steps)
forecast_vals = np.asarray(forecast)

# Fiziksel sınırlar
forecast_vals = np.clip(forecast_vals, 0, 100)

future_years = np.arange(FUTURE_START, FUTURE_END + 1)

print(
    f"\nGelecek tahminleri "
    f"({FUTURE_START}–{FUTURE_END}, ARIMA):"
)
for y, v in zip(future_years, forecast_vals):
    print(
        f"Yıl: {int(y)}, "
        f"Tahmin edilen yenilenebilir enerji oranı: "
        f"{float(v):.2f} %"
    )


# ======================================================
# Grafik: Tarihsel veriler + ARIMA gelecek tahmini
# ======================================================
plt.figure()

hist_years = df_clean["Year"].to_numpy()
hist_values = df_clean["Value"].to_numpy()

plt.plot(
    hist_years,
    hist_values,
    marker="o",
    label="Gerçek değerler (ölçüm)"
)
plt.plot(
    future_years,
    forecast_vals,
    linestyle="--",
    marker="o",
    label=f"ARIMA({P},{D},{Q}) tahmini"
)

# Ölçüm ve tahmin sınırı
plt.axvline(
    x=last_obs_year,
    linestyle=":",
    color="gray",
    label="Ölçüm / Tahmin Sınırı (2020)"
)

plt.xlabel("Yıl")
plt.ylabel("Yenilenebilir enerji oranı (%)")
plt.title(
    f"Almanya – ARIMA Tahmini "
    f"({FUTURE_START}–{FUTURE_END})"
)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
