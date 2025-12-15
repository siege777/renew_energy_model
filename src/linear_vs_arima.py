import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Makine öğrenmesi ve hata metrikleri
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Zaman serisi modeli
from statsmodels.tsa.arima.model import ARIMA

# Veri temizleme, interpolasyon ve outlier kontrol fonksiyonu
from data_preprocessing import load_prepare_and_check

import warnings

# ARIMA başlangıç parametrelerine dair uyarıları bastırma
# (model yine düzgün çalışır, sadece konsol çıktısını sadeleştirmek için)
warnings.filterwarnings(
    "ignore",
    message="Non-stationary starting autoregressive parameters"
)

# Ayarlar
# ======================================================

# Veri dosyası yolu
CSV_PATH = "data/renew_energy_consumption.csv"

# Analiz edilecek ülke (ISO country code)
COUNTRY_CODE = "DEU"  # Avusturya

# Analize dahil edilecek ilk yıl
START_YEAR = 1990

# Eksik yıllar için interpolasyon ayarları
DO_INTERPOLATE = True
INTERP_UPTO = 2021     # 2022 ve sonrası doldurulmaz (tahmin dönemi)

# Train / test ayrım oranı (zaman sırası korunur)
TEST_SIZE = 0.3

# ARIMA(p, d, q) parametreleri
ARIMA_ORDER = (1, 1, 1)


# ======================================================
# Veri hazırlama
# ======================================================
# Veri okunur, temizlenir, interpolasyon uygulanır ve outlier kontrolleri yapılır
df_clean, summary, flagged = load_prepare_and_check(
    csv_path=CSV_PATH,
    country_code=COUNTRY_CODE,
    start_year=START_YEAR,
    do_interpolate=DO_INTERPOLATE,
    interpolate_upto_year=INTERP_UPTO,
)

# Year verisini integer hale getir
df_clean["Year"] = pd.to_numeric(
    df_clean["Year"], errors="coerce"
).astype(int)

# Value eksik olan satırları çıkar
df_clean = df_clean.dropna(subset=["Value"]).copy()

# Yıla göre sırala
df_clean = df_clean.sort_values("Year").reset_index(drop=True)

# ------------------------
# Regresyon için X (yıl) ve y (oran) oluştur
# ------------------------
X = df_clean["Year"].values.reshape(-1, 1)
y = df_clean["Value"].values

# Zaman serisi olduğu için shuffle=False
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, shuffle=False
)

# Eğitim ve test yıllarını ayrı ayrı tut
train_years = X_train.ravel().astype(int)
test_years = X_test.ravel().astype(int)

print("\n--- Split Bilgisi ---")
print(
    "Eğitim seti yıl aralığı:",
    int(train_years.min()),
    "-",
    int(train_years.max())
)
print(
    "Test seti yıl aralığı:",
    int(test_years.min()),
    "-",
    int(test_years.max())
)


# ======================================================
# Model 1: Lineer Regresyon (Year → Value)
# ======================================================
# Basit trend modeli: yıl → enerji oranı
lin = LinearRegression()
lin.fit(X_train, y_train)

# Test seti için tahmin
y_pred_lin = lin.predict(X_test)

# Hata metrikleri
rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))
mae_lin = mean_absolute_error(y_test, y_pred_lin)


# ======================================================
# Model 2: ARIMA (Value zaman serisi)
# ======================================================
# PeriodIndex kullanımı:
# - Yıllık frekansı açıkça belirtir

y_train_series = pd.Series(
    y_train,
    index=pd.PeriodIndex(train_years, freq="Y")
)

# ARIMA modelini kur ve eğit
arima_model = ARIMA(y_train_series, order=ARIMA_ORDER)
arima_fit = arima_model.fit()

# Test uzunluğu kadar ileri tahmin al
steps = len(y_test)
y_pred_arima = (
    arima_fit
    .get_forecast(steps=steps)
    .predicted_mean
    .to_numpy()
)

# Yüzdelik oran olduğu için 0-100 sınırları içinde tut
y_pred_arima = np.clip(y_pred_arima, 0, 100)

# Hata metrikleri
rmse_arima = np.sqrt(mean_squared_error(y_test, y_pred_arima))
mae_arima = mean_absolute_error(y_test, y_pred_arima)


# ======================================================
# Sonuç tablosu (test seti karşılaştırması)
# ======================================================
results = pd.DataFrame([
    {"Model": f"ARIMA{ARIMA_ORDER}", "RMSE": rmse_arima, "MAE": mae_arima},
    {"Model": "LinearRegression", "RMSE": rmse_lin, "MAE": mae_lin},
]).sort_values("RMSE").reset_index(drop=True)

print("\n--- Model Karşılaştırma (Test Seti) ---")
print(results.to_string(index=False))


# ======================================================
# Test döneminde: gerçek vs tahmin (kontrol amaçlı)
# ======================================================
print("\n--- Test seti: Gerçek vs Tahmin ---")
for year, real, pl, pa in zip(
    test_years, y_test, y_pred_lin, y_pred_arima
):
    print(
        f"Yıl: {int(year)} | "
        f"Gerçek: {real:.2f} | "
        f"Lineer: {pl:.2f} | "
        f"ARIMA: {pa:.2f}"
    )


# ======================================================
# Grafik: Tüm veri + model tahminleri
# ======================================================
years_all = df_clean["Year"].values.astype(int)
values_all = df_clean["Value"].values

# Lineer regresyon için tüm yıllara tahmin
yhat_lin_all = lin.predict(X)

# ------------------------
# ARIMA: train fit + test forecast birleşimi
# ------------------------
train_index = y_train_series.index  # PeriodIndex (yıllık)

# Eğitim dönemi in-sample tahminleri
pred_train = arima_fit.predict(
    start=train_index[0],
    end=train_index[-1]
)
pred_train = pd.Series(pred_train, index=train_index)

# d=1 nedeniyle ilk tahmin NaN olabilir → gerçek değerle doldur
if pd.isna(pred_train.iloc[0]):
    pred_train.iloc[0] = y_train[0]

# Test dönemi forecast (PeriodIndex üretir)
fc_test = (
    arima_fit
    .get_forecast(steps=len(y_test))
    .predicted_mean
)

# Tüm yıllar için boş bir dizi oluştur
yhat_arima_all = np.full_like(
    values_all,
    fill_value=np.nan,
    dtype=float
)

# Yıl → indeks eşlemesi
year_to_pos = {
    int(y): i
    for i, y in enumerate(years_all)
}

# Train tahminlerini yerleştir
for period, val in pred_train.items():
    yr = int(period.year)
    if yr in year_to_pos and pd.notna(val):
        yhat_arima_all[year_to_pos[yr]] = float(val)

# Test tahminlerini yerleştir
for period, val in fc_test.items():
    yr = int(period.year)
    if yr in year_to_pos and pd.notna(val):
        yhat_arima_all[year_to_pos[yr]] = float(val)

# Fiziksel sınırlar
yhat_arima_all = np.clip(yhat_arima_all, 0, 100)


# ------------------------
# Grafik çizimi
# ------------------------
plt.figure()

plt.plot(
    years_all,
    values_all,
    marker="o",
    label="Gerçek (Tüm Veri)"
)

split_year = int(train_years.max())
plt.axvline(
    split_year,
    linestyle="--",
    label="Train/Test Ayrımı"
)

plt.plot(
    years_all,
    yhat_lin_all,
    linestyle="--",
    label="Lineer Tahmin"
)
plt.plot(
    years_all,
    yhat_arima_all,
    linestyle="--",
    label=f"ARIMA{ARIMA_ORDER} (Fit/Forecast)"
)

plt.xlabel("Yıl")
plt.ylabel("Yenilenebilir enerji oranı (%)")
plt.title(
    "Almanya için Lineer Regresyon ve ARIMA Modellerinin Karşılaştırılması"
)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
