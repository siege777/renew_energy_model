import pandas as pd


def detect_outliers_iqr(series: pd.Series, k: float = 1.5):
    # 1. ve 3. çeyrekleri (Q1, Q3) hesapla
    q1, q3 = series.quantile([0.25, 0.75])

    # Interquartile Range (IQR) = Q3 - Q1
    iqr = q3 - q1

    # Alt ve üst eşikleri belirle
    # Varsayılan k=1.5, John Tukey'in outlier tanımı
    low = q1 - k * iqr
    high = q3 + k * iqr

    # Eşiklerin dışında kalan değerleri işaretle
    mask = (series < low) | (series > high)

    # Outlier maskesi ve eşik değerlerini döndür
    return mask, float(low), float(high)


def detect_outliers_zscore(series: pd.Series, z_thresh: float = 3.0):
    # Serinin ortalamasını al
    mu = series.mean()

    # Standart sapma (population std, ddof=0)
    sigma = series.std(ddof=0)

    # Eğer standart sapma 0 ise (yani tüm değerler aynıysa)
    # Z-score tanımsız olur → sıfır vektörü üret
    if sigma == 0 or pd.isna(sigma):
        z = (series - mu) * 0.0
    else:
        # Standart Z-score hesabı
        z = (series - mu) / sigma

    # |z| > z_thresh olanları outlier kabul et
    mask = z.abs() > z_thresh

    # Outlier maskesi ve z-score değerlerini döndür
    return mask, z


def load_prepare_and_check(
    csv_path: str,
    country_code: str,
    start_year: int = 1990,
    do_interpolate: bool = True,
    interpolate_upto_year=None
):
    # World Bank'in' CSV dosyasını oku
    # İlk 4 satır metadata olduğu için atlanması gerekir
    df = pd.read_csv(csv_path, skiprows=4)

    # İlgili ülkeyi Country Code üzerinden filtrele
    df_country = df[df["Country Code"] == country_code].copy()
    if df_country.empty:
        raise ValueError(f"Country Code bulunamadı: {country_code}")

    # Geniş formattaki veriyi uzun formata çevir
    df_long = df_country.melt(
        id_vars=[
            "Country Name",
            "Country Code",
            "Indicator Name",
            "Indicator Code"
        ],
        var_name="Year",
        value_name="Value"
    )

    # Year sütununu sayısala çevir (hatalı olanlar NaN olur)
    df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce")

    # Başlangıç yılından önceki verileri at
    df_long = df_long[df_long["Year"] >= start_year].copy()

    # Yıla göre sırala
    df_long = df_long.sort_values("Year").reset_index(drop=True)

    # -----------------------------
    # Interpolasyon (isteğe bağlıdır)
    # -----------------------------
    # Amaç: eksik yılları lineer şekilde doldurmak
    # interpolate_upto_year verilirse, sadece o yıla kadar yapılır
    if do_interpolate:
        if interpolate_upto_year is None:
            df_long["Value"] = df_long["Value"].interpolate(method="linear")
        else:
            mask = df_long["Year"] <= interpolate_upto_year
            df_long.loc[mask, "Value"] = (
                df_long.loc[mask, "Value"]
                .interpolate(method="linear")
            )

    # Value değeri NaN olan satırları tamamen çıkar
    df_clean = df_long.dropna(subset=["Value"]).copy()

    # Tekrar yıla göre sırala
    df_clean = df_clean.sort_values("Year").reset_index(drop=True)

    # Yıllık değişimi (Delta) hesapla
    # İlk satır NaN olur
    df_clean["Delta"] = df_clean["Value"].diff()

    # -----------------------------
    # Value (seviye) outlier kontrolleri
    # -----------------------------
    # IQR yöntemine göre outlier'lar
    out_iqr_value, low_v, high_v = detect_outliers_iqr(df_clean["Value"])

    # Z-score yöntemine göre outlier'lar
    out_z_value, z_vals = detect_outliers_zscore(
        df_clean["Value"], z_thresh=3.0
    )

    # -----------------------------
    # Delta (yıllık değişim) outlier kontrolü
    # -----------------------------
    # Delta NaN olmayan değerleri al
    delta_nonan = df_clean["Delta"].dropna()

    # IQR için en az birkaç tane gözlem olması gerekir
    if len(delta_nonan) >= 4:
        out_iqr_delta, low_d, high_d = detect_outliers_iqr(delta_nonan)

        # Delta üzerinden bulunan outlier'ları
        # ana dataframe indislerine geri eşle
        out_iqr_delta_full = df_clean["Delta"].isin(
            delta_nonan[out_iqr_delta]
        )
    else:
        # Veri yetersizse IQR uygulanmaz
        low_d, high_d = float("nan"), float("nan")
        out_iqr_delta_full = pd.Series(
            [False] * len(df_clean),
            index=df_clean.index
        )

    df_clean["Z_Value"] = z_vals
    df_clean["Outlier_IQR_Value"] = out_iqr_value
    df_clean["Outlier_Z_Value"] = out_z_value
    df_clean["Outlier_IQR_Delta"] = out_iqr_delta_full

    # Özet istatistikler (raporlama için)
    outlier_summary = {
        "value_iqr_low": low_v,
        "value_iqr_high": high_v,
        "delta_iqr_low": low_d,
        "delta_iqr_high": high_d,
        "value_iqr_count": int(df_clean["Outlier_IQR_Value"].sum()),
        "value_z_count": int(df_clean["Outlier_Z_Value"].sum()),
        "delta_iqr_count": int(df_clean["Outlier_IQR_Delta"].sum()),
    }

    # En az bir yönteme göre outlier olan satırları filtrele
    flagged = df_clean[
        df_clean["Outlier_IQR_Value"]
        | df_clean["Outlier_Z_Value"]
        | df_clean["Outlier_IQR_Delta"]
    ].copy()

    # Sadece önemli bilgileri içeren outlier tablosu
    flagged_years_df = (
        flagged[
            [
                "Year",
                "Value",
                "Delta",
                "Z_Value",
                "Outlier_IQR_Value",
                "Outlier_Z_Value",
                "Outlier_IQR_Delta",
            ]
        ].copy()
        if not flagged.empty
        else flagged
    )

    # Temiz veri, özet ve outlier detay tablosunu döndür
    return df_clean, outlier_summary, flagged_years_df
