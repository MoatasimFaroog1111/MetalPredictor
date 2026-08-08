# حزمة بيانات المعادن النفيسة الساعية — USD/kg

هذه الحزمة مخصصة لجلب وتنظيف أسعار:

- الذهب XAU
- الفضة XAG
- البلاتين XPT
- البلاديوم XPD

بتردد ساعة واحدة، للفترة الافتراضية:

- البداية: 2021-08-08 12:00 UTC
- النهاية: 2026-08-08 12:00 UTC

## الوحدة

المصدر غالبًا يعرض الأسعار بالدولار لكل أونصة تروي `USD/troy oz`.
الحزمة تحول جميع حقول OHLC إلى:

`USD/kg = USD/troy_oz × 32.15074656862798`

وتحتفظ أيضًا بالسعر الأصلي بالدولار لكل أونصة للمراجعة.

## مهم جدًا

هذه الحزمة **لا تحتوي على خمس سنوات من الأسعار الفعلية** لأن تنزيل البيانات الكاملة من المصادر الموثوقة يتطلب مفتاح API/اشتراك صالح. الملفات الموجودة هنا هي أدوات الجلب والتنظيف والتحقق الجاهزة للتشغيل.

## الخيار الموصى به للـ Spot

Twelve Data:

- XAU/USD
- XAG/USD
- XPT/USD
- XPD/USD

تشغيل:

```bash
pip install -r requirements.txt
export TWELVEDATA_API_KEY="YOUR_KEY"
python fetch_twelvedata.py
```

في PowerShell:

```powershell
$env:TWELVEDATA_API_KEY="YOUR_KEY"
python .\fetch_twelvedata.py
```

المخرجات:

- `output_twelvedata/metals_hourly_usd_per_kg.csv`
- `output_twelvedata/metals_hourly_usd_per_kg.json`
- `output_twelvedata/metals_hourly_usd_per_kg.parquet`

## خيار Futures للتحقق وميزات الحجم

Databento/CME continuous futures:

- Gold: GC.v.0
- Silver: SI.v.0
- Platinum: PL.v.0
- Palladium: PA.v.0

تشغيل:

```bash
export DATABENTO_API_KEY="YOUR_KEY"
python fetch_databento.py
```

## التنظيف والتحقق

```bash
python clean_validate.py output_twelvedata/metals_hourly_usd_per_kg.csv
python validate_and_plot.py cleaned/metals_hourly_clean.csv
```

لا يتم forward-fill لساعات الإغلاق أو الفجوات تلقائيًا، حتى لا يتم خلق أسعار وهمية في بيانات التدريب.
