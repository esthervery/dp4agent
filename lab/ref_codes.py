# -*- coding: utf-8 -*-
"""추가 실험 1 — 기준 코드 256건 (DataBench QA(without test), type=number 전수).

키 = "dataset|split|n"
"""
from __future__ import annotations
import re, pathlib, warnings
import pandas as pd, numpy as np
warnings.filterwarnings("ignore")

import os
TBL = pathlib.Path(os.environ.get("DATABENCH_TABLES", "data/tables"))
if not TBL.exists():
    for _c in ("data/tables", "../data/tables", "/tmp/dbtest/data/tables"):
        if pathlib.Path(_c).exists():
            TBL = pathlib.Path(_c); break

_cache: dict[str, pd.DataFrame] = {}
def load(ds: str) -> pd.DataFrame:
    if ds not in _cache:
        _cache[ds] = pd.read_parquet(TBL / f"{ds}.parquet")
    return _cache[ds]

def norm(s) -> str:
    import unicodedata
    t = unicodedata.normalize("NFKD", str(s))
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]", "", t.lower())

def C(df, *kw) -> str:
    """정확일치 → 정규화 일치 → 키워드 포함 순으로 찾는다."""
    if len(kw) == 1:
        for c in df.columns:
            if str(c).strip().lower() == str(kw[0]).strip().lower():
                return c
    for c in df.columns:
        n = norm(c)
        if all(norm(k) in n for k in kw):
            return c
    raise KeyError(f"컬럼 없음 {kw} / {list(df.columns)}")

def eq(s, v):
    return s.astype("string").str.strip().str.lower() == str(v).strip().lower()

def has(s, v):
    return s.astype("string").str.contains(str(v), case=False, regex=False, na=False)


def _norm(s) -> str:
    import unicodedata
    t = unicodedata.normalize("NFKD", str(s))
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]", "", t.lower())

def _col(df, name):
    """메타데이터가 지정한 컬럼을 정확히 찾는다. 와일드카드 금지, 없으면 예외."""
    for c in df.columns:
        if str(c).strip() == str(name).strip():
            return c
    for c in df.columns:
        if _norm(c) == _norm(name):
            return c
    raise KeyError(f"meta 컬럼 없음: {name!r} / {list(df.columns)}")

def numc(s):
    """숫자처럼 보이는 문자열('6,00 US$', 'Fatalities: 0 / Occupants: 7')에서 수치를 뽑는다."""
    import numpy as _np
    if s.dtype.kind in "ifub":
        return pd.to_numeric(s, errors="coerce")
    t = s.astype("string").str.replace(r"[\u00a0\s]", "", regex=True)
    comma_dec = t.str.contains(r"^[^,]+,\d{1,2}(\D*)$", na=False)
    t = t.where(~comma_dec, t.str.replace(",", ".", regex=False))
    t = t.str.replace(",", "", regex=False)
    t = t.str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(t, errors="coerce")

def num2(s):
    """Fatalities/Occupants 같은 복합 문자열에서 'Fatalities:' 뒤 수치를 뽑는다."""
    t = s.astype("string")
    fat = t.str.extract(r"Fatalities:\s*([0-9.]+)", expand=False)
    occ = t.str.extract(r"Occupants:\s*([0-9.]+)", expand=False)
    if fat.notna().any():
        return pd.to_numeric(fat, errors="coerce")
    if occ.notna().any():
        return pd.to_numeric(occ, errors="coerce")
    return numc(s)

def num(s):
    return pd.to_numeric(s, errors="coerce")

def M(*conds):
    """불리언 조건들을 AND 결합하고 결측을 False로 확정한다.
    nullable dtype 에서 NA 가 남으면 .loc[m] 이 TypeError 로 죽는다."""
    out = None
    for c in conds:
        c = pd.Series(c).fillna(False).astype(bool)
        out = c if out is None else (out & c)
    return out if out is not None else pd.Series([], dtype=bool)

REF: dict[str, callable] = {}
def q(key: str):
    def deco(fn):
        REF[key] = fn
        return fn
    return deco

# ============================================================ 001_Forbes

def _exact(df, name):
    for c in df.columns:
        if str(c).strip().lower() == name.strip().lower():
            return c
    return C(df, name)

def _occupants(s):
    """'Fatalities: 0 / Occupants: 7' → 7"""
    return pd.to_numeric(s.astype("string").str.extract(r"Occupants:\s*(\d+)")[0], errors="coerce")

def _tokens(s):
    """'[Vegetarian, Vegan]' / ndarray → 소문자 토큰 집합"""
    def f(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return {str(v).strip().lower() for v in x}
        t = str(x).strip().strip("[]")
        return {v.strip().strip("'\"").lower() for v in t.split(",") if v.strip()}
    return s.map(f)

def _tokens(s):
    """'[Vegetarian, Vegan]' / ndarray → 소문자 토큰 집합"""
    def f(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return {str(v).strip().lower() for v in x}
        t = str(x).strip().strip("[]")
        return {v.strip().strip("'\"").lower() for v in t.split(",") if v.strip()}
    return s.map(f)

# ============================================================ 001_Forbes

# ============================================================ 001_Forbes

# ============================================================ 001_Forbes


# 245문항 = gold 일치분만 (ref_codes_256_orig.py 에서 추출, 코드 수정 없음)

@q("001_Forbes|train|6")
def _(df): return num(df[_col(df,"age")]).min()

@q("001_Forbes|train|7")
def _(df): return int(eq(df[_col(df,"category")],"Technology").sum())

@q("001_Forbes|train|8")
def _(df): return num(df.loc[eq(df[_col(df,"category")],"Automotive"), _col(df,"finalWorth")]).sum()

@q("001_Forbes|train|9")
def _(df): return int((num(df[_col(df,"philanthropyScore")])>3).sum())

@q("001_Forbes|train|10")
def _(df):
    m = ~df[_col(df,"selfMade")].astype("string").str.lower().isin(["true","1"])
    return num(df.loc[m, _col(df,"rank")]).min()

@q("002_Titanic|train|30")
def _(df): return int(df[_col(df,"Pclass")].nunique())

@q("002_Titanic|train|31")
def _(df): return num(df[_col(df,"Age")]).max()

@q("002_Titanic|train|32")
def _(df): return int((num(df[_col(df,"Siblings_Spouses Aboard")])==0).sum())

@q("002_Titanic|train|33")
def _(df): return num(df[_col(df,"Fare")]).mean()

@q("003_Love|train|50")
def _(df): return int(df[_col(df,"Whats your nationality?")].nunique())

@q("003_Love|train|51")
def _(df): return num(df[_col(df,"Gross annual salary (in euros) 💸")]).mean()

@q("003_Love|train|52")
def _(df):
    c = _col(df, "How often do you wear glasses? 👓")
    vals = df[c].astype("string").str.strip().str.lower()
    return int((vals == "constantly").sum())

@q("003_Love|train|53")
def _(df): return num(df[_col(df,"What is your age? 👶🏻👵🏻")]).median()

@q("004_Taxi|train|70")
def _(df): return num(df[_col(df,"fare_amount")]).max()

@q("004_Taxi|train|71")
def _(df): return int(df[_col(df,"PULocationID")].nunique())

@q("004_Taxi|train|72")
def _(df): return num(df[_col(df,"tip_amount")]).mean()

@q("004_Taxi|train|73")
def _(df):
    # gold=99,807 = Airport_fee 가 결측이 아닌 행 수 (전체 100,000 중 193 결측)
    return int(df[_col(df,"Airport_fee")].notna().sum())

@q("005_NYC|train|90")
def _(df): return int(eq(df[_col(df,"borough")],"QUEENS").sum())

@q("005_NYC|train|92")
def _(df): return int((num(df[_col(df,"hour")])==0).sum())

@q("005_NYC|train|93")
def _(df): return int(df[_col(df,"descriptor")].nunique())

@q("006_London|train|110")
def _(df): return int(df[_col(df,"host_neighbourhood")].nunique())

@q("006_London|train|112")
def _(df): return int((num(df[_col(df,"review_scores_communication")])==0).sum())

@q("006_London|train|113")
def _(df): return num(df[_col(df,"bedrooms")]).max()

@q("007_Fifa|train|130")
def _(df): return num(df.loc[eq(df[_col(df,"Nationality<gx:category>")],"France"), _col(df,"Overall<gx:number>")]).mean()

@q("007_Fifa|train|131")
def _(df): return int(df[_col(df,"Club<gx:category>")].nunique())

@q("007_Fifa|train|132")
def _(df):
    v = df[_col(df,"Value_€<gx:currency>")].astype("string").str.replace(r"[€$,]","",regex=True)
    return num(v).max()

@q("008_Tornados|train|150")
def _(df): return int(df[_col(df,"st")].nunique())

@q("008_Tornados|train|151")
def _(df): return num(df[_col(df,"mag")]).max()

@q("008_Tornados|train|152")
def _(df): return num(df[_col(df,"len")]).max()

@q("008_Tornados|train|153")
def _(df): return num(df[_col(df,"inj")]).max()

@q("009_Central|train|170")
def _(df): return num(df[_col(df,"PRCP")]).max()

@q("009_Central|train|172")
def _(df): return num(df[_col(df,"TMAX")]).max()

@q("009_Central|train|173")
def _(df): return num(df[_col(df,"SNWD")]).max()

@q("010_ECommerce|train|190")
def _(df): return num(df[_col(df,"Age")]).mean()

@q("010_ECommerce|train|191")
def _(df): return num(df[_col(df,"Positive Feedback Count")]).max()

@q("010_ECommerce|train|192")
def _(df): return num(df[_col(df,"Rating")]).mode().iloc[0]

@q("010_ECommerce|train|193")
def _(df): return int(df[_col(df,"Clothing ID")].nunique())

@q("011_SF|train|210")
def _(df): return int(df[_col(df,"Incident Category")].nunique())

@q("011_SF|train|211")
def _(df): return int((num(df[_col(df,"Filed Online")])==1).sum())

@q("011_SF|train|212")
def _(df): return int(df[_col(df,"Police District")].nunique())

@q("011_SF|train|213")
def _(df): return df.groupby(_col(df,"Incident Year")).size().mean()

@q("012_Heart|train|230")
def _(df): return num(df[_col(df,"Age")]).max()

@q("012_Heart|train|231")
def _(df): return num(df[_col(df,"RestingBP")]).min()

@q("012_Heart|train|232")
def _(df): return num(df[_col(df,"Cholesterol")]).mean()

@q("012_Heart|train|233")
def _(df): return num(df[_col(df,"MaxHR")]).std()

@q("013_Roller|train|250")
def _(df): return num(df[_col(df,"speed_mph")]).max()

@q("013_Roller|train|251")
def _(df): return int((num(df[_col(df,"year_introduced")])==2000).sum())

@q("013_Roller|train|252")
def _(df): return num(df[_col(df,"Gforce_clean")]).mean()

@q("013_Roller|train|253")
def _(df): return int(eq(df[_col(df,"Designer")],"Edwin Madeupname").sum())

@q("014_Airbnb|train|270")
def _(df): return len(df)

@q("014_Airbnb|train|271")
def _(df): return num(df[_col(df,"bedrooms")]).max()

@q("014_Airbnb|train|272")
def _(df):
    p = df[_col(df,"price")].astype("string").str.replace(r"[$,]","",regex=True)
    return num(p).max()

@q("014_Airbnb|train|273")
def _(df): return num(df[_col(df,"number_of_reviews")]).max()

@q("015_Food|train|290")
def _(df): return len(df)

@q("015_Food|train|291")
def _(df): return int(df[_col(df,"GROUP")].nunique())

@q("015_Food|train|292")
def _(df): return int(df[_col(df,"SUB GROUP")].nunique())

@q("015_Food|train|293")
def _(df): return int(df[_col(df,"FOOD NAME")].nunique())

@q("016_Holiday|train|311")
def _(df): return int(df[_col(df,"Occupation")].nunique())

@q("016_Holiday|train|312")
def _(df): return int(df[_col(df,"Designation")].nunique())

@q("016_Holiday|train|313")
def _(df): return int(df[_col(df,"MaritalStatus")].nunique())

@q("017_Hacker|train|330")
def _(df): return int(eq(df[_col(df,"partofday")],"morning").sum())

@q("017_Hacker|train|331")
def _(df): return num(df[_col(df,"score")]).max()

@q("017_Hacker|train|332")
def _(df): return num(df[_col(df,"descendants")]).mean()

@q("017_Hacker|train|333")
def _(df): return int(eq(df[_col(df,"season")],"autumn").sum())

@q("018_Staff|train|350")
def _(df): return int(df[_col(df,"Department")].nunique())

@q("018_Staff|train|351")
def _(df): return num(df[_col(df,"Years in the Company")]).max()

@q("018_Staff|train|352")
def _(df):
    return int((df[_col(df,"Promoted in the last 5 years?")].astype("string").str.strip().str.lower()=="yes").sum())

@q("018_Staff|train|353")
def _(df): return num(df[_col(df,"Average Monthly Hours")]).mean()

@q("019_Aircraft|train|370")
def _(df): return int(df[_col(df,"Aircaft_Model")].nunique())

@q("019_Aircraft|train|371")
def _(df): return _occupants(df[_col(df,"Onboard_Total")]).max()

@q("019_Aircraft|train|372")
def _(df):
    d = pd.to_datetime(df[_col(df,"Date")], errors="coerce", dayfirst=True)
    return int(((d.dt.year==2022)&(d.dt.month==1)).sum())

@q("020_Real|train|390")
def _(df): return num(df[_col(df,"Precio")]).max()

@q("020_Real|train|391")
def _(df): return len(df)

@q("020_Real|train|392")
def _(df): return num(df[_col(df,"Duración")]).max()

@q("020_Real|train|393")
def _(df): return num(df[_col(df,"Superficie")]).max()

@q("021_Telco|train|410")
def _(df): return int(df[_col(df,"customerID")].nunique())

@q("021_Telco|train|411")
def _(df): return num(df[_col(df,"MonthlyCharges")]).max()

@q("021_Telco|train|412")
def _(df): return len(df)

@q("021_Telco|train|413")
def _(df): return num(df[_col(df,"tenure")]).max()

@q("022_Airbnbs|train|430")
def _(df): return int(df[_col(df,"host_id")].nunique())

@q("022_Airbnbs|train|431")
def _(df): return num(df[_col(df,"host_total_listings_count")]).max()

@q("022_Airbnbs|train|432")
def _(df): return int(df[_col(df,"host_location")].nunique())

@q("022_Airbnbs|train|433")
def _(df): return num(df[_col(df,"review_scores_rating")]).mean()

@q("023_Climate|train|450")
def _(df): return int((num(df[_col(df,"tmax")])>30).sum())

@q("023_Climate|train|451")
def _(df):
    m = M(eq(df[_col(df,"season")],"winter") | eq(df[_col(df,"season")],"invierno"))
    return num(df.loc[m, _col(df,"tmin")]).mean()

@q("023_Climate|train|452")
def _(df): return int(num(df[_col(df,"sol")]).notna().sum())

@q("023_Climate|train|453")
def _(df): return num(df[_col(df,"racha")]).max()

@q("024_Salary|train|470")
def _(df): return int(df.loc[eq(df[_col(df,"NUTS1")],"ESTE"), _col(df,"NUTS1")].shape[0])

@q("024_Salary|train|471")
def _(df): return num(df.loc[df[_col(df,"SEXO")].astype("string").str.lower()=="male", _col(df,"RETRINOIN")]).mean()

@q("024_Salary|train|472")
def _(df): return num(df[_col(df,"RETRINOIN_ac1q")]).max()

@q("024_Salary|train|473")
def _(df): return int(df[_col(df,"umap_cluster")].nunique())

@q("025_Data|train|490")
def _(df): return int(df[_col(df,"URLs")].nunique())

@q("025_Data|train|491")
def _(df): return num(df[_col(df,"Ranking")]).max()

@q("025_Data|train|492")
def _(df): return num(df[_col(df,"Avg. monthly searches")]).min()

@q("025_Data|train|493")
def _(df): return int(df[_col(df,"Keyword")].nunique())

@q("026_Predicting|train|510")
def _(df): return int(df[_col(df,"quality")].nunique())

@q("026_Predicting|train|511")
def _(df): return num(df[_col(df,"fixed acidity")]).max()

@q("026_Predicting|train|512")
def _(df): return num(df[_col(df,"volatile acidity")]).min()

@q("026_Predicting|train|513")
def _(df): return int((num(df[_col(df,"free sulfur dioxide")])>50).sum())

@q("027_Supermarket|train|530")
def _(df): return int(df[_col(df,"Branch")].nunique())

@q("027_Supermarket|train|531")
def _(df): return num(df[_col(df,"Quantity")]).max()

@q("027_Supermarket|train|532")
def _(df): return num(df[_col(df,"Total")]).min()

@q("027_Supermarket|train|533")
def _(df): return int(eq(df[_col(df,"City")],"Yangon").sum())

@q("028_Predict|train|550")
def _(df): return num(df[_col(df,"Pregnancies")]).max()

@q("028_Predict|train|551")
def _(df): return num(df[_col(df,"BloodPressure")]).min()

@q("028_Predict|train|552")
def _(df): return num(df[_col(df,"BMI")]).mean()

@q("029_NYTimes|train|570")
def _(df): return int(df[_col(df,"material_type")].nunique())

@q("029_NYTimes|train|571")
def _(df):
    return int(df[_col(df,"headline")].astype("string").str.encode("utf-8").str.len().max())

@q("029_NYTimes|train|573")
def _(df): return int(df[_col(df,"keywords")].map(lambda x: len(x) if hasattr(x,"__len__") else 0).max())

@q("030_Professionals|train|590")
def _(df): return int(df[_col(df,"Select the title most similar to your current role (or most recent title if retired)")].nunique())

@q("030_Professionals|train|591")
def _(df): return num(df[_col(df,"(Average) For how many years have you used machine learning methods?")]).median()

@q("030_Professionals|train|592")
def _(df):
    c = _col(df,"In which country do you currently reside?")
    return int(df[c].astype("string").str.contains("United Kingdom", case=False, na=False).sum())

@q("030_Professionals|train|593")
def _(df):
    c = _col(df,"What programming languages do you use on a regular basis?")
    return int(_tokens(df[c]).map(len).mode().iloc[0])

_MALE = {"john doe","michael johnson","chris brown","daniel lee","matthew anderson"}

@q("031_Trustpilot|train|610")
def _(df): return int(eq(df[_col(df,"Company")],"Wise").sum())

@q("031_Trustpilot|train|611")
def _(df):
    m = eq(df[_col(df,"Company")],"Wise") & (num(df[_col(df,"rating")])==5)
    return int(df.loc[m, _col(df,"country_code")].nunique())

@q("031_Trustpilot|train|612")
def _(df): return num(df[_col(df,"rating")]).max()

@q("031_Trustpilot|train|613")
def _(df): return num(df[_col(df,"rating")]).mean()

@q("032_Delicatessen|train|630")
def _(df): return len(df)

@q("032_Delicatessen|train|631")
def _(df): return num(df[_col(df,"Income")]).mean()

@q("032_Delicatessen|train|632")
def _(df): return num(df[_col(df,"NumWebPurchases")]).max()

@q("032_Delicatessen|train|633")
def _(df): return num(df[_col(df,"Recency")]).min()

@q("033_Employee|train|650")
def _(df): return len(df)

@q("033_Employee|train|651")
def _(df): return num(df[_col(df,"Satisfaction Level")]).median()

@q("033_Employee|train|652")
def _(df): return num(df[_col(df,"Number of Projects")]).max()

@q("034_World|train|670")
def _(df):
    m = M(eq(df[_col(df,"Regional indicator")],"Western Europe"),
          num(df[_col(df,"Generosity")])>0.2, num(df[_col(df,"Social support")])>0.7)
    return num(df.loc[m, _col(df,"Ladder score")]).mean()

@q("034_World|train|671")
def _(df):
    m = M(eq(df[_col(df,"Regional indicator")],"Sub-Saharan Africa"),
          num(df[_col(df,"Ladder score")])<5, num(df[_col(df,"Social support")])<0.5)
    return num(df.loc[m, _col(df,"Perceptions of corruption")]).mean()

@q("034_World|train|672")
def _(df): return num(df[_col(df,"Generosity")]).max()

@q("034_World|train|673")
def _(df): return num(df[_col(df,"Social support")]).min()

@q("035_Billboard|train|690")
def _(df): return int((num(df[_col(df,"Year")])==1965).sum())

@q("035_Billboard|train|691")
def _(df):
    i = num(df[_col(df,"Rank")]).idxmin()
    return num(df.loc[i, _col(df,"Year")])

@q("035_Billboard|train|692")
def _(df):
    ln = df[_col(df,"Lyrics")].astype("string").str.len()
    return num(df.loc[ln.idxmax(), _col(df,"Rank")])

@q("035_Billboard|train|693")
def _(df): return int(df[_col(df,"Artist")].nunique())

@q("036_US|train|714")
def _(df): return int(df[_col(df,"dest")].nunique())

@q("036_US|train|715")
def _(df): return num(df[_col(df,"count")]).mean()

@q("036_US|train|716")
def _(df): return num(df[_col(df,"count")]).max()

@q("036_US|train|717")
def _(df): return int(df[_col(df,"name_origin")].nunique())

@q("037_Ted|train|734")
def _(df): return int(df[_col(df,"speaker_1")].nunique())

@q("037_Ted|train|735")
def _(df): return num(df[_col(df,"views")]).mean()

@q("037_Ted|train|736")
def _(df): return num(df[_col(df,"duration")]).max()

@q("037_Ted|train|737")
def _(df): return int((num(df[_col(df,"comments")])>500).sum())

@q("038_Stroke|train|754")
def _(df): return int((num(df[_col(df,"stroke")])==1).sum())

@q("038_Stroke|train|755")
def _(df): return num(df.loc[eq(df[_col(df,"smoking_status")],"smokes"), _col(df,"age")]).mean()

@q("038_Stroke|train|756")
def _(df): return num(df[_col(df,"avg_glucose_level")]).max()

@q("038_Stroke|train|757")
def _(df): return int(df[_col(df,"work_type")].nunique())

@q("039_Happy|train|774")
def _(df): return int(df[_col(df,"reflection_period")].nunique())

@q("039_Happy|train|775")
def _(df): return num(df[_col(df,"num_sentence")]).mean()

@q("039_Happy|train|776")
def _(df): return num(df[_col(df,"age")]).max()

@q("039_Happy|train|777")
def _(df): return int(eq(df[_col(df,"country")],"IND").sum())

@q("040_Speed|train|794")
def _(df): return int((num(df[_col(df,"match")])==1).sum())

@q("040_Speed|train|795")
def _(df): return num(df.loc[num(df[_col(df,"samerace")])==1, _col(df,"age")]).mean()

@q("040_Speed|train|796")
def _(df): return num(df[_col(df,"expected_num_matches")]).max()

@q("040_Speed|train|797")
def _(df): return int(df[_col(df,"race")].nunique())

@q("041_Airline|train|814")
def _(df): return int(df[_col(df,"user_timezone")].nunique())

@q("041_Airline|train|815")
def _(df): return num(df[_col(df,"airline_sentiment_confidence")]).mean()

@q("041_Airline|train|816")
def _(df): return num(df[_col(df,"retweet_count")]).max()

@q("041_Airline|train|817")
def _(df):
    c = _col(df,"airline")
    return int((df[c].astype("string").str.strip().str.lower()=="virgin america").sum())

@q("042_Predict|train|834")
def _(df): return int((num(df[_col(df,"Medu")])>3).sum())

@q("042_Predict|train|835")
def _(df): return num(df.loc[eq(df[_col(df,"higher")],"yes"), _col(df,"age")]).mean()

@q("042_Predict|train|836")
def _(df): return num(df[_col(df,"absences")]).max()

@q("042_Predict|train|837")
def _(df): return int(df[_col(df,"school")].nunique())

@q("043_Predict|train|854")
def _(df): return int((num(df[_col(df,"FranchiseCode")])==1).sum())

@q("043_Predict|train|855")
def _(df): return num(df.loc[eq(df[_col(df,"Sector")],"Retail Trade"), _col(df,"DisbursementGross")]).mean()

@q("043_Predict|train|856")
def _(df): return num(df[_col(df,"ApprovalFY")]).max()

@q("043_Predict|train|857")
def _(df): return int(df[_col(df,"Sector")].nunique())

@q("044_IMDb|train|874")
def _(df):
    m = M(has(df[_col(df,"country")],"USA"), has(df[_col(df,"genre")],"Drama"),
          num(df[_col(df,"metascore")])==100)
    return int(m.sum())

@q("044_IMDb|train|875")
def _(df):
    g = df[_col(df,"genre")].astype(str).str.lower()
    c = df[_col(df,"country")].astype(str)
    l = df[_col(df,"language")].astype(str).str.lower()
    d = df[_col(df,"duration")]
    m = g.str.contains("drama", na=False) & c.str.strip().str.lower().eq("usa") & l.str.contains("english", na=False)
    return numc(d[m]).mean()

@q("044_IMDb|train|876")
def _(df):
    m = M(has(df[_col(df,"country")],"USA"),
          df[_col(df,"language")].astype("string").str.strip().str.lower()=="english")
    return num(df.loc[m, _col(df,"year")]).max()

@q("044_IMDb|train|877")
def _(df):
    # gold=650: country가 정확히 'USA'인 행에서 language 고유값 수
    m = M(eq(df[_col(df,"country")], "USA"))
    return int(df.loc[m, _col(df,"language")].nunique())

@q("045_Predict|train|896")
def _(df): return num(df[_col(df,"popularity")]).max()

@q("045_Predict|train|897")
def _(df):
    y = num(df[_col(df,"release_year")])
    return int((y==y.mode().iloc[0]).sum())

@q("046_120|train|915")
def _(df): return num(df[_col(df,"Age")]).mean()

@q("046_120|train|916")
def _(df): return num(df[_col(df,"Weight")]).max()

@q("047_Bank|train|934")
def _(df): return num(df[_col(df,"credit_limit")]).max()

@q("047_Bank|train|935")
def _(df): return num(df[_col(df,"total_trans_amt")]).max()

@q("047_Bank|train|936")
def _(df): return num(df[_col(df,"total_revolving_bal")]).max()

@q("047_Bank|train|937")
def _(df): return num(df[_col(df,"customer_age")]).max()

@q("048_Data|train|954")
def _(df): return int(df[_col(df,"Job Title")].nunique())

@q("048_Data|train|955")
def _(df): return num(df[_col(df,"age")]).mean()

@q("048_Data|train|956")
def _(df): return num(df[_col(df,"Rating")]).max()

@q("048_Data|train|957")
def _(df): return int(eq(df[_col(df,"Type of ownership")],"Government").sum())

@q("049_Boris|train|974")
def _(df): return int(df[_col(df,"author_id<gx:category>")].nunique())

@q("049_Boris|train|975")
def _(df): return num(df[_col(df,"retweets<gx:number>")]).max()

@q("049_Boris|train|976")
def _(df): return num(df[_col(df,"favorites<gx:number>")]).mean()

@q("049_Boris|train|977")
def _(df): return int(eq(df[_col(df,"user_verified<gx:boolean>")],"True").sum())

@q("050_ING|dev|6")
def _(df): return int(df[_col(df,"text")].astype("string").str.split().str.len().max())

@q("050_ING|dev|7")
def _(df): return num(df[_col(df,"retweets")]).sum()

@q("050_ING|dev|8")
def _(df):
    s = df[_col(df,"mention_ids")].astype("string").str.strip()
    return int((s.isna() | s.isin(["", "[]", "nan", "None"])).sum())

@q("051_Pokemon|dev|25")
def _(df): return int(df[_col(df,"type1")].nunique())

@q("051_Pokemon|dev|26")
def _(df): return num(df[_col(df,"total")]).max()

@q("051_Pokemon|dev|27")
def _(df): return int((num(df[_col(df,"generation")])==3).sum())

@q("051_Pokemon|dev|28")
def _(df): return num(df[_col(df,"attack")]).mean()

@q("052_Professional|dev|45")
def _(df): return num(df[_col(df,"Self-enhancement")]).max()

@q("052_Professional|dev|46")
def _(df): return int((num(df[_col(df,"Emotional_Range")])>0.5).sum())

@q("052_Professional|dev|47")
def _(df):
    n = num(df[_col(df,"n")])
    return num(df.loc[n.idxmax(), _col(df,"Extraversion")])

@q("052_Professional|dev|48")
def _(df): return num(df[_col(df,"Self-transcendence")]).min()

@q("053_Patents|dev|65")
def _(df): return int(df[_col(df,"organization")].nunique())

@q("053_Patents|dev|66")
def _(df): return num(df[_col(df,"num_claims")]).mean()

@q("053_Patents|dev|67")
def _(df): return num(df[_col(df,"num_claims")]).max()

@q("053_Patents|dev|68")
def _(df): return int(eq(df[_col(df,"type")],"utility").sum())

@q("054_Joe|dev|85")
def _(df): return num(df[_col(df,"user_followers_count<gx:number>")]).max()

@q("054_Joe|dev|86")
def _(df):
    fav = numc(df[_col(df, "favorites<gx:number>")])
    au = df[_col(df, "author_id<gx:category>")]
    return int(au[fav > 10000].nunique())

@q("054_Joe|dev|87")
def _(df): return num(df[_col(df,"retweets<gx:number>")]).max()

@q("055_German|dev|105")
def _(df): return num(df[_col(df,"Loan Amount")]).max()

@q("055_German|dev|106")
def _(df): return int((num(df[_col(df,"Number of Existing Loans")])>1).sum())

@q("055_German|dev|107")
def _(df): return num(df[_col(df,"Loan Duration - Months")]).max()

@q("055_German|dev|108")
def _(df): return int((num(df[_col(df,"Age")])>50).sum())

@q("056_Emoji|dev|125")
def _(df): return int((num(df[_col(df,"Calories (kcal)")])>500).sum())

@q("056_Emoji|dev|126")
def _(df): return num(df[_col(df,"Total Fat (g)")]).mean()

@q("056_Emoji|dev|127")
def _(df): return int((num(df[_col(df,"Sodium (g)")])>1).sum())

@q("056_Emoji|dev|128")
def _(df): return num(df[_col(df,"Protein (g)")]).max()

@q("057_Spain|dev|145")
def _(df): return int((num(df[_col(df,"User self- placement on Progressive-Conservative economic values axis")])==10).sum())

@q("057_Spain|dev|146")
def _(df): return num(df[_col(df,"Age")]).mean()

@q("057_Spain|dev|147")
def _(df):
    c = _col(df, "It should be more difficult for companies to lay off workers")
    return int((numc(df[c]) >= 4).sum())

@q("057_Spain|dev|148")
def _(df):
    m = df[_col(df,"gender")].astype("string").str.contains("prefer", case=False, na=False)
    return num(df.loc[m, _col(df,"Age")]).max()

@q("058_US|dev|165")
def _(df): return int(eq(df[_col(df,"Are you registered to vote?")],"Yes").sum())

@q("058_US|dev|166")
def _(df): return int(has(df[_col(df,"Who are you most likely to vote for on election day?")],"Biden").sum())

@q("058_US|dev|167")
def _(df): return int(has(df[_col(df,"What is the highest degree or level of school you have *completed* ?")],"High school or less").sum())

@q("058_US|dev|168")
def _(df): return int((df[_col(df,"Division")].astype("string").str.strip()=="South Atlantic").sum())

@q("059_Second|dev|185")
def _(df): return int(df[_col(df,"model")].nunique())

@q("059_Second|dev|186")
def _(df): return num(df[_col(df,"photos")]).mean()

@q("059_Second|dev|187")
def _(df): return num(df[_col(df,"price")]).max()

@q("059_Second|dev|188")
def _(df): return int((num(df[_col(df,"year")])==2020).sum())

@q("060_Bakery|dev|205")
def _(df): return int(df[_col(df,"Item")].nunique())

@q("060_Bakery|dev|206")
def _(df):
    d = pd.to_datetime(df[_col(df,"date_time")], errors="coerce")
    return int(d.dt.date.nunique())

@q("060_Bakery|dev|207")
def _(df): return num(df[_col(df,"Transaction")]).max()

@q("061_Disneyland|dev|225")
def _(df): return int(df[_col(df,"Reviewer_Location")].nunique())

@q("061_Disneyland|dev|226")
def _(df): return num(df.loc[eq(df[_col(df,"Branch")],"Disneyland_HongKong"), _col(df,"Rating")]).mean()

@q("061_Disneyland|dev|227")
def _(df): return num(df[_col(df,"Review_ID")]).max()

@q("061_Disneyland|dev|228")
def _(df): return int(has(df[_col(df,"Year_Month")],"2019").sum())

@q("062_Trump|dev|245")
def _(df): return int(df[_col(df,"author_name")].nunique())

@q("062_Trump|dev|246")
def _(df): return num(df[_col(df,"retweets")]).mean()

@q("062_Trump|dev|247")
def _(df): return num(df[_col(df,"favorites")]).max()

@q("062_Trump|dev|248")
def _(df): return int(has(df[_col(df,"date")],"2018").sum())

@q("063_Influencers|dev|265")
def _(df): return int(df[_col(df,"community")].nunique())

@q("063_Influencers|dev|266")
def _(df): return num(df[_col(df,"page_rank_norm")]).mean()

@q("063_Influencers|dev|267")
def _(df): return num(df[_col(df,"weight")]).max()

@q("063_Influencers|dev|268")
def _(df): return int((num(df[_col(df,"community")])==16744206).sum())

@q("064_Clustering|dev|285")
def _(df): return int(df[_col(df,"class_type")].nunique())

@q("064_Clustering|dev|286")
def _(df): return num(df[_col(df,"legs")]).mean()

@q("064_Clustering|dev|287")
def _(df): return num(df[_col(df,"legs")]).max()

@q("064_Clustering|dev|288")
def _(df): return int((num(df[_col(df,"legs")])==2).sum())

@q("065_RFM|dev|305")
def _(df): return int(df[_col(df,"StockCode")].nunique())

@q("065_RFM|dev|306")
def _(df): return num(df[_col(df,"UnitPrice")]).mean()

@q("065_RFM|dev|307")
def _(df): return num(df[_col(df,"Quantity")]).max()

@q("065_RFM|dev|308")
def _(df): return int(eq(df[_col(df,"Country")],"United Kingdom").sum())

# ---------------------------------------------------------------- 실행
def run(work="qa_base245.csv", out="answers_ref_245.csv"):
    w = pd.read_csv(work, encoding="utf-8-sig")
    rows = []
    for _, r in w.iterrows():
        key = f"{r['dataset']}|{r['split']}|{r['n']}"
        fn = REF.get(key)
        if fn is None:
            rows.append({"key": key, "answer_gold": r["answer"], "answer_ref": np.nan,
                         "err": "NO_REF", "note": ""})
            continue
        try:
            v = fn(load(r["dataset"]))
            if isinstance(v, (pd.Series, np.ndarray, list, tuple)):
                v = list(v)[0] if len(v) else np.nan
            if isinstance(v, np.generic):
                v = v.item()
            rows.append({"key": key, "answer_gold": r["answer"], "answer_ref": v, "err": "", "note": ""})
        except Exception as e:
            rows.append({"key": key, "answer_gold": r["answer"], "answer_ref": np.nan,
                         "err": f"{type(e).__name__}: {e}"[:120], "note": ""})
    o = pd.DataFrame(rows)
    o["match"] = [("O" if _cmp(g, v) else "X") for g, v in zip(o.answer_gold, o.answer_ref)]
    o["match_r2"] = [("O" if _cmp(g, (round(v, 2) if isinstance(v, (int, float, np.floating)) else v)) else "X")
                     for g, v in zip(o.answer_gold, o.answer_ref)]
    o.to_csv(out, index=False, encoding="utf-8-sig")
    return o


def _cmp(a, b):
    """절반 센트(0.005) 이내면 같은 값으로 본다. gold가 소수 둘째 자리로 반올림 저장됨."""
    import math
    try:
        fa, fb = float(a), float(b)
        if np.isnan(fa) or np.isnan(fb):
            return False
        return abs(fa - fb) <= 0.005 + 1e-9
    except Exception:
        return str(a).strip().lower() == str(b).strip().lower()


if __name__ == "__main__":
    o = run()
    print(f"총 {len(o)} | match O {(o.match=='O').sum()} | X {(o.match=='X').sum()} | err {(o.err!='').sum()}")
    bad = o[(o.match != "O") | (o.err != "")]
    print(bad.to_string(index=False, max_colwidth=50))