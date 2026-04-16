import os

import numpy as np
import pandas as pd

try:
    import config_clearml  # noqa: F401
    from clearml import Task
except ImportError:
    Task = None


class _NoOpLogger:
    def report_scalar(self, *args, **kwargs):
        return None


class _NoOpTask:
    @staticmethod
    def init(*args, **kwargs):
        return _NoOpTask()

    def upload_artifact(self, *args, **kwargs):
        return None

    def get_logger(self):
        return _NoOpLogger()

    def connect(self, *args, **kwargs):
        return None

    def flush(self, *args, **kwargs):
        return None

    def close(self):
        return None


Task = Task or _NoOpTask


def cap_and_normalize(series):
    lower = series.quantile(0.01)
    upper = series.quantile(0.99)
    capped = series.clip(lower, upper)
    denom = capped.max() - capped.min()
    if denom == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (capped - capped.min()) / denom


def text_length_feature(series):
    return series.fillna("").astype(str).str.len()


task = Task.init(project_name="AI Studio Property", task_name="Preprocessing Stage")
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/half_treated.csv")
listings_df = pd.read_csv(
    "data/listings.csv",
    usecols=[
        "id",
        "name",
        "description",
        "host_since",
        "host_about",
        "host_listings_count",
        "host_total_listings_count",
        "host_verifications",
        "host_has_profile_pic",
        "neighbourhood_cleansed",
        "latitude",
        "longitude",
        "property_type",
        "room_type",
        "bathrooms_text",
        "number_of_reviews_l30d",
        "number_of_reviews_ly",
        "availability_eoy",
        "estimated_occupancy_l365d",
        "estimated_revenue_l365d",
        "instant_bookable",
        "calculated_host_listings_count",
    ],
)
df = df.merge(listings_df, on="id", how="left")

task.connect({
    "input_main_dataset": "data/half_treated.csv",
    "input_listing_dataset": "data/listings.csv",
    "merge_key": "id",
    "target_column": "review_scores_rating",
})

df["accommodates_norm"] = cap_and_normalize(df["accommodates"])
df["bathrooms_norm"] = cap_and_normalize(df["bathrooms"].fillna(df["bathrooms"].median()))
df["bedrooms_norm"] = cap_and_normalize(df["bedrooms"].fillna(df["bedrooms"].median()))
df["beds_norm"] = cap_and_normalize(df["beds"].fillna(df["beds"].median()))
df["price_filled"] = df["price"].fillna(df["price"].median())
df["price_norm"] = cap_and_normalize(np.log1p(df["price_filled"]))

df["host_response_rate_filled"] = df["host_response_rate"].fillna(0.5)
df["host_acceptance_rate_filled"] = df["host_acceptance_rate"].fillna(0.5)
df["host_response_time_filled"] = df["host_response_time"].map({
    "within an hour": 1.0,
    "within a few hours": 0.75,
    "within a day": 0.5,
    "a few days or more": 0.25,
}).fillna(0.5)
df["host_is_superhost_filled"] = df["host_is_superhost"].map({"t": 1, "f": 0}).fillna(0.5)
df["host_identity_verified_filled"] = df["host_identity_verified"].map({"t": 1, "f": 0}).fillna(0.5)
df["host_has_profile_pic_filled"] = df["host_has_profile_pic"].map({"t": 1, "f": 0}).fillna(0.5)
df["instant_bookable_filled"] = df["instant_bookable"].map({"t": 1, "f": 0}).fillna(0.5)

df["amenities_weight_norm"] = cap_and_normalize(df["amenities_weight"])
df["minimum_nights_norm"] = cap_and_normalize(1 / np.log1p(df["minimum_nights"]))
df["availability_365_norm"] = 1 - (df["availability_365"] / 365)

df["number_of_reviews_log"] = cap_and_normalize(np.log1p(df["number_of_reviews"]))
df["number_of_reviews_ltm_log"] = cap_and_normalize(np.log1p(df["number_of_reviews_ltm"]))
df["reviews_per_month_log"] = cap_and_normalize(np.log1p(df["reviews_per_month"].fillna(0)))
df["has_reviews_filled"] = df["has_reviews"].fillna(0)
df["review_activity_score"] = cap_and_normalize(
    0.4 * np.log1p(df["number_of_reviews"]) +
    0.4 * np.log1p(df["number_of_reviews_ltm"]) +
    0.2 * np.log1p(df["reviews_per_month"].fillna(0))
)

df["price_per_guest"] = df["price_filled"] / df["accommodates"].replace(0, np.nan)
df["price_per_guest"] = df["price_per_guest"].fillna(df["price_per_guest"].median())
df["price_per_guest_norm"] = cap_and_normalize(np.log1p(df["price_per_guest"]))
df["beds_per_bedroom"] = df["beds"].fillna(df["beds"].median()) / df["bedrooms"].replace(0, np.nan)
df["beds_per_bedroom"] = df["beds_per_bedroom"].replace([np.inf, -np.inf], np.nan).fillna(df["beds"].median())
df["beds_per_bedroom_norm"] = cap_and_normalize(df["beds_per_bedroom"])
df["superhost_response_combo"] = df["host_is_superhost_filled"] * df["host_response_rate_filled"]

df["review_recency_ratio"] = df["number_of_reviews_ltm"] / df["number_of_reviews"].replace(0, np.nan)
df["review_recency_ratio"] = df["review_recency_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0)
df["review_recency_ratio_norm"] = cap_and_normalize(df["review_recency_ratio"])
df["response_acceptance_gap_norm"] = cap_and_normalize(df["host_response_rate_filled"] - df["host_acceptance_rate_filled"])
df["host_reliability_score"] = cap_and_normalize(
    0.35 * df["host_response_rate_filled"] +
    0.35 * df["host_acceptance_rate_filled"] +
    0.15 * df["host_is_superhost_filled"] +
    0.15 * df["host_identity_verified_filled"]
)
df["demand_pressure"] = cap_and_normalize(np.log1p(df["number_of_reviews_ltm"] + 1) * (1 - df["availability_365_norm"] + 1e-6))
df["value_signal"] = cap_and_normalize((1 - df["price_per_guest_norm"]) * (df["amenities_weight_norm"] + 0.1))
df["capacity_efficiency"] = cap_and_normalize(0.5 * df["bathrooms_norm"] + 0.5 * df["bedrooms_norm"] - 0.2 * df["accommodates_norm"])

df["latitude_norm"] = cap_and_normalize(df["latitude"].fillna(df["latitude"].median()))
df["longitude_norm"] = cap_and_normalize(df["longitude"].fillna(df["longitude"].median()))
df["neighbourhood_listing_density"] = cap_and_normalize(
    df["neighbourhood_cleansed"].fillna("Unknown").map(df["neighbourhood_cleansed"].fillna("Unknown").value_counts())
)
df["room_type_encoded"] = df["room_type"].map({
    "Entire home/apt": 1.0,
    "Private room": 0.5,
    "Hotel room": 0.75,
    "Shared room": 0.25,
}).fillna(0.5)

top_property_types = df["property_type"].fillna("Unknown").value_counts().head(12).index
df["property_type_grouped"] = df["property_type"].fillna("Unknown")
df.loc[~df["property_type_grouped"].isin(top_property_types), "property_type_grouped"] = "Other"
property_type_codes = {
    name: idx / max(len(top_property_types), 1)
    for idx, name in enumerate(sorted(set(df["property_type_grouped"])))
}
df["property_type_encoded"] = df["property_type_grouped"].map(property_type_codes).fillna(0)

df["bathrooms_text_length"] = cap_and_normalize(text_length_feature(df["bathrooms_text"]))
df["name_length_norm"] = cap_and_normalize(text_length_feature(df["name"]))
df["description_length_norm"] = cap_and_normalize(text_length_feature(df["description"]))
df["host_about_length_norm"] = cap_and_normalize(text_length_feature(df["host_about"]))

host_since = pd.to_datetime(df["host_since"], errors="coerce")
host_tenure_days = (pd.Timestamp("2026-04-16") - host_since).dt.days
df["host_tenure_norm"] = cap_and_normalize(np.log1p(host_tenure_days.fillna(host_tenure_days.median())))
df["host_listings_count_norm"] = cap_and_normalize(np.log1p(df["host_listings_count"].fillna(df["host_listings_count"].median())))
df["host_total_listings_count_norm"] = cap_and_normalize(np.log1p(df["host_total_listings_count"].fillna(df["host_total_listings_count"].median())))
df["calculated_host_listings_count_norm"] = cap_and_normalize(np.log1p(df["calculated_host_listings_count"].fillna(df["calculated_host_listings_count"].median())))
df["host_verification_count_norm"] = cap_and_normalize(df["host_verifications"].fillna("").astype(str).str.count("'") / 2)

df["number_of_reviews_l30d_norm"] = cap_and_normalize(np.log1p(df["number_of_reviews_l30d"].fillna(0)))
df["number_of_reviews_ly_norm"] = cap_and_normalize(np.log1p(df["number_of_reviews_ly"].fillna(0)))
df["availability_eoy_norm"] = cap_and_normalize(df["availability_eoy"].fillna(df["availability_365"]))
df["estimated_occupancy_l365d_norm"] = cap_and_normalize(df["estimated_occupancy_l365d"].fillna(df["estimated_occupancy_l365d"].median()))
df["estimated_revenue_l365d_norm"] = cap_and_normalize(np.log1p(df["estimated_revenue_l365d"].fillna(df["estimated_revenue_l365d"].median())))
df["recent_review_momentum"] = cap_and_normalize(np.log1p(df["number_of_reviews_l30d"].fillna(0)) + np.log1p(df["number_of_reviews_ly"].fillna(0)))
df["host_scale_signal"] = cap_and_normalize(np.log1p(df["host_total_listings_count"].fillna(df["host_total_listings_count"].median())) * (0.5 + df["host_is_superhost_filled"]))
df["geo_price_signal"] = cap_and_normalize(df["price_norm"] * (0.5 + df["latitude_norm"]) * (0.5 + df["longitude_norm"]))

selected_columns = [
    "privacy_type", "is_apartment", "is_house", "is_nature", "is_unique", "is_hotel",
    "accommodates_norm", "bathrooms_norm", "bedrooms_norm", "beds_norm",
    "price_norm", "price_per_guest_norm", "price_originally_empty",
    "amenities_weight_norm", "minimum_nights_norm", "availability_365_norm",
    "host_response_rate_filled", "host_acceptance_rate_filled", "host_response_time_filled",
    "host_is_superhost_filled", "host_identity_verified_filled",
    "number_of_reviews_log", "number_of_reviews_ltm_log", "reviews_per_month_log",
    "has_reviews_filled", "review_activity_score", "beds_per_bedroom_norm",
    "superhost_response_combo", "review_recency_ratio_norm", "response_acceptance_gap_norm",
    "host_reliability_score", "demand_pressure", "value_signal", "capacity_efficiency",
    "latitude_norm", "longitude_norm", "neighbourhood_listing_density",
    "property_type_encoded", "room_type_encoded", "bathrooms_text_length",
    "name_length_norm", "description_length_norm", "host_about_length_norm",
    "host_tenure_norm", "host_has_profile_pic_filled", "instant_bookable_filled",
    "host_listings_count_norm", "host_total_listings_count_norm",
    "calculated_host_listings_count_norm", "host_verification_count_norm",
    "number_of_reviews_l30d_norm", "number_of_reviews_ly_norm", "availability_eoy_norm",
    "estimated_occupancy_l365d_norm", "estimated_revenue_l365d_norm",
    "recent_review_momentum", "host_scale_signal", "geo_price_signal",
    "review_scores_rating",
]

df_clean = df.loc[df["review_scores_rating"].notna(), selected_columns].copy()
os.makedirs("outputs", exist_ok=True)
df_clean.to_csv("outputs/processed.csv", index=False)
task.upload_artifact("processed_dataset", "outputs/processed.csv")
print("Preprocessing complete")
task.flush(wait_for_uploads=True)
task.close()
