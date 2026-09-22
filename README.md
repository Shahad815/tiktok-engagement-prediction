# TikTok Video Engagement Prediction — Day 30 Views

**WeCloudData DS Bootcamp — In-Class Competition**

Predicts the cumulative view count a TikTok video will reach at Day 30,
using only video metadata, creator statistics, and daily engagement
metrics from the first 5 days after posting (no data leakage).

## Approach
Instead of predicting the raw view count directly, the model predicts
the **growth ratio** (target views / day-5 views) in log-space using
LightGBM with a Huber loss. This handles the heavy power-law
distribution of the target much better than predicting raw views.

**Key steps:**
1. Merge video metadata, daily engagement (days 0–5), and creator stats
2. Forward-fill missing days, build ratio/growth features
3. Add creator features (followers, avg likes/video) as of day 5
4. Train LightGBM (Huber loss, α=0.2) to predict log(growth ratio)
5. Validate with GroupKFold on `author_id` (no creator leakage)
6. Train final model on all data (5 seeds averaged), predict on test set

## Results

| Model | RMSE |
|---|---|
| Mean baseline | 324,033 |
| Day-5 × median growth | 145,815 |
| LightGBM (log-target) | 173,415 |
| LightGBM (log-ratio, L2) | 128,291 |
| **LightGBM (log-ratio, Huber α=0.2)** | **59,197 (CV)** |

**Kaggle Public Score:** 86,303.49

## Key Findings
- The target follows a heavy power-law distribution — a few viral
  videos dominate the RMSE (~70% of squared error from ~20 videos).
- Predicting the growth ratio in log-space, rather than the raw
  target, was the single biggest improvement.
- Huber loss (less sensitive to outliers than L2) further improved
  results over standard L2 regression.
- Creator features gave a small, consistent improvement on the
  log-scale metric.

## Files
- `TikTok_Engagement_Prediction_Final.ipynb` — full notebook (EDA,
  feature engineering, model training, CV, final predictions)
- `submission.csv` — final Kaggle submission

## How to Run
1. Open the notebook in Google Colab
2. Download the competition data zip from Kaggle
3. Run all cells top to bottom — it will prompt you to upload the zip
   and will reproduce the exact final submission
