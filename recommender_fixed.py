# Databricks notebook source
# MAGIC %md
# MAGIC # Gaming Recommender System (revised)
# MAGIC
# MAGIC Changes from the first version, and why:
# MAGIC
# MAGIC | # | Problem in v1 | Fix |
# MAGIC |---|---------------|-----|
# MAGIC | 1 | `Game ID` built from `uuid()` + `hash()`: random, lazily re-evaluated, collision-prone | Deterministic `dense_rank` over the game name |
# MAGIC | 2 | Purchase rows (`Hours = 1`) mixed with real play hours | Train on `play` rows only; keep purchases only to know what a player owns |
# MAGIC | 3 | Raw hours are heavily skewed | Use `log1p(hours)` as the implicit-feedback strength |
# MAGIC | 4 | RMSE against hours is meaningless for implicit ALS | Ranking metrics (Precision@K, Recall@K) on held-out games, plus a popularity baseline |
# MAGIC | 5 | Default `coldStartStrategy="nan"` and a null check that can't see NaN | `coldStartStrategy="drop"` |
# MAGIC | 6 | Recommendations could include games the player already has | Anti-join against what the player has seen/owns |
# MAGIC | 7 | Hyperparameters picked by RMSE | Grid search scored by Precision@10 on a validation split |

# COMMAND ----------

import logging
import mlflow
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.ml.recommendation import ALS

logging.getLogger("mlflow").setLevel(logging.ERROR)
mlflow.pyspark.ml.autolog()

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
nb = ctx.notebookPath().get().split("/")[2]

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
experiment_name = f"/Shared/{nb}_recommender_system"
if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(name=experiment_name)
mlflow.set_experiment(experiment_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load the data
# MAGIC `games.csv` was loaded but never used in v1 (every game name already appears in `steam-200k.csv`), so it is dropped here.

# COMMAND ----------

path = "/Volumes/teaching/datasets/assignment_2/task_2/"

activitySchema = StructType([
    StructField("player_id", StringType()),   # unique player identifier
    StructField("game", StringType()),        # game name
    StructField("action", StringType()),      # 'purchase' or 'play'
    StructField("hours", DoubleType()),       # hours played (always 1 for 'purchase')
])

activityDf = (
    spark.read.options(delimiter=",")
    .schema(activitySchema)
    .csv(f"{path}steam-200k.csv", header=False, lineSep="\r\n")
    .withColumn("player_id", F.col("player_id").cast("int"))
)
activityDf.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Preprocessing
# MAGIC
# MAGIC **Game IDs (fix 1).** v1 used `hash(first4chars + uuid())`. `uuid()` is non-deterministic and Spark is lazy,
# MAGIC so the mapping can be *recomputed with different values* each time it is used, and then IDs in the model and the
# MAGIC lookup table no longer match. The `% 90000000` also allowed collisions.
# MAGIC `dense_rank()` over the sorted game name gives each distinct name one stable integer, with no collisions.

# COMMAND ----------

gamesWithID = (
    activityDf.select("game").distinct()
    .withColumn("game_id", F.dense_rank().over(Window.orderBy("game")).cast("int"))
    .cache()
)
gamesWithID.count()  # materialise once

activityDf = activityDf.join(gamesWithID, on="game", how="left")

# COMMAND ----------

# MAGIC %md
# MAGIC **Interactions vs ownership (fixes 2 and 3).**
# MAGIC
# MAGIC * `owned`: every (player, game) pair that appears at all (purchase or play). Used later to avoid recommending games a player already has.
# MAGIC * `interactions`: only `play` rows with `hours > 0`, summed per pair so each (player, game) appears once.
# MAGIC   Purchase rows are excluded because their "1" is not hours.
# MAGIC * `strength = log1p(hours)`: a few players have thousands of hours, which would otherwise dominate the loss.

# COMMAND ----------

owned = activityDf.select("player_id", "game_id").distinct().cache()

interactions = (
    activityDf.filter((F.col("action") == "play") & (F.col("hours") > 0))
    .groupBy("player_id", "game_id")
    .agg(F.sum("hours").alias("hours"))
    .withColumn("strength", F.log1p("hours"))
    .select("player_id", "game_id", "strength")
    .cache()
)
print("play interactions:", interactions.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Train / validation / test split
# MAGIC Random row-level split (70/15/15) of the played games. Each held-out row is a game the player really played but the
# MAGIC model never saw, which is what we need to measure ranking quality.

# COMMAND ----------

train, val, test = interactions.randomSplit([0.7, 0.15, 0.15], seed=100)
train, val, test = train.cache(), val.cache(), test.cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Evaluation helpers (fix 4)
# MAGIC RMSE compares ALS's preference score (roughly 0 to 1) with raw hours, so it says nothing about recommendation quality.
# MAGIC Instead we ask: *of the top-K games we recommend, how many did the player actually go on to play?*
# MAGIC
# MAGIC * **Precision@K** = hits / K
# MAGIC * **Recall@K** = hits / (number of held-out games for that player)
# MAGIC
# MAGIC Recommended games the player already saw in training are removed first (fix 6), otherwise the model scores by
# MAGIC "recommending" what it was trained on.

# COMMAND ----------

K = 10
CANDIDATES = 200  # ALS candidates per user before filtering seen games (must be > K + most games any one user has seen)


def als_recs(model, seen, n=CANDIDATES):
    """Top-n ALS candidates per user, minus games in `seen`, ranked 1..n."""
    raw = model.recommendForAllUsers(n)
    exploded = raw.select(
        "player_id", F.explode("recommendations").alias("r")
    ).select(
        "player_id", F.col("r.game_id").alias("game_id"), F.col("r.rating").alias("score")
    )
    unseen = exploded.join(seen, ["player_id", "game_id"], "left_anti")
    w = Window.partitionBy("player_id").orderBy(F.col("score").desc())
    return unseen.withColumn("rank", F.row_number().over(w))


def popularity_recs(train_df, seen, n=CANDIDATES):
    """Non-personalised baseline: the most-played games overall, minus what each user has seen."""
    top = (
        train_df.groupBy("game_id").agg(F.count("*").alias("score"))
        .orderBy(F.col("score").desc()).limit(n)
    )
    users = train_df.select("player_id").distinct()
    cand = users.crossJoin(top).join(seen, ["player_id", "game_id"], "left_anti")
    w = Window.partitionBy("player_id").orderBy(F.col("score").desc())
    return cand.withColumn("rank", F.row_number().over(w))


def precision_recall_at_k(recs, heldout, k=K):
    """Mean Precision@k / Recall@k over users that have recommendations AND held-out games."""
    users = recs.select("player_id").distinct()
    relevant = heldout.join(users, "player_id", "left_semi")
    hits = (
        recs.filter(F.col("rank") <= k)
        .join(relevant.select("player_id", "game_id").withColumn("hit", F.lit(1)),
              ["player_id", "game_id"], "inner")
        .groupBy("player_id").agg(F.sum("hit").alias("hits"))
    )
    per_user = (
        relevant.groupBy("player_id").agg(F.count("*").alias("n_relevant"))
        .join(hits, "player_id", "left").fillna(0, subset=["hits"])
        .select(
            (F.col("hits") / k).alias("precision"),
            (F.col("hits") / F.col("n_relevant")).alias("recall"),
        )
    )
    row = per_user.agg(F.avg("precision").alias("p"), F.avg("recall").alias("r")).first()
    return row["p"], row["r"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Baseline: always recommend the most popular games
# MAGIC A recommender that cannot beat this is not adding value, so we measure it first on the validation set.

# COMMAND ----------

p, r = precision_recall_at_k(popularity_recs(train, seen=train), val)
print(f"Popularity baseline (validation)  P@{K}={p:.4f}  R@{K}={r:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. ALS and hyperparameter search (fixes 5 and 7)
# MAGIC * `coldStartStrategy="drop"`: unseen users/games return NaN by default (not null), which poisons averages. Dropping them is explicit.
# MAGIC * `rank` is now tuned too, and `alpha` uses values suited to log-scaled strengths. The old grid (0.01-1.0) barely weighted observed interactions.
# MAGIC * Each combination is trained on `train` and scored by Precision@10 on `val`. Plain loop instead of `TrainValidationSplit`
# MAGIC   because Spark's built-in evaluators don't do per-user top-K.

# COMMAND ----------

import itertools

grid = list(itertools.product([10, 20], [0.01, 0.1], [10.0, 40.0]))  # rank, regParam, alpha
results = []

for rank, reg, alpha in grid:
    als = ALS(
        userCol="player_id", itemCol="game_id", ratingCol="strength",
        implicitPrefs=True, coldStartStrategy="drop", nonnegative=True,
        maxIter=10, rank=rank, regParam=reg, alpha=alpha, seed=100,
    )
    with mlflow.start_run(run_name=f"als_r{rank}_reg{reg}_a{alpha}"):
        model = als.fit(train)
        p, r = precision_recall_at_k(als_recs(model, seen=train), val)
        mlflow.log_metrics({f"val_precision_at_{K}": p, f"val_recall_at_{K}": r})
    results.append((p, r, rank, reg, alpha))
    print(f"rank={rank:>2} regParam={reg:<5} alpha={alpha:<5}  P@{K}={p:.4f}  R@{K}={r:.4f}")

best_p, best_r, best_rank, best_reg, best_alpha = max(results)
print(f"\nBest: rank={best_rank} regParam={best_reg} alpha={best_alpha}  (val P@{K}={best_p:.4f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Final evaluation on the untouched test set
# MAGIC Refit the best configuration on train + validation, then score once on `test`. Seen games = everything in train + validation.

# COMMAND ----------

train_val = train.union(val).cache()

final_als = ALS(
    userCol="player_id", itemCol="game_id", ratingCol="strength",
    implicitPrefs=True, coldStartStrategy="drop", nonnegative=True,
    maxIter=10, rank=best_rank, regParam=best_reg, alpha=best_alpha, seed=100,
)
with mlflow.start_run(run_name="als_final"):
    final_model = final_als.fit(train_val)
    p, r = precision_recall_at_k(als_recs(final_model, seen=train_val), test)
    pb, rb = precision_recall_at_k(popularity_recs(train_val, seen=train_val), test)
    mlflow.log_metrics({
        f"test_precision_at_{K}": p, f"test_recall_at_{K}": r,
        f"baseline_precision_at_{K}": pb, f"baseline_recall_at_{K}": rb,
    })

print(f"ALS         (test)  P@{K}={p:.4f}  R@{K}={r:.4f}")
print(f"Popularity  (test)  P@{K}={pb:.4f}  R@{K}={rb:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Recommendations for one player
# MAGIC Uses a widget instead of `input()` (which blocks in non-interactive runs), and excludes every game the player
# MAGIC already owns (purchased or played), not just those used in training.

# COMMAND ----------

dbutils.widgets.text("player_id", "", "Player ID")
player = dbutils.widgets.get("player_id")

if player:
    pid = int(player)
    recs = (
        als_recs(final_model, seen=owned)
        .filter(F.col("player_id") == pid)
        .filter(F.col("rank") <= K)
        .join(gamesWithID, "game_id")
        .select("rank", "game", "score")
        .orderBy("rank")
    )
    recs.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Trending games (non-personalised fallback, e.g. for brand-new players)
# MAGIC Counts **distinct players** and keeps the average hours alongside the play/purchase ratio.
# MAGIC The ratio can exceed 100% because some games are played without a purchase record (for example free-to-play titles),
# MAGIC so it measures engagement, not conversion, and should not be read as a strict percentage.

# COMMAND ----------

gameStats = activityDf.groupBy("game").agg(
    F.sum(F.when(F.col("action") == "purchase", 1).otherwise(0)).alias("purchases"),
    F.sum(F.when(F.col("action") == "play", 1).otherwise(0)).alias("plays"),
    F.round(F.avg(F.when(F.col("action") == "play", F.col("hours"))), 1).alias("avg_hours"),
)

trending = (
    gameStats.filter(F.col("plays") > 500)
    .withColumn("play_to_purchase_pct", F.round(F.col("plays") / F.col("purchases") * 100, 2))
    .orderBy(F.col("play_to_purchase_pct").desc())
)
trending.display()
