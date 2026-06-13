import argparse
import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader

SVD_PARAMS = {
    "lr_all":    0.0062939,
    "n_epochs":  69,
    "n_factors": 215,
    "reg_bi":    0.31902932,
    "reg_bu":    0.03736959,
    "reg_pu":    0.0458803,
    "reg_qi":    0.0457921065,
}

def parse_args():
    p = argparse.ArgumentParser(description="Train community SVD model offline")
    p.add_argument("--data",       default="data/ratings.csv")
    p.add_argument("--out",        default="community_model.pkl",
                   help="Output path for trained model")
    p.add_argument("--sample",     type=int, default=0,
                   help="Sample size (0 = full dataset)")
    p.add_argument("--min-ratings",type=int, default=200,
                   help="Min ratings per film")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.data):
        sys.exit(f"Ratings file not found: {args.data}")

    print(f"Loading {args.data} ...")
    df = pd.read_csv(args.data, low_memory=False)
    df = df[["user_id", "movie_id", "rating_val"]].dropna()
    df["rating_val"] = pd.to_numeric(df["rating_val"], errors="coerce").clip(1, 10)
    df.dropna(subset=["rating_val"], inplace=True)
    df["rating_val"] = df["rating_val"].astype(int)
    print(f"  {len(df):,} ratings loaded")

    if args.min_ratings > 0:
        counts   = df["movie_id"].value_counts()
        eligible = counts[counts >= args.min_ratings].index
        before   = df["movie_id"].nunique()
        df       = df[df["movie_id"].isin(eligible)]
        print(f"  Min {args.min_ratings} filter: {df['movie_id'].nunique():,} / {before:,} films")

    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=42)
        print(f"  Sampled {args.sample:,} ratings")

    print(f"\nTraining SVD on {len(df):,} ratings, {df['movie_id'].nunique():,} films ...")
    random.seed(42); np.random.seed(42)

    reader = Reader(rating_scale=(1, 10))
    data   = Dataset.load_from_df(df[["user_id", "movie_id", "rating_val"]], reader)
    algo   = SVD(**SVD_PARAMS, random_state=42)
    algo.fit(data.build_full_trainset())

    bundle = {
        "key":          ("community_v1", len(df)),
        "algo":         algo,
        "film_list":    df["movie_id"].unique().tolist(),
    }

    with open(args.out, "wb") as f:
        pickle.dump(bundle, f)

    size_mb = os.path.getsize(args.out) / 1024 / 1024
    print(f"\nSaved to '{args.out}' ({size_mb:.1f} MB)")
    print(f"Films in model: {len(bundle['film_list']):,}")
    print("\nNext steps:")
    print(f"  Upload to S3:  aws s3 cp {args.out} s3://your-bucket/models/community_model.pkl")
    print(f"  Or R2:         rclone copy {args.out} r2:your-bucket/models/")


if __name__ == "__main__":
    main()