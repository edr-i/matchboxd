import argparse
import os
import sys
import time
import pickle
import random

import numpy as np
import pandas as pd
import requests

TMDB_BASE = "https://api.themoviedb.org/3"

                                                                                

def parse_args():
    p = argparse.ArgumentParser(description="Letterboxd recommender (CF / content / hybrid)")
    p.add_argument("--user",     required=True,          help="Letterboxd username (first user)")
    p.add_argument("--user2",    default=None,           help="Second username for blend mode")
    p.add_argument("--mode",     default="cf",
                   choices=["cf", "content", "hybrid"],  help="Recommendation mode")
    p.add_argument("--data",     default="data/ratings.csv",
                                                         help="Community ratings CSV (cf/hybrid)")
    p.add_argument("--sample",   type=int, default=500_000,
                                                         help="Community ratings sample size")
    p.add_argument("--movie-data", default="data/movie_data.csv",
                                                         help="Path to movie_data.csv (content/hybrid)")
    p.add_argument("--tmdb-key", default=None,
                                                         help="TMDB API key to supplement missing recent films (optional)")
    p.add_argument("--alpha",    type=float, default=0.7,
                                                         help="CF weight in hybrid (0–1)")
    p.add_argument("--top",      type=int, default=25,   help="Number of recommendations")
    p.add_argument("--cache",    default=".cache_community.pkl",
                                                         help="Community model cache file")
    p.add_argument("--cache-content", default=".cache_content.pkl",
                                                         help="Content model cache file")
    p.add_argument("--no-cache", action="store_true",    help="Retrain ignoring cache")
    p.add_argument("--watchlist-boost", type=float, default=0.5,
                                                         help="Score bonus for watchlisted films")
    p.add_argument("--decades",  type=int, nargs="+", default=None,
                                                         help="Only show films from these decades e.g. --decades 1990 2000 2010 2020")
    p.add_argument("--min-ratings", type=int, default=200,
                                                         help="Min community ratings a film must have to be recommended (default 200)")
    p.add_argument("--min-runtime", type=int, default=40,
                                                         help="Min runtime in minutes (default 40, filters shorts/concerts)")
    p.add_argument("--content-pool",  type=int, default=5000,
                                                         help="# candidate films to score in content mode (default 5000)")
    args = p.parse_args()

    if args.mode in ("content", "hybrid") and not os.path.exists(args.movie_data):
        p.error(f"Movie metadata file not found: {args.movie_data}\n"
                "Expected data/movie_data.csv from the Kaggle dataset.")
    if args.mode in ("cf", "hybrid") and not os.path.exists(args.data):
        p.error(f"Community ratings file not found: {args.data}\n"
                "Download from: kaggle datasets download samlearner/letterboxd-movie-ratings-data")
    return args

                                                                                

def fetch_user_data(username: str, include_favorites: bool = False):
    print(f"\n[1] Fetching Letterboxd data for '{username}' ...")
    from letterboxdpy.user import User

    user      = User(username)
    films_data = user.get_films()
    movies    = films_data.get("movies", {})
    rated     = {slug: m for slug, m in movies.items() if m.get("rating") is not None}

    ratings = []
    for slug, m in rated.items():
        val = int(round(float(m["rating"]) * 2))
        val = max(1, min(10, val))
        ratings.append({"user_id": username, "movie_id": slug, "rating_val": val})

    print(f"    Rated films : {len(ratings)}")
    print(f"    Avg rating  : {films_data.get('rating_average', 'n/a')}")

    try:
        wl_data   = user.get_watchlist()
        wl_items  = wl_data.get("data", {})
        watchlist = set()
        for item in wl_items.values():
            slug = item.get("slug") or item.get("film_slug") or item.get("movie_slug")
            if slug:
                watchlist.add(slug)
        if not watchlist:
            watchlist = set(wl_items.keys())
        print(f"    Watchlist   : {len(watchlist)} films")
    except Exception as e:
        print(f"    Watchlist fetch failed ({e}), skipping.")
        watchlist = set()

    if len(ratings) < 20:
        print("  ⚠  Fewer than 20 rated films — quality will be limited.")

    if not include_favorites:
        return ratings, watchlist

    favorites = []
    try:
        favs = user.get_favorites()
        favorites = [{"slug": v["slug"], "name": v["name"], "year": v.get("year", 0)}
                     for v in favs.values()]
        print(f"    Favorites   : {len(favorites)} films")
    except Exception as e:
        print(f"    Favorites fetch failed ({e}), skipping.")

    return ratings, watchlist, favorites

                                                                               

def merge_user_ratings(ratings1: list[dict], ratings2: list[dict],
                       user1: str, user2: str) -> list[dict]:
    map1 = {r["movie_id"]: r["rating_val"] for r in ratings1}
    map2 = {r["movie_id"]: r["rating_val"] for r in ratings2}
    all_slugs = set(map1) | set(map2)
    merged = []
    for slug in all_slugs:
        if slug in map1 and slug in map2:
            val = round((map1[slug] + map2[slug]) / 2)
        elif slug in map1:
            val = map1[slug]
        else:
            val = map2[slug]
        merged.append({"user_id": f"{user1}+{user2}", "movie_id": slug, "rating_val": val})
    return merged


def blend_cf_scores(algo, user1: str, user2: str,
                    candidates: list[str]) -> dict[str, float]:
    scores = {}
    for movie_id in candidates:
        s1 = algo.predict(user1, movie_id).est
        s2 = algo.predict(user2, movie_id).est
        scores[movie_id] = (s1 + s2) / 2
    return scores


def load_community_ratings(path: str, sample_size: int, min_ratings: int = 50) -> pd.DataFrame:
    print(f"\n    Loading community ratings from '{path}' ...")
    df = pd.read_csv(path, low_memory=False)

                                                        
    RENAME = {}                                                       
    if RENAME:
        df.rename(columns=RENAME, inplace=True)

    required = {"user_id", "movie_id", "rating_val"}
    missing  = required - set(df.columns)
    if missing:
        sys.exit(f"❌  CSV missing columns: {missing}. Found: {list(df.columns)}")

    df = df[["user_id", "movie_id", "rating_val"]].dropna()
    df["rating_val"] = pd.to_numeric(df["rating_val"], errors="coerce").clip(1, 10)
    df.dropna(subset=["rating_val"], inplace=True)
    df["rating_val"] = df["rating_val"].astype(int)

    total = len(df)
    if sample_size and sample_size < total:
        df = df.sample(n=sample_size, random_state=42)
        print(f"    Sampled {sample_size:,} / {total:,} ratings")
    else:
        print(f"    Loaded {total:,} ratings")

    if min_ratings > 0:
        counts = df["movie_id"].value_counts()
        eligible = counts[counts >= min_ratings].index
        before = df["movie_id"].nunique()
        df = df[df["movie_id"].isin(eligible)]
        print(f"    Min {min_ratings} ratings filter: {df['movie_id'].nunique():,} / {before:,} films kept")

    return df.reset_index(drop=True)

                                                                                

SVD_PARAMS = {
    "lr_all":    0.0062939,
    "n_epochs":  69,
    "n_factors": 215,
    "reg_bi":    0.31902932,
    "reg_bu":    0.03736959,
    "reg_pu":    0.0458803,
    "reg_qi":    0.0457921065,
}

def train_community_model(community_df: pd.DataFrame,
                          cache_path: str, no_cache: bool):
    from surprise import SVD, Dataset, Reader

    cache_key = ("community_v1", len(community_df))
    if not no_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("key") == cache_key:
                print("\n[3-CF] Loaded cached community SVD model ✓")
                return cached["algo"]
        except Exception:
            pass

    print(f"\n[3-CF] Training community SVD model on {len(community_df):,} ratings ...")
    reader = Reader(rating_scale=(1, 10))
    data   = Dataset.load_from_df(
                community_df[["user_id", "movie_id", "rating_val"]], reader)

    random.seed(42); np.random.seed(42)
    algo = SVD(**SVD_PARAMS, random_state=42)
    algo.fit(data.build_full_trainset())

    with open(cache_path, "wb") as f:
        pickle.dump({"key": cache_key, "algo": algo}, f)
    print(f"    Model cached to '{cache_path}' ✓")
    return algo


def fold_in_user(algo, user_ratings: list[dict],
                 n_epochs: int = 50, lr: float = 0.005, reg: float = 0.02
                 ) -> tuple[np.ndarray, float]:
    mu  = algo.trainset.global_mean
    pu  = np.zeros(algo.n_factors, dtype=np.float64)
    bu  = 0.0
    r2i = algo.trainset._raw2inner_id_items

    known = [(r2i[r["movie_id"]], float(r["rating_val"]))
             for r in user_ratings if r["movie_id"] in r2i]

    if not known:
        return pu, bu

    for _ in range(n_epochs):
        for (iid, r_ui) in known:
            qi  = algo.qi[iid]
            err = r_ui - (mu + bu + algo.bi[iid] + np.dot(pu, qi))
            pu  += lr * (err * qi - reg * pu)
            bu  += lr * (err      - reg * bu)

    return pu, bu


def predict_folded(algo, pu: np.ndarray, bu: float,
                   candidates: list[str]) -> dict[str, float]:
    mu   = algo.trainset.global_mean
    r2i  = algo.trainset._raw2inner_id_items
    known = [c for c in candidates if c in r2i]
    if not known:
        return {c: mu for c in candidates}
    iids   = np.array([r2i[c] for c in known])
    scores = mu + bu + algo.bi[iids] + algo.qi[iids] @ pu
    scores = np.clip(scores, 1, 10)
    result = {c: mu for c in candidates}
    result.update(zip(known, scores.tolist()))
    return result


def predict_cf_blend(algo, ratings1: list[dict], ratings2: list[dict],
                     candidates: list[str]) -> dict[str, float]:
    pu1, bu1 = fold_in_user(algo, ratings1)
    pu2, bu2 = fold_in_user(algo, ratings2)
    s1 = predict_folded(algo, pu1, bu1, candidates)
    s2 = predict_folded(algo, pu2, bu2, candidates)
    return {c: (s1[c] + s2[c]) / 2 for c in candidates}

                                                                               

def load_movie_metadata(path: str) -> pd.DataFrame:
    print(f"    Loading movie metadata from '{path}' ...")
    cols = ["movie_id", "movie_title", "genres", "original_language",
            "runtime", "year_released", "overview", "tmdb_id"]
    header = pd.read_csv(path, nrows=0, engine="python")
    if "image_url" in header.columns:
        cols.append("image_url")
    df = pd.read_csv(path, engine="python", on_bad_lines="skip", usecols=cols)
    df["movie_id"] = df["movie_id"].astype(str).str.strip()
    df.drop_duplicates(subset="movie_id", inplace=True)
    df["director"] = ""
    if "image_url" in df.columns:
        df["poster_url"] = df["image_url"].apply(
            lambda x: f"https://a.ltrbxd.com/resized/{x}.jpg"
                      if pd.notna(x) and str(x).strip() else ""
        )
        df.drop(columns=["image_url"], inplace=True)
    else:
        df["poster_url"] = ""
    df.reset_index(drop=True, inplace=True)
    print(f"    {len(df):,} films loaded")
    return df

def _tmdb_fetch_one(slug: str, api_key: str) -> dict | None:
    """
    Fetches full metadata for one slug from TMDB.
    If slug is a numeric TMDB ID, skips the search step.
    Returns a row dict or None on failure.
    """
    try:
        if slug.isdigit():
            tmdb_id = int(slug)
        else:
            title_guess = slug.replace("-", " ").strip()
            parts = title_guess.rsplit(" ", 1)
            year  = None
            if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 4:
                title_guess, year = parts[0], parts[1]
            params = {"query": title_guess, "api_key": api_key}
            if year:
                params["year"] = year
            r       = requests.get(f"{TMDB_BASE}/search/movie", params=params, timeout=10)
            results = r.json().get("results", [])
            if not results:
                return None
            tmdb_id = results[0]["id"]

        r2 = requests.get(f"{TMDB_BASE}/movie/{tmdb_id}",
                          params={"api_key": api_key,
                                  "append_to_response": "credits"}, timeout=10)
        d  = r2.json()
        if "id" not in d:
            return None

        genres    = str([g["name"] for g in d.get("genres", [])])
        directors = [c["name"] for c in d.get("credits", {}).get("crew", [])
                     if c.get("job") == "Director"]
        poster    = d.get("poster_path") or ""
        if poster:
            poster = f"https://image.tmdb.org/t/p/w500{poster}"

        return {
            "movie_id":          slug,
            "tmdb_id":           tmdb_id,
            "movie_title":       d.get("title", "") or "",
            "genres":            genres,
            "original_language": d.get("original_language", ""),
            "runtime":           d.get("runtime", 0) or 0,
            "year_released":     int((d.get("release_date") or "0")[:4] or 0),
            "overview":          d.get("overview", "") or "",
            "director":          ", ".join(directors),
            "poster_url":        poster,
        }
    except Exception:
        return None


def supplement_from_tmdb(missing_slugs: list[str], api_key: str,
                          existing_df: pd.DataFrame,
                          workers: int = 20) -> pd.DataFrame:
    if not missing_slugs or not api_key:
        return pd.DataFrame(columns=existing_df.columns)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    print(f"    Fetching TMDB metadata for {len(missing_slugs)} films (concurrent) ...")
    rows  = []
    count = [0]
    lock  = threading.Lock()

    def fetch(slug):
        row = _tmdb_fetch_one(slug, api_key)
        if row:
            with lock:
                rows.append(row)
                count[0] += 1
                if count[0] % 20 == 0:
                    print(f"      {count[0]}/{len(missing_slugs)} ...")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, s) for s in missing_slugs]
        for f in as_completed(futures):
            try: f.result()
            except Exception: pass

    if rows:
        print(f"    Supplemented {len(rows)} / {len(missing_slugs)} films ✓")
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=existing_df.columns)

def parse_genres(raw) -> list[str]:
    import json
    if raw is None or isinstance(raw, float):
        return []
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        result = json.loads(raw)
        return result if isinstance(result, list) else []
    except Exception:
        return [g.strip().strip('"\'') for g in raw.strip("[]").split(",") if g.strip()]

def build_tfidf_features(meta_df: pd.DataFrame,
                          rated_slugs: list[str],
                          all_slugs: list[str]) -> tuple:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import hstack, csr_matrix

    idx = meta_df.set_index("movie_id")

                                                                                
    all_genres = sorted({g
                         for raw in meta_df["genres"]
                         for g in parse_genres(raw)})
    genre_index = {g: i for i, g in enumerate(all_genres)}

                                                                                
    all_langs  = sorted(meta_df["original_language"].dropna().unique())
    lang_index = {l: i for i, l in enumerate(all_langs)}

    def row_for(slug):
        if slug not in idx.index:
            return None
        r       = idx.loc[slug]
        genres  = parse_genres(r.get("genres", ""))
        raw_lang = r.get("original_language", "")
        lang    = "" if pd.isna(raw_lang) else str(raw_lang)
        raw_runtime = r.get("runtime", 0)
        raw_year    = r.get("year_released", 2000)
        runtime = 0.0   if pd.isna(raw_runtime) else float(raw_runtime)
        year    = 2000.0 if pd.isna(raw_year)   else float(raw_year)
        raw_overview = r.get("overview", "")
        overview = "" if pd.isna(raw_overview) else str(raw_overview)

        g_vec = [0.0] * len(all_genres)
        for g in genres:
            if g in genre_index:
                g_vec[genre_index[g]] = 1.0

        l_vec = [0.0] * len(all_langs)
        if lang in lang_index:
            l_vec[lang_index[lang]] = 1.0

        numeric = [runtime / 200.0, (year - 1900) / 120.0]
        return g_vec + l_vec + numeric, overview

                                                               
    def build_rows(slugs):
        dense_rows, texts = [], []
        valid_slugs = []
        for s in slugs:
            result = row_for(s)
            if result is None:
                continue
            dense_rows.append(result[0])
            texts.append(result[1])
            valid_slugs.append(s)
        return dense_rows, texts, valid_slugs

    rated_dense, rated_texts, rated_valid = build_rows(rated_slugs)
    all_dense,   all_texts,   all_valid   = build_rows(all_slugs)

                                                          
    tfidf = TfidfVectorizer(max_features=200, stop_words="english")
    tfidf.fit(all_texts + rated_texts)

    X_rated = hstack([csr_matrix(rated_dense), tfidf.transform(rated_texts)])
    X_all   = hstack([csr_matrix(all_dense),   tfidf.transform(all_texts)])

    return X_rated, X_all, rated_valid, all_valid

def build_feature_vector(meta: dict, all_tags: list[str],
                         tag_index: dict[str, int]) -> np.ndarray:
    vec = np.zeros(len(all_tags) + 2, dtype=np.float32)
    tags = (meta.get("genres", []) + meta.get("keywords", []) +
            meta.get("cast", [])   + meta.get("directors", []) +
            ([meta.get("language", "")] if meta.get("language") else []))
    for tag in tags:
        idx = tag_index.get(tag)
        if idx is not None:
            vec[idx] = 1.0
    vec[-2] = meta.get("runtime", 0) / 200.0                      
    vec[-1] = (meta.get("year", 2000) - 1900) / 120.0             
    return vec

def train_content_model(user_ratings: list[dict], meta_df: pd.DataFrame,
                        cache_path: str, no_cache: bool, tmdb_key: str = None):
    from sklearn.linear_model import Ridge

    cache_key = ("content_local", tuple(sorted(r["movie_id"] for r in user_ratings)))
    if not no_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("key") == cache_key:
                print("\n[2-Content] Loaded cached content model ✓")
                # Refresh poster_url from current meta_df (fixes stale caches)
                if meta_df is not None and "meta_df" in cached and "poster_url" in meta_df.columns:
                    url_map = meta_df.set_index("movie_id")["poster_url"]
                    cm = cached["meta_df"]
                    if "poster_url" in cm.columns:
                        cm["poster_url"] = cm["movie_id"].map(url_map).fillna(cm["poster_url"])
                    else:
                        cm["poster_url"] = cm["movie_id"].map(url_map).fillna("")
                return cached
        except Exception:
            pass

    rated_slugs = [r["movie_id"] for r in user_ratings]
    rating_map  = {r["movie_id"]: r["rating_val"] for r in user_ratings}

                                                                            
    missing = [s for s in rated_slugs if s not in set(meta_df["movie_id"])]
    if missing:
        print(f"    {len(missing)} rated films not in local CSV (likely post-2022)")
        if tmdb_key:
            supplement = supplement_from_tmdb(missing, tmdb_key, meta_df)
            if not supplement.empty:
                meta_df = pd.concat([meta_df, supplement], ignore_index=True)
        else:
            print(f"    ⚠  Pass --tmdb-key to include them (improves model quality)")

    print(f"\n[2-Content] Building content features for {len(rated_slugs)} rated films ...")
    try:
        X_rated, _, rated_valid, _ = build_tfidf_features(meta_df, rated_slugs, rated_slugs)
    except ValueError:
        print(f"    ⚠  Not enough text/metadata to build content model — skipping CB")
        return {"key": cache_key, "model": None, "meta_df": meta_df}

    y = [rating_map[s] for s in rated_valid]
    if len(y) < 2:
        print(f"    ⚠  Only {len(y)} film(s) with metadata — skipping CB")
        return {"key": cache_key, "model": None, "meta_df": meta_df}

    print(f"    Training on {len(y)} / {len(rated_slugs)} rated films "
          f"({'%.0f' % (100*len(y)/len(rated_slugs))}% coverage)")
    model = Ridge(alpha=1.0)
    model.fit(X_rated, y)
    print(f"    Ridge model trained ✓")

    bundle = {"key": cache_key, "model": model, "meta_df": meta_df}
    with open(cache_path, "wb") as f:
        pickle.dump(bundle, f)
    return bundle

def predict_content(bundle: dict, user_ratings: list[dict],
                    candidates: list[str]) -> dict[str, float]:
    model   = bundle["model"]
    if model is None:
        return {}

    meta_df = bundle["meta_df"]

    print(f"\n[3-Content] Scoring {len(candidates):,} candidate films ...")
    rated_slugs = [r["movie_id"] for r in user_ratings]

    _, X_all, _, all_valid = build_tfidf_features(meta_df, rated_slugs, candidates)

    preds  = model.predict(X_all)
    scores = {slug: float(score) for slug, score in zip(all_valid, preds)}
    print(f"    Scored {len(scores):,} films ✓")
    return scores

                                                                                

def normalise(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return scores
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {k: 5.0 for k in scores}
    return {k: 1 + 9 * (v - lo) / (hi - lo) for k, v in scores.items()}

def build_recommendations(cf_scores:      dict[str, float] | None,
                           content_scores: dict[str, float] | None,
                           watchlist:      set[str],
                           mode:           str,
                           alpha:          float,
                           watchlist_boost: float,
                           top_n:          int,
                           decades:        list[int] | None = None,
                           meta_df:        pd.DataFrame | None = None,
                           watchlist_both: set[str] | None = None,
                           min_runtime:    int = 40) -> pd.DataFrame:
    print(f"\n[5] Blending scores (mode={mode}, α={alpha}) ...")

    if meta_df is not None:
        meta_idx = meta_df.set_index("movie_id")
    else:
        meta_idx = None

    def meta_val(slug, col, default):
        if meta_idx is None or slug not in meta_idx.index:
            return default
        v = meta_idx.at[slug, col]
        return default if pd.isna(v) else v

    allowed_slugs = None
    if meta_df is not None:
        mask = pd.Series(True, index=meta_df.index)

        if decades:
            decade_set = set(decades)
            mask &= meta_df["year_released"].apply(
                lambda y: (int(float(y)) // 10 * 10) in decade_set
                          if pd.notna(y) and str(y).strip() not in ("", "0") else False
            )

        if min_runtime > 0:
            mask &= meta_df["runtime"].apply(
                lambda r: float(r) >= min_runtime
                          if pd.notna(r) and str(r).strip() not in ("", "0") else False
            )

        allowed_slugs = set(meta_df[mask]["movie_id"])
        print(f"    Quality filter → {len(allowed_slugs):,} eligible films "
              f"(runtime≥{min_runtime}min" +
              (f", decades={sorted(decades)}" if decades else "") + ")")

    cf_norm      = normalise(cf_scores)      if cf_scores      else {}
    content_norm = normalise(content_scores) if content_scores else {}

    all_movies = set(cf_norm) | set(content_norm)

    if allowed_slugs is not None:
        all_movies = all_movies & allowed_slugs

    if mode in ("content", "hybrid") and content_scores:
        all_movies = {s for s in all_movies
                      if content_scores.get(s, 5.0) != 5.0}

    rows = []
    for slug in all_movies:
        cf  = cf_norm.get(slug, 5.0)
        cb  = content_norm.get(slug, 5.0)

        if mode == "cf":
            score = cf
        elif mode == "content":
            score = cb
        else:
            score = alpha * cf + (1 - alpha) * cb

        if slug in watchlist:
            score += watchlist_boost

        rows.append({
            "movie_id":        slug,
            "cf_score":        round(cf, 2) if cf_scores      else None,
            "content_score":   round(cb, 2) if content_scores else None,
            "final_score":     round(score, 2),
            "in_watchlist":    slug in watchlist,
            "in_both_wl":      slug in watchlist_both if watchlist_both else False,
        })

    df = (pd.DataFrame(rows)
            .sort_values("final_score", ascending=False)
            .head(top_n)
            .reset_index(drop=True))
    df.index += 1
    return df

def get_film_info(slug: str, meta_df: pd.DataFrame | None,
                  api_key: str | None = None) -> dict:
    """
    Returns a dict with title, year, director, poster_url, overview for a slug.
    Looks up meta_df first; if not found and api_key provided, fetches from TMDB.
    """
    row = None
    if meta_df is not None:
        r = meta_df[meta_df["movie_id"] == slug]
        if r.empty and slug.isdigit() and "tmdb_id" in meta_df.columns:
            r = meta_df[meta_df["tmdb_id"].astype(str) == slug]
        if not r.empty:
            row = r.iloc[0]

    if row is None and api_key:
        fetched = _tmdb_fetch_one(slug, api_key)
        if fetched:
            return {
                "slug":      slug,
                "title":     fetched.get("movie_title", ""),
                "year":      fetched.get("year_released", 0),
                "director":  fetched.get("director", ""),
                "poster_url":fetched.get("poster_url", ""),
                "overview":  fetched.get("overview", ""),
                "genres":    parse_genres(str(fetched.get("genres", ""))),
            }

    if row is None:
        fallback = slug.replace("-", " ").title() if not slug.isdigit() else f"Film {slug}"
        return {"slug": slug, "title": fallback, "year": 0,
                "director": "", "poster_url": "", "overview": "", "genres": []}

    def safe(col, default=""):
        v = row.get(col, default)
        return default if pd.isna(v) else v

    title = str(safe("movie_title")).strip() or slug.replace("-", " ").title()
    year  = safe("year_released", 0)
    year  = int(float(year)) if year and str(year).strip() not in ("", "0") else 0

    return {
        "slug":       slug,
        "title":      title,
        "year":       year,
        "director":   str(safe("director")),
        "poster_url": str(safe("poster_url")),
        "overview":   str(safe("overview")),
        "genres":     parse_genres(str(safe("genres", ""))),
    }


def build_user_profile(ratings: list[dict], meta_df: pd.DataFrame,
                       watchlist_size: int = 0) -> dict:
    if not ratings or meta_df is None:
        return {}

    from collections import defaultdict

    rating_map = {r["movie_id"]: r["rating_val"] for r in ratings}
    slugs = list(rating_map.keys())

    sub = meta_df[meta_df["movie_id"].isin(slugs)].set_index("movie_id")

    genre_ratings   = defaultdict(list)
    runtime_buckets = {"short": [], "standard": [], "long": []}
    decade_ratings  = defaultdict(list)
    lang_en, lang_intl = [], []

    for slug, rating in rating_map.items():
        if slug not in sub.index:
            continue
        row = sub.loc[slug]

        for g in parse_genres(str(row.get("genres", ""))):
            genre_ratings[g].append(rating)

        rt = row.get("runtime", 0)
        if pd.notna(rt) and str(rt).strip() not in ("", "0"):
            rt = float(rt)
            if rt < 90:
                runtime_buckets["short"].append(rating)
            elif rt <= 130:
                runtime_buckets["standard"].append(rating)
            else:
                runtime_buckets["long"].append(rating)

        yr = row.get("year_released", 0)
        if pd.notna(yr) and str(yr).strip() not in ("", "0"):
            decade_ratings[(int(float(yr)) // 10) * 10].append(rating)

        lang = row.get("original_language", "")
        if pd.notna(lang) and lang:
            (lang_en if lang == "en" else lang_intl).append(rating)

    top_genres = sorted(
        {g: sum(v) / len(v) for g, v in genre_ratings.items() if len(v) >= 5},
        key=lambda g: sum(genre_ratings[g]) / len(genre_ratings[g]),
        reverse=True,
    )[:5]

    rt_avgs = {k: sum(v) / len(v) for k, v in runtime_buckets.items() if len(v) >= 3}
    best_runtime = max(rt_avgs, key=rt_avgs.get) if rt_avgs else None

    decade_avgs = {d: sum(v) / len(v) for d, v in decade_ratings.items() if len(v) >= 5}
    top_decade = max(decade_avgs, key=decade_avgs.get) if decade_avgs else None

    all_vals = list(rating_map.values())
    avg_rating = round(sum(all_vals) / len(all_vals) / 2, 2)

    total_lang = len(lang_en) + len(lang_intl)
    intl_pct = round(len(lang_intl) / total_lang, 2) if total_lang > 0 else 0.0

    return {
        "total_films":   len(ratings),
        "avg_rating":    avg_rating,
        "top_genres":    top_genres,
        "runtime_pref":  best_runtime,
        "top_decade":    top_decade,
        "intl_pct":      intl_pct,
        "watchlist_size": watchlist_size,
    }


def make_display_name(slug: str, meta_df: pd.DataFrame | None,
                      api_key: str | None = None) -> str:
    info = get_film_info(slug, meta_df, api_key)
    if info["year"]:
        return f"{info['title']} ({info['year']})"
    return info["title"]

                                                                                

def print_results(df: pd.DataFrame, username: str, mode: str,
                  meta_df: pd.DataFrame | None = None,
                  watchlist_both: set[str] | None = None,
                  api_key: str | None = None) -> None:
    show_cf      = "cf_score"      in df.columns and df["cf_score"].notna().any()
    show_content = "content_score" in df.columns and df["content_score"].notna().any()

    header = f"{'#':>3}  {'Movie':<45}  {'Final':>5}"
    if show_cf:      header += f"  {'CF':>5}"
    if show_content: header += f"  {'CB':>5}"
    header += f"  {'WL':>3}"
    sep = "─" * len(header)

    print(f"\n── Top {len(df)} recommendations for '{username}' (mode: {mode}) ──")
    print(sep); print(header); print(sep)
    for rank, row in df.iterrows():
        name = make_display_name(row["movie_id"], meta_df, api_key=api_key)
        line = f"{rank:>3}  {name:<45}  {row['final_score']:>5.2f}"
        if show_cf:      line += f"  {row['cf_score']:>5.2f}"
        if show_content: line += f"  {row['content_score']:>5.2f}"
        line += f"  {'✓' if row.get('in_watchlist') else '':>3}"
        print(line)
    print(sep)
    print("  CF=collaborative, CB=content-based, WL=in a watchlist")

    if watchlist_both:
        print(f"\n── Films on both watchlists ──────────────────────────────────────")
        for slug in sorted(watchlist_both):
            print(f"  {make_display_name(slug, meta_df, api_key=api_key)}")
        print(sep)

                                                                                

def main():
    args = parse_args()

    is_blend = args.user2 is not None
    label    = f"{args.user}+{args.user2}" if is_blend else args.user

    ratings1, watchlist1 = fetch_user_data(args.user)
    if is_blend:
        ratings2, watchlist2 = fetch_user_data(args.user2)
        watchlist    = watchlist1 | watchlist2
        merged_ratings = merge_user_ratings(ratings1, ratings2, args.user, args.user2)
        rated_slugs  = {r["movie_id"] for r in ratings1} | {r["movie_id"] for r in ratings2}
        print(f"\n    User 1: {len(ratings1)} ratings, User 2: {len(ratings2)} ratings, "
              f"Union: {len(rated_slugs)} unique films")
    else:
        ratings2     = None
        watchlist    = watchlist1
        merged_ratings = ratings1
        rated_slugs  = {r["movie_id"] for r in ratings1}

    cf_scores      = None
    content_scores = None
    content_bundle = None
    meta_df        = None

    if args.mode in ("cf", "hybrid"):
        print("\n[2] Loading community ratings ...")
        community_df = load_community_ratings(args.data, args.sample, args.min_ratings)
        algo         = train_community_model(community_df, args.cache, args.no_cache)
        candidates   = [m for m in community_df["movie_id"].unique()
                        if m not in rated_slugs]
        print(f"\n[4-CF] Folding in user(s) and scoring {len(candidates):,} films ...")
        t0 = time.time()
        if is_blend:
            cf_scores = predict_cf_blend(algo, ratings1, ratings2, candidates)
        else:
            pu, bu    = fold_in_user(algo, ratings1)
            cf_scores = predict_folded(algo, pu, bu, candidates)
        print(f"    Done in {time.time()-t0:.1f}s")

    if args.mode in ("content", "hybrid"):
        meta_df        = load_movie_metadata(args.movie_data)
        content_bundle = train_content_model(merged_ratings, meta_df,
                                             args.cache_content, args.no_cache,
                                             tmdb_key=args.tmdb_key)
        meta_df        = content_bundle["meta_df"]
        if cf_scores:
            cb_candidates = list(cf_scores.keys())
        else:
            cb_candidates = [s for s in meta_df["movie_id"].tolist()
                             if s not in rated_slugs]
            print(f"  ℹ  Content mode: scoring {len(cb_candidates):,} films from metadata")

        content_scores = predict_content(content_bundle, merged_ratings, cb_candidates)

    watchlist_both = (watchlist1 & watchlist2) if is_blend else None

    recs = build_recommendations(
        cf_scores, content_scores, watchlist,
        args.mode, args.alpha, args.watchlist_boost, args.top,
        decades=args.decades, meta_df=meta_df,
        watchlist_both=watchlist_both,
        min_runtime=args.min_runtime
    )

    display_meta    = content_bundle["meta_df"] if content_bundle else None
    display_api_key = args.tmdb_key
    print_results(recs, label, args.mode, meta_df=display_meta,
                  watchlist_both=watchlist_both, api_key=display_api_key)

if __name__ == "__main__":
    main()

    # abi is gay
    