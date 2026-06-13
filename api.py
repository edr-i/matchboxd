import os
import sys
import pickle
import threading
import queue
import json
from contextlib import asynccontextmanager

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))

from letterboxd_recommender import (
    fetch_user_data,
    load_community_ratings,
    train_community_model,
    fold_in_user,
    predict_folded,
    predict_cf_blend,
    load_movie_metadata,
    train_content_model,
    predict_content,
    merge_user_ratings,
    supplement_from_tmdb,
    build_recommendations,
    get_film_info,
    make_display_name,
    build_user_profile,
    compute_because_reasons,
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH       = os.environ.get("RATINGS_CSV",   "data/ratings.csv")
MOVIE_DATA_PATH = os.environ.get("MOVIE_DATA_CSV", "data/movie_data.csv")
TMDB_KEY        = os.environ.get("TMDB_KEY", "")
TMDB_BASE       = "https://api.themoviedb.org/3"
if not TMDB_KEY:
    print("⚠  TMDB_KEY not set — recent film metadata will be limited")
COMMUNITY_CACHE = ".cache_community.pkl"
SAMPLE_SIZE     = int(os.environ.get("SAMPLE_SIZE", 11_000_000))
MIN_RATINGS     = int(os.environ.get("MIN_RATINGS", 200))

_state_lock      = threading.Lock()
_meta_lock       = threading.Lock()
_community_algo  = None
_meta_df         = None
_startup_done    = threading.Event()
_startup_status  = {"step": "starting", "message": "Initializing..."}


def get_community_algo():
    global _community_algo
    with _state_lock:
        if _community_algo is None:
            _startup_status["message"] = "Loading community ratings (this takes a minute)..."
            print("Loading community ratings...")
            community_df    = load_community_ratings(DATA_PATH, SAMPLE_SIZE, MIN_RATINGS)
            _startup_status["message"] = "Training SVD model..."
            _community_algo = train_community_model(community_df, COMMUNITY_CACHE, no_cache=False)
        return _community_algo


def get_meta_df():
    global _meta_df
    with _meta_lock:
        if _meta_df is None:
            _startup_status["message"] = "Loading movie metadata..."
            _meta_df = load_movie_metadata(MOVIE_DATA_PATH)
        return _meta_df


class RecommendRequest(BaseModel):
    user1:      str
    user2:      str | None = None
    mode:       str        = "hybrid"
    alpha:      float      = 0.7
    top:        int        = 25
    decades:    list[int] | None = None
    min_runtime: int       = 40


def _run_startup():
    try:
        get_community_algo()
        get_meta_df()
        _startup_status["message"] = "Ready"
        _startup_done.set()
        print("Server ready ✓")
    except Exception as e:
        _startup_status["message"] = f"Startup failed: {e}"
        print(f"Startup error: {e}")
        _startup_done.set()  # set anyway so the site unblocks and shows error


@app.on_event("startup")
async def startup():
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_startup)


@app.get("/startup-status")
async def startup_status():
    return {
        "ready":   _startup_done.is_set(),
        "message": _startup_status["message"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))

@app.get("/letterboxd_logo_long.svg")
def lb_logo_long():
    return FileResponse(os.path.join(os.path.dirname(__file__), "letterboxd_logo_long.svg"),
                        media_type="image/svg+xml")


@app.get("/providers/{slug}")
def get_providers(slug: str, country: str = "US"):
    if not TMDB_KEY:
        return {"country": country, "flatrate": [], "rent": [], "buy": []}

    meta_df = get_meta_df()
    tmdb_id = None
    if meta_df is not None:
        row = meta_df[meta_df["movie_id"] == slug]
        if not row.empty and "tmdb_id" in meta_df.columns:
            v = row.iloc[0].get("tmdb_id")
            if v and not pd.isna(v):
                tmdb_id = str(int(float(v)))

    if not tmdb_id:
        from letterboxd_recommender import _tmdb_fetch_one
        fetched = _tmdb_fetch_one(slug, TMDB_KEY)
        if fetched:
            tmdb_id = str(fetched.get("tmdb_id", ""))

    if not tmdb_id:
        return {"country": country, "flatrate": [], "rent": [], "buy": []}

    try:
        r = requests.get(
            f"{TMDB_BASE}/movie/{tmdb_id}/watch/providers",
            params={"api_key": TMDB_KEY},
            timeout=8,
        )
        r.raise_for_status()
        country_data = r.json().get("results", {}).get(country.upper(), {})

        def parse(entries):
            return [
                {"name": e["provider_name"],
                 "logo": f"https://image.tmdb.org/t/p/original{e['logo_path']}"}
                for e in (entries or []) if e.get("logo_path")
            ]

        return {
            "country":  country.upper(),
            "flatrate": parse(country_data.get("flatrate")),
            "rent":     parse(country_data.get("rent")),
            "buy":      parse(country_data.get("buy")),
        }
    except Exception:
        return {"country": country, "flatrate": [], "rent": [], "buy": []}


_PROXY_ALLOWED_HOSTS = {"a.ltrbxd.com", "image.tmdb.org"}

@app.get("/img-proxy")
def img_proxy(url: str):
    from urllib.parse import urlparse
    from fastapi.responses import Response
    if urlparse(url).hostname not in _PROXY_ALLOWED_HOSTS:
        raise HTTPException(status_code=403, detail="Host not allowed")
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return Response(
            content=r.content,
            media_type=r.headers.get("content-type", "image/jpeg"),
            headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "public, max-age=86400"},
        )
    except Exception:
        raise HTTPException(status_code=502, detail="Could not fetch image")


@app.post("/recommend")
def recommend(req: RecommendRequest):
    if not _startup_done.is_set():
        raise HTTPException(status_code=503, detail="Server is still initializing, please wait.")
    is_blend = req.user2 is not None and req.user2.strip() != ""

    try:
        ratings1, watchlist1 = fetch_user_data(req.user1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch user '{req.user1}': {e}")

    if is_blend:
        try:
            ratings2, watchlist2 = fetch_user_data(req.user2)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not fetch user '{req.user2}': {e}")
        watchlist      = watchlist1 | watchlist2
        watchlist_both = watchlist1 & watchlist2
        merged_ratings = merge_user_ratings(ratings1, ratings2, req.user1, req.user2)
        rated_slugs    = {r["movie_id"] for r in ratings1} | {r["movie_id"] for r in ratings2}
    else:
        ratings2       = None
        watchlist      = watchlist1
        watchlist_both = set()
        merged_ratings = ratings1
        rated_slugs    = {r["movie_id"] for r in ratings1}

    algo     = get_community_algo()
    meta_df  = get_meta_df()

    cf_scores      = None
    content_scores = None

    if req.mode in ("cf", "hybrid"):
        candidates = [m for m in algo.trainset._raw2inner_id_items.keys()
                      if m not in rated_slugs]
        if is_blend:
            cf_scores = predict_cf_blend(algo, ratings1, ratings2, candidates)
        else:
            pu, bu    = fold_in_user(algo, ratings1)
            cf_scores = predict_folded(algo, pu, bu, candidates)

    if req.mode in ("content", "hybrid"):
        missing = [s for s in rated_slugs if s not in set(meta_df["movie_id"])]
        if missing and TMDB_KEY:
            supp    = supplement_from_tmdb(missing, TMDB_KEY, meta_df)
            meta_df = pd.concat([meta_df, supp], ignore_index=True) if not supp.empty else meta_df

        content_cache = f".cache_content_{req.user1}{'_' + req.user2 if is_blend else ''}.pkl"
        bundle        = train_content_model(merged_ratings, meta_df,
                                            content_cache, no_cache=False,
                                            tmdb_key=TMDB_KEY)
        meta_df       = bundle["meta_df"]

        cb_candidates = list(cf_scores.keys()) if cf_scores else [
            s for s in meta_df["movie_id"].tolist() if s not in rated_slugs
        ]
        content_scores = predict_content(bundle, merged_ratings, cb_candidates)

    recs = build_recommendations(
        cf_scores, content_scores, watchlist,
        req.mode, req.alpha, watchlist_boost=0.5, top_n=req.top,
        decades=req.decades, meta_df=meta_df,
        watchlist_both=watchlist_both, min_runtime=req.min_runtime,
    )

    # Collect all slugs needing enrichment
    result_slugs  = list(recs["movie_id"])
    wl_slugs      = sorted(watchlist_both) if is_blend else []
    all_slugs     = list(dict.fromkeys(result_slugs + wl_slugs))  # deduped, ordered

    # Fetch poster + director concurrently for all slugs
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    info_cache = {}
    lock       = threading.Lock()

    def fetch_info(slug):
        info = get_film_info(slug, meta_df, api_key=None)  # get base info from CSV
        # if poster or director missing, hit TMDB directly using tmdb_id
        if TMDB_KEY and (not info.get("poster_url") or not info.get("director")):
            # find tmdb_id from meta_df
            tmdb_id = None
            if meta_df is not None:
                row = meta_df[meta_df["movie_id"] == slug]
                if not row.empty and "tmdb_id" in meta_df.columns:
                    v = row.iloc[0].get("tmdb_id")
                    if v and not pd.isna(v):
                        tmdb_id = str(int(float(v)))
            lookup_slug = tmdb_id if tmdb_id else slug
            from letterboxd_recommender import _tmdb_fetch_one
            fetched = _tmdb_fetch_one(lookup_slug, TMDB_KEY)
            if fetched:
                info["poster_url"] = fetched.get("poster_url", "")
                info["director"]   = fetched.get("director", "")
        with lock:
            info_cache[slug] = info

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(fetch_info, s) for s in all_slugs]
        for f in as_completed(futures):
            try: f.result()
            except Exception: pass

    def enrich(slug):
        return info_cache.get(slug, get_film_info(slug, meta_df))

    results = []
    for _, row in recs.iterrows():
        slug = row["movie_id"]
        info = enrich(slug)
        results.append({
            "slug":          slug,
            "title":         info["title"],
            "year":          info["year"],
            "director":      info["director"],
            "poster_url":    info["poster_url"],
            "overview":      info["overview"],
            "final_score":   row["final_score"],
            "cf_score":      row.get("cf_score"),
            "content_score": row.get("content_score"),
            "in_watchlist":  row.get("in_watchlist", False),
            "in_both_wl":    row.get("in_both_wl", False),
            "in_wl1":        slug in watchlist1,
            "in_wl2":        slug in watchlist2 if is_blend else False,
        })

    both_wl = []
    if is_blend:
        for slug in wl_slugs:
            info = enrich(slug)
            both_wl.append({
                "slug":       slug,
                "title":      info["title"],
                "year":       info["year"],
                "director":   info["director"],
                "poster_url": info["poster_url"],
            })

    return {
        "user1":            req.user1,
        "user2":            req.user2,
        "mode":             req.mode,
        "recommendations":  results,
        "both_watchlist":   both_wl,
    }


def _compute_blend_score(p1: dict, p2: dict,
                         ratings1: list = None, ratings2: list = None) -> dict:
    g1 = set(p1.get("top_genres", []))
    g2 = set(p2.get("top_genres", []))
    shared_genres = sorted(g1 & g2)
    union = g1 | g2
    genre_score = round((len(shared_genres) / len(union)) * 30) if union else 15

    d1, d2 = p1.get("top_decade"), p2.get("top_decade")
    era_score = round(max(0, 1 - abs(d1 - d2) / 50) * 15) if (d1 and d2) else 7

    rt1, rt2 = p1.get("runtime_pref"), p2.get("runtime_pref")
    runtime_score = (10 if rt1 == rt2 else 3) if (rt1 and rt2) else 5

    r1 = p1.get("avg_rating", 2.5)
    r2 = p2.get("avg_rating", 2.5)
    rating_score = round(max(0, 1 - abs(r1 - r2) / 2.5) * 15)

    # Film overlap — the real taste signal
    films_both_rated = 0
    films_both_loved = 0
    agreement_score  = 0
    if ratings1 and ratings2:
        map1 = {r["movie_id"]: r["rating_val"] for r in ratings1}
        map2 = {r["movie_id"]: r["rating_val"] for r in ratings2}
        shared = set(map1) & set(map2)
        films_both_rated = len(shared)
        films_both_loved = sum(1 for s in shared if map1[s] >= 8 and map2[s] >= 8)
        if films_both_rated >= 5:
            agreement_rate = films_both_loved / films_both_rated
            agreement_score = round(agreement_rate * 30)
        else:
            agreement_score = 0

    total = min(100, genre_score + era_score + runtime_score + rating_score + agreement_score)
    return {
        "total":            total,
        "shared_genres":    shared_genres,
        "films_both_rated": films_both_rated,
        "films_both_loved": films_both_loved,
    }


@app.post("/recommend-stream")
def recommend_stream(req: RecommendRequest):
    if not _startup_done.is_set():
        raise HTTPException(status_code=503, detail="Server is still initializing.")

    def generate():
        def emit(msg):
            yield f"data: {json.dumps({'type': 'progress', 'message': msg})}\n\n"

        try:
            is_blend = req.user2 is not None and req.user2.strip() != ""

            yield from emit(f"Fetching Letterboxd data for {req.user1}…")
            try:
                ratings1, watchlist1, favs1 = fetch_user_data(req.user1, include_favorites=True)
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                return

            if is_blend:
                yield from emit(f"Fetching Letterboxd data for {req.user2}…")
                try:
                    ratings2, watchlist2, favs2 = fetch_user_data(req.user2, include_favorites=True)
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                    return
                watchlist      = watchlist1 | watchlist2
                watchlist_both = watchlist1 & watchlist2
                merged_ratings = merge_user_ratings(ratings1, ratings2, req.user1, req.user2)
                rated_slugs    = {r["movie_id"] for r in ratings1} | {r["movie_id"] for r in ratings2}
            else:
                ratings2       = None
                watchlist      = watchlist1
                watchlist_both = set()
                merged_ratings = ratings1
                rated_slugs    = {r["movie_id"] for r in ratings1}

            algo    = get_community_algo()
            meta_df = get_meta_df()

            def enrich_favs(favs):
                enriched = []
                for f in favs:
                    info = get_film_info(f["slug"], meta_df, api_key=None)
                    poster = info.get("poster_url", "")
                    if not poster and TMDB_KEY:
                        from letterboxd_recommender import _tmdb_fetch_one
                        fetched = _tmdb_fetch_one(f["slug"], TMDB_KEY)
                        if fetched:
                            poster = fetched.get("poster_url", "")
                    enriched.append({**f, "poster_url": poster})
                return enriched

            profile1 = build_user_profile(ratings1, meta_df, watchlist_size=len(watchlist1))
            profile1["favorites"] = enrich_favs(favs1)
            profile_payload = {"user1": req.user1, "profile1": profile1}
            if is_blend:
                profile2 = build_user_profile(ratings2, meta_df, watchlist_size=len(watchlist2))
                profile2["favorites"] = enrich_favs(favs2)
                profile_payload["user2"] = req.user2
                profile_payload["profile2"] = profile2
                profile_payload["blend_score"] = _compute_blend_score(profile1, profile2, ratings1, ratings2)
            yield f"data: {json.dumps({'type': 'profile', 'data': profile_payload})}\n\n"

            cf_scores      = None
            content_scores = None

            if req.mode in ("cf", "hybrid"):
                yield from emit("Calculating taste profile…")
                candidates = [m for m in algo.trainset._raw2inner_id_items.keys()
                              if m not in rated_slugs]
                if is_blend:
                    cf_scores = predict_cf_blend(algo, ratings1, ratings2, candidates)
                else:
                    pu, bu    = fold_in_user(algo, ratings1)
                    cf_scores = predict_folded(algo, pu, bu, candidates)

            if req.mode in ("content", "hybrid"):
                missing = [s for s in rated_slugs if s not in set(meta_df["movie_id"])]
                if missing and TMDB_KEY:
                    yield from emit(f"Fetching metadata for {len(missing)} recent films…")
                    supp    = supplement_from_tmdb(missing, TMDB_KEY, meta_df)
                    meta_df = pd.concat([meta_df, supp], ignore_index=True) if not supp.empty else meta_df

                yield from emit("Building content model…")
                content_cache = f".cache_content_{req.user1}{'_' + req.user2 if is_blend else ''}.pkl"
                bundle        = train_content_model(merged_ratings, meta_df,
                                                    content_cache, no_cache=False,
                                                    tmdb_key=TMDB_KEY)
                meta_df       = bundle["meta_df"]

                cb_candidates = list(cf_scores.keys()) if cf_scores else [
                    s for s in meta_df["movie_id"].tolist() if s not in rated_slugs
                ]
                yield from emit(f"Scoring {len(cb_candidates):,} films…")
                content_scores = predict_content(bundle, merged_ratings, cb_candidates)

            yield from emit("Ranking recommendations…")
            recs = build_recommendations(
                cf_scores, content_scores, watchlist,
                req.mode, req.alpha, watchlist_boost=0.5, top_n=req.top,
                decades=req.decades, meta_df=meta_df,
                watchlist_both=watchlist_both, min_runtime=req.min_runtime,
            )

            yield from emit(f"Fetching posters for {len(recs)} films…")
            result_slugs = list(recs["movie_id"])
            wl_slugs     = sorted(watchlist_both) if is_blend else []
            all_slugs    = list(dict.fromkeys(result_slugs + wl_slugs))

            info_cache = {}
            lock       = threading.Lock()

            def fetch_info(slug):
                info = get_film_info(slug, meta_df, api_key=None)
                if TMDB_KEY and (not info.get("poster_url") or not info.get("director")):
                    tmdb_id = None
                    if meta_df is not None:
                        row = meta_df[meta_df["movie_id"] == slug]
                        if not row.empty and "tmdb_id" in meta_df.columns:
                            v = row.iloc[0].get("tmdb_id")
                            if v and not pd.isna(v):
                                tmdb_id = str(int(float(v)))
                    lookup = tmdb_id if tmdb_id else slug
                    from letterboxd_recommender import _tmdb_fetch_one
                    fetched = _tmdb_fetch_one(lookup, TMDB_KEY)
                    if fetched:
                        info["poster_url"] = fetched.get("poster_url", "")
                        info["director"]   = fetched.get("director", "")
                with lock:
                    info_cache[slug] = info

            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=20) as pool:
                futures = [pool.submit(fetch_info, s) for s in all_slugs]
                for f in as_completed(futures):
                    try: f.result()
                    except Exception: pass

            def enrich(slug):
                return info_cache.get(slug, get_film_info(slug, meta_df))

            # Compute "because you loved/liked X · you love Drama" reasons
            rec_slugs_list = list(recs["movie_id"])
            reasons1: dict = {}
            reasons2: dict = {}
            if req.mode in ("cf", "hybrid") and cf_scores:
                reasons1 = compute_because_reasons(
                    algo, ratings1, rec_slugs_list, meta_df,
                    user_profile=profile1,
                    username=req.user1 if is_blend else None,
                )
                if is_blend:
                    reasons2 = compute_because_reasons(
                        algo, ratings2, rec_slugs_list, meta_df,
                        user_profile=profile2,
                        username=req.user2,
                    )

            results = []
            for _, row in recs.iterrows():
                slug = row["movie_id"]
                info = enrich(slug)
                b1 = reasons1.get(slug)
                b2 = reasons2.get(slug) if is_blend else None
                parts = [p for p in [b1, b2] if p]
                because = " · ".join(parts) if parts else None
                results.append({
                    "slug":          slug,
                    "title":         info["title"],
                    "year":          info["year"],
                    "director":      info["director"],
                    "poster_url":    info["poster_url"],
                    "overview":      info["overview"],
                    "final_score":   row["final_score"],
                    "cf_score":      row.get("cf_score"),
                    "content_score": row.get("content_score"),
                    "in_watchlist":  row.get("in_watchlist", False),
                    "in_both_wl":    row.get("in_both_wl", False),
                    "in_wl1":        slug in watchlist1,
                    "in_wl2":        slug in watchlist2 if is_blend else False,
                    "genres":        info.get("genres", []),
                    "because":       because,
                })

            both_wl = []
            if is_blend:
                for slug in wl_slugs:
                    info = enrich(slug)
                    both_wl.append({
                        "slug":       slug,
                        "title":      info["title"],
                        "year":       info["year"],
                        "director":   info["director"],
                        "poster_url": info["poster_url"],
                    })

            payload = {
                "user1":           req.user1,
                "user2":           req.user2,
                "mode":            req.mode,
                "recommendations": results,
                "both_watchlist":  both_wl,
            }
            yield f"data: {json.dumps({'type': 'result', 'data': payload})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/pdf")
def pdf_page(user1: str, mode: str, user2: str = None):
    """Returns a print-ready HTML page for the given user combo."""
    is_blend = user2 is not None
    accent   = "#00e054" if is_blend else "#ff8000"
    label    = f"{user1} + {user2}" if is_blend else user1
    mode_label = "Blend" if is_blend else "Single"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Matchboxd — {label}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@700&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
  :root {{ --accent: {accent}; --black: #0a0a0a; --white: #f0ede6; --border: 2px solid #0a0a0a; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--white); color: var(--black); font-family: 'Barlow Condensed', sans-serif; padding: 2rem; }}
  header {{ display: flex; justify-content: space-between; align-items: baseline; border-bottom: 3px solid var(--black); padding-bottom: 0.75rem; margin-bottom: 1.5rem; }}
  .logo {{ font-size: 2.5rem; font-weight: 700; text-transform: uppercase; letter-spacing: -0.02em; }}
  .logo span {{ color: var(--accent); }}
  .header-right {{ text-align: right; }}
  .mode-badge {{ font-family: 'Space Mono', monospace; font-size: 0.6rem; text-transform: uppercase; letter-spacing: 0.15em; background: var(--accent); color: {'#0a0a0a' if is_blend else '#fff'}; padding: 2px 8px; }}
  .sub {{ font-family: 'Space Mono', monospace; font-size: 0.58rem; text-transform: uppercase; letter-spacing: 0.15em; color: #555; margin-top: 0.3rem; }}
  .section-title {{ font-size: 1.6rem; font-weight: 700; text-transform: uppercase; margin: 1.5rem 0 1rem; border-bottom: var(--border); padding-bottom: 0.4rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 0.75rem; }}
  .card {{ display: flex; flex-direction: column; }}
  .card-poster {{ border: var(--border); aspect-ratio: 2/3; overflow: hidden; position: relative; background: #ddd; }}
  .card-poster img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
  .card-no-poster {{ width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; font-family: 'Space Mono', monospace; font-size: 0.5rem; color: #999; text-align: center; padding: 4px; text-transform: uppercase; }}
  .card-rank {{ position: absolute; top: 0; left: 0; font-size: 0.7rem; font-weight: 700; background: var(--black); color: var(--white); padding: 1px 4px; line-height: 1.4; }}
  .wl-badge {{ position: absolute; top: 0; right: 0; font-family: 'Space Mono', monospace; font-size: 0.42rem; font-weight: 700; padding: 1px 4px; text-transform: uppercase; background: var(--accent); color: {'#0a0a0a' if is_blend else '#fff'}; line-height: 1.4; }}
  .wl-green {{ background: #00e054; color: #0a0a0a; }}
  .card-body {{ padding: 0.3rem 0; }}
  .card-title {{ font-size: 0.75rem; font-weight: 700; text-transform: uppercase; line-height: 1.2; }}
  .card-meta {{ font-family: 'Space Mono', monospace; font-size: 0.5rem; color: #555; margin-top: 0.1rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .card-score {{ font-family: 'Space Mono', monospace; font-size: 0.5rem; color: #555; margin-top: 0.2rem; }}
  footer {{ margin-top: 2rem; border-top: var(--border); padding-top: 0.75rem; font-family: 'Space Mono', monospace; font-size: 0.55rem; text-transform: uppercase; letter-spacing: 0.12em; color: #999; text-align: center; }}
  @media print {{
    body {{ padding: 1rem; }}
    .no-print {{ display: none; }}
    @page {{ margin: 1cm; size: A4; }}
  }}
  .print-btn {{ font-family: 'Barlow Condensed', sans-serif; font-size: 1rem; font-weight: 700; text-transform: uppercase; background: var(--accent); color: {'#0a0a0a' if is_blend else '#fff'}; border: 2px solid var(--black); padding: 0.5rem 1.5rem; cursor: pointer; box-shadow: 3px 3px 0 var(--black); margin-bottom: 1.5rem; }}
</style>
</head>
<body>
<div class="no-print" style="margin-bottom:1rem">
  <button class="print-btn" onclick="window.print()">Print / Save as PDF</button>
</div>
<header>
  <div class="logo">match<span>boxd</span></div>
  <div class="header-right">
    <div class="mode-badge">{mode_label}</div>
    <div class="sub">made with ♡ by kopliku.dev</div>
  </div>
</header>
<div id="content"><p style="font-family:monospace;font-size:0.8rem;color:#555">Loading recommendations…</p></div>
<footer>matchboxd · kopliku.dev · {label} · {mode}</footer>
<script>
async function load() {{
  const res  = await fetch('/recommend', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{ user1: {json.dumps(user1)}, user2: {json.dumps(user2)}, mode: {json.dumps(mode)}, top: 50 }})
  }});
  const data = await res.json();
  const u1 = data.user1, u2 = data.user2, isBlend = !!u2;
  let html = '<div class="section-title">Recommendations for {label}</div><div class="grid">';
  data.recommendations.forEach((f, i) => {{
    const title = f.title || f.slug;
    const year  = f.year ? '(' + f.year + ')' : '';
    const score = f.final_score ? f.final_score.toFixed(2) : '';
    const dir   = f.director || '';
    const lbUrl = 'https://letterboxd.com/film/' + f.slug + '/';
    const img   = f.poster_url ? '<img src="' + f.poster_url + '">' : '<div class="card-no-poster">' + title + '</div>';
    let wl = '';
    if (isBlend) {{
      if (f.in_both_wl || (f.in_wl1 && f.in_wl2)) wl = '<div class="wl-badge wl-green">WL: ' + u1 + ' & ' + u2 + '</div>';
      else if (f.in_wl1) wl = '<div class="wl-badge">WL: ' + u1 + '</div>';
      else if (f.in_wl2) wl = '<div class="wl-badge">WL: ' + u2 + '</div>';
    }} else if (f.in_watchlist || f.in_wl1) {{
      wl = '<div class="wl-badge">WL</div>';
    }}
    html += '<div class="card"><div class="card-poster"><img src="' + (f.poster_url||'') + '" onerror="this.parentNode.innerHTML=\'<div class=card-no-poster>' + title.replace(/'/g,"\\'") + '</div>\'">' +
      '<div class="card-rank">#' + (i+1) + '</div>' + wl + '</div>' +
      '<div class="card-body"><div class="card-title">' + title + ' ' + year + '</div>' +
      (score ? '<div class="card-score">' + score + '</div>' : '') +
      (dir   ? '<div class="card-meta">'  + dir   + '</div>' : '') +
      '</div></div>';
  }});
  html += '</div>';
  if (data.both_watchlist && data.both_watchlist.length) {{
    html += '<div class="section-title">Both watchlists</div><div class="grid">';
    data.both_watchlist.forEach(f => {{
      const title = f.title || f.slug;
      const year  = f.year ? '(' + f.year + ')' : '';
      html += '<div class="card"><div class="card-poster">' +
        (f.poster_url ? '<img src="' + f.poster_url + '">' : '<div class="card-no-poster">' + title + '</div>') +
        '</div><div class="card-body"><div class="card-title">' + title + ' ' + year + '</div>' +
        (f.director ? '<div class="card-meta">' + f.director + '</div>' : '') + '</div></div>';
    }});
    html += '</div>';
  }}
  document.getElementById('content').innerHTML = html;
}}
load();
</script>
</body></html>"""
    return HTMLResponse(content=html)