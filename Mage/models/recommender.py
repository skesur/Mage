import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from scipy.sparse import hstack, vstack, csr_matrix
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class RecommendationEngine:
    
    def __init__(self):
        # TF-IDF for content-based filtering
        self.movie_tfidf = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        self.show_tfidf = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        self.book_tfidf = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        # Count Vectorizer for genre/vibe analysis
        self.genre_vectorizer = CountVectorizer(
            max_features=100,
            token_pattern=r'[^,]+',
            lowercase=True
        )
        
        # Numerical scalers
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        
        # K-Nearest Neighbors for collaborative filtering
        self.knn_movie = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='cosine', n_jobs=-1)
        self.knn_show = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='cosine', n_jobs=-1)
        self.knn_book = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='cosine', n_jobs=-1)
        
        # Dimensionality reduction
        self.svd_movie = TruncatedSVD(n_components=50, random_state=42)
        self.svd_show = TruncatedSVD(n_components=50, random_state=42)
        self.svd_book = TruncatedSVD(n_components=50, random_state=42)
        
        # Clustering
        self.kmeans_movie = KMeans(n_clusters=10, random_state=42, n_init=10)
        self.kmeans_show = KMeans(n_clusters=10, random_state=42, n_init=10)
        self.kmeans_book = KMeans(n_clusters=10, random_state=42, n_init=10)
        
        # Feature matrices
        self.movie_features = None
        self.show_features = None
        self.book_features = None
        
        self.movie_features_reduced = None
        self.show_features_reduced = None
        self.book_features_reduced = None
        
        # Similarity matrices
        self.movie_similarity = None
        self.show_similarity = None
        self.book_similarity = None
        
        # Data storage
        self.movies_df = None
        self.shows_df = None
        self.books_df = None
        
        # Cluster assignments
        self.movie_clusters = None
        self.show_clusters = None
        self.book_clusters = None
        
        self.is_trained = False
    
    # ============================================================
    # TRAINING METHODS
    # ============================================================
    
    def fit(self, movies_df: pd.DataFrame, shows_df: pd.DataFrame, books_df: pd.DataFrame) -> None:
        print("\n🚀 Training Recommendation Engine...")
        
        self.movies_df = movies_df.copy()
        self.shows_df = shows_df.copy()
        self.books_df = books_df.copy()
        
        if not movies_df.empty:
            print("\n🎬 Training Movie Recommender...")
            self._fit_movies(movies_df)
        
        if not shows_df.empty:
            print("\n📺 Training TV Show Recommender...")
            self._fit_shows(shows_df)
        
        if not books_df.empty:
            print("\n📚 Training Book Recommender...")
            self._fit_books(books_df)
        
        self.is_trained = True
        print("\n✅ Recommendation Engine Fully Trained!")
    
    def _fit_movies(self, df: pd.DataFrame) -> None:
        # Text features
        df['combined_text'] = (
            df['genres'].fillna('') + ' ' +
            df['vibes'].fillna('') + ' ' +
            df['logline'].fillna('')
        )
        text_features = self.movie_tfidf.fit_transform(df['combined_text'])
        
        # Genre features
        genre_features = self.genre_vectorizer.fit_transform(df['genres'].fillna('unknown'))
        
        # Numerical features
        numerical_cols = ['duration', 'releaseYear', 'rating']
        numerical_data = df[numerical_cols].values
        numerical_data = np.nan_to_num(numerical_data, nan=0.0)
        numerical_features = self.standard_scaler.fit_transform(numerical_data)
        
        # Decade features
        df['decade'] = (df['releaseYear'] // 10) * 10
        decade_dummies = pd.get_dummies(df['decade'], prefix='decade').values
        
        # Rating categories
        df['rating_category'] = pd.cut(df['rating'], bins=[0, 2, 3.5, 4.5, 5], labels=['low', 'medium', 'high', 'excellent'])
        rating_dummies = pd.get_dummies(df['rating_category'], prefix='rating').values
        
        # Combine features
        self.movie_features = hstack([text_features, genre_features, numerical_features, decade_dummies, rating_dummies])
        
        # Dimensionality reduction
        if self.movie_features.shape[0] > 50:
            self.movie_features_reduced = self.svd_movie.fit_transform(self.movie_features)
        
        # Similarity matrix
        self.movie_similarity = cosine_similarity(self.movie_features)
        
        # Clustering
        if self.movie_features.shape[0] >= 10:
            if self.movie_features_reduced is not None:
                self.movie_clusters = self.kmeans_movie.fit_predict(self.movie_features_reduced)
            else:
                dense_features = self.movie_features.toarray()
                self.movie_clusters = self.kmeans_movie.fit_predict(dense_features)
        
        # KNN
        self.knn_movie.fit(self.movie_features)
        
        print(f"✅ Movie model complete: {len(df)} movies")
    
    def _fit_shows(self, df: pd.DataFrame) -> None:
        df['release_year_numeric'] = df['releaseYears'].str.extract(r'(\d{4})').astype(float).fillna(2000)
        
        df['combined_text'] = (
            df['genres'].fillna('') + ' ' +
            df['vibes'].fillna('') + ' ' +
            df['logline'].fillna('')
        )
        text_features = self.show_tfidf.fit_transform(df['combined_text'])
        
        genre_features = self.genre_vectorizer.fit_transform(df['genres'].fillna('unknown'))
        
        numerical_cols = ['seasons', 'episodesPerSeason', 'release_year_numeric', 'rating']
        numerical_data = df[numerical_cols].values
        numerical_data = np.nan_to_num(numerical_data, nan=0.0)
        numerical_features = self.standard_scaler.fit_transform(numerical_data)
        
        df['season_category'] = pd.cut(df['seasons'], bins=[0, 2, 5, 10, 100], labels=['short', 'medium', 'long', 'epic'])
        season_dummies = pd.get_dummies(df['season_category'], prefix='seasons').values
        
        self.show_features = hstack([text_features, genre_features, numerical_features, season_dummies])
        
        if self.show_features.shape[0] > 50:
            self.show_features_reduced = self.svd_show.fit_transform(self.show_features)
        
        self.show_similarity = cosine_similarity(self.show_features)
        
        if self.show_features.shape[0] >= 10:
            if self.show_features_reduced is not None:
                self.show_clusters = self.kmeans_show.fit_predict(self.show_features_reduced)
            else:
                self.show_clusters = self.kmeans_show.fit_predict(self.show_features.toarray())
        
        self.knn_show.fit(self.show_features)
        
        print(f"✅ Show model complete: {len(df)} shows")
    
    def _fit_books(self, df: pd.DataFrame) -> None:
        df['combined_text'] = (
            df['type'].fillna('') + ' ' +
            df['vibes'].fillna('') + ' ' +
            df['logline'].fillna('')
        )
        text_features = self.book_tfidf.fit_transform(df['combined_text'])
        
        type_features = self.genre_vectorizer.fit_transform(df['type'].fillna('unknown'))
        
        numerical_cols = ['pages', 'releaseYear', 'rating']
        numerical_data = df[numerical_cols].values
        numerical_data = np.nan_to_num(numerical_data, nan=0.0)
        numerical_features = self.standard_scaler.fit_transform(numerical_data)
        
        df['page_category'] = pd.cut(df['pages'], bins=[0, 200, 400, 600, 10000], labels=['short', 'medium', 'long', 'epic'])
        page_dummies = pd.get_dummies(df['page_category'], prefix='pages').values
        
        self.book_features = hstack([text_features, type_features, numerical_features, page_dummies])
        
        if self.book_features.shape[0] > 50:
            self.book_features_reduced = self.svd_book.fit_transform(self.book_features)
        
        self.book_similarity = cosine_similarity(self.book_features)
        
        if self.book_features.shape[0] >= 10:
            if self.book_features_reduced is not None:
                self.book_clusters = self.kmeans_book.fit_predict(self.book_features_reduced)
            else:
                self.book_clusters = self.kmeans_book.fit_predict(self.book_features.toarray())
        
        self.knn_book.fit(self.book_features)
        
        print(f"✅ Book model complete: {len(df)} books")
    
    # ============================================================
    # SEARCH BY TITLE
    # ============================================================
    
    def search_by_title(self, title: str, content_type: str) -> Optional[Dict]:
        if content_type == 'movie' and self.movies_df is not None:
            df = self.movies_df
            id_col = 'movieId'
        elif content_type == 'show' and self.shows_df is not None:
            df = self.shows_df
            id_col = 'showId'
        elif content_type == 'book' and self.books_df is not None:
            df = self.books_df
            id_col = 'bookId'
        else:
            return None
        
        # Case-insensitive partial match
        mask = df['title'].str.lower().str.contains(title.lower(), na=False)
        matches = df[mask]
        
        if matches.empty:
            return None
        
        # Return the best match (first one)
        item = matches.iloc[0]
        result = {
            id_col: int(item[id_col]),
            'title': item['title'],
            'rating': float(item['rating']),
            'platform': item['platform']
        }
        
        if content_type == 'movie':
            result.update({
                'genres': item['genres'],
                'vibes': item['vibes'],
                'duration': int(item['duration']),
                'releaseYear': int(item['releaseYear']),
                'logline': item['logline']
            })
        elif content_type == 'show':
            result.update({
                'genres': item['genres'],
                'vibes': item['vibes'],
                'seasons': int(item['seasons']),
                'episodesPerSeason': int(item['episodesPerSeason']),
                'releaseYears': item['releaseYears'],
                'logline': item['logline']
            })
        elif content_type == 'book':
            result.update({
                'type': item['type'],
                'vibes': item['vibes'],
                'pages': int(item['pages']),
                'releaseYear': int(item['releaseYear']),
                'logline': item['logline']
            })
        
        return result
    
    # ============================================================
    # FILTER-BASED RECOMMENDATIONS - ALL FIXED
    # ============================================================
    
    def recommend_movies_by_filters(self, 
                                    genres: Optional[List[str]] = None,
                                    vibes: Optional[List[str]] = None,
                                    max_duration: Optional[int] = None,
                                    min_year: Optional[int] = None,
                                    max_year: Optional[int] = None,
                                    min_rating: Optional[float] = None,
                                    platform: Optional[str] = None,
                                    n: int = 10,
                                    genres_and_logic: bool = False,  
                                    vibes_and_logic: bool = False,    
                                    diversity: bool = True,
                                    session_seed: Optional[int] = None) -> List[Dict]:
    
        if self.movies_df is None or self.movies_df.empty:
            return []
        
        df = self.movies_df.copy()
        
        # Apply genre filter with AND/OR logic
        if genres:
            if genres_and_logic:
                # ALL genres must be present (AND logic)
                genre_mask = df['genres'].apply(
                    lambda x: all(g.lower() in str(x).lower() for g in genres) if pd.notna(x) else False
                )
            else:
                # ANY genre can be present (OR logic - default)
                genre_mask = df['genres'].apply(
                    lambda x: any(g.lower() in str(x).lower() for g in genres) if pd.notna(x) else False
                )
            df = df[genre_mask]
        
        # Apply vibe filter with AND/OR logic
        if vibes:
            if vibes_and_logic:
                # ALL vibes must be present (AND logic)
                vibe_mask = df['vibes'].apply(
                    lambda x: all(v.lower() in str(x).lower() for v in vibes) if pd.notna(x) else False
                )
            else:
                # ANY vibe can be present (OR logic - default)
                vibe_mask = df['vibes'].apply(
                    lambda x: any(v.lower() in str(x).lower() for v in vibes) if pd.notna(x) else False
                )
            df = df[vibe_mask]
        
        # Apply duration filter
        if max_duration:
            df = df[df['duration'] <= max_duration]
        
        # Apply year filters
        if min_year:
            df = df[df['releaseYear'] >= min_year]
        if max_year:
            df = df[df['releaseYear'] <= max_year]
        
        # Apply rating filter
        if min_rating:
            df = df[df['rating'] >= min_rating]
        
        # Apply platform filter
        if platform:
            platform_mask = df['platform'].str.lower().str.contains(platform.lower(), na=False)
            df = df[platform_mask]
        
        # Return empty list instead of error
        if df.empty:
            return []
        
        # Apply diversity sampling
        if diversity and len(df) > n:
            if session_seed is None:
                import time
                session_seed = int(time.time() * 1000) % 100000
            
            rng = np.random.RandomState(session_seed)
            df = df.sort_values('rating', ascending=False).reset_index(drop=True)
            
            tier_size = max(len(df) // 3, 1)
            top_tier = df.iloc[:tier_size]
            mid_tier = df.iloc[tier_size:tier_size*2] if len(df) > tier_size else pd.DataFrame()
            lower_tier = df.iloc[tier_size*2:] if len(df) > tier_size*2 else pd.DataFrame()
            
            n_top = min(int(n * 0.6), len(top_tier))
            n_mid = min(int(n * 0.3), len(mid_tier))
            n_lower = n - n_top - n_mid
            
            selected = []
            
            if not top_tier.empty:
                top_weights = (top_tier['rating'].values ** 1.5)
                top_weights = top_weights / top_weights.sum()
                top_indices = rng.choice(len(top_tier), size=min(n_top, len(top_tier)), replace=False, p=top_weights)
                selected.append(top_tier.iloc[top_indices])
            
            if not mid_tier.empty and n_mid > 0:
                mid_indices = rng.choice(len(mid_tier), size=min(n_mid, len(mid_tier)), replace=False)
                selected.append(mid_tier.iloc[mid_indices])
            
            if not lower_tier.empty and n_lower > 0:
                lower_indices = rng.choice(len(lower_tier), size=min(n_lower, len(lower_tier)), replace=False)
                selected.append(lower_tier.iloc[lower_indices])
            
            df = pd.concat(selected, ignore_index=True)
            df = df.sample(frac=1.0, random_state=session_seed).reset_index(drop=True)
        else:
            df = df.sort_values('rating', ascending=False).head(n)
        
        # Convert to results
        results = []
        for _, movie in df.iterrows():
            results.append({
                'movieId': int(movie['movieId']),
                'title': movie['title'],
                'genres': movie['genres'],
                'vibes': movie['vibes'],
                'rating': float(movie['rating']),
                'duration': int(movie['duration']),
                'year': int(movie['releaseYear']),
                'platform': movie['platform']
            })
        
        return results
    
    def recommend_shows_by_filters(self,
                                   genres: Optional[List[str]] = None,
                                   vibes: Optional[List[str]] = None,
                                   min_year: Optional[int] = None,
                                   max_year: Optional[int] = None,
                                   min_rating: Optional[float] = None,
                                   max_seasons: Optional[int] = None,
                                   platform: Optional[str] = None,
                                   n: int = 10,
                                   genres_and_logic: bool = False,
                                   vibes_and_logic: bool = False,
                                   diversity: bool = True,
                                   session_seed: Optional[int] = None) -> List[Dict]:
        
        if self.shows_df is None or self.shows_df.empty:
            return []
        
        df = self.shows_df.copy()
        
        # Apply genre filter with AND/OR logic
        if genres:
            if genres_and_logic:
                genre_mask = df['genres'].apply(
                    lambda x: all(g.lower() in str(x).lower() for g in genres) if pd.notna(x) else False
                )
            else:
                genre_mask = df['genres'].apply(
                    lambda x: any(g.lower() in str(x).lower() for g in genres) if pd.notna(x) else False
                )
            df = df[genre_mask]
        
        # Apply vibe filter with AND/OR logic
        if vibes:
            if vibes_and_logic:
                vibe_mask = df['vibes'].apply(
                    lambda x: all(v.lower() in str(x).lower() for v in vibes) if pd.notna(x) else False
                )
            else:
                vibe_mask = df['vibes'].apply(
                    lambda x: any(v.lower() in str(x).lower() for v in vibes) if pd.notna(x) else False
                )
            df = df[vibe_mask]
        
        # Apply year filters
        if min_year or max_year:
            df['release_year_numeric'] = df['releaseYears'].str.extract(r'(\d{4})').astype(float)
            if min_year:
                df = df[df['release_year_numeric'] >= min_year]
            if max_year:
                df = df[df['release_year_numeric'] <= max_year]
        
        # Apply rating filter
        if min_rating:
            df = df[df['rating'] >= min_rating]
        
        # Apply season filter
        if max_seasons:
            df = df[df['seasons'] <= max_seasons]
        
        # Apply platform filter
        if platform:
            platform_mask = df['platform'].str.lower().str.contains(platform.lower(), na=False)
            df = df[platform_mask]
        
        # Return empty list gracefully
        if df.empty:
            return []
        
        # Apply diversity sampling (same logic as movies)
        if diversity and len(df) > n:
            if session_seed is None:
                import time
                session_seed = int(time.time() * 1000) % 100000
            
            rng = np.random.RandomState(session_seed)
            df = df.sort_values('rating', ascending=False).reset_index(drop=True)
            
            tier_size = max(len(df) // 3, 1)
            top_tier = df.iloc[:tier_size]
            mid_tier = df.iloc[tier_size:tier_size*2] if len(df) > tier_size else pd.DataFrame()
            lower_tier = df.iloc[tier_size*2:] if len(df) > tier_size*2 else pd.DataFrame()
            
            n_top = min(int(n * 0.6), len(top_tier))
            n_mid = min(int(n * 0.3), len(mid_tier))
            n_lower = n - n_top - n_mid
            
            selected = []
            
            if not top_tier.empty:
                top_weights = (top_tier['rating'].values ** 1.5)
                top_weights = top_weights / top_weights.sum()
                top_indices = rng.choice(len(top_tier), size=min(n_top, len(top_tier)), replace=False, p=top_weights)
                selected.append(top_tier.iloc[top_indices])
            
            if not mid_tier.empty and n_mid > 0:
                mid_indices = rng.choice(len(mid_tier), size=min(n_mid, len(mid_tier)), replace=False)
                selected.append(mid_tier.iloc[mid_indices])
            
            if not lower_tier.empty and n_lower > 0:
                lower_indices = rng.choice(len(lower_tier), size=min(n_lower, len(lower_tier)), replace=False)
                selected.append(lower_tier.iloc[lower_indices])
            
            df = pd.concat(selected, ignore_index=True)
            df = df.sample(frac=1.0, random_state=session_seed).reset_index(drop=True)
        else:
            df = df.sort_values('rating', ascending=False).head(n)
        
        results = []
        for _, show in df.iterrows():
            results.append({
                'showId': int(show['showId']),
                'title': show['title'],
                'genres': show['genres'],
                'vibes': show['vibes'],
                'rating': float(show['rating']),
                'seasons': int(show['seasons']),
                'platform': show['platform']
            })
        
        return results
    
    def recommend_books_by_filters(self,
                                   book_type: Optional[str] = None,
                                   vibes: Optional[List[str]] = None,
                                   max_pages: Optional[int] = None,
                                   min_year: Optional[int] = None,
                                   max_year: Optional[int] = None,
                                   min_rating: Optional[float] = None,
                                   platform: Optional[str] = None,
                                   n: int = 10,
                                   vibes_and_logic: bool = False,
                                   diversity: bool = True,
                                   session_seed: Optional[int] = None) -> List[Dict]:
        
        if self.books_df is None or self.books_df.empty:
            return []
        
        df = self.books_df.copy()
        
        # Apply book type filter
        if book_type:
            type_mask = df['type'].str.lower() == book_type.lower()
            df = df[type_mask]
        
        # Apply vibe filter for books with AND/OR logic
        if vibes:
            if vibes_and_logic:
                # ALL vibes must be present (AND logic)
                vibe_mask = df['vibes'].apply(
                    lambda x: all(v.lower() in str(x).lower() for v in vibes) if pd.notna(x) else False
                )
            else:
                # ANY vibe can be present (OR logic - default)
                vibe_mask = df['vibes'].apply(
                    lambda x: any(v.lower() in str(x).lower() for v in vibes) if pd.notna(x) else False
                )
            df = df[vibe_mask]
        
        # Apply pages filter
        if max_pages:
            df = df[df['pages'] <= max_pages]
        
        # Apply year filters
        if min_year:
            df = df[df['releaseYear'] >= min_year]
        if max_year:
            df = df[df['releaseYear'] <= max_year]
        
        # Apply rating filter
        if min_rating:
            df = df[df['rating'] >= min_rating]
        
        # Apply platform filter
        if platform:
            platform_mask = df['platform'].str.lower().str.contains(platform.lower(), na=False)
            df = df[platform_mask]
        
        # Return empty list gracefully
        if df.empty:
            return []
        
        # Apply diversity sampling (same logic as movies)
        if diversity and len(df) > n:
            if session_seed is None:
                import time
                session_seed = int(time.time() * 1000) % 100000
            
            rng = np.random.RandomState(session_seed)
            df = df.sort_values('rating', ascending=False).reset_index(drop=True)
            
            tier_size = max(len(df) // 3, 1)
            top_tier = df.iloc[:tier_size]
            mid_tier = df.iloc[tier_size:tier_size*2] if len(df) > tier_size else pd.DataFrame()
            lower_tier = df.iloc[tier_size*2:] if len(df) > tier_size*2 else pd.DataFrame()
            
            n_top = min(int(n * 0.6), len(top_tier))
            n_mid = min(int(n * 0.3), len(mid_tier))
            n_lower = n - n_top - n_mid
            
            selected = []
            
            if not top_tier.empty:
                top_weights = (top_tier['rating'].values ** 1.5)
                top_weights = top_weights / top_weights.sum()
                top_indices = rng.choice(len(top_tier), size=min(n_top, len(top_tier)), replace=False, p=top_weights)
                selected.append(top_tier.iloc[top_indices])
            
            if not mid_tier.empty and n_mid > 0:
                mid_indices = rng.choice(len(mid_tier), size=min(n_mid, len(mid_tier)), replace=False)
                selected.append(mid_tier.iloc[mid_indices])
            
            if not lower_tier.empty and n_lower > 0:
                lower_indices = rng.choice(len(lower_tier), size=min(n_lower, len(lower_tier)), replace=False)
                selected.append(lower_tier.iloc[lower_indices])
            
            df = pd.concat(selected, ignore_index=True)
            df = df.sample(frac=1.0, random_state=session_seed).reset_index(drop=True)
        else:
            df = df.sort_values('rating', ascending=False).head(n)
        
        results = []
        for _, book in df.iterrows():
            results.append({
                'bookId': int(book['bookId']),
                'title': book['title'],
                'type': book['type'],
                'vibes': book['vibes'],
                'rating': float(book['rating']),
                'pages': int(book['pages']),
                'year': int(book['releaseYear']),
                'platform': book['platform']
            })
        
        return results
    
    # ============================================================
    # PERSONALIZED RECOMMENDATIONS - FIXED INDEXING
    # ============================================================
    
    def recommend_for_user(self, user_profile: Dict, content_type: str, n: int = 10, session_seed: Optional[int] = None) -> List[Dict]:
        
        if content_type == 'movie':
            return self._recommend_movies_for_user(user_profile, n, session_seed)
        elif content_type == 'show':
            return self._recommend_shows_for_user(user_profile, n, session_seed)
        elif content_type == 'book':
            return self._recommend_books_for_user(user_profile, n, session_seed)
        else:
            return []
    
    def _recommend_movies_for_user(self, user_profile: Dict, n: int, session_seed: Optional[int] = None) -> List[Dict]:
        """
        FIXED: Corrected indexing issue by using enumerate with position index
        """
        if self.movies_df is None or self.movies_df.empty:
            return []
    
        df = self.movies_df.copy()
        
        # Filter out movies user has already added to their list
        watched_movie_ids = user_profile.get('watched_movie_ids', [])
        if watched_movie_ids:
            df = df[~df['movieId'].isin(watched_movie_ids)]
        
        if df.empty:
            return []
        
        # FIXED: Reset index to ensure continuous 0-based indexing
        df = df.reset_index(drop=True)
    
        genre_scores = np.zeros(len(df))
        vibe_scores = np.zeros(len(df))
    
        preferred_genres = user_profile.get('preferred_genres', [])
        preferred_vibes = user_profile.get('preferred_vibes', [])
        genre_weights = user_profile.get('genre_scores', {})
        vibe_weights = user_profile.get('vibe_scores', {})
    
        # FIXED: Use position-based iteration instead of iterrows()
        for pos in range(len(df)):
            row = df.iloc[pos]
            
            # Genre scoring
            genres = str(row['genres']).lower().split(',')
            for genre in preferred_genres:
                if any(genre.lower() in g for g in genres):
                    weight = genre_weights.get(genre, 1.0)
                    genre_scores[pos] += weight
            
            # Vibe scoring
            vibes = str(row['vibes']).lower().split(',')
            for vibe in preferred_vibes:
                if any(vibe.lower() in v for v in vibes):
                    weight = vibe_weights.get(vibe, 1.0)
                    vibe_scores[pos] += weight
    
        if genre_scores.max() > 0:
            genre_scores = genre_scores / genre_scores.max()
        if vibe_scores.max() > 0:
            vibe_scores = vibe_scores / vibe_scores.max()
    
        # Movies are rated out of 10
        rating_scores = df['rating'].values / 10.0
    
        final_scores = (
            0.4 * genre_scores +
            0.3 * vibe_scores +
            0.3 * rating_scores
        )
    
        df['recommendation_score'] = final_scores
        
        if len(df) > n:
            if session_seed is None:
                import time
                session_seed = int(time.time() * 1000) % 100000
            
            rng = np.random.RandomState(session_seed)
            df = df.sort_values('recommendation_score', ascending=False).reset_index(drop=True)
            
            tier_size = max(len(df) // 3, 1)
            top_tier = df.iloc[:tier_size]
            mid_tier = df.iloc[tier_size:tier_size*2] if len(df) > tier_size else pd.DataFrame()
            lower_tier = df.iloc[tier_size*2:] if len(df) > tier_size*2 else pd.DataFrame()
            
            n_top = min(int(n * 0.6), len(top_tier))
            n_mid = min(int(n * 0.3), len(mid_tier))
            n_lower = n - n_top - n_mid
            
            selected = []
            
            if not top_tier.empty:
                top_weights = (top_tier['recommendation_score'].values ** 1.5)
                top_weights = top_weights / top_weights.sum()
                top_indices = rng.choice(len(top_tier), size=min(n_top, len(top_tier)), replace=False, p=top_weights)
                selected.append(top_tier.iloc[top_indices])
            
            if not mid_tier.empty and n_mid > 0:
                mid_indices = rng.choice(len(mid_tier), size=min(n_mid, len(mid_tier)), replace=False)
                selected.append(mid_tier.iloc[mid_indices])
            
            if not lower_tier.empty and n_lower > 0:
                lower_indices = rng.choice(len(lower_tier), size=min(n_lower, len(lower_tier)), replace=False)
                selected.append(lower_tier.iloc[lower_indices])
            
            df = pd.concat(selected, ignore_index=True)
            df = df.sample(frac=1.0, random_state=session_seed).reset_index(drop=True)
        else:
            df = df.sort_values('recommendation_score', ascending=False).head(n)
    
        results = []
        for _, movie in df.iterrows():
            results.append({
                'movieId': int(movie['movieId']),
                'title': movie['title'],
                'genres': movie['genres'],
                'vibes': movie['vibes'],
                'rating': float(movie['rating']),
                'duration': int(movie['duration']),
                'year': int(movie['releaseYear']),
                'platform': movie['platform']
            })
    
        return results

    def _recommend_shows_for_user(self, user_profile: Dict, n: int, session_seed: Optional[int] = None) -> List[Dict]:
        """
        FIXED: Corrected indexing issue by using enumerate with position index
        """
        if self.shows_df is None or self.shows_df.empty:
            return []
    
        df = self.shows_df.copy()
        
        # Filter out shows user has already added to their list
        watched_show_ids = user_profile.get('watched_show_ids', [])
        if watched_show_ids:
            df = df[~df['showId'].isin(watched_show_ids)]
        
        if df.empty:
            return []
        
        # FIXED: Reset index to ensure continuous 0-based indexing
        df = df.reset_index(drop=True)
    
        genre_scores = np.zeros(len(df))
        vibe_scores = np.zeros(len(df))
    
        preferred_genres = user_profile.get('preferred_genres', [])
        preferred_vibes = user_profile.get('preferred_vibes', [])
        genre_weights = user_profile.get('genre_scores', {})
        vibe_weights = user_profile.get('vibe_scores', {})
    
        # FIXED: Use position-based iteration
        for pos in range(len(df)):
            row = df.iloc[pos]
            
            # Genre scoring
            genres = str(row['genres']).lower().split(',')
            for genre in preferred_genres:
                if any(genre.lower() in g for g in genres):
                    weight = genre_weights.get(genre, 1.0)
                    genre_scores[pos] += weight
            
            # Vibe scoring
            vibes = str(row['vibes']).lower().split(',')
            for vibe in preferred_vibes:
                if any(vibe.lower() in v for v in vibes):
                    weight = vibe_weights.get(vibe, 1.0)
                    vibe_scores[pos] += weight
        
        if genre_scores.max() > 0:
            genre_scores = genre_scores / genre_scores.max()
        if vibe_scores.max() > 0:
            vibe_scores = vibe_scores / vibe_scores.max()
    
        # Shows are rated out of 10
        rating_scores = df['rating'].values / 10.0
    
        final_scores = (
            0.4 * genre_scores +
            0.3 * vibe_scores +
            0.3 * rating_scores
        )
    
        df['recommendation_score'] = final_scores
        
        if len(df) > n:
            if session_seed is None:
                import time
                session_seed = int(time.time() * 1000) % 100000
            
            rng = np.random.RandomState(session_seed)
            df = df.sort_values('recommendation_score', ascending=False).reset_index(drop=True)
            
            tier_size = max(len(df) // 3, 1)
            top_tier = df.iloc[:tier_size]
            mid_tier = df.iloc[tier_size:tier_size*2] if len(df) > tier_size else pd.DataFrame()
            lower_tier = df.iloc[tier_size*2:] if len(df) > tier_size*2 else pd.DataFrame()
            
            n_top = min(int(n * 0.6), len(top_tier))
            n_mid = min(int(n * 0.3), len(mid_tier))
            n_lower = n - n_top - n_mid
            
            selected = []
            
            if not top_tier.empty:
                top_weights = (top_tier['recommendation_score'].values ** 1.5)
                top_weights = top_weights / top_weights.sum()
                top_indices = rng.choice(len(top_tier), size=min(n_top, len(top_tier)), replace=False, p=top_weights)
                selected.append(top_tier.iloc[top_indices])
            
            if not mid_tier.empty and n_mid > 0:
                mid_indices = rng.choice(len(mid_tier), size=min(n_mid, len(mid_tier)), replace=False)
                selected.append(mid_tier.iloc[mid_indices])
            
            if not lower_tier.empty and n_lower > 0:
                lower_indices = rng.choice(len(lower_tier), size=min(n_lower, len(lower_tier)), replace=False)
                selected.append(lower_tier.iloc[lower_indices])
            
            df = pd.concat(selected, ignore_index=True)
            df = df.sample(frac=1.0, random_state=session_seed).reset_index(drop=True)
        else:
            df = df.sort_values('recommendation_score', ascending=False).head(n)
    
        results = []
        for _, show in df.iterrows():
            results.append({
                'showId': int(show['showId']),
                'title': show['title'],
                'genres': show['genres'],
                'vibes': show['vibes'],
                'rating': float(show['rating']),
                'seasons': int(show['seasons']),
                'platform': show['platform']
            })
    
        return results

    def _recommend_books_for_user(self, user_profile: Dict, n: int, session_seed: Optional[int] = None) -> List[Dict]:
        """
        FIXED: Corrected indexing issue by using enumerate with position index
        """
        if self.books_df is None or self.books_df.empty:
            return []
    
        df = self.books_df.copy()
        
        # Filter out books user has already added to their list
        read_book_ids = user_profile.get('read_book_ids', [])
        if read_book_ids:
            df = df[~df['bookId'].isin(read_book_ids)]
        
        if df.empty:
            return []
        
        # FIXED: Reset index to ensure continuous 0-based indexing
        df = df.reset_index(drop=True)
    
        vibe_scores = np.zeros(len(df))
    
        preferred_vibes = user_profile.get('preferred_vibes', [])
        vibe_weights = user_profile.get('vibe_scores', {})
    
        # FIXED: Use position-based iteration
        for pos in range(len(df)):
            row = df.iloc[pos]
            
            # Vibe scoring
            vibes = str(row['vibes']).lower().split(',')
            for vibe in preferred_vibes:
                if any(vibe.lower() in v for v in vibes):
                    weight = vibe_weights.get(vibe, 1.0)
                    vibe_scores[pos] += weight
    
        if vibe_scores.max() > 0:
            vibe_scores = vibe_scores / vibe_scores.max()
    
        # Books are rated out of 5
        rating_scores = df['rating'].values / 5.0
    
        final_scores = (
            0.5 * vibe_scores +
            0.5 * rating_scores
        )
        
        df['recommendation_score'] = final_scores
        
        if len(df) > n:
            if session_seed is None:
                import time
                session_seed = int(time.time() * 1000) % 100000
            
            rng = np.random.RandomState(session_seed)
            df = df.sort_values('recommendation_score', ascending=False).reset_index(drop=True)
            
            tier_size = max(len(df) // 3, 1)
            top_tier = df.iloc[:tier_size]
            mid_tier = df.iloc[tier_size:tier_size*2] if len(df) > tier_size else pd.DataFrame()
            lower_tier = df.iloc[tier_size*2:] if len(df) > tier_size*2 else pd.DataFrame()
            
            n_top = min(int(n * 0.6), len(top_tier))
            n_mid = min(int(n * 0.3), len(mid_tier))
            n_lower = n - n_top - n_mid
            
            selected = []
            
            if not top_tier.empty:
                top_weights = (top_tier['recommendation_score'].values ** 1.5)
                top_weights = top_weights / top_weights.sum()
                top_indices = rng.choice(len(top_tier), size=min(n_top, len(top_tier)), replace=False, p=top_weights)
                selected.append(top_tier.iloc[top_indices])
            
            if not mid_tier.empty and n_mid > 0:
                mid_indices = rng.choice(len(mid_tier), size=min(n_mid, len(mid_tier)), replace=False)
                selected.append(mid_tier.iloc[mid_indices])
            
            if not lower_tier.empty and n_lower > 0:
                lower_indices = rng.choice(len(lower_tier), size=min(n_lower, len(lower_tier)), replace=False)
                selected.append(lower_tier.iloc[lower_indices])
            
            df = pd.concat(selected, ignore_index=True)
            df = df.sample(frac=1.0, random_state=session_seed).reset_index(drop=True)
        else:
            df = df.sort_values('recommendation_score', ascending=False).head(n)
        
        results = []
        for _, book in df.iterrows():
            results.append({
                'bookId': int(book['bookId']),
                'title': book['title'],
                'type': book['type'],
                'vibes': book['vibes'],
                'rating': float(book['rating']),
                'pages': int(book['pages']),
                'year': int(book['releaseYear']),
                'platform': book['platform']
            })
    
        return results