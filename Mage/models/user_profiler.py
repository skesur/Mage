import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class UserProfiler:
    
    def __init__(self):
        
        # Classification models for preference prediction
        self.genre_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.rating_predictor = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
        
        # Clustering for user segmentation
        self.user_clusterer = KMeans(n_clusters=5, random_state=42)
        
        # Label encoders
        self.genre_encoder = LabelEncoder()
        self.vibe_encoder = LabelEncoder()
        
        # Trained flag
        self.is_trained = False
    
    def analyze_user_history(self, movie_list: pd.DataFrame, 
                            show_list: pd.DataFrame, 
                            book_list: pd.DataFrame) -> Dict:
        
        profile = {
            'preferred_genres': [],
            'preferred_vibes': [],
            'avg_watch_time': 0,
            'avg_reading_pages': 0,
            'genre_scores': {},
            'vibe_scores': {},
            'consumption_patterns': {},
            'rating_distribution': {},
            'watched_movie_ids': [],
            'watched_show_ids': [],
            'read_book_ids': []
        }
        
        # Collect all IDs from user's lists (not just completed)
        if not movie_list.empty and 'movieId' in movie_list.columns:
            profile['watched_movie_ids'] = movie_list['movieId'].tolist()
        
        if not show_list.empty and 'showId' in show_list.columns:
            profile['watched_show_ids'] = show_list['showId'].tolist()
        
        if not book_list.empty and 'bookId' in book_list.columns:
            profile['read_book_ids'] = book_list['bookId'].tolist()
        
        # Analyze movies
        if not movie_list.empty:
            completed_movies = movie_list[movie_list['status'] == 'Completed']
            
            if not completed_movies.empty:
                # Extract genres - Demonstrates: String parsing, Counter
                all_genres = []
                for genres_str in completed_movies['genres']:
                    if pd.notna(genres_str):
                        genres = [g.strip() for g in str(genres_str).split(',')]
                        all_genres.extend(genres)
                
                genre_counts = Counter(all_genres)
                profile['preferred_genres'].extend([g for g, _ in genre_counts.most_common(5)])
                
                # Calculate genre scores - Demonstrates: Dictionary operations
                total_genres = sum(genre_counts.values())
                for genre, count in genre_counts.items():
                    profile['genre_scores'][genre] = count / total_genres
                
                # Extract vibes
                all_vibes = []
                for vibes_str in completed_movies['vibes']:
                    if pd.notna(vibes_str):
                        vibes = [v.strip() for v in str(vibes_str).split(',')]
                        all_vibes.extend(vibes)
                
                vibe_counts = Counter(all_vibes)
                profile['preferred_vibes'].extend([v for v, _ in vibe_counts.most_common(5)])
                
                # Calculate vibe scores
                total_vibes = sum(vibe_counts.values())
                for vibe, count in vibe_counts.items():
                    profile['vibe_scores'][vibe] = count / total_vibes
                
                # Average watch time - Demonstrates: Mean calculation
                if 'duration' in completed_movies.columns:
                    profile['avg_watch_time'] = completed_movies['duration'].mean()
                
                # Rating distribution - Demonstrates: Value counts, normalization
                if 'userRating' in completed_movies.columns:
                    ratings = completed_movies['userRating'].dropna()
                    if not ratings.empty:
                        profile['rating_distribution']['movies'] = {
                            'mean': float(ratings.mean()),
                            'std': float(ratings.std()),
                            'median': float(ratings.median())
                        }
        
        # Analyze shows
        if not show_list.empty:
            completed_shows = show_list[show_list['status'] == 'Completed']
            
            if not completed_shows.empty:
                # Extract genres
                all_genres = []
                for genres_str in completed_shows['genres']:
                    if pd.notna(genres_str):
                        genres = [g.strip() for g in str(genres_str).split(',')]
                        all_genres.extend(genres)
                
                genre_counts = Counter(all_genres)
                profile['preferred_genres'].extend([g for g, _ in genre_counts.most_common(5)])
                
                # Update genre scores
                total_genres = sum(genre_counts.values())
                for genre, count in genre_counts.items():
                    if genre in profile['genre_scores']:
                        profile['genre_scores'][genre] = (profile['genre_scores'][genre] + count / total_genres) / 2
                    else:
                        profile['genre_scores'][genre] = count / total_genres
                
                # Extract vibes
                all_vibes = []
                for vibes_str in completed_shows['vibes']:
                    if pd.notna(vibes_str):
                        vibes = [v.strip() for v in str(vibes_str).split(',')]
                        all_vibes.extend(vibes)
                
                vibe_counts = Counter(all_vibes)
                profile['preferred_vibes'].extend([v for v, _ in vibe_counts.most_common(5)])
                
                # Update vibe scores
                total_vibes = sum(vibe_counts.values())
                for vibe, count in vibe_counts.items():
                    if vibe in profile['vibe_scores']:
                        profile['vibe_scores'][vibe] = (profile['vibe_scores'][vibe] + count / total_vibes) / 2
                    else:
                        profile['vibe_scores'][vibe] = count / total_vibes
                
                # Rating distribution
                if 'userRating' in completed_shows.columns:
                    ratings = completed_shows['userRating'].dropna()
                    if not ratings.empty:
                        profile['rating_distribution']['shows'] = {
                            'mean': float(ratings.mean()),
                            'std': float(ratings.std()),
                            'median': float(ratings.median())
                        }
        
        # Analyze books
        if not book_list.empty:
            completed_books = book_list[book_list['status'] == 'Completed']
            
            if not completed_books.empty:
                # Extract vibes
                all_vibes = []
                for vibes_str in completed_books['vibes']:
                    if pd.notna(vibes_str):
                        vibes = [v.strip() for v in str(vibes_str).split(',')]
                        all_vibes.extend(vibes)
                
                vibe_counts = Counter(all_vibes)
                profile['preferred_vibes'].extend([v for v, _ in vibe_counts.most_common(5)])
                
                # Update vibe scores
                total_vibes = sum(vibe_counts.values())
                for vibe, count in vibe_counts.items():
                    if vibe in profile['vibe_scores']:
                        profile['vibe_scores'][vibe] = (profile['vibe_scores'][vibe] + count / total_vibes) / 2
                    else:
                        profile['vibe_scores'][vibe] = count / total_vibes
                
                # Average reading pages
                if 'pages' in completed_books.columns:
                    profile['avg_reading_pages'] = completed_books['pages'].mean()
                
                # Rating distribution
                if 'userRating' in completed_books.columns:
                    ratings = completed_books['userRating'].dropna()
                    if not ratings.empty:
                        profile['rating_distribution']['books'] = {
                            'mean': float(ratings.mean()),
                            'std': float(ratings.std()),
                            'median': float(ratings.median())
                        }
        
        # Remove duplicates and get unique preferences
        profile['preferred_genres'] = list(set(profile['preferred_genres']))[:10]
        profile['preferred_vibes'] = list(set(profile['preferred_vibes']))[:10]
        
        # Calculate consumption patterns - Demonstrates: Aggregation
        total_completed = (
            len(movie_list[movie_list['status'] == 'Completed']) +
            len(show_list[show_list['status'] == 'Completed']) +
            len(book_list[book_list['status'] == 'Completed'])
        )
        
        profile['consumption_patterns'] = {
            'total_completed': total_completed,
            'movies_completed': len(movie_list[movie_list['status'] == 'Completed']),
            'shows_completed': len(show_list[show_list['status'] == 'Completed']),
            'books_completed': len(book_list[book_list['status'] == 'Completed']),
            'completion_rate': self._calculate_completion_rate(movie_list, show_list, book_list)
        }
        
        return profile
    
    def _calculate_completion_rate(self, movie_list: pd.DataFrame, 
                                   show_list: pd.DataFrame, 
                                   book_list: pd.DataFrame) -> float:
        
        total_items = len(movie_list) + len(show_list) + len(book_list)
        if total_items == 0:
            return 0.0
        
        completed_items = (
            len(movie_list[movie_list['status'] == 'Completed']) +
            len(show_list[show_list['status'] == 'Completed']) +
            len(book_list[book_list['status'] == 'Completed'])
        )
        
        return (completed_items / total_items) * 100
        user_avg_rating = rated_items['userRating'].mean()
        
        # Adjust based on genre/vibe match
        adjustment = 0
        
        if 'genres' in item_features:
            item_genres = set([g.strip().lower() for g in str(item_features['genres']).split(',')])
            user_genres = set()
            for genres_str in rated_items['genres']:
                if pd.notna(genres_str):
                    user_genres.update([g.strip().lower() for g in str(genres_str).split(',')])
            
            # If genres match, increase prediction
            if item_genres & user_genres:
                adjustment += 0.5
        
        predicted_rating = user_avg_rating + adjustment
        
        # Clip to valid range - Demonstrates: NumPy clip
        return float(np.clip(predicted_rating, 1.0, 5.0))
    
    def cluster_users(self, user_profiles: List[Dict]) -> np.ndarray:
        
        if len(user_profiles) < 5:
            return np.zeros(len(user_profiles))
        
        # Create feature vectors - Demonstrates: Feature engineering
        features = []
        for profile in user_profiles:
            feature_vector = [
                profile.get('avg_watch_time', 0),
                profile.get('avg_reading_pages', 0),
                len(profile.get('preferred_genres', [])),
                len(profile.get('preferred_vibes', [])),
                profile.get('consumption_patterns', {}).get('total_completed', 0)
            ]
            features.append(feature_vector)
        
        features = np.array(features)
        
        # Normalize features - Demonstrates: StandardScaler
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        
        # Fit KMeans - Demonstrates: Clustering
        clusters = self.user_clusterer.fit_predict(features_normalized)
        
        return clusters
    
    def detect_viewing_patterns(self, user_history: pd.DataFrame) -> Dict:
        
        patterns = {
            'temporal': {},
            'behavioral': {},
            'trends': {}
        }
        
        if user_history.empty or 'addedDate' not in user_history.columns:
            return patterns
        
        # Convert to datetime - Demonstrates: DateTime operations
        user_history['addedDate'] = pd.to_datetime(user_history['addedDate'])
        
        # Analyze by day of week - Demonstrates: DateTime extraction
        if not user_history.empty:
            user_history['day_of_week'] = user_history['addedDate'].dt.day_name()
            day_counts = user_history['day_of_week'].value_counts()
            patterns['temporal']['most_active_day'] = day_counts.idxmax() if not day_counts.empty else 'Unknown'
            
            # Analyze by month
            user_history['month'] = user_history['addedDate'].dt.month_name()
            month_counts = user_history['month'].value_counts()
            patterns['temporal']['most_active_month'] = month_counts.idxmax() if not month_counts.empty else 'Unknown'
        
        # Behavioral patterns
        if 'status' in user_history.columns:
            status_counts = user_history['status'].value_counts()
            patterns['behavioral']['status_distribution'] = status_counts.to_dict()
            
            # Calculate abandonment rate
            total = len(user_history)
            watching = len(user_history[user_history['status'].isin(['Watching', 'Reading'])])
            patterns['behavioral']['active_items_percentage'] = (watching / total * 100) if total > 0 else 0
        
        # Detect trends - Demonstrates: Rolling statistics
        if len(user_history) > 5:
            user_history_sorted = user_history.sort_values('addedDate')
            user_history_sorted['items_added'] = 1
            user_history_sorted['cumulative_items'] = user_history_sorted['items_added'].cumsum()
            
            # Calculate growth rate
            recent_items = len(user_history[user_history['addedDate'] > (pd.Timestamp.now() - pd.Timedelta(days=30))])
            older_items = len(user_history[user_history['addedDate'] <= (pd.Timestamp.now() - pd.Timedelta(days=30))])
            
            if older_items > 0:
                growth_rate = ((recent_items - older_items) / older_items) * 100
                patterns['trends']['monthly_growth_rate'] = growth_rate
            else:
                patterns['trends']['monthly_growth_rate'] = 0
        
        return patterns
    
    def calculate_similarity_score(self, user_profile: Dict, item_features: Dict) -> float:
        
        # Extract genres and vibes
        item_genres = set([g.strip().lower() for g in str(item_features.get('genres', '')).split(',')])
        item_vibes = set([v.strip().lower() for v in str(item_features.get('vibes', '')).split(',')])
        
        user_genres = set([g.lower() for g in user_profile.get('preferred_genres', [])])
        user_vibes = set([v.lower() for v in user_profile.get('preferred_vibes', [])])
        
        # Calculate Jaccard similarity - Demonstrates: Set operations
        genre_similarity = len(item_genres & user_genres) / len(item_genres | user_genres) if item_genres | user_genres else 0
        vibe_similarity = len(item_vibes & user_vibes) / len(item_vibes | user_vibes) if item_vibes | user_vibes else 0
        
        # Combine scores
        similarity_score = (genre_similarity * 0.6 + vibe_similarity * 0.4)
        
        return similarity_score
    
    def recommend_similar_users(self, user_id: int, all_users_data: List[Tuple[int, Dict]], n: int = 5) -> List[int]:
        
        if len(all_users_data) < 2:
            return []
        
        current_user_profile = None
        for uid, profile in all_users_data:
            if uid == user_id:
                current_user_profile = profile
                break
        
        if not current_user_profile:
            return []
        
        # Calculate similarity with all users
        similarities = []
        for uid, profile in all_users_data:
            if uid != user_id:
                # Simple similarity based on shared preferences
                shared_genres = set(current_user_profile.get('preferred_genres', [])) & set(profile.get('preferred_genres', []))
                shared_vibes = set(current_user_profile.get('preferred_vibes', [])) & set(profile.get('preferred_vibes', []))
                
                similarity = len(shared_genres) + len(shared_vibes)
                similarities.append((uid, similarity))
        
        # Sort and return top N - Demonstrates: Sorting with lambda
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [uid for uid, _ in similarities[:n]]
    
    def generate_insights(self, user_profile: Dict) -> List[str]:
        
        insights = []
        
        # Genre insights
        if user_profile.get('preferred_genres'):
            top_genres = user_profile['preferred_genres'][:3]
            insights.append(f"You love {', '.join(top_genres)} content")
        
        # Vibe insights
        if user_profile.get('preferred_vibes'):
            top_vibes = user_profile['preferred_vibes'][:3]
            insights.append(f"Your preferred vibes: {', '.join(top_vibes)}")
        
        # Watch time insights
        avg_watch_time = user_profile.get('avg_watch_time', 0)
        if avg_watch_time > 0:
            if avg_watch_time < 90:
                insights.append(f"You prefer shorter content (avg {int(avg_watch_time)} min)")
            elif avg_watch_time < 150:
                insights.append(f"You enjoy standard-length content (avg {int(avg_watch_time)} min)")
            else:
                insights.append(f"You're into epic, longer content (avg {int(avg_watch_time)} min)")
        
        # Reading insights
        avg_pages = user_profile.get('avg_reading_pages', 0)
        if avg_pages > 0:
            if avg_pages < 200:
                insights.append(f"You prefer shorter books (avg {int(avg_pages)} pages)")
            elif avg_pages < 400:
                insights.append(f"You read medium-length books (avg {int(avg_pages)} pages)")
            else:
                insights.append(f"You tackle longer books (avg {int(avg_pages)} pages)")
        
        # Completion rate insights
        consumption = user_profile.get('consumption_patterns', {})
        completion_rate = consumption.get('completion_rate', 0)
        if completion_rate > 0:
            if completion_rate > 80:
                insights.append(f"Excellent completion rate: {completion_rate:.1f}%")
            elif completion_rate > 50:
                insights.append(f"Good completion rate: {completion_rate:.1f}%")
            else:
                insights.append(f"Completion rate: {completion_rate:.1f}% - Maybe try shorter content?")
        
        # Activity insights
        total_completed = consumption.get('total_completed', 0)
        if total_completed > 0:
            insights.append(f"You've completed {total_completed} items so far!")
        
        return insights
    
    def predict_next_preference(self, user_profile: Dict) -> Dict:
        
        predictions = {
            'recommended_genres': [],
            'recommended_vibes': [],
            'predicted_content_type': '',
            'confidence': 0.0
        }
        
        # Analyze consumption patterns
        consumption = user_profile.get('consumption_patterns', {})
        movies_completed = consumption.get('movies_completed', 0)
        shows_completed = consumption.get('shows_completed', 0)
        books_completed = consumption.get('books_completed', 0)
        
        total = movies_completed + shows_completed + books_completed
        
        if total == 0:
            return predictions
        
        # Predict content type based on history - Demonstrates: Probability estimation
        content_probs = {
            'movie': movies_completed / total,
            'show': shows_completed / total,
            'book': books_completed / total
        }
        
        predictions['predicted_content_type'] = max(content_probs, key=content_probs.get)
        predictions['confidence'] = max(content_probs.values())
        
        # Recommend genres based on scores
        genre_scores = user_profile.get('genre_scores', {})
        if genre_scores:
            sorted_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
            predictions['recommended_genres'] = [g for g, _ in sorted_genres[:5]]
        
        # Recommend vibes
        vibe_scores = user_profile.get('vibe_scores', {})
        if vibe_scores:
            sorted_vibes = sorted(vibe_scores.items(), key=lambda x: x[1], reverse=True)
            predictions['recommended_vibes'] = [v for v, _ in sorted_vibes[:5]]
        
        return predictions