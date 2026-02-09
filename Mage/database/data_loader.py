import pandas as pd
import numpy as np
from typing import Tuple
import os

class DataLoader:
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        self.movies_file = os.path.join(data_dir, 'movies.txt')
        self.shows_file = os.path.join(data_dir, 'tv_shows.txt')
        self.books_file = os.path.join(data_dir, 'books.txt')
    
    def load_movies(self) -> pd.DataFrame:
        try:
            # Read TSV file
            df = pd.read_csv(self.movies_file, sep='\t', encoding='utf-8')
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Rename columns to match database schema
            column_mapping = {
                'movie_title': 'title',
                'movie_logline': 'logline',
                'movie_genres': 'genres',
                'movie_vibes': 'vibes',
                'movie_release_year': 'releaseYear',
                'movie_duration': 'duration',
                'movie_ratings': 'rating',
                'movie_platform': 'platform'
            }
            df = df.rename(columns=column_mapping)
            
            # Handle missing values - Demonstrates: fillna, data imputation
            df['logline'] = df['logline'].fillna('No description available')
            df['duration'] = pd.to_numeric(df['duration'], errors='coerce').fillna(120)
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(3.5)
            df['releaseYear'] = pd.to_numeric(df['releaseYear'], errors='coerce').fillna(2000)
            
            # Clean string columns - Demonstrates: String operations
            df['genres'] = df['genres'].str.strip().str.replace('"', '')
            df['vibes'] = df['vibes'].str.strip().str.replace('"', '')
            df['platform'] = df['platform'].str.strip().str.replace('"', '')
            
            # Remove duplicates - Demonstrates: duplicate handling
            df = df.drop_duplicates(subset=['title'])
            
            # Select only needed columns in correct order
            df = df[[
                'title', 'logline', 'genres', 'vibes', 
                'releaseYear', 'duration', 'rating', 'platform'
            ]]
            
            return df
        except FileNotFoundError:
            print(f"Error: {self.movies_file} not found")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading movies: {str(e)}")
            return pd.DataFrame()
    
    def load_shows(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.shows_file, sep='\t', encoding='utf-8')
            df.columns = df.columns.str.strip()
            
            # Rename columns to match database schema
            column_mapping = {
                'show_title': 'title',
                'show_logline': 'logline',
                'show_genres': 'genres',
                'show_vibes': 'vibes',
                'show_release_years': 'releaseYears',
                'show_season': 'seasons',
                'show_eps_per_season': 'episodesPerSeason',
                'show_ratings': 'rating',
                'show_platform': 'platform'
            }
            df = df.rename(columns=column_mapping)
            
            # Handle missing values
            df['logline'] = df['logline'].fillna('No description available')
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(7.0)
            df['seasons'] = pd.to_numeric(df['seasons'], errors='coerce').fillna(1)
            df['episodesPerSeason'] = pd.to_numeric(df['episodesPerSeason'], errors='coerce').fillna(10)
            
            # Clean strings
            df['genres'] = df['genres'].str.strip().str.replace('"', '')
            df['vibes'] = df['vibes'].str.strip().str.replace('"', '')
            df['platform'] = df['platform'].str.strip().str.replace('"', '')
            
            df = df.drop_duplicates(subset=['title'])
            
            # Select only needed columns in correct order
            df = df[[
                'title', 'logline', 'genres', 'vibes', 
                'releaseYears', 'seasons', 'episodesPerSeason', 'rating', 'platform'
            ]]
            
            return df
        except FileNotFoundError:
            print(f"Error: {self.shows_file} not found")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading shows: {str(e)}")
            return pd.DataFrame()
    
    def load_books(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.books_file, sep='\t', encoding='utf-8')
            df.columns = df.columns.str.strip()
            
            # Rename columns to match database schema
            column_mapping = {
                'book_title': 'title',
                'book_logline': 'logline',
                'book_type': 'type',
                'book_vibes': 'vibes',
                'book_release_year': 'releaseYear',
                'book_pages': 'pages',
                'book_ratings': 'rating',
                'book_platform': 'platform'
            }
            df = df.rename(columns=column_mapping)
            
            # Handle missing values
            df['logline'] = df['logline'].fillna('No description available')
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(4.0)
            df['pages'] = pd.to_numeric(df['pages'], errors='coerce').fillna(300)
            df['releaseYear'] = pd.to_numeric(df['releaseYear'], errors='coerce').fillna(2000)
            
            # Clean strings
            df['type'] = df['type'].str.strip().str.replace('"', '')
            df['vibes'] = df['vibes'].str.strip().str.replace('"', '')
            df['platform'] = df['platform'].str.strip().str.replace('"', '')
            
            df = df.drop_duplicates(subset=['title'])
            
            # Select only needed columns in correct order
            df = df[[
                'title', 'logline', 'type', 'vibes', 
                'releaseYear', 'pages', 'rating', 'platform'
            ]]
            
            return df
        except FileNotFoundError:
            print(f"Error: {self.books_file} not found")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading books: {str(e)}")
            return pd.DataFrame()
    
    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        movies = self.load_movies()
        shows = self.load_shows()
        books = self.load_books()
        
        return movies, shows, books
    
    def get_statistics(self) -> dict:
        movies = self.load_movies()
        shows = self.load_shows()
        books = self.load_books()
        
        return {
            'movies': {
                'count': len(movies),
                'avg_rating': movies['rating'].mean() if not movies.empty else 0,
                'avg_duration': movies['duration'].mean() if not movies.empty else 0
            },
            'shows': {
                'count': len(shows),
                'avg_rating': shows['rating'].mean() if not shows.empty else 0,
                'total_seasons': shows['seasons'].sum() if not shows.empty else 0
            },
            'books': {
                'count': len(books),
                'avg_rating': books['rating'].mean() if not books.empty else 0,
                'avg_pages': books['pages'].mean() if not books.empty else 0
            }
        }