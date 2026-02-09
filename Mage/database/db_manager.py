import sqlite3
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
import pandas as pd
import numpy as np
import threading

class DatabaseManager:
    
    def __init__(self, db_name: str = 'Mage.db'):
        self.db_name = db_name
        self._local = threading.local()
        self.initialize_database()
    
    @property
    def connection(self) -> sqlite3.Connection:
        # Each thread gets its own connection
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(self.db_name, check_same_thread=False)
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection
    
    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_name, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
    
    def initialize_database(self) -> None:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                userId INTEGER PRIMARY KEY AUTOINCREMENT,
                userName TEXT UNIQUE NOT NULL,
                userEmail TEXT UNIQUE NOT NULL,
                userPassword TEXT NOT NULL,
                createdAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Movies table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS movies (
                movieId INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                logline TEXT,
                genres TEXT,
                vibes TEXT,
                releaseYear INTEGER,
                duration INTEGER,
                rating REAL,
                platform TEXT
            )
        ''')
        
        # TV Shows table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tvShows (
                showId INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                logline TEXT,
                genres TEXT,
                vibes TEXT,
                releaseYears TEXT,
                seasons INTEGER,
                episodesPerSeason INTEGER,
                rating REAL,
                platform TEXT
            )
        ''')
        
        # Books table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS books (
                bookId INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                logline TEXT,
                type TEXT,
                vibes TEXT,
                releaseYear INTEGER,
                pages INTEGER,
                rating REAL,
                platform TEXT
            )
        ''')
        
        # User Movie List
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS userMovieList (
                listId INTEGER PRIMARY KEY AUTOINCREMENT,
                userId INTEGER NOT NULL,
                movieId INTEGER NOT NULL,
                status TEXT DEFAULT 'Want to Watch',
                userRating REAL,
                addedDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                watchedDate TIMESTAMP,
                FOREIGN KEY (userId) REFERENCES users(userId),
                FOREIGN KEY (movieId) REFERENCES movies(movieId),
                UNIQUE(userId, movieId)
            )
        ''')
        
        # User TV Show List
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS userShowList (
                listId INTEGER PRIMARY KEY AUTOINCREMENT,
                userId INTEGER NOT NULL,
                showId INTEGER NOT NULL,
                status TEXT DEFAULT 'Want to Watch',
                userRating REAL,
                currentSeason INTEGER DEFAULT 1,
                currentEpisode INTEGER DEFAULT 1,
                addedDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completedDate TIMESTAMP,
                FOREIGN KEY (userId) REFERENCES users(userId),
                FOREIGN KEY (showId) REFERENCES tvShows(showId),
                UNIQUE(userId, showId)
            )
        ''')
        
        # User Book List
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS userBookList (
                listId INTEGER PRIMARY KEY AUTOINCREMENT,
                userId INTEGER NOT NULL,
                bookId INTEGER NOT NULL,
                status TEXT DEFAULT 'Want to Read',
                userRating REAL,
                currentPage INTEGER DEFAULT 0,
                addedDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                finishedDate TIMESTAMP,
                FOREIGN KEY (userId) REFERENCES users(userId),
                FOREIGN KEY (bookId) REFERENCES books(bookId),
                UNIQUE(userId, bookId)
            )
        ''')
        
        # User Chat History
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS userHistory (
                historyId INTEGER PRIMARY KEY AUTOINCREMENT,
                userId INTEGER NOT NULL,
                userChatNo INTEGER NOT NULL,
                chatMessage TEXT NOT NULL,
                messageType TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (userId) REFERENCES users(userId)
            )
        ''')
        
        # User Notification Settings table - FIXED: Using snake_case column names
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS userNotificationSettings (
                setting_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER UNIQUE NOT NULL,
                email_enabled INTEGER DEFAULT 0,
                weekly_digest INTEGER DEFAULT 1,
                completion_reminders INTEGER DEFAULT 1,
                new_recommendations INTEGER DEFAULT 0,
                sender_email TEXT,
                sender_password TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(userId)
            )
        ''')
        
        # User Preferences (ML-derived)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS userPreferences (
                prefId INTEGER PRIMARY KEY AUTOINCREMENT,
                userId INTEGER UNIQUE NOT NULL,
                preferredGenres TEXT,
                preferredVibes TEXT,
                avgWatchTime REAL,
                avgReadingPages REAL,
                genreScores TEXT,
                vibeScores TEXT,
                lastUpdated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (userId) REFERENCES users(userId)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    # ============ INSERT METHODS ============
    
    def insert_movies(self, movies_df: pd.DataFrame) -> int:
        conn = self.get_connection()
        cursor = conn.cursor()
        count = 0
        
        try:
            for _, row in movies_df.iterrows():
                cursor.execute('''
                    INSERT OR IGNORE INTO movies 
                    (title, logline, genres, vibes, releaseYear, duration, rating, platform)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['title'],
                    row['logline'],
                    row['genres'],
                    row['vibes'],
                    row['releaseYear'],
                    row['duration'],
                    row['rating'],
                    row['platform']
                ))
                count += 1
            conn.commit()
            return count
        except Exception as e:
            conn.rollback()
            raise Exception(f"Error inserting movies: {str(e)}")
        finally:
            conn.close()
    
    def insert_shows(self, shows_df: pd.DataFrame) -> int:
        conn = self.get_connection()
        cursor = conn.cursor()
        count = 0
        
        try:
            for _, row in shows_df.iterrows():
                cursor.execute('''
                    INSERT OR IGNORE INTO tvShows 
                    (title, logline, genres, vibes, releaseYears, seasons, episodesPerSeason, rating, platform)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['title'],
                    row['logline'],
                    row['genres'],
                    row['vibes'],
                    row['releaseYears'],
                    row['seasons'],
                    row['episodesPerSeason'],
                    row['rating'],
                    row['platform']
                ))
                count += 1
            conn.commit()
            return count
        except Exception as e:
            conn.rollback()
            raise Exception(f"Error inserting shows: {str(e)}")
        finally:
            conn.close()
    
    def insert_books(self, books_df: pd.DataFrame) -> int:
        conn = self.get_connection()
        cursor = conn.cursor()
        count = 0
        
        try:
            for _, row in books_df.iterrows():
                cursor.execute('''
                    INSERT OR IGNORE INTO books 
                    (title, logline, type, vibes, releaseYear, pages, rating, platform)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['title'],
                    row['logline'],
                    row['type'],
                    row['vibes'],
                    row['releaseYear'],
                    row['pages'],
                    row['rating'],
                    row['platform']
                ))
                count += 1
            conn.commit()
            return count
        except Exception as e:
            conn.rollback()
            raise Exception(f"Error inserting books: {str(e)}")
        finally:
            conn.close()
    
    # ============ QUERY METHODS ============
    
    def get_all_movies(self) -> pd.DataFrame:
        conn = self.get_connection()
        try:
            query = "SELECT * FROM movies"
            df = pd.read_sql_query(query, conn)
            return df
        finally:
            conn.close()
    
    def get_all_shows(self) -> pd.DataFrame:
        conn = self.get_connection()
        try:
            query = "SELECT * FROM tvShows"
            df = pd.read_sql_query(query, conn)
            return df
        finally:
            conn.close()
    
    def get_all_books(self) -> pd.DataFrame:
        conn = self.get_connection()
        try:
            query = "SELECT * FROM books"
            df = pd.read_sql_query(query, conn)
            return df
        finally:
            conn.close()
    
    def get_user_movie_list(self, user_id: int) -> pd.DataFrame:
        conn = self.get_connection()
        try:
            query = '''
                SELECT 
                    uml.listId, uml.status, uml.userRating, uml.watchedDate, uml.addedDate,
                    m.movieId, m.title, m.genres, m.vibes, m.releaseYear, m.duration, m.rating, m.platform
                FROM userMovieList uml
                JOIN movies m ON uml.movieId = m.movieId
                WHERE uml.userId = ?
                ORDER BY uml.addedDate DESC
            '''
            df = pd.read_sql_query(query, conn, params=(user_id,))
            return df
        finally:
            conn.close()
    
    def get_user_show_list(self, user_id: int) -> pd.DataFrame:
        conn = self.get_connection()
        try:
            query = '''
                SELECT 
                    usl.listId, usl.status, usl.userRating, usl.currentSeason, usl.currentEpisode,
                    usl.addedDate, usl.completedDate,
                    s.showId, s.title, s.genres, s.vibes, s.seasons, s.episodesPerSeason, s.rating, s.platform
                FROM userShowList usl
                JOIN tvShows s ON usl.showId = s.showId
                WHERE usl.userId = ?
                ORDER BY usl.addedDate DESC
            '''
            df = pd.read_sql_query(query, conn, params=(user_id,))
            return df
        finally:
            conn.close()
    
    def get_user_book_list(self, user_id: int) -> pd.DataFrame:
        conn = self.get_connection()
        try:
            query = '''
                SELECT 
                    ubl.listId, ubl.status, ubl.userRating, ubl.currentPage, 
                    ubl.addedDate, ubl.finishedDate,
                    b.bookId, b.title, b.type, b.vibes, b.pages, b.rating, b.platform
                FROM userBookList ubl
                JOIN books b ON ubl.bookId = b.bookId
                WHERE ubl.userId = ?
                ORDER BY ubl.addedDate DESC
            '''
            df = pd.read_sql_query(query, conn, params=(user_id,))
            return df
        finally:
            conn.close()
    
    # ============ USER LIST MANAGEMENT ============
    
    def add_to_movie_list(self, user_id: int, movie_id: int, status: str = 'Want to Watch') -> bool:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO userMovieList (userId, movieId, status)
                VALUES (?, ?, ?)
            ''', (user_id, movie_id, status))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()
    
    def add_to_show_list(self, user_id: int, show_id: int, status: str = 'Want to Watch') -> bool:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO userShowList (userId, showId, status)
                VALUES (?, ?, ?)
            ''', (user_id, show_id, status))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()
    
    def add_to_book_list(self, user_id: int, book_id: int, status: str = 'Want to Read') -> bool:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO userBookList (userId, bookId, status)
                VALUES (?, ?, ?)
            ''', (user_id, book_id, status))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()
    
    def update_movie_status(self, user_id: int, movie_id: int, status: str, rating: Optional[float] = None) -> bool:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if status == 'Completed' and rating is not None:
                cursor.execute('''
                    UPDATE userMovieList 
                    SET status = ?, userRating = ?, watchedDate = CURRENT_TIMESTAMP
                    WHERE userId = ? AND movieId = ?
                ''', (status, rating, user_id, movie_id))
            else:
                cursor.execute('''
                    UPDATE userMovieList 
                    SET status = ?
                    WHERE userId = ? AND movieId = ?
                ''', (status, user_id, movie_id))
            conn.commit()
            return True
        except Exception:
            return False
        finally:
            conn.close()
    
    def update_show_progress(self, user_id: int, show_id: int, season: int, episode: int, 
                           status: Optional[str] = None, rating: Optional[float] = None) -> bool:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if status and rating:
                cursor.execute('''
                    UPDATE userShowList 
                    SET currentSeason = ?, currentEpisode = ?, status = ?, userRating = ?
                    WHERE userId = ? AND showId = ?
                ''', (season, episode, status, rating, user_id, show_id))
            else:
                cursor.execute('''
                    UPDATE userShowList 
                    SET currentSeason = ?, currentEpisode = ?
                    WHERE userId = ? AND showId = ?
                ''', (season, episode, user_id, show_id))
            conn.commit()
            return True
        except Exception:
            return False
        finally:
            conn.close()
    
    def update_book_progress(self, user_id: int, book_id: int, current_page: int, 
                           status: Optional[str] = None, rating: Optional[float] = None) -> bool:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if status and rating:
                cursor.execute('''
                    UPDATE userBookList 
                    SET currentPage = ?, status = ?, userRating = ?
                    WHERE userId = ? AND bookId = ?
                ''', (current_page, status, rating, user_id, book_id))
            else:
                cursor.execute('''
                    UPDATE userBookList 
                    SET currentPage = ?
                    WHERE userId = ? AND bookId = ?
                ''', (current_page, user_id, book_id))
            conn.commit()
            return True
        except Exception:
            return False
        finally:
            conn.close()
    
    # ============ CHAT HISTORY ============
    
    def save_chat_message(self, user_id: int, chat_no: int, message: str, message_type: str) -> None:
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO userHistory (userId, userChatNo, chatMessage, messageType)
                VALUES (?, ?, ?, ?)
            ''', (user_id, chat_no, message, message_type))
            conn.commit()
        finally:
            conn.close()
    
    def get_chat_history(self, user_id: int) -> List[int]:
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                SELECT DISTINCT userChatNo 
                FROM userHistory 
                WHERE userId = ? 
                ORDER BY userChatNo DESC
            ''', (user_id,))
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def get_chat_messages(self, user_id: int, chat_no: int) -> List[Tuple]:
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                SELECT chatMessage, messageType, timestamp 
                FROM userHistory 
                WHERE userId = ? AND userChatNo = ? 
                ORDER BY timestamp
            ''', (user_id, chat_no))
            return cursor.fetchall()
        finally:
            conn.close()
    
    # ============ USER PREFERENCES ============
    
    def save_user_preferences(self, user_id: int, preferences: Dict) -> None:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        import json
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO userPreferences 
                (userId, preferredGenres, preferredVibes, avgWatchTime, avgReadingPages, 
                 genreScores, vibeScores, lastUpdated)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                user_id,
                json.dumps(preferences.get('preferred_genres', [])),
                json.dumps(preferences.get('preferred_vibes', [])),
                preferences.get('avg_watch_time', 0),
                preferences.get('avg_reading_pages', 0),
                json.dumps(preferences.get('genre_scores', {})),
                json.dumps(preferences.get('vibe_scores', {}))
            ))
            conn.commit()
        finally:
            conn.close()
    
    def get_user_preferences(self, user_id: int) -> Optional[Dict]:
        import json
        
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                SELECT preferredGenres, preferredVibes, avgWatchTime, avgReadingPages,
                       genreScores, vibeScores
                FROM userPreferences
                WHERE userId = ?
            ''', (user_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'preferred_genres': json.loads(row[0]) if row[0] else [],
                    'preferred_vibes': json.loads(row[1]) if row[1] else [],
                    'avg_watch_time': row[2],
                    'avg_reading_pages': row[3],
                    'genre_scores': json.loads(row[4]) if row[4] else {},
                    'vibe_scores': json.loads(row[5]) if row[5] else {}
                }
            return None
        finally:
            conn.close()
    
    # ============ ANALYTICS ============
    
    def get_user_statistics(self, user_id: int) -> Dict:
        stats = {}
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Movies stats
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) as completed,
                    AVG(userRating) as avg_rating
                FROM userMovieList
                WHERE userId = ?
            ''', (user_id,))
            row = cursor.fetchone()
            stats['movies'] = {
                'total': row[0] or 0,
                'completed': row[1] or 0,
                'avg_rating': round(row[2], 2) if row[2] else 0
            }
            
            # Shows stats
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) as completed,
                    AVG(userRating) as avg_rating
                FROM userShowList
                WHERE userId = ?
            ''', (user_id,))
            row = cursor.fetchone()
            stats['shows'] = {
                'total': row[0] or 0,
                'completed': row[1] or 0,
                'avg_rating': round(row[2], 2) if row[2] else 0
            }
            
            # Books stats
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) as completed,
                    AVG(userRating) as avg_rating
                FROM userBookList
                WHERE userId = ?
            ''', (user_id,))
            row = cursor.fetchone()
            stats['books'] = {
                'total': row[0] or 0,
                'completed': row[1] or 0,
                'avg_rating': round(row[2], 2) if row[2] else 0
            }
            
            return stats
        finally:
            conn.close()
    
    # ============ NOTIFICATION SETTINGS METHODS ============
    
    def get_notification_settings(self, user_id: int) -> Optional[Dict]:
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT email_enabled, weekly_digest, completion_reminders, new_recommendations, 
                       sender_email, sender_password
                FROM userNotificationSettings
                WHERE user_id = ?
            ''', (user_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'email_enabled': bool(row[0]),
                    'weekly_digest': bool(row[1]),
                    'completion_reminders': bool(row[2]),
                    'new_recommendations': bool(row[3]),
                    'sender_email': str(row[4]) if row[4] else '',
                    'sender_password': str(row[5]) if row[5] else ''
                }
            return None
        except sqlite3.Error as e:
            print(f"❌ SQLite Error getting notification settings: {e}")
            import traceback
            traceback.print_exc()
            return None
        except Exception as e:
            print(f"❌ General Error getting notification settings: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            if conn:
                conn.close()
    
    def save_notification_settings(self, user_id: int, settings: Dict) -> bool:
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Convert booleans to integers explicitly (0 or 1)
            email_enabled = 1 if settings.get('email_enabled', False) else 0
            weekly_digest = 1 if settings.get('weekly_digest', True) else 0
            completion_reminders = 1 if settings.get('completion_reminders', True) else 0
            new_recommendations = 1 if settings.get('new_recommendations', False) else 0
            sender_email = str(settings.get('sender_email', '')) if settings.get('sender_email') else ''
            sender_password = str(settings.get('sender_password', '')) if settings.get('sender_password') else ''
            
            # Debug logging
            print(f"💾 Saving notification settings for user {user_id}:")
            print(f"   email_enabled: {email_enabled}")
            print(f"   weekly_digest: {weekly_digest}")
            print(f"   completion_reminders: {completion_reminders}")
            print(f"   new_recommendations: {new_recommendations}")
            print(f"   sender_email: {sender_email}")
            print(f"   sender_password: {'***' if sender_password else '(empty)'}")
            
            # Check if record exists
            cursor.execute('''
                SELECT setting_id FROM userNotificationSettings WHERE user_id = ?
            ''', (user_id,))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing record
                cursor.execute('''
                    UPDATE userNotificationSettings 
                    SET email_enabled = ?, 
                        weekly_digest = ?, 
                        completion_reminders = ?, 
                        new_recommendations = ?,
                        sender_email = ?, 
                        sender_password = ?, 
                        last_updated = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                ''', (
                    email_enabled,
                    weekly_digest,
                    completion_reminders,
                    new_recommendations,
                    sender_email,
                    sender_password,
                    user_id
                ))
                print(f"✅ Updated existing settings for user {user_id}")
            else:
                # Insert new record
                cursor.execute('''
                    INSERT INTO userNotificationSettings 
                    (user_id, email_enabled, weekly_digest, completion_reminders, new_recommendations, 
                     sender_email, sender_password, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    user_id,
                    email_enabled,
                    weekly_digest,
                    completion_reminders,
                    new_recommendations,
                    sender_email,
                    sender_password
                ))
                print(f"✅ Inserted new settings for user {user_id}")
            
            conn.commit()
            print(f"✅ Successfully saved notification settings for user {user_id}")
            return True
            
        except sqlite3.IntegrityError as e:
            print(f"❌ SQLite Integrity Error: {e}")
            import traceback
            traceback.print_exc()
            if conn:
                conn.rollback()
            return False
        except sqlite3.Error as e:
            print(f"❌ SQLite Error: {e}")
            import traceback
            traceback.print_exc()
            if conn:
                conn.rollback()
            return False
        except Exception as e:
            print(f"❌ Unexpected Error: {e}")
            import traceback
            traceback.print_exc()
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()
    
    # ============ REMOVE FROM LIST METHODS ============
    
    def remove_from_movie_list(self, user_id: int, movie_id: int) -> bool:
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                DELETE FROM userMovieList
                WHERE userId = ? AND movieId = ?
            ''', (user_id, movie_id))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error removing movie: {e}")
            return False
        finally:
            conn.close()
    
    def remove_from_show_list(self, user_id: int, show_id: int) -> bool:
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                DELETE FROM userShowList
                WHERE userId = ? AND showId = ?
            ''', (user_id, show_id))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error removing show: {e}")
            return False
        finally:
            conn.close()
    
    def remove_from_book_list(self, user_id: int, book_id: int) -> bool:
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                DELETE FROM userBookList
                WHERE userId = ? AND bookId = ?
            ''', (user_id, book_id))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error removing book: {e}")
            return False
        finally:
            conn.close()