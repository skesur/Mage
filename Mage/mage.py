import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Import custom modules
from database.db_manager import DatabaseManager
from database.data_loader import DataLoader
from models.recommender import RecommendationEngine
from models.user_profiler import UserProfiler
from utils.auth import authenticate_user, create_user
from utils.helpers import (
    parse_user_query, format_recommendations,
    calculate_watch_time, calculate_reading_time, generate_summary_stats, format_single_item
)
from utils.notifications import NotificationManager

# ============ PAGE CONFIGURATION ============

st.set_page_config(
    page_title="Mage - Entertainment Manager",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ CUSTOM CSS ============

def load_css():
    
    st.markdown("""
        <style>
        :root {
            --primary-bg: #0e1117;
            --primary-orange: #FF5F15;
            --success-green: #00C896;
            --text-primary: #FFFFFF;
        }
        
        [data-testid="collapsedControl"] {
            display: block !important;
            color: #FF5F15 !important;
        }
        
        footer {visibility: hidden;}
        
        header[data-testid="stHeader"] {
            background-color: transparent;
            height: 3rem;
        }
        
        .stApp {
            background-color: #0e1117;
        }
        
        [data-testid="stSidebar"] {
            background-color: #141922;
        }
        
        .stTextInput input, .stTextArea textarea {
            background-color: #2A2828;
            color: white;
            border: 1px solid #FF5F15;
            border-radius: 8px;
        }
        
        [data-testid="stChatInput"] button {
            background-color: #FF5F15 !important;
            color: white !important;
        }
        
        .stButton button {
            background-color: #FF5F15;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: 600;
        }
        
        .stButton button:hover {
            background-color: #E54D0C;
        }
        
        .recommendation-card {
            background-color: #2A2828;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 4px solid #FF5F15;
        }
        
        .stat-box {
            background: linear-gradient(135deg, #2A2828 0%, #1E1C1C 100%);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        
        .stat-number {
            font-size: 36px;
            font-weight: bold;
            color: #FF5F15;
        }
        
        .stat-label {
            color: #B8B8B8;
            font-size: 14px;
            margin-top: 5px;
        }
        
        .insight-box {
            background-color: rgba(255, 95, 21, 0.1);
            border-left: 4px solid #FF5F15;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
        }
        
        .rating-box {
            background-color: rgba(255, 95, 21, 0.15);
            border: 2px solid #FF5F15;
            padding: 20px;
            border-radius: 12px;
            margin: 15px 0;
        }
        </style>
    """, unsafe_allow_html=True)

# ============ INITIALIZATION ============

@st.cache_resource
def initialize_app():
    
    db = DatabaseManager()
    loader = DataLoader()
    movies_df, shows_df, books_df = loader.load_all()
    
    if db.get_all_movies().empty and not movies_df.empty:
        db.insert_movies(movies_df)
        print("Loaded movies into database")
    
    if db.get_all_shows().empty and not shows_df.empty:
        db.insert_shows(shows_df)
        print("Loaded shows into database")
    
    if db.get_all_books().empty and not books_df.empty:
        db.insert_books(books_df)
        print("Loaded books into database")
    
    recommender = RecommendationEngine()
    
    movies_df = db.get_all_movies()
    shows_df = db.get_all_shows()
    books_df = db.get_all_books()
    
    if not movies_df.empty or not shows_df.empty or not books_df.empty:
        recommender.fit(movies_df, shows_df, books_df)
        print("Trained recommendation model")
    
    profiler = UserProfiler()
    notifier = NotificationManager()
    
    return db, recommender, profiler, notifier

db_manager, recommendation_engine, user_profiler, notification_manager = initialize_app()

# ============ SESSION STATE ============

def init_session_state():
    
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_data' not in st.session_state:
        st.session_state.user_data = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_chat' not in st.session_state:
        st.session_state.current_chat = 1
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'chat'
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = None
    if 'last_recommendations' not in st.session_state:
        st.session_state.last_recommendations = []
    if 'last_content_type' not in st.session_state:
        st.session_state.last_content_type = None
    
    # FIXED: Session state for searched items
    if 'last_searched_item' not in st.session_state:
        st.session_state.last_searched_item = None
    if 'last_searched_type' not in st.session_state:
        st.session_state.last_searched_type = None
    
    # UPDATED: Notification preferences
    if 'email_notifications_enabled' not in st.session_state:
        st.session_state.email_notifications_enabled = False
    if 'completion_reminders' not in st.session_state:
        st.session_state.completion_reminders = True
    if 'new_recommendations_notify' not in st.session_state:
        st.session_state.new_recommendations_notify = False
    
    # UPDATED: Email configuration status
    if 'email_configured' not in st.session_state:
        st.session_state.email_configured = False
    if 'configured_email' not in st.session_state:
        st.session_state.configured_email = ''

init_session_state()

# ============ AUTHENTICATION PAGES ============

def auth_page():
    
    st.markdown("<h1 style='text-align: center; color: #FF5F15;'>🎬 Mage</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #B8B8B8;'>Your Intelligent Entertainment Manager</p>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        st.markdown("### Welcome Back!")
        
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                if not username or not password:
                    st.error("All fields are required!")
                else:
                    success, user_data = authenticate_user(username, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user_data = user_data
                        
                        chat_history = db_manager.get_chat_history(user_data['userId'])
                        st.session_state.current_chat = max(chat_history, default=0) + 1
                        st.session_state.messages = []
                        
                        _load_user_profile()
                        
                        # UPDATED: Load notification settings and email credentials on login
                        notif_settings = db_manager.get_notification_settings(user_data['userId'])
                        if notif_settings:
                            st.session_state.email_notifications_enabled = notif_settings.get('email_enabled', False)
                            st.session_state.completion_reminders = notif_settings.get('completion_reminders', True)
                            st.session_state.new_recommendations_notify = notif_settings.get('new_recommendations', False)
                            
                            # UPDATED: Load and configure email credentials from database
                            sender_email = notif_settings.get('sender_email', '')
                            sender_password = notif_settings.get('sender_password', '')
                            
                            if sender_email and sender_password:
                                notification_manager.sender_email = sender_email
                                notification_manager.sender_password = sender_password
                                notification_manager.enabled = True
                                st.session_state.email_configured = True
                                st.session_state.configured_email = sender_email
                        
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password!")
    
    with tab2:
        st.markdown("### Create Account")
        
        with st.form("signup_form"):
            username = st.text_input("Username", key="signup_username")
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")
            submit = st.form_submit_button("Sign Up", use_container_width=True)
            
            if submit:
                if not username or not email or not password:
                    st.error("All fields are required!")
                elif password != confirm_password:
                    st.error("Passwords do not match!")
                else:
                    success, message, user_id = create_user(username, email, password)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

# ============ HELPER FUNCTIONS ============

def _load_user_profile():
    
    if not st.session_state.logged_in:
        return
    
    user_id = st.session_state.user_data['userId']
    profile = db_manager.get_user_preferences(user_id)
    
    if not profile:
        movie_list = db_manager.get_user_movie_list(user_id)
        show_list = db_manager.get_user_show_list(user_id)
        book_list = db_manager.get_user_book_list(user_id)
        
        profile = user_profiler.analyze_user_history(movie_list, show_list, book_list)
        db_manager.save_user_preferences(user_id, profile)
    
    st.session_state.user_profile = profile

def _update_user_profile():
    
    if not st.session_state.logged_in:
        return
    
    user_id = st.session_state.user_data['userId']
    
    movie_list = db_manager.get_user_movie_list(user_id)
    show_list = db_manager.get_user_show_list(user_id)
    book_list = db_manager.get_user_book_list(user_id)
    
    profile = user_profiler.analyze_user_history(movie_list, show_list, book_list)
    db_manager.save_user_preferences(user_id, profile)
    st.session_state.user_profile = profile

def _send_notification_if_enabled(notification_type: str, **kwargs):
    
    # Check if notifications are enabled
    if not st.session_state.email_notifications_enabled:
        return False
    
    if not notification_manager.enabled:
        return False
    
    user_email = st.session_state.user_data['userEmail']
    user_name = st.session_state.user_data['userName']
    
    try:
        if notification_type == 'add':
            # When item is added to list
            if not st.session_state.new_recommendations_notify:
                return False
            
            item_title = kwargs.get('item_title', 'Unknown')
            content_type = kwargs.get('content_type', 'item')
            
            subject = f"Added to Your {content_type.capitalize()} List"
            body = f"""
Hello {user_name}!

You've added a new {content_type} to your list:

{item_title}

Great choice! Don't forget to update your progress as you watch/read.

View your list: Log in to Mage

Best regards,
Mage Team
            """
            
            return notification_manager.send_email(user_email, subject, body)
        
        elif notification_type == 'complete':
            # When item is marked as completed
            if not st.session_state.completion_reminders:
                return False
            
            item_title = kwargs.get('item_title', 'Unknown')
            content_type = kwargs.get('content_type', 'item')
            user_rating = kwargs.get('user_rating')
            
            return notification_manager.send_completion_congratulations(
                user_email, 
                item_title, 
                user_rating, 
                content_type
            )
        
        elif notification_type == 'recommendation':
            # When new recommendations are generated
            if not st.session_state.new_recommendations_notify:
                return False
            
            recommendations = kwargs.get('recommendations', [])
            content_type = kwargs.get('content_type', 'movie')
            
            if recommendations:
                return notification_manager.send_recommendations_email(
                    user_email,
                    recommendations,
                    content_type
                )
        
        return False
        
    except Exception as e:
        print(f"Error sending notification: {e}")
        return False

# ============ CHAT BOT ============

def process_chat_message(user_message: str) -> str:
    
    intent = parse_user_query(user_message)
    
    action = intent['action']
    content_type = intent['content_type']
    filters = intent['filters']
    search_title = intent.get('search_title')
    item_number = intent.get('item_number')
    
    user_id = st.session_state.user_data['userId']
    user_profile = st.session_state.user_profile or {}
    
    # ========== HANDLE ADD SEARCHED ITEM ==========
    if action == 'add_to_list' and not item_number:
        # Check if user wants to add the last searched item
        if hasattr(st.session_state, 'last_searched_item') and st.session_state.last_searched_item:
            selected_item = st.session_state.last_searched_item
            content_type = st.session_state.last_searched_type
            
            success = False
            if content_type == 'movie':
                movie_id = selected_item.get('movieId')
                success = db_manager.add_to_movie_list(user_id, movie_id)
            elif content_type == 'show':
                show_id = selected_item.get('showId')
                success = db_manager.add_to_show_list(user_id, show_id)
            elif content_type == 'book':
                book_id = selected_item.get('bookId')
                success = db_manager.add_to_book_list(user_id, book_id)
            
            if success:
                title = selected_item.get('title', 'Item')
                _update_user_profile()
                
                # Send notification if enabled
                _send_notification_if_enabled(
                    'add',
                    item_title=title,
                    content_type=content_type
                )
                
                # Clear the searched item after adding
                st.session_state.last_searched_item = None
                st.session_state.last_searched_type = None
                
                return f" '{title}' has been added to your {content_type} list!"
            else:
                return f"This {content_type} is already in your list!"
        else:
            return "No item found to add. Please search for a specific item first (e.g., 'tell me about Inception')"
    
    # ========== HANDLE ADD BY NUMBER ==========
    if action == 'add_by_number' and item_number is not None:
        if not hasattr(st.session_state, 'last_recommendations') or not st.session_state.last_recommendations:
            return "No recent recommendations found. Please ask for recommendations first!"
        
        if item_number < 1 or item_number > len(st.session_state.last_recommendations):
            return f"Invalid number. Please choose between 1 and {len(st.session_state.last_recommendations)}."
        
        selected_item = st.session_state.last_recommendations[item_number - 1]
        content_type = st.session_state.last_content_type
        
        success = False
        if content_type == 'movie':
            movie_id = selected_item.get('movieId')
            success = db_manager.add_to_movie_list(user_id, movie_id)
        elif content_type == 'show':
            show_id = selected_item.get('showId')
            success = db_manager.add_to_show_list(user_id, show_id)
        elif content_type == 'book':
            book_id = selected_item.get('bookId')
            success = db_manager.add_to_book_list(user_id, book_id)
        
        if success:
            title = selected_item.get('title', 'Item')
            _update_user_profile()
            
            # Send notification if enabled
            _send_notification_if_enabled(
                'add',
                item_title=title,
                content_type=content_type
            )
            return f"'{title}' has been added to your {content_type} list!"
        else:
            return f"This {content_type} is already in your list!"
    
    # ========== HANDLE SEARCH BY TITLE ==========
    if action == 'search_title' and search_title:
        content_types_to_try = [content_type] if content_type else ['movie', 'show', 'book']
        
        for ct in content_types_to_try:
            result = recommendation_engine.search_by_title(search_title, ct)
            if result:
                # FIXED: Store the searched item for easy adding
                st.session_state.last_searched_item = result
                st.session_state.last_searched_type = ct
                
                formatted_result = format_single_item(result, ct)
                # Add instructions for adding the item
                formatted_result += "\n\n To add this to your list, type 'add' or 'add it'"
                return formatted_result
        
        return f"Sorry, I couldn't find '{search_title}' in our database. Try a different search term!"
    
    # ========== HANDLE RECOMMENDATIONS ==========
    if action == 'recommend':
        if not content_type:
            return "Please specify whether you want movie, show, or book recommendations!"
        
        # FIXED: Get the requested count from intent (default to 5 if not specified)
        requested_count = intent.get('requested_count', 5)
        
        if filters:
            if content_type == 'movie':
                recommendations = recommendation_engine.recommend_movies_by_filters(
                    genres=filters.get('genres'),
                    vibes=filters.get('vibes'),
                    max_duration=filters.get('max_duration'),
                    min_year=filters.get('min_year'),
                    max_year=filters.get('max_year'),
                    min_rating=filters.get('min_rating'),
                    platform=filters.get('platform'),
                    n=requested_count  # FIXED: Use requested count
                )
            elif content_type == 'show':
                recommendations = recommendation_engine.recommend_shows_by_filters(
                    genres=filters.get('genres'),
                    vibes=filters.get('vibes'),
                    min_year=filters.get('min_year'),
                    max_year=filters.get('max_year'),
                    min_rating=filters.get('min_rating'),
                    platform=filters.get('platform'),
                    n=requested_count  # FIXED: Use requested count
                )
            else:
                recommendations = recommendation_engine.recommend_books_by_filters(
                    book_type=filters.get('book_type'),
                    vibes=filters.get('vibes'),
                    max_pages=filters.get('max_duration'),
                    min_year=filters.get('min_year'),
                    max_year=filters.get('max_year'),
                    min_rating=filters.get('min_rating'),
                    platform=filters.get('platform'),
                    n=requested_count  # FIXED: Use requested count
                )
        else:
            # Personalized recommendations (excludes watched items)
            recommendations = recommendation_engine.recommend_for_user(
                user_profile, content_type, n=requested_count  # FIXED: Use requested count
            )
        
        if recommendations:
            st.session_state.last_recommendations = recommendations
            st.session_state.last_content_type = content_type
            
            return format_recommendations(recommendations, content_type)
        else:
            return f"Sorry, I couldn't find any {content_type}s matching your criteria. Try different filters!"
    
    # ========== HANDLE GET STATUS ==========
    elif action == 'get_status':
        stats = db_manager.get_user_statistics(user_id)
        
        response = "**Your Entertainment Stats:**\n\n"
        response += f"**Movies:**\n"
        response += f"- Total: {stats['movies']['total']}\n"
        response += f"- Completed: {stats['movies']['completed']}\n"
        response += f"- Avg Rating: {stats['movies']['avg_rating']}/10\n\n"
        
        response += f"**TV Shows:**\n"
        response += f"- Total: {stats['shows']['total']}\n"
        response += f"- Completed: {stats['shows']['completed']}\n"
        response += f"- Avg Rating: {stats['shows']['avg_rating']}/10\n\n"
        
        response += f"**Books:**\n"
        response += f"- Total: {stats['books']['total']}\n"
        response += f"- Completed: {stats['books']['completed']}\n"
        response += f"- Avg Rating: {stats['books']['avg_rating']}/5\n"
        
        return response
    
    # ========== DEFAULT HELP MESSAGE ==========
    else:
        return """I can help you with:
- Recommending movies, shows, or books (e.g., "recommend action movies")
- Searching for specific titles (e.g., "tell me about Jurassic Park")
- Adding items to your list (type the number after recommendations)
- Viewing your statistics (e.g., "show my stats")

Try asking: "Recommend action movies under 2 hours" or "Tell me about Inception"
"""

# ============ MAIN CHAT PAGE ============

def chat_page():
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask me about movies, shows, or books..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        db_manager.save_chat_message(
            st.session_state.user_data['userId'],
            st.session_state.current_chat,
            prompt,
            "user"
        )
        
        response = process_chat_message(prompt)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
        
        db_manager.save_chat_message(
            st.session_state.user_data['userId'],
            st.session_state.current_chat,
            response,
            "assistant"
        )
                            
# ============ MY LISTS PAGE  ============

def my_lists_page():
    
    st.markdown("## My Lists")
    
    tab1, tab2, tab3 = st.tabs(["Movies", "Shows", "Books"])
    
    user_id = st.session_state.user_data['userId']
    
    # ========== TAB 1: MOVIES (NO CHANGES - WORKING CORRECTLY) ==========
    with tab1:
        movie_list = db_manager.get_user_movie_list(user_id)
        
        if movie_list.empty:
            st.info("No movies in your list yet! Ask for recommendations to get started.")
        else:
            for _, movie in movie_list.iterrows():
                with st.container():
                    
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.markdown(f"### {movie['title']}")
                        st.caption(f"{movie['genres']} | {movie['vibes']}")
                        st.caption(f"{movie['rating']}/10 | {calculate_watch_time(movie['duration'])}")
                        # FIXED: Show current progress/completion status
                        if movie['status'] == 'Completed' and movie['userRating']:
                            st.caption(f"✅ Completed | Your Rating: {movie['userRating']}/10")
                    
                    with col2:
                        current_status = movie['status']
                        rating_mode_key = f"rating_mode_{movie['listId']}"
            
                        if current_status != "Completed":
                            if rating_mode_key not in st.session_state:
                                st.session_state[rating_mode_key] = False
                        
                            if not st.session_state[rating_mode_key]:
                                new_status = st.selectbox(
                                    "Status",
                                    ["Want to Watch", "Watching", "Completed"],
                                    index=["Want to Watch", "Watching", "Completed"].index(current_status),
                                    key=f"movie_status_{movie['listId']}"
                                )
                            
                                if new_status == "Completed" and current_status != "Completed":
                                    st.session_state[rating_mode_key] = True
                                    st.rerun()
                                elif new_status != current_status:
                                    db_manager.update_movie_status(user_id, movie['movieId'], new_status)
                                    _update_user_profile()
                                    st.success("Status updated!")
                                    st.rerun()
                            else:
                                st.info("Rate this movie!")
                                rating = st.slider(
                                    "Your Rating",
                                    1.0, 10.0, 5.0, 0.5,
                                    key=f"movie_rating_{movie['listId']}"
                                )
                            
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    if st.button("✅", key=f"submit_rating_{movie['listId']}"):
                                        db_manager.update_movie_status(
                                            user_id, movie['movieId'], "Completed", rating
                                        )
                                        _update_user_profile()
                                    
                                        _send_notification_if_enabled(
                                            'complete',
                                            item_title=movie['title'],
                                            content_type='movie',
                                            user_rating=rating
                                        )
                                        st.session_state[rating_mode_key] = False
                                        st.success(f"Rated {rating}/10!")
                                        st.rerun()
                            
                                with col_b:
                                    if st.button("❌", key=f"cancel_rating_{movie['listId']}"):
                                        st.session_state[rating_mode_key] = False
                                        st.rerun()
                    
                    with col3:
                        if st.button("Remove", key=f"remove_movie_{movie['listId']}"):
                            db_manager.remove_from_movie_list(user_id, movie['movieId'])
                            st.success("Removed!")
                            st.rerun()
                    
    
    # ========== TAB 2: TV SHOWS (FIXED VERSION) ==========
    with tab2:
        show_list = db_manager.get_user_show_list(user_id)
        
        if show_list.empty:
            st.info("No shows in your list yet!")
        else:
            for _, show in show_list.iterrows():
                with st.container():
                    
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.markdown(f"### {show['title']}")
                        st.caption(f"{show['genres']} | {show['vibes']}")
                        st.caption(f"{show['rating']}/10 | {show['seasons']} seasons")
                        
                        # FIXED: Show current progress/completion status
                        if show['status'] == 'Completed' and show['userRating']:
                            st.caption(f"✅ Completed | Your Rating: {show['userRating']}/10")
                        else:
                            st.caption(f"Progress: S{show['currentSeason']}E{show['currentEpisode']}")
                    
                    with col2:
                        rating_mode_key = f"show_rating_mode_{show['listId']}"
                        
                        # FIXED: Initialize rating mode state
                        if rating_mode_key not in st.session_state:
                            st.session_state[rating_mode_key] = False
                        
                        # FIXED: Check if already completed - show status instead of inputs
                        if show['status'] == 'Completed' and show['userRating']:
                            pass
                        elif st.session_state[rating_mode_key]:
                            # Rating mode - show is complete, get rating
                            st.info("Rate this show!")
                            show_rating = st.slider(
                                "Your Rating",
                                1.0, 10.0, 7.0, 0.5,
                                key=f"show_final_rating_{show['listId']}"
                            )
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                if st.button("✅", key=f"submit_show_rating_{show['listId']}"):
                                    # FIXED: Use the ACTUAL current values from the database row
                                    current_season = int(show['currentSeason'])
                                    current_episode = int(show['currentEpisode'])
                                    
                                    # Mark as Completed with rating
                                    success = db_manager.update_show_progress(
                                        user_id, show['showId'], 
                                        current_season, current_episode,
                                        status="Completed", 
                                        rating=show_rating
                                    )
                                    
                                    if success:
                                        st.session_state[rating_mode_key] = False
                                        _update_user_profile()
                                        
                                        _send_notification_if_enabled(
                                            'complete',
                                            item_title=show['title'],
                                            content_type='show',
                                            user_rating=show_rating
                                        )
                                        st.success(f"Rated {show_rating}/10!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to save rating")
                            
                            with col_b:
                                if st.button("❌", key=f"skip_show_rating_{show['listId']}"):
                                    st.session_state[rating_mode_key] = False
                                    st.rerun()
                        else:
                            # Normal progress tracking mode
                            current_season = st.number_input(
                                "Season",
                                min_value=1,
                                max_value=int(show['seasons']),
                                value=int(show['currentSeason']),
                                key=f"show_season_{show['listId']}"
                            )
                            
                            current_episode = st.number_input(
                                "Episode",
                                min_value=1,
                                max_value=int(show['episodesPerSeason']),
                                value=int(show['currentEpisode']),
                                key=f"show_episode_{show['listId']}"
                            )
                            
                            if st.button("Update Progress", key=f"update_show_{show['listId']}"):
                                total_episodes = show['seasons'] * show['episodesPerSeason']
                                watched_episodes = (current_season - 1) * show['episodesPerSeason'] + current_episode
                                
                                # Check if show is finished
                                if watched_episodes >= total_episodes:
                                    # FIXED: First update the progress, THEN trigger rating mode
                                    db_manager.update_show_progress(
                                        user_id, show['showId'], 
                                        current_season, current_episode,
                                        status="Watching"  # Keep as Watching until rated
                                    )
                                    # Trigger rating mode
                                    st.session_state[rating_mode_key] = True
                                    st.rerun()
                                else:
                                    # Normal progress update
                                    success = db_manager.update_show_progress(
                                        user_id, show['showId'], 
                                        current_season, current_episode,
                                        status="Watching"
                                    )
                                    if success:
                                        st.success("Progress updated!")
                                        _update_user_profile()
                                        st.rerun()
                                    else:
                                        st.error("Failed to update progress")
                    
                    with col3:
                        if st.button("Remove", key=f"remove_show_{show['listId']}"):
                            db_manager.remove_from_show_list(user_id, show['showId'])
                            st.success("Removed!")
                            st.rerun()
                    
    
    # ========== TAB 3: BOOKS (FIXED VERSION) ==========
    with tab3:
        book_list = db_manager.get_user_book_list(user_id)
        
        if book_list.empty:
            st.info("No books in your list yet!")
        else:
            for _, book in book_list.iterrows():
                with st.container():
                    
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.markdown(f"### {book['title']}")
                        st.caption(f"{book['type']} | {book['vibes']}")
                        st.caption(f"{book['rating']}/5 | {book['pages']} pages")
                        
                        # FIXED: Show current progress/completion status
                        if book['status'] == 'Completed' and book['userRating']:
                            st.caption(f"✅ Completed | Your Rating: {book['userRating']}/5")
                        else:
                            progress_pct = (book['currentPage'] / book['pages'] * 100) if book['pages'] > 0 else 0
                            st.caption(f"Progress: {book['currentPage']}/{book['pages']} pages ({progress_pct:.1f}%)")
                    
                    with col2:
                        rating_mode_key = f"book_rating_mode_{book['listId']}"
                        
                        # FIXED: Initialize rating mode state
                        if rating_mode_key not in st.session_state:
                            st.session_state[rating_mode_key] = False
                        
                        # FIXED: Check if already completed - show status instead of inputs
                        if book['status'] == 'Completed' and book['userRating']:
                            pass
                        elif st.session_state[rating_mode_key]:
                            # Rating mode - book is complete, get rating
                            st.info("Rate this book!")
                            book_rating = st.slider(
                                "Your Rating",
                                1.0, 5.0, 3.5, 0.5,
                                key=f"book_final_rating_{book['listId']}"
                            )
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                if st.button("✅", key=f"submit_book_rating_{book['listId']}"):
                                    # FIXED: Use the ACTUAL current page from the database row
                                    current_page = int(book['currentPage'])
                                    
                                    # Mark as Completed with rating
                                    success = db_manager.update_book_progress(
                                        user_id, book['bookId'], 
                                        current_page,
                                        status="Completed", 
                                        rating=book_rating
                                    )
                                    
                                    if success:
                                        st.session_state[rating_mode_key] = False
                                        _update_user_profile()
                                        
                                        _send_notification_if_enabled(
                                            'complete',
                                            item_title=book['title'],
                                            content_type='book',
                                            user_rating=book_rating
                                        )
                                        st.success(f"Rated {book_rating}/5!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to save rating")
                            
                            with col_b:
                                if st.button("❌", key=f"skip_book_rating_{book['listId']}"):
                                    st.session_state[rating_mode_key] = False
                                    st.rerun()
                        else:
                            # Normal progress tracking mode
                            current_page = st.number_input(
                                "Current Page",
                                min_value=0,
                                max_value=int(book['pages']),
                                value=int(book['currentPage']),
                                key=f"book_page_{book['listId']}"
                            )
                            
                            if st.button("Update Progress", key=f"update_book_{book['listId']}"):
                                # Check if book is finished
                                if current_page >= book['pages']:
                                    # FIXED: First update the progress, THEN trigger rating mode
                                    db_manager.update_book_progress(
                                        user_id, book['bookId'], 
                                        current_page, 
                                        status="Reading"  # Keep as Reading until rated
                                    )
                                    # Trigger rating mode
                                    st.session_state[rating_mode_key] = True
                                    st.rerun()
                                else:
                                    # Normal progress update
                                    success = db_manager.update_book_progress(
                                        user_id, book['bookId'], 
                                        current_page, 
                                        status="Reading"
                                    )
                                    if success:
                                        st.success("Progress updated!")
                                        _update_user_profile()
                                        st.rerun()
                                    else:
                                        st.error("Failed to update progress")
                    
                    with col3:
                        if st.button("Remove", key=f"remove_book_{book['listId']}"):
                            db_manager.remove_from_book_list(user_id, book['bookId'])
                            st.success("Removed!")
                            st.rerun()
                            st.rerun()

# ============ DASHBOARD PAGE ============

def dashboard_page():
    
    st.markdown("## Your Dashboard")
    
    user_id = st.session_state.user_data['userId']
    stats = db_manager.get_user_statistics(user_id)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f'<div class="stat-number">{stats["movies"]["completed"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Movies Watched</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="stat-number">{stats["shows"]["completed"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Shows Completed</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="stat-number">{stats["books"]["completed"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Books Finished</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.session_state.user_profile:
        st.markdown("### Your Insights")
        
        insights = user_profiler.generate_insights(st.session_state.user_profile)
        
        for insight in insights:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

        st.markdown("---")
        
        st.markdown("### Your Preferences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Favorite Genres:**")
            genres = st.session_state.user_profile.get('preferred_genres', [])[:5]
            for genre in genres:
                st.markdown(f"{genre}")
        
        with col2:
            st.markdown("**Favorite Vibes:**")
            vibes = st.session_state.user_profile.get('preferred_vibes', [])[:5]
            for vibe in vibes:
                st.markdown(f"{vibe}")

# ============ SETTINGS PAGE (UPDATED WITH PERSISTENT NOTIFICATIONS) ============

def settings_page():
    
    st.markdown("## Settings")
    
    tab1, tab2, tab3 = st.tabs(["Profile", "Notifications", "About"])
    
    user_id = st.session_state.user_data['userId']
    
    with tab1:
        st.markdown("### User Profile")
        user = st.session_state.user_data
        
        st.text_input("Username", value=user['userName'], disabled=True)
        st.text_input("Email", value=user['userEmail'], disabled=True)
        
        st.markdown("---")
        
        st.markdown("### Change Password")
        with st.form("change_password_form"):
            old_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_new = st.text_input("Confirm New Password", type="password")
            
            if st.form_submit_button("Update Password"):
                if new_password != confirm_new:
                    st.error("Passwords don't match!")
                else:
                    from utils.auth import update_password
                    success, msg = update_password(user['userId'], old_password, new_password)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
    
    with tab2:
        st.markdown("### Notification Preferences")
        
        # ========== UPDATED EMAIL CONFIGURATION SECTION ==========
        st.markdown("#### Email Configuration")
        
        # Show user's signup email (read-only)
        st.text_input(
            "Your Email (from signup)",
            value=st.session_state.user_data['userEmail'],
            disabled=True,
            help="This email will be used to receive notifications"
        )
        
        # Email configuration expander
        with st.expander("Configure Email Settings (One-time setup)", expanded=not st.session_state.email_configured):
            st.info("""
            **How to get Gmail App Password:**
            1. Go to your Google Account settings
            2. Enable 2-Factor Authentication
            3. Visit: https://myaccount.google.com/apppasswords
            4. Generate a new App Password (select "Mail" and "Other")
            5. Copy the 16-character password and paste below
            
            **Note:** Use your App Password, NOT your regular Gmail password!
            This password will be saved securely and you'll only need to enter it once.
            """)
            
            # UPDATED: Pre-fill with user's signup email
            email_input = st.text_input(
                "Sender Email Address (for sending notifications)",
                value=st.session_state.configured_email or st.session_state.user_data['userEmail'],
                placeholder="your-email@gmail.com",
                key="email_config_input",
                help="The Gmail address you'll use to send notifications (usually same as your signup email)"
            )
            
            password_input = st.text_input(
                "Gmail App Password (16 characters)",
                type="password",
                placeholder="xxxx xxxx xxxx xxxx",
                key="password_config_input",
                help="16-character app-specific password from Google - Enter once and it will be saved"
            )
            
            col_save, col_test = st.columns([1, 1])
            
            with col_save:
                if st.button("Save Email Configuration", use_container_width=True, type="primary"):
                    if email_input and password_input:
                        # Update notification manager
                        notification_manager.sender_email = email_input
                        notification_manager.sender_password = password_input
                        notification_manager.enabled = True
                        
                        # Update session state
                        st.session_state.email_configured = True
                        st.session_state.configured_email = email_input
                        
                        # UPDATED: Save email credentials to database
                        settings = {
                            'email_enabled': st.session_state.email_notifications_enabled,
                            'completion_reminders': st.session_state.completion_reminders,
                            'new_recommendations': st.session_state.new_recommendations_notify,
                            'sender_email': email_input,
                            'sender_password': password_input
                        }
                        db_manager.save_notification_settings(user_id, settings)
                        
                        st.success("Email configuration saved permanently!")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("Please provide both email and app password")
            
            with col_test:
                if st.button("Send Test Email", use_container_width=True):
                    if notification_manager.enabled:
                        with st.spinner("Sending test email..."):
                            test_subject = "🎬 Mage Test Email"
                            test_body = f"""
Hello {st.session_state.user_data['userName']}!

This is a test email from Mage Entertainment Manager.

If you received this, your email configuration is working correctly!

Configured sender email: {notification_manager.sender_email}
Your email: {st.session_state.user_data['userEmail']}

Best regards,
Mage Team
                            """
                            
                            if notification_manager.send_email(
                                st.session_state.user_data['userEmail'],
                                test_subject,
                                test_body
                            ):
                                st.success(f"Test email sent to {st.session_state.user_data['userEmail']}!")
                                st.info("Check your inbox (and spam folder if needed)")
                            else:
                                st.error("Failed to send test email. Check credentials and App Password.")
                    else:
                        st.warning("Please configure and save email settings first")
        
        # Show current email configuration status
        if st.session_state.email_configured:
            st.success(f"Email configured: {st.session_state.configured_email}")
        else:
            st.warning("Email not configured. Expand the section above to set up email notifications.")
        
        st.markdown("---")
        
        # UPDATED: Notification toggle with persistent settings
        st.markdown("#### Enable/Disable Notifications")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            enable_email = st.checkbox(
                "Enable email notifications", 
                value=st.session_state.email_notifications_enabled,
                key="email_checkbox",
                disabled=not st.session_state.email_configured,
                help="Email must be configured first" if not st.session_state.email_configured else "Toggle to enable/disable all email notifications"
            )
        
        with col2:
            if st.button("Save Settings", key="save_notif_btn", type="primary"):
                # IMPROVED: Better handling of email credentials with explicit type conversion
                sender_email = ''
                sender_password = ''
                
                # Get credentials from notification_manager if available
                if notification_manager.sender_email:
                    sender_email = str(notification_manager.sender_email)
                if notification_manager.sender_password:
                    sender_password = str(notification_manager.sender_password)
                
                # Build settings dictionary with explicit type conversion
                settings = {
                    'email_enabled': bool(st.session_state.email_notifications_enabled),
                    'completion_reminders': bool(st.session_state.completion_reminders),
                    'new_recommendations': bool(st.session_state.new_recommendations_notify),
                    'sender_email': sender_email,
                    'sender_password': sender_password
                }
                
                # Debug logging
                print(f"Attempting to save notification settings for user {user_id}")
                print(f"Settings: {settings}")
                
                try:
                    result = db_manager.save_notification_settings(user_id, settings)
                    print(f"Save result: {result}")
                    
                    if result:
                        st.success("Settings saved successfully!")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("Failed to save settings - Database returned False")
                        st.info("Check the terminal/console for detailed error logs")
                except Exception as e:
                    st.error(f"Failed to save settings: {str(e)}")
                    import traceback
                    print("Full traceback:")
                    traceback.print_exc()
                    st.info("Check the terminal/console for detailed error information")
        # Update session state when checkbox changes
        if enable_email != st.session_state.email_notifications_enabled:
            st.session_state.email_notifications_enabled = enable_email
        
        # Show notification type settings only if notifications are enabled
        if st.session_state.email_notifications_enabled:
            st.markdown("---")
            st.markdown("#### Notification Types")
            
            completion = st.checkbox(
                "Completion reminders", 
                value=st.session_state.completion_reminders,
                key="completion_checkbox",
                help="Get congratulations when you complete an item"
            )
            if completion != st.session_state.completion_reminders:
                st.session_state.completion_reminders = completion
            
            recommendations = st.checkbox(
                "New recommendations", 
                value=st.session_state.new_recommendations_notify,
                key="recommendations_checkbox",
                help="Be notified when you add new items to your list"
            )
            if recommendations != st.session_state.new_recommendations_notify:
                st.session_state.new_recommendations_notify = recommendations
            
            st.markdown("---")
            
            st.info(f"""
            **Current Notification Settings:**
            - Email Notifications: {'Enabled' if st.session_state.email_notifications_enabled else 'Disabled'}
            - Completion Reminders: {'Enabled' if st.session_state.completion_reminders else 'Disabled'}
            - New Recommendations: {'Enabled' if st.session_state.new_recommendations_notify else 'Disabled'}
            
            **Tip:** Your settings are automatically saved and will persist across sessions. 
            Click "Save Settings" button above after making any changes.
            """)
    
    with tab3:
        st.markdown("### About Mage")
        st.markdown("""
        **Mage - Entertainment Recommendations Manager**
        
        Version 1.0.0 
        
        Built with:
        - Python
        - Streamlit
        - NumPy & Pandas
        - Scikit-learn
        - SQLite
        
        Features:
        - AI-powered recommendations
        - Progress tracking with persistent ratings
        - Personal insights
        - Persistent email notifications
        - One-time email setup
        - Settings persist across sessions
        - Improved personalization
        
        © 2026 Mage Team
        """)

# ============ SIDEBAR ============

def render_sidebar():
    
    with st.sidebar:
        st.title(f"Hi, {st.session_state.user_data['userName']}")
        st.markdown("---")
        
        if st.button("Chat", use_container_width=True):
            st.session_state.current_page = 'chat'
            st.rerun()
        
        if st.button("My Lists", use_container_width=True):
            st.session_state.current_page = 'lists'
            st.rerun()
        
        if st.button("Dashboard", use_container_width=True):
            st.session_state.current_page = 'dashboard'
            st.rerun()
        
        if st.button("Settings", use_container_width=True):
            st.session_state.current_page = 'settings'
            st.rerun()
        
        st.markdown("---")
        
        if st.session_state.current_page == 'chat':
            st.markdown("### Chat History")
            
            if st.button("New Chat", use_container_width=True):
                chat_history = db_manager.get_chat_history(st.session_state.user_data['userId'])
                st.session_state.current_chat = max(chat_history, default=0) + 1
                st.session_state.messages = []
                st.rerun()
            
            chat_history = db_manager.get_chat_history(st.session_state.user_data['userId'])
            for chat_no in chat_history[:10]:
                if st.button(f"Chat #{chat_no}", key=f"chat_{chat_no}", use_container_width=True):
                    st.session_state.current_chat = chat_no
                    messages = db_manager.get_chat_messages(st.session_state.user_data['userId'], chat_no)
                    st.session_state.messages = [
                        {"role": msg[1], "content": msg[0]} for msg in messages
                    ]
                    st.rerun()
        
        st.markdown("---")
        
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_data = None
            st.session_state.messages = []
            st.session_state.user_profile = None
            st.rerun()

# ============ MAIN APPLICATION ============

def main():
    
    load_css()
    
    if not st.session_state.logged_in:
        auth_page()
    else:
        render_sidebar()
        
        if st.session_state.current_page == 'chat':
            st.markdown(f"## Chat #{st.session_state.current_chat}")
            chat_page()
        elif st.session_state.current_page == 'lists':
            my_lists_page()
        elif st.session_state.current_page == 'dashboard':
            dashboard_page()
        elif st.session_state.current_page == 'settings':
            settings_page()

if __name__ == "__main__":
    main()