import hashlib
import sqlite3
from typing import Tuple, Optional, Dict
import re

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def validate_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password: str) -> Tuple[bool, str]:
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"
    
    if not any(c.isalpha() for c in password):
        return False, "Password must contain at least one letter"
    
    return True, "Password is valid"

def create_user(username: str, email: str, password: str, db_name: str = 'Mage.db') -> Tuple[bool, str, Optional[int]]:
    # Validate inputs
    if not username or not email or not password:
        return False, "All fields are required", None
    
    if len(username) < 3:
        return False, "Username must be at least 3 characters long", None
    
    if not validate_email(email):
        return False, "Invalid email format", None
    
    is_valid, msg = validate_password(password)
    if not is_valid:
        return False, msg, None
    
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        hashed_pwd = hash_password(password)
        
        cursor.execute(
            "INSERT INTO users (userName, userEmail, userPassword) VALUES (?, ?, ?)",
            (username, email, hashed_pwd)
        )
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return True, "Account created successfully!", user_id
        
    except sqlite3.IntegrityError as e:
        if 'userName' in str(e):
            return False, "Username already exists", None
        elif 'userEmail' in str(e):
            return False, "Email already exists", None
        else:
            return False, "Error creating account", None
    except Exception as e:
        return False, f"Error: {str(e)}", None

def authenticate_user(username: str, password: str, db_name: str = 'Mage.db') -> Tuple[bool, Optional[Dict]]:
    if not username or not password:
        return False, None
    
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        hashed_pwd = hash_password(password)
        
        cursor.execute(
            "SELECT userId, userName, userEmail FROM users WHERE userName=? AND userPassword=?",
            (username, hashed_pwd)
        )
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return True, {
                "userId": user[0],
                "userName": user[1],
                "userEmail": user[2]
            }
        
        return False, None
        
    except Exception as e:
        print(f"Authentication error: {str(e)}")
        return False, None

def update_password(user_id: int, old_password: str, new_password: str, db_name: str = 'Mage.db') -> Tuple[bool, str]:
    # Validate new password
    is_valid, msg = validate_password(new_password)
    if not is_valid:
        return False, msg
    
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        # Verify old password
        old_hashed = hash_password(old_password)
        cursor.execute(
            "SELECT userId FROM users WHERE userId=? AND userPassword=?",
            (user_id, old_hashed)
        )
        
        if not cursor.fetchone():
            conn.close()
            return False, "Current password is incorrect"
        
        # Update to new password
        new_hashed = hash_password(new_password)
        cursor.execute(
            "UPDATE users SET userPassword=? WHERE userId=?",
            (new_hashed, user_id)
        )
        
        conn.commit()
        conn.close()
        
        return True, "Password updated successfully"
        
    except Exception as e:
        return False, f"Error updating password: {str(e)}"

def get_user_by_id(user_id: int, db_name: str = 'Mage.db') -> Optional[Dict]:
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT userId, userName, userEmail, createdAt FROM users WHERE userId=?",
            (user_id,)
        )
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return {
                "userId": user[0],
                "userName": user[1],
                "userEmail": user[2],
                "createdAt": user[3]
            }
        
        return None
        
    except Exception as e:
        print(f"Error fetching user: {str(e)}")
        return None