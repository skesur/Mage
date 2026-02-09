import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import os

class NotificationManager:
    
    def __init__(self):
        
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        
        # Try multiple sources for email credentials
        self.sender_email = ''
        self.sender_password = ''
        
        # Method 1: Try Streamlit secrets first
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'email' in st.secrets:
                self.sender_email = st.secrets['email'].get('SENDER_EMAIL', '')
                self.sender_password = st.secrets['email'].get('SENDER_PASSWORD', '')
                if st.secrets['email'].get('SMTP_SERVER'):
                    self.smtp_server = st.secrets['email']['SMTP_SERVER']
                if st.secrets['email'].get('SMTP_PORT'):
                    self.smtp_port = st.secrets['email']['SMTP_PORT']
        except:
            pass
        
        # Method 2: Fall back to environment variables
        if not self.sender_email:
            self.sender_email = os.getenv('SENDER_EMAIL', '')
            self.sender_password = os.getenv('SENDER_PASSWORD', '')
        
        self.enabled = False
        if self.sender_email and self.sender_password:
            self.enabled = True
    
    def send_email(self, recipient_email: str, subject: str, body: str, html: bool = False) -> bool:
        
        if not self.enabled:
            print("Email notifications not configured")
            return False
        
        try:
            # Create message - Demonstrates: MIME message creation
            message = MIMEMultipart('alternative')
            message['From'] = self.sender_email
            message['To'] = recipient_email
            message['Subject'] = subject
            
            # Attach body
            if html:
                part = MIMEText(body, 'html')
            else:
                part = MIMEText(body, 'plain')
            
            message.attach(part)
            
            # Send email - Demonstrates: SMTP connection
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(message)
            
            return True
            
        except Exception as e:
            print(f"Error sending email: {str(e)}")
            return False
    
    def send_recommendations_email(self, user_email: str, recommendations: List[Dict], content_type: str) -> bool:
        
        subject = f"Your Personalized {content_type.capitalize()} Recommendations"
        
        # Create HTML body - Demonstrates: HTML generation
        html_body = self._create_recommendations_html(recommendations, content_type)
        
        return self.send_email(user_email, subject, html_body, html=True)
    
    def _create_recommendations_html(self, recommendations: List[Dict], content_type: str) -> str:
        
        html = """
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                .header { background: linear-gradient(135deg, #FF5F15 0%, #E54D0C 100%); 
                          color: white; padding: 20px; text-align: center; }
                .recommendation { background: #f4f4f4; padding: 15px; margin: 10px 0; 
                                 border-radius: 8px; border-left: 4px solid #FF5F15; }
                .title { font-size: 18px; font-weight: bold; color: #FF5F15; }
                .details { color: #666; margin-top: 5px; }
                .rating { color: #FFB800; font-weight: bold; }
                .footer { text-align: center; padding: 20px; color: #999; font-size: 12px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎬 Mage Recommendations</h1>
                <p>Personalized picks just for you!</p>
            </div>
            <div style="padding: 20px;">
        """
        
        for i, item in enumerate(recommendations[:5], 1):
            title = item.get('title', 'Unknown')
            rating = item.get('rating', 0)
            
            html += f'<div class="recommendation">'
            html += f'<div class="title">{i}. {title}</div>'
            html += f'<div class="rating">{rating}/5</div>'
            
            if content_type == 'movie':
                duration = item.get('duration', 0)
                genres = item.get('genres', '')
                html += f'<div class="details">Duration: {duration} min | Genres: {genres}</div>'
            elif content_type == 'show':
                seasons = item.get('seasons', 0)
                genres = item.get('genres', '')
                html += f'<div class="details">Seasons: {seasons} | Genres: {genres}</div>'
            elif content_type == 'book':
                pages = item.get('pages', 0)
                book_type = item.get('type', '')
                html += f'<div class="details">Pages: {pages} | Type: {book_type}</div>'
            
            html += '</div>'
        
        html += """
            </div>
            <div class="footer">
                <p>Powered by Mage - Your Entertainment Manager</p>
                <p>This is an automated message. Please do not reply.</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def send_progress_reminder(self, user_email: str, items: List[Dict]) -> bool:
        
        if not items:
            return False
        
        subject = "Don't forget to finish your shows/books!"
        
        body = "Hello!\n\n"
        body += "You have some items in progress:\n\n"
        
        for item in items[:5]:
            title = item.get('title', 'Unknown')
            status = item.get('status', '')
            body += f"â€¢ {title} ({status})\n"
        
        body += "\nKeep going!\n\n"
        body += "Best regards,\nMage Team"
        
        return self.send_email(user_email, subject, body)
    
    def send_completion_congratulations(self, user_email: str, item_title: str, 
                                       user_rating: Optional[float], content_type: str) -> bool:
        
        subject = f"Congrats on completing {item_title}!"
        
        body = f"""
        Congratulations!
        
        You've completed: {item_title}
        """
        
        body += f"""
        
        Keep exploring! Check out our recommendations for more {content_type}s you might enjoy.
        
        Best regards,
        Mage Team
        """
        
        return self.send_email(user_email, subject, body)
    
    