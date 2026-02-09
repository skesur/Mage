import re
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import pandas as pd


def parse_user_query(query: str) -> Dict:
    
    query_lower = query.lower()

    intent = {
        "action": "unknown",
        "content_type": None,
        "filters": {},
        "entity_id": None,
        "search_title": None,
        "item_number": None,
        "requested_count": 5,  # Default is 5
    }

    # ========== DETECT "ADD NUMBER" ACTION FIRST ==========
    add_number_patterns = [
        r"(?:add|save)\s+(?:movie|show|book)?\s*(\d+)",
        r"^(\d+)$",
        r"(?:number|#)\s*(\d+)",
    ]
    
    for pattern in add_number_patterns:
        match = re.search(pattern, query_lower)
        if match:
            intent["action"] = "add_by_number"
            intent["item_number"] = int(match.group(1))
            return intent
    
    # ========== DETECT "ADD" ACTION FOR SEARCHED ITEMS ==========
    # Detect standalone "add" or "add it" commands (for last searched item)
    if query_lower.strip() in ['add', 'add it', 'add this', 'save it', 'save this', 'save']:
        intent["action"] = "add_to_list"
        return intent

    # ========== PARSE EXPLICIT COUNT ==========
    count_patterns = [
        r"(?:recommend|suggest|show|give)\s+(?:me\s+)?(\d+)\s+",
        r"(\d+)\s+(?:best|great|good|top)?\s*(?:movies|shows|books)",
        r"(?:top|best)\s+(\d+)",
    ]
    
    for pattern in count_patterns:
        match = re.search(pattern, query_lower)
        if match:
            count = int(match.group(1))
            # Limit to reasonable range (1-20)
            intent["requested_count"] = min(max(count, 1), 20)
            break

    # ========== DETECT SPECIFIC ITEM SEARCH ==========
    specific_patterns = [
        r"(?:tell me about|what is|info about|details of|information on|describe)\s+['\"]?(.+?)['\"]?\s*(?:\?|$)",
        r"(?:search for|find|show me)\s+['\"]?(.+?)['\"]?\s*(?:\?|$)",
        r"(?:do you know|have you heard of)\s+['\"]?(.+?)['\"]?\s*(?:\?|$)",
    ]
    
    for pattern in specific_patterns:
        match = re.search(pattern, query_lower)
        if match:
            potential_title = match.group(1).strip()
            potential_title = re.sub(r'\s+(movie|show|book|film|series|the movie|the show|the book)$', '', potential_title)
            potential_title = potential_title.strip('"\'')
            
            if potential_title and len(potential_title) > 2:
                intent["search_title"] = potential_title
                intent["action"] = "search_title"
                
                if any(word in query_lower for word in ["movie", "film"]):
                    intent["content_type"] = "movie"
                elif any(word in query_lower for word in ["show", "series", "tv"]):
                    intent["content_type"] = "show"
                elif any(word in query_lower for word in ["book", "novel"]):
                    intent["content_type"] = "book"
                
                return intent

    # ========== DETECT GENERAL ACTION ==========
    if any(word in query_lower for word in ["recommend", "suggest", "show me", "find", 
                                             "what should i", "give me", "looking for", "want to watch",
                                             "want to read", "need some", "any good"]):
        intent["action"] = "recommend"
    elif any(word in query_lower for word in ["add", "save", "put in", "add to my list"]):
        intent["action"] = "add_to_list"
    elif any(word in query_lower for word in ["track", "progress", "update", "currently", "watching", "reading"]):
        intent["action"] = "update_progress"
    elif any(word in query_lower for word in ["status", "what am i", "my list", "my stats", "statistics"]):
        intent["action"] = "get_status"
    elif any(word in query_lower for word in ["similar", "like", "related to"]):
        intent["action"] = "find_similar"

    # ========== DETECT CONTENT TYPE ==========
    if any(word in query_lower for word in ["movie", "film", "cinema", "flick"]):
        intent["content_type"] = "movie"
    elif any(word in query_lower for word in ["show", "series", "tv", "television", "episode"]):
        intent["content_type"] = "show"
    elif any(word in query_lower for word in ["book", "novel", "reading", "read"]):
        intent["content_type"] = "book"

    # ========== BOOK TYPE FILTER ==========
    if intent["content_type"] == "book":
        if any(word in query_lower for word in ["non-fiction", "nonfiction", "non fiction"]):
            intent["filters"]["book_type"] = "Non-Fiction"
        elif "fiction" in query_lower and "non" not in query_lower:
            intent["filters"]["book_type"] = "Fiction"

    # ========== PLATFORM FILTER ==========
    platforms_map = {
        "netflix": ["netflix"],
        "amazon prime": ["amazon prime", "prime video", "amazon"],
        "disney+": ["disney+", "disney plus", "disney"],
        "hbo": ["hbo", "hbo max"],
        "hulu": ["hulu"],
        "apple tv": ["apple tv", "apple tv+"],
        "paramount": ["paramount", "paramount+"],
        "peacock": ["peacock"],
        "audible": ["audible"],
        "kindle": ["kindle", "kindle unlimited"],
        "scribd": ["scribd"],
        "goodreads": ["goodreads"],
    }
    
    for platform, keywords in platforms_map.items():
        if any(keyword in query_lower for keyword in keywords):
            intent["filters"]["platform"] = platform
            break

    # ========== ENHANCED GENRE DETECTION WITH AND LOGIC ==========
    genres_map = {
        "action": ["action"],
        "comedy": ["comedy", "funny", "humorous"],
        "drama": ["drama", "dramatic"],
        "horror": ["horror", "scary", "spooky"],
        "sci-fi": ["sci-fi", "science fiction", "scifi", "futuristic"],
        "romance": ["romance", "romantic", "love"],
        "thriller": ["thriller", "suspense"],
        "fantasy": ["fantasy", "magical"],
        "mystery": ["mystery", "detective", "crime"],
        "adventure": ["adventure", "exploration"],
        "animation": ["animation", "animated", "cartoon"],
        "war": ["war", "military"],
        "western": ["western"],
        "documentary": ["documentary", "docuseries"],
        "biography": ["biography", "biopic", "biographical"]
    }
    
    found_genres = []
    for genre, keywords in genres_map.items():
        if any(keyword in query_lower for keyword in keywords):
            found_genres.append(genre)
    
    if found_genres:
        intent["filters"]["genres"] = found_genres
        # Set AND logic flag when multiple genres specified
        intent["filters"]["genres_and_logic"] = True

    # ========== ENHANCED VIBE DETECTION ==========
    vibes_map = {
        "dark": ["dark", "gritty", "noir"],
        "light-hearted": ["light-hearted", "lighthearted", "fun", "cheerful"],
        "intense": ["intense", "gripping"],
        "emotional": ["emotional", "touching", "heartwarming"],
        "epic": ["epic", "grand", "sweeping"],
        "wholesome": ["wholesome", "feel-good", "uplifting"],
        "suspenseful": ["suspenseful", "tense"],
        "thought-provoking": ["thought-provoking", "cerebral", "intellectual"],
        "fast-paced": ["fast-paced", "action-packed"],
        "slow-burn": ["slow-burn", "atmospheric"],
        "philosophical": ["philosophical", "deep", "contemplative"],
        "psychological": ["psychological", "mind-bending"],
        "inspiring": ["inspiring", "inspirational", "motivational"],
        "melancholic": ["melancholic", "sad", "depressing"],
        "nostalgic": ["nostalgic", "retro"],
    }
    
    found_vibes = []
    for vibe, keywords in vibes_map.items():
        if any(keyword in query_lower for keyword in keywords):
            found_vibes.append(vibe)
    
    if found_vibes:
        intent["filters"]["vibes"] = found_vibes
        # FIX #4: Set AND logic flag when multiple vibes specified
        intent["filters"]["vibes_and_logic"] = True

    # ========== TIME CONSTRAINTS ==========
    time_patterns = [
        r"(?:under|less than|shorter than|below|max)\s*(\d+)\s*(hour|hr|h|minute|min|m)s?",
        r"(\d+)\s*(hour|hr|h|minute|min|m)s?\s*(?:or less|or under|max)",
    ]
    
    for pattern in time_patterns:
        time_match = re.search(pattern, query_lower)
        if time_match:
            duration = int(time_match.group(1))
            unit = time_match.group(2)

            if "hour" in unit or unit in ["hr", "h"]:
                duration *= 60

            intent["filters"]["max_duration"] = duration
            break

    # ========== YEAR CONSTRAINTS ==========
    year_patterns = [
        r"(?:after|since|from)\s*(\d{4})",
        r"(?:before|prior to|until)\s*(\d{4})",
        r"(?:in|from)\s*(?:the\s*)?(\d{4})s?",
        r"(?:between)\s*(\d{4})\s*(?:and|to|-)\s*(\d{4})",
        r"(?:of|from)\s*(\d{4})",
    ]
    
    if any(word in query_lower for word in ["recent", "new", "latest", "modern"]):
        intent["filters"]["min_year"] = 2015
    elif any(word in query_lower for word in ["old", "classic", "vintage", "retro"]):
        intent["filters"]["max_year"] = 2000
    else:
        for pattern in year_patterns:
            year_match = re.search(pattern, query_lower)
            if year_match:
                if "between" in pattern:
                    intent["filters"]["min_year"] = int(year_match.group(1))
                    intent["filters"]["max_year"] = int(year_match.group(2))
                elif "s" in year_match.group(0) and not any(word in year_match.group(0) for word in ["after", "since", "before"]):
                    decade = int(year_match.group(1))
                    intent["filters"]["min_year"] = decade
                    intent["filters"]["max_year"] = decade + 9
                elif any(word in pattern for word in ["after", "since", "from"]) or "of" in year_match.group(0):
                    year_val = int(year_match.group(1))
                    intent["filters"]["min_year"] = year_val
                    intent["filters"]["max_year"] = year_val
                else:
                    intent["filters"]["max_year"] = int(year_match.group(1))
                break

    # ========== RATING CONSTRAINTS - "BEST/GREAT/GOOD" KEYWORDS ==========
    quality_keywords_high = ["best", "great", "top rated", "highly rated", "excellent", "top"]
    quality_keywords_good = ["good", "decent", "quality"]
    
    if any(word in query_lower for word in quality_keywords_high):
        # High quality threshold
        if intent["content_type"] == "book":
            intent["filters"]["min_rating"] = 4.0  # Books: 4+/5
        else:
            intent["filters"]["min_rating"] = 8.0  # Movies/shows: 8+/10
    elif any(word in query_lower for word in quality_keywords_good):
        # Good quality threshold
        if intent["content_type"] == "book":
            intent["filters"]["min_rating"] = 3.5
        else:
            intent["filters"]["min_rating"] = 7.0
    else:
        # Check for explicit rating patterns
        rating_patterns = [
            r"rating\s*(?:above|over|at least|greater than)\s*(\d+\.?\d*)",
            r"(?:above|over|at least)\s*(\d+\.?\d*)\s*(?:stars|rating)",
        ]
        
        for pattern in rating_patterns:
            rating_match = re.search(pattern, query_lower)
            if rating_match:
                rating_value = float(rating_match.group(1))
                intent["filters"]["min_rating"] = rating_value
                break

    # ========== PAGE CONSTRAINTS FOR BOOKS ==========
    if intent["content_type"] == "book":
        pages_patterns = [
            r"(?:under|less than|shorter than)\s*(\d+)\s*pages?",
            r"(?:quick|short)\s*read",
            r"(?:long|lengthy)\s*read"
        ]
        
        for pattern in pages_patterns:
            pages_match = re.search(pattern, query_lower)
            if pages_match:
                if "quick" in pattern or "short" in pattern:
                    intent["filters"]["max_duration"] = 250
                elif "long" in pattern or "lengthy" in pattern:
                    intent["filters"]["min_duration"] = 400
                else:
                    intent["filters"]["max_duration"] = int(pages_match.group(1))
                break

    return intent


def format_recommendations(recommendations: List[Dict], content_type: str) -> str:
    
    if not recommendations:
        return "I couldn't find any recommendations matching your criteria. Try different filters or ask for general recommendations!"

    output = []
    output.append(f"\n Top {len(recommendations)} {content_type.capitalize()} Recommendations:\n")

    for i, item in enumerate(recommendations, 1):
        title = item.get("title", "Unknown")
        rating = item.get("rating", 0)

        if content_type == "movie":
            duration = item.get("duration", 0)
            genres = item.get("genres", "")
            year = item.get("year", "")
            platform = item.get("platform", "")
            output.append(f"{i}. {title} | {rating:.1f}/10")
            output.append(f"   Duration: {duration} min | Genres: {genres}")
            if year:
                output.append(f" | Year: {year}")
            if platform:
                output.append(f" | Platform: {platform}")

        elif content_type == "show":
            seasons = item.get("seasons", 0)
            genres = item.get("genres", "")
            platform = item.get("platform", "")
            output.append(f"{i}. {title} | {rating:.1f}/10")
            output.append(f"   Seasons: {seasons} | Genres: {genres}")
            if platform:
                output.append(f" | Platform: {platform}")

        elif content_type == "book":
            pages = item.get("pages", 0)
            book_type = item.get("type", "")
            year = item.get("year", "")
            platform = item.get("platform", "")
            output.append(f"{i}. {title} | {rating:.1f}/5")
            output.append(f"   Pages: {pages} | Type: {book_type}")
            if year:
                output.append(f" | Year: {year}")
            if platform:
                output.append(f" | Platform: {platform}")

        output.append("")

    output.append("\n To add to your list, type the number (e.g., '1' or 'add 1')\n")

    return "\n".join(output)


def format_single_item(item: Dict, content_type: str) -> str:
    
    if not item:
        return "Item not found. Try a different search term or check the spelling!"
    
    output = []
    
    title = item.get("title", "Unknown")
    rating = item.get("rating", 0)
    
    output.append(f"\n {content_type.capitalize()} Details:\n")
    output.append(f"**{title}**\n")
    
    if content_type == "movie":
        output.append(f" Rating: {rating:.1f}/10\n")
        output.append(f" Duration: {item.get('duration', 0)} minutes\n")
        output.append(f" Genres: {item.get('genres', 'N/A')}\n")
        output.append(f" Vibes: {item.get('vibes', 'N/A')}\n")
        output.append(f" Release Year: {item.get('releaseYear', 'N/A')}\n")
        output.append(f" Platform: {item.get('platform', 'N/A')}\n")
        if item.get('logline'):
            output.append(f"\n Description:\n{item.get('logline')}\n")
    
    elif content_type == "show":
        output.append(f" Rating: {rating:.1f}/10\n")
        output.append(f" Seasons: {item.get('seasons', 0)}\n")
        output.append(f" Episodes per Season: {item.get('episodesPerSeason', 0)}\n")
        output.append(f" Genres: {item.get('genres', 'N/A')}\n")
        output.append(f" Vibes: {item.get('vibes', 'N/A')}\n")
        output.append(f" Release Years: {item.get('releaseYears', 'N/A')}\n")
        output.append(f" Platform: {item.get('platform', 'N/A')}\n")
        if item.get('logline'):
            output.append(f"\n Description:\n{item.get('logline')}\n")
    
    elif content_type == "book":
        output.append(f" Rating: {rating:.1f}/5\n")
        output.append(f" Type: {item.get('type', 'N/A')}\n")
        output.append(f" Pages: {item.get('pages', 0)}\n")
        output.append(f" Vibes: {item.get('vibes', 'N/A')}\n")
        output.append(f" Release Year: {item.get('releaseYear', 'N/A')}\n")
        output.append(f" Platform: {item.get('platform', 'N/A')}\n")
        if item.get('logline'):
            output.append(f"\n Description:\n{item.get('logline')}\n")
    
    return "\n".join(output)


def calculate_time_remaining(total_duration: int, current_progress: int) -> str:
    
    remaining = total_duration - current_progress

    if remaining <= 0:
        return "Completed!"

    if remaining < 60:
        return f"{remaining} minutes remaining"
    elif remaining < 1440:
        hours = remaining // 60
        minutes = remaining % 60
        return f"{hours}h {minutes}m remaining"
    else:
        days = remaining // 1440
        return f"~{days} days remaining"


def extract_numbers_from_text(text: str) -> List[int]:
    
    numbers = re.findall(r"\d+", text)
    return [int(n) for n in numbers]


def clean_text(text: str) -> str:
    
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?-]", "", text)
    return text.strip()


def calculate_reading_time(pages: int, reading_speed: int = 50) -> str:
    
    hours = pages / reading_speed

    if hours < 1:
        return f"~{int(hours * 60)} minutes"
    elif hours < 24:
        return f"~{int(hours)} hours"
    else:
        return f"~{int(hours / 24)} days"


def calculate_watch_time(duration: int) -> str:
    
    hours = duration // 60
    minutes = duration % 60

    if hours == 0:
        return f"{minutes} min"
    elif minutes == 0:
        return f"{hours}h"
    else:
        return f"{hours}h {minutes}m"


def generate_summary_stats(items: List[Dict]) -> Dict:
    
    if not items:
        return {"total": 0, "avg_rating": 0, "total_time": 0}

    ratings = [item.get("rating", 0) for item in items if item.get("rating")]
    avg_rating = sum(ratings) / len(ratings) if ratings else 0

    total_time = 0
    if "duration" in items[0]:
        total_time = sum(item.get("duration", 0) for item in items)
    elif "pages" in items[0]:
        total_time = sum(item.get("pages", 0) for item in items)

    return {
        "total": len(items),
        "avg_rating": round(avg_rating, 2),
        "total_time": total_time,
    }


def sort_by_multiple_criteria(
    items: List[Dict], criteria: List[Tuple[str, bool]]
) -> List[Dict]:
    
    sorted_items = items.copy()

    for key, ascending in reversed(criteria):
        sorted_items = sorted(
            sorted_items,
            key=lambda x: x.get(key, 0),
            reverse=not ascending,
        )

    return sorted_items


def paginate_results(
    items: List, page: int = 1, per_page: int = 10
) -> Tuple[List, Dict]:
    
    total_items = len(items)
    total_pages = (total_items + per_page - 1) // per_page

    page = max(1, min(page, total_pages if total_pages > 0 else 1))

    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page

    return items[start_idx:end_idx], {
        "current_page": page,
        "total_pages": total_pages,
        "per_page": per_page,
        "total_items": total_items,
        "has_next": page < total_pages,
        "has_prev": page > 1,
    }