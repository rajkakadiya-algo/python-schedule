# Fix for Python 3.13 compatibility - imghdr was removed
# This MUST be done before importing tweepy
import sys
try:
    import imghdr
except ImportError:
    # Create a minimal imghdr compatibility shim
    from types import ModuleType
    
    # Create a fake imghdr module
    imghdr = ModuleType('imghdr')
    
    def what(file, h=None):
        """Minimal imghdr.what implementation for tweepy compatibility"""
        if hasattr(file, 'read'):
            # File-like object
            header = file.read(32)
            file.seek(0)  # Reset file position
        else:
            # File path string
            try:
                with open(file, 'rb') as f:
                    header = f.read(32)
            except:
                return None
        
        # Simple image format detection based on file headers
        if header.startswith(b'\xff\xd8\xff'):
            return 'jpeg'
        elif header.startswith(b'\x89PNG\r\n\x1a\n'):
            return 'png'
        elif header.startswith(b'GIF8'):
            return 'gif'
        elif header.startswith(b'RIFF') and b'WEBP' in header:
            return 'webp'
        elif header.startswith(b'BM'):
            return 'bmp'
        return None
    
    imghdr.what = what
    sys.modules['imghdr'] = imghdr

from flask import Flask, request, jsonify, send_from_directory
import requests
import tweepy
import json
import os
import time
import random
import tempfile
import urllib.request
import uuid
import logging
import schedule
import pytz
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union, List
from threading import Thread
from PIL import Image
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pymongo
from pymongo import MongoClient
from bson import ObjectId

# Google Cloud Storage imports
from google.cloud import storage
from google.oauth2 import service_account
from google.api_core.exceptions import Conflict

# Configure logging BEFORE using it
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Instagram posting function
# from app import instagram_post

def resize_image_for_instagram(image_url: str) -> str:
    """
    Resize image to Instagram-compatible aspect ratio
    Returns the URL of the resized image
    """
    try:
        import requests
        from PIL import Image
        import tempfile
        import os
        
        # Download the image
        response = requests.get(image_url, timeout=30)
        if response.status_code != 200:
            logger.error(f"Failed to download image for resizing: {response.status_code}")
            return image_url  # Return original URL if download fails
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as original_temp:
            original_temp.write(response.content)
            original_path = original_temp.name
        
        # Open and process the image
        with Image.open(original_path) as img:
            # Convert to RGB if necessary (for RGBA, P mode images)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Create a white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            original_width, original_height = img.size
            original_ratio = original_width / original_height
            
            logger.info(f"Original image dimensions: {original_width}x{original_height}, aspect ratio: {original_ratio:.2f}")
            
            # Instagram aspect ratio requirements
            # Square: 1:1, Portrait: 4:5 (0.8), Landscape: 1.91:1 (1.91)
            
            target_width = 1080
            
            if 0.8 <= original_ratio <= 1.91:
                # Image is already within acceptable range, just resize to target width
                target_height = int(target_width / original_ratio)
                logger.info(f"Image aspect ratio is acceptable, resizing to: {target_width}x{target_height}")
            elif original_ratio < 0.8:
                # Too tall, make it portrait (4:5)
                target_height = int(target_width / 0.8)  # 1350px for 1080px width
                logger.info(f"Image too tall ({original_ratio:.2f}), converting to portrait: {target_width}x{target_height}")
            else:
                # Too wide, make it landscape (1.91:1)
                target_height = int(target_width / 1.91)  # ~566px for 1080px width
                logger.info(f"Image too wide ({original_ratio:.2f}), converting to landscape: {target_width}x{target_height}")
            
            # Resize the image
            resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            # Save the resized image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as resized_temp:
                resized_img.save(resized_temp, format='JPEG', quality=95, optimize=True)
                resized_path = resized_temp.name
            
            # Upload the resized image to Google Cloud Storage
            try:
                with open(resized_path, 'rb') as file:
                    original_filename = image_url.split('/')[-1].split('?')[0]  # Extract filename from URL
                    filename_parts = original_filename.rsplit('.', 1)
                    if len(filename_parts) == 2:
                        base_name, ext = filename_parts
                        resized_filename = f"{base_name}_instagram_resized.jpg"
                    else:
                        resized_filename = f"{original_filename}_instagram_resized.jpg"
                    
                    resized_url = upload_to_gcs(file, resized_filename)
                    
                    if resized_url:
                        logger.info(f"Successfully resized image for Instagram: {original_width}x{original_height} -> {target_width}x{target_height}")
                        return resized_url
                    else:
                        logger.error("Failed to upload resized image to GCS")
                        return image_url
            finally:
                # Clean up temporary files
                try:
                    os.unlink(original_path)
                    os.unlink(resized_path)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temp files: {cleanup_error}")
            
    except Exception as e:
        logger.error(f"Error resizing image for Instagram: {e}")
        return image_url  # Return original URL if resizing fails

def instagram_post(image_path: str, caption: str, user_id: str, access_token: str) -> Optional[str]:
    """
    Post an image to Instagram
    Returns media_id if successful, None if failed
    """
    try:
        # Instagram Graph API endpoint for media upload
        base_url = "https://graph.facebook.com/v22.0"
        
        # First, create a media container
        create_container_url = f"{base_url}/{user_id}/media"
        
        # Prepare the image URL (must be publicly accessible)
        image_url = None
        if image_path.startswith('http'):
            # Use the URL directly for remote images (including Google Cloud Storage URLs)
            image_url = image_path
            logger.info(f"Using remote image URL for Instagram: {image_url[:100]}...")
            
            # Resize the image to Instagram-compatible aspect ratio
            resized_url = resize_image_for_instagram(image_url)
            if resized_url != image_url:
                image_url = resized_url
                logger.info(f"Using resized image URL: {image_url[:100]}...")
        else:
            # For local files, we need to upload to a public URL first
            # Check if the file exists locally
            if os.path.exists(image_path):
                logger.info(f"Local file found, uploading to Google Cloud Storage: {image_path}")
                
                # Upload local file to Google Cloud Storage
                try:
                    with open(image_path, 'rb') as file:
                        filename = os.path.basename(image_path)
                        gcs_url = upload_to_gcs(file, filename)
                        if gcs_url:
                            # Resize the uploaded image for Instagram
                            image_url = resize_image_for_instagram(gcs_url)
                            logger.info(f"Successfully uploaded to GCS and resized for Instagram: {image_url[:100]}...")
                        else:
                            logger.error("Failed to upload local file to Google Cloud Storage")
                            return None
                except Exception as upload_error:
                    logger.error(f"Error uploading local file to GCS: {upload_error}")
                    return None
            else:
                logger.error(f"Local file not found: {image_path}")
                return None
        
        if not image_url:
            logger.error("No valid image URL available for Instagram posting")
            return None
        
        container_params = {
            'image_url': image_url,
            'caption': caption,
            'access_token': access_token
        }
        
        logger.info(f"Creating Instagram media container with URL: {image_url[:100]}...")
        response = requests.post(create_container_url, data=container_params)
        result = response.json()
        
        if 'id' not in result:
            error_msg = result.get('error', {})
            error_code = error_msg.get('code')
            error_message = error_msg.get('message', 'Unknown error')
            
            # Specific handling for aspect ratio errors
            if error_code == 36003:
                logger.error(f"Instagram aspect ratio error: {error_message}")
                logger.info("Attempting to resize image for Instagram compatibility...")
                
                # Try resizing the image again with more aggressive settings
                try:
                    resized_url = resize_image_for_instagram(image_url)
                    if resized_url != image_url:
                        logger.info(f"Retrying with newly resized image: {resized_url[:100]}...")
                        container_params['image_url'] = resized_url
                        response = requests.post(create_container_url, data=container_params)
                        result = response.json()
                        
                        if 'id' in result:
                            creation_id = result['id']
                            logger.info(f"Media container created successfully after resize with ID: {creation_id}")
                        else:
                            logger.error(f"Still failed after resize: {result}")
                            return None
                    else:
                        logger.error("Image resizing did not produce a different URL")
                        return None
                except Exception as resize_error:
                    logger.error(f"Error during retry resize: {resize_error}")
                    return None
            else:
                logger.error(f"Failed to create media container: {result}")
                return None
        else:
            creation_id = result['id']
            logger.info(f"Media container created with ID: {creation_id}")
        
        # Now publish the container
        publish_url = f"{base_url}/{user_id}/media_publish"
        publish_params = {
            'creation_id': creation_id,
            'access_token': access_token
        }
        
        logger.info(f"Publishing Instagram media container: {creation_id}")
        response = requests.post(publish_url, data=publish_params)
        result = response.json()
        
        if 'id' in result:
            media_id = result['id']
            logger.info(f"Instagram post published successfully with media ID: {media_id}")
            return media_id
        else:
            logger.error(f"Failed to publish media: {result}")
            return None
            
    except Exception as e:
        logger.error(f"Instagram post error: {e}")
        return None

app = Flask(__name__, static_folder='.')

# Enhanced CORS configuration for better compatibility
CORS(app, resources={
    r"/api/*": {
        "origins": ["https://algosocialai.vercel.app", "http://localhost:3000", "http://localhost:5000"],
        "methods": ["GET", "POST", "DELETE", "OPTIONS", "PUT", "PATCH"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With", "Accept", "Origin"],
        "supports_credentials": True,
        "expose_headers": ["Content-Type", "Authorization"]
    }
})

# Add manual CORS headers for additional compatibility
@app.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    if origin in ["https://algosocialai.vercel.app", "http://localhost:3000", "http://localhost:5000"]:
        response.headers.add('Access-Control-Allow-Origin', origin)
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With,Accept,Origin')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,DELETE,OPTIONS,PUT,PATCH')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        response.headers.add('Access-Control-Expose-Headers', 'Content-Type,Authorization')
    return response

# Explicit OPTIONS handler for preflight requests
@app.route('/api/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    origin = request.headers.get('Origin')
    if origin in ["https://algosocialai.vercel.app", "http://localhost:3000", "http://localhost:5000"]:
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', origin)
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With,Accept,Origin')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,DELETE,OPTIONS,PUT,PATCH')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        response.headers.add('Access-Control-Max-Age', '86400')
        return response
    return jsonify({'error': 'Origin not allowed'}), 403

# Add these configurations after app initialization
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# MongoDB Configuration
MONGODB_URI = "mongodb+srv://rahultummaalgobrainai:Iqq4Vs9zpWHoIfLK@algowhatsapp.84eyv9q.mongodb.net/Al-SocialMedia?retryWrites=true&w=majority"
DB_NAME = "Al-SocialMedia"
COLLECTION_NAME = "posts"

# Google Cloud Storage Configuration
JSON_KEY_PATH = "sm1.json"  # Path to service account key file
GCS_BUCKET_NAME = "my-new-static-bucket-123456"  # Use a globally unique bucket name

# Initialize MongoDB client
try:
    mongo_client = MongoClient(MONGODB_URI)
    db = mongo_client[DB_NAME]
    scheduled_posts_collection = db[COLLECTION_NAME]
    logger.info("MongoDB connection established successfully")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    mongo_client = None
    db = None
    scheduled_posts_collection = None

# Initialize Google Cloud Storage client
try:
    # Check if we have environment variable for credentials (Production)
    gcs_credentials_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    if gcs_credentials_json:
        # Production: Use environment variable
        import json
        credentials_info = json.loads(gcs_credentials_json)
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        gcs_client = storage.Client(credentials=credentials, project=credentials.project_id)
        logger.info("Google Cloud Storage initialized with environment credentials")
    else:
        # Development: Use JSON file (fallback)
        if os.path.exists(JSON_KEY_PATH):
            credentials = service_account.Credentials.from_service_account_file(JSON_KEY_PATH)
            gcs_client = storage.Client(credentials=credentials, project=credentials.project_id)
            logger.info("Google Cloud Storage initialized with JSON file credentials")
        else:
            logger.error("No Google Cloud credentials found. Please set GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable or add sm1.json file.")
            gcs_client = None
            gcs_bucket = None
    
    # Try to create bucket if client was initialized successfully
    if gcs_client:
        try:
            gcs_bucket = gcs_client.create_bucket(GCS_BUCKET_NAME)
            logger.info(f"✅ Bucket '{GCS_BUCKET_NAME}' created.")
        except Conflict:
            gcs_bucket = gcs_client.bucket(GCS_BUCKET_NAME)
            logger.info(f"ℹ️ Bucket '{GCS_BUCKET_NAME}' already exists. Using it.")
        except Exception as create_error:
            # If creation fails with other error, just get bucket reference
            gcs_bucket = gcs_client.bucket(GCS_BUCKET_NAME)
            logger.info(f"Using existing Google Cloud Storage bucket '{GCS_BUCKET_NAME}'")
        
        logger.info("Google Cloud Storage connection established successfully")
except Exception as e:
    logger.error(f"Failed to connect to Google Cloud Storage: {e}")
    gcs_client = None
    gcs_bucket = None

# Timezone configuration for India (IST)
IST = pytz.timezone('Asia/Kolkata')

def convert_ist_to_utc(ist_datetime_str: str) -> datetime:
    """Convert IST datetime string to UTC datetime object"""
    try:
        # Parse the datetime string
        naive_datetime = datetime.strptime(ist_datetime_str, '%Y-%m-%d %H:%M')
        # Localize to IST
        ist_datetime = IST.localize(naive_datetime)
        # Convert to UTC
        utc_datetime = ist_datetime.astimezone(pytz.UTC)
        return utc_datetime
    except Exception as e:
        logger.error(f"Error converting IST to UTC: {e}")
        raise ValueError(f"Invalid datetime format: {ist_datetime_str}")

def get_current_ist_time() -> datetime:
    """Get current time in IST"""
    return datetime.now(IST)

def validate_schedule_time_ist(schedule_time_str: str, platform: str = 'facebook') -> Optional[datetime]:
    """
    Validate schedule time in IST and return UTC datetime object.
    Returns None if invalid, UTC datetime if valid.
    """
    try:
        # Parse the datetime string in IST
        naive_datetime = datetime.strptime(schedule_time_str, '%Y-%m-%d %H:%M')
        ist_datetime = IST.localize(naive_datetime)
        
        # Get current time in IST
        current_ist = get_current_ist_time()
        
        # Debug logging
        logger.info(f"Schedule time: {ist_datetime}")
        logger.info(f"Current IST time: {current_ist}")
        logger.info(f"Platform: {platform}")
        
        # For Facebook, check if schedule time is at least 10 minutes in the future
        if platform.lower() == 'facebook':
            min_schedule_time = current_ist + timedelta(minutes=10)
            if ist_datetime < min_schedule_time:
                logger.error(f"Facebook posts must be scheduled at least 10 minutes in the future. Required: {min_schedule_time}, Provided: {ist_datetime}")
                return None
        else:
            # For other platforms, just check if it's in the future (allow 1 minute tolerance)
            min_schedule_time = current_ist + timedelta(minutes=1)
            if ist_datetime < min_schedule_time:
                logger.error(f"Schedule time must be in the future. Current: {current_ist}, Provided: {ist_datetime}")
                return None
        
        # Check if schedule time is not more than 6 months in the future
        max_schedule_time = current_ist + timedelta(days=180)
        if ist_datetime > max_schedule_time:
            logger.error(f"Schedule time cannot be more than 6 months in the future")
            return None
        
        # Convert to UTC for storage
        utc_datetime = ist_datetime.astimezone(pytz.UTC)
        return utc_datetime
        
    except ValueError as e:
        logger.error(f"Invalid datetime format: {e}")
        return None

class MongoDBHelper:
    """Helper class for MongoDB operations related to scheduled posts"""
    
    @staticmethod
    def create_scheduled_post(user_id: str, content: Dict, platforms: List[str], 
                            schedule_time: datetime, post_type: str = "scheduled",
                            ai_generation: Optional[Dict] = None) -> Optional[str]:
        """
        Create a new scheduled post in MongoDB
        Returns: post_id if successful, None if failed
        """
        try:
            if scheduled_posts_collection is None:
                logger.error("MongoDB collection not available")
                return None
            
            # Prepare platform data
            platform_data = []
            for platform in platforms:
                platform_data.append({
                    "_id": ObjectId(),
                    "platform": platform,
                    "status": "scheduled"
                })
            
            # Create post document
            post_document = {
                "userId": ObjectId(user_id) if user_id else None,
                "content": content,
                "postType": post_type,
                "schedule": {
                    "publishAt": schedule_time
                },
                "platform": platform_data,
                "creditUsed": 0,  # Will be updated when post is published
                "createdAt": datetime.now(pytz.UTC),
                "updatedAt": datetime.now(pytz.UTC)
            }
            
            # Add AI generation info if provided
            if ai_generation:
                post_document["aiGeneration"] = ai_generation
            
            # Insert document
            result = scheduled_posts_collection.insert_one(post_document)
            
            if result.inserted_id:
                logger.info(f"Scheduled post created with ID: {result.inserted_id}")
                return str(result.inserted_id)
            else:
                logger.error("Failed to create scheduled post")
                return None
                
        except Exception as e:
            logger.error(f"Error creating scheduled post in MongoDB: {e}")
            return None
    
    @staticmethod
    def create_scheduled_post_with_times(user_id: str, content: Dict, platforms: List[str], 
                            platform_schedule_times: Dict[str, Dict], post_type: str = "scheduled",
                            ai_generation: Optional[Dict] = None) -> Optional[str]:
        """
        Create a new scheduled post in MongoDB with different schedule times for each platform
        
        Args:
            user_id: User ID
            content: Post content (caption, hashtags, images)
            platforms: List of platform names
            platform_schedule_times: Dict with platform as key and dict with ist_time and utc_time
            post_type: Type of post (default: "scheduled")
            ai_generation: Optional AI generation metadata
            
        Returns: post_id if successful, None if failed
        """
        try:
            if scheduled_posts_collection is None:
                logger.error("MongoDB collection not available")
                return None
            
            # Prepare platform data with individual schedule times
            platform_data = []
            for platform in platforms:
                time_data = platform_schedule_times.get(platform, {})
                platform_data.append({
                    "_id": ObjectId(),
                    "platform": platform,
                    "status": "scheduled",
                    "scheduledTime": time_data.get('utc_time'),  # Store UTC time for execution
                    "scheduledTimeIST": time_data.get('ist_time')  # Store IST time for reference
                })
            
            # Create post document
            post_document = {
                "userId": ObjectId(user_id) if user_id else None,
                "content": content,
                "postType": post_type,
                "schedule": {
                    "publishAt": min([platform_schedule_times[p]['utc_time'] for p in platforms])  # Use earliest time for general reference
                },
                "platform": platform_data,
                "creditUsed": 0,  # Will be updated when post is published
                "createdAt": datetime.now(pytz.UTC),
                "updatedAt": datetime.now(pytz.UTC),
                "hasCustomTimes": True  # Flag to indicate platform-specific times
            }
            
            # Add AI generation info if provided
            if ai_generation:
                post_document["aiGeneration"] = ai_generation
            
            # Insert document
            result = scheduled_posts_collection.insert_one(post_document)
            
            if result.inserted_id:
                logger.info(f"Scheduled post with platform times created with ID: {result.inserted_id}")
                return str(result.inserted_id)
            else:
                logger.error("Failed to create scheduled post with platform times")
                return None
                
        except Exception as e:
            logger.error(f"Error creating scheduled post with platform times in MongoDB: {e}")
            return None
    
    @staticmethod
    def update_post_status(post_id: str, platform: str, status: str, 
                          published_id: Optional[str] = None) -> bool:
        """
        Update the status of a specific platform for a scheduled post
        """
        try:
            if scheduled_posts_collection is None:
                return False
            
            update_data = {
                "platform.$.status": status,
                "updatedAt": datetime.now(pytz.UTC)
            }
            
            if published_id:
                update_data["platform.$.publishedId"] = published_id
            
            result = scheduled_posts_collection.update_one(
                {
                    "_id": ObjectId(post_id),
                    "platform.platform": platform
                },
                {
                    "$set": update_data
                }
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error updating post status in MongoDB: {e}")
            return False
    
    @staticmethod
    def get_scheduled_posts_for_execution() -> List[Dict]:
        """
        Get all scheduled posts that are ready for execution
        Handles both old format (single time) and new format (platform-specific times)
        """
        try:
            if scheduled_posts_collection is None:
                return []
            
            # Get current time in UTC with proper timezone information
            current_time = datetime.now(pytz.UTC)
            ist_time = current_time.astimezone(pytz.timezone('Asia/Kolkata'))
            logger.info(f"Getting scheduled posts for execution at: {current_time} UTC ({ist_time.strftime('%Y-%m-%d %H:%M:%S')} IST)")
            
            # Find posts that are scheduled and due for execution
            # This query handles both old and new formats
            query = {
                "postType": "scheduled",
                "$or": [
                    # Old format: posts with single schedule time
                    {
                        "hasCustomTimes": {"$ne": True},
                        "schedule.publishAt": {"$lte": current_time},
                        "platform": {
                            "$elemMatch": {
                                "status": "scheduled"
                            }
                        }
                    },
                    # New format: posts with platform-specific times  
                    {
                        "hasCustomTimes": True,
                        "platform": {
                            "$elemMatch": {
                                "status": "scheduled",
                                "scheduledTime": {"$lte": current_time}
                            }
                        }
                    }
                ]
            }
            
            logger.info(f"MongoDB query: {query}")
            posts = scheduled_posts_collection.find(query)
            
            posts_list = list(posts)
            logger.info(f"Found {len(posts_list)} posts ready for execution")
            
            # Post-process to ensure datetime fields are timezone-aware
            for post in posts_list:
                # Fix schedule.publishAt if it exists and is naive
                if post.get('schedule', {}).get('publishAt'):
                    publish_at = post['schedule']['publishAt']
                    if hasattr(publish_at, 'tzinfo') and publish_at.tzinfo is None:
                        post['schedule']['publishAt'] = pytz.UTC.localize(publish_at)
                
                # Fix platform scheduledTime fields if they exist and are naive
                if post.get('platform'):
                    for platform_info in post['platform']:
                        if platform_info.get('scheduledTime'):
                            scheduled_time = platform_info['scheduledTime']
                            if hasattr(scheduled_time, 'tzinfo') and scheduled_time.tzinfo is None:
                                platform_info['scheduledTime'] = pytz.UTC.localize(scheduled_time)
            
            return posts_list
            
        except Exception as e:
            logger.error(f"Error fetching scheduled posts for execution: {e}")
            return []
    
    @staticmethod
    def get_user_scheduled_posts(user_id: str, limit: int = 50) -> List[Dict]:
        """
        Get scheduled posts for a specific user
        """
        try:
            if scheduled_posts_collection is None:
                return []
            
            posts = scheduled_posts_collection.find(
                {"userId": ObjectId(user_id)},
                sort=[("createdAt", -1)],
                limit=limit
            )
            
            return list(posts)
            
        except Exception as e:
            logger.error(f"Error fetching user scheduled posts: {e}")
            return []
    
    @staticmethod
    def delete_scheduled_post(post_id: str, user_id: Optional[str] = None) -> bool:
        """
        Delete a scheduled post
        """
        try:
            if scheduled_posts_collection is None:
                return False
            
            query = {"_id": ObjectId(post_id)}
            if user_id:
                query["userId"] = ObjectId(user_id)
            
            result = scheduled_posts_collection.delete_one(query)
            return result.deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error deleting scheduled post: {e}")
            return False

def allowed_file(filename):
    """Check if the file extension is allowed"""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def save_uploaded_file(file):
    """Save uploaded file and return the path"""
    try:
        if file and allowed_file(file.filename):
            # Create uploads directory if it doesn't exist
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

            # Generate unique filename
            original_filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{original_filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save the file
            file.save(filepath)
            logger.info(f"File saved successfully: {filepath}")
            
            # Return the absolute path
            return os.path.abspath(filepath)
        return None
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise Exception(f"Failed to save file: {str(e)}")

def upload_to_gcs(file, filename: str) -> Optional[str]:
    """Upload file to Google Cloud Storage and return public URL - following working example pattern"""
    try:
        if gcs_bucket is None:
            logger.error("Google Cloud Storage not configured")
            return None
            
        if not filename:
            logger.error("No filename provided")
            return None
        
        # Generate unique destination blob name (like the working example)
        unique_filename = f"images/{uuid.uuid4()}_{secure_filename(filename)}"
        
        # Create blob (like the working example)
        blob = gcs_bucket.blob(unique_filename)
        
        # Set proper content type based on file extension
        file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
        content_type_map = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg', 
            'png': 'image/png',
            'gif': 'image/gif',
            'webp': 'image/webp'
        }
        
        content_type = content_type_map.get(file_ext, 'image/jpeg')
        blob.content_type = content_type
        logger.info(f"Setting content type to: {content_type} for file: {filename}")
        
        # Upload from file object (similar to upload_from_filename in working example)
        blob.upload_from_file(file, content_type=content_type)
        
        # Try to make the bucket publicly readable if possible
        try:
            # First try to make the blob public (works if uniform access is disabled)
            blob.make_public()
            public_url = blob.public_url
            logger.info(f"✅ Blob made public using ACL")
        except Exception as acl_error:
            logger.info(f"ℹ️ Cannot use ACL (uniform bucket access enabled): {acl_error}")
            # If uniform bucket access is enabled, try to make bucket public
            try:
                from google.cloud import storage
                # Make bucket publicly readable
                policy = gcs_bucket.get_iam_policy(requested_policy_version=3)
                policy.bindings.append({
                    "role": "roles/storage.objectViewer",
                    "members": ["allUsers"]
                })
                gcs_bucket.set_iam_policy(policy)
                logger.info(f"✅ Bucket made publicly readable")
                public_url = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{unique_filename}"
            except Exception as bucket_error:
                logger.warning(f"⚠️ Cannot make bucket public: {bucket_error}")
                # Generate signed URL as fallback (valid for 7 days)
                from datetime import timedelta
                signed_url = blob.generate_signed_url(
                    version="v4",
                    expiration=datetime.now(pytz.UTC) + timedelta(days=7),
                    method="GET"
                )
                public_url = signed_url
                logger.info(f"🔗 Generated signed URL (7 days validity)")
        
        logger.info(f"📤 Uploaded '{filename}' to 'gs://{GCS_BUCKET_NAME}/{unique_filename}'.")
        logger.info(f"File uploaded to GCS successfully: {public_url}")
        return public_url
        
    except Exception as e:
        logger.error(f"❌ Error uploading to GCS: {str(e)}")
        return None

def validate_schedule_time(schedule_time_str: str, platform: str = 'facebook') -> Optional[int]:
    """
    Validate and convert schedule time to Unix timestamp.
    Returns None if invalid, Unix timestamp if valid.
    Platform-specific validation rules:
    - Facebook: Must be at least 10 minutes in future
    - Twitter: Can be any future time
    """
    try:
        # Parse the datetime string in local timezone
        schedule_time = datetime.strptime(schedule_time_str, '%Y-%m-%d %H:%M')
        
        # Get current time in UTC timezone
        current_time = datetime.now(pytz.UTC)
        # Convert to IST for comparison with schedule time
        ist = pytz.timezone('Asia/Kolkata')
        current_time_ist = current_time.astimezone(ist).replace(tzinfo=None)
        
        # For Facebook, check if schedule time is at least 10 minutes in the future
        if platform.lower() == 'facebook':
            min_schedule_time = current_time_ist + timedelta(minutes=10)
            if schedule_time < min_schedule_time:
                logger.error(f"Facebook posts must be scheduled at least 10 minutes in the future. Current time: {current_time_ist.strftime('%Y-%m-%d %H:%M')}, Minimum allowed time: {min_schedule_time.strftime('%Y-%m-%d %H:%M')}")
                return None
        else:
            # For Twitter, just check if it's in the future
            if schedule_time <= current_time_ist:
                logger.error(f"Schedule time must be in the future. Current time: {current_time_ist.strftime('%Y-%m-%d %H:%M')}")
                return None
            
        # Check if schedule time is not more than 6 months in the future
        max_schedule_time = current_time_ist + timedelta(days=180)
        if schedule_time > max_schedule_time:
            logger.error(f"Schedule time cannot be more than 6 months in the future. Maximum time: {max_schedule_time.strftime('%Y-%m-%d %H:%M')}")
            return None
        
        # Convert to Unix timestamp while preserving local time
        return int(schedule_time.timestamp())
        
    except ValueError as e:
        logger.error(f"Invalid datetime format: {e}")
        return None

class SocialMediaScheduler:
    def __init__(self):
        self.fb_user_access_token = "EAAJZBvtBx1esBPKACDtsnTM9yK0dW2XQYleQsQ23iTWZBcZAL0WEZBjsOm1aj7UVPPOjjMXiUuXv11ZCeTZAAatLZAtEqj9hYn6shTNMVfZBOmhbHwTYycVvAyv8GuqGFbzCcZAEORHUlpNcquEHqq6piTDhiam6PVZAKuJuxvb3GW5lpBibM8E5EzDVVYewYaGIZBS"
        self.fb_base_url = "https://graph.facebook.com/v22.0"
        self.fb_page_access_token = None
        self.fb_page_id = "654470507748535"
        
        # Twitter credentials - replace with your actual credentials
        self.tw_consumer_key = "PR7xsEwNEuSpqTI2dRF8nfS1h"
        self.tw_consumer_secret = "qd3xHOxs9xfZ0VD4z8RRWHvASJRP9NnIWrxwBFAUIdVO7QeEff"
        self.tw_access_token = "1925047617152811009-T4rohuOAtf0OT9fxZBDvkZdxbjzsHy"
        self.tw_access_token_secret = "vH6QfxtpdAOfKcdhoaAneJ41OGAwzOdpmA8GtH8Llfnm4"
        self.tw_bearer_token = "AAAAAAAAAAAAAAAAAAAAALFa2AEAAAAAYnv%2F5X%2FYOTYR6TVyBvA1Nfgeem8%3DnWn9WDZcwB8CXjvcHRuusw0CoRV0KtH1boOUtMpGTGqvUeFTQU"
        
        # Instagram credentials - replace with your actual credentials
        self.ig_user_id = "17841474183245806"
        self.ig_access_token = "EAAYhQbxuBtMBOZCntJJnbPXGPpIaubva1DndF7RZCdZBloAJjsvG0zpLrcw32xy0xK8LKUQVdE4Pu1PapMywsnRR11Iz4ug6gZAS74IAKHNeeTpO5I3u3Utan73ZBlyeNiFPkrthLuXjXdKVDWwJaMujRbR1AstEENRM15R1v0tODPS5CCcYLeRUvE4I0"

        # LinkedIn credentials - replace with your actual credentials
        self.linkedin_client_id = "77h94rbzkymv01"
        self.linkedin_client_secret = "WPL_AP1.naGaLVGv600B929L.IdPw4A=="
        self.linkedin_access_token = "AQWtrslnEPA-u-0m_yerMlDKc-oy31oA90EssCtOl7NZFL6I63Ss2HowAiUf4PqQz4UF27Mon7osj2AfqjIhUQVuK-iEmbJ8Ck0f0R9FrnjpsSpbuQzSrmHacV0pTGCSKWpI0YhC9jDqwkwOrIbcnl38-uYfr48cvazk9-iYxON32PsTAoCORkg4U75Bk1Mg8NVmWv2v0fhB5_76HK3pmdoUXzz-abUubWGL7SlEH7XgsQXHlSo5tI3XfHa_gindLE6crLTl2sBXs-l0t_4J0yAzMiHxhUxADUwYbNNIVbr9eGTZ7oWlIwoGQP4wFJHlGrqFe176IyVyYgwUmOrHyeAdrNa5RQ"  # Will be set when user authenticates
        self.linkedin_api_base = "https://api.linkedin.com/v2"
        self.linkedin_user_id = "pQ8VCwYg4Y"  # Will be set when user authenticates
        
        # Initialize scheduled posts storage
        self.scheduled_posts = {}
        self.load_scheduled_posts()
        
        # Initialize Facebook page access
        self.setup_facebook_page_access()
    
    def format_content_with_hashtags(self, caption: str, hashtags: List[str], platform: str = '') -> str:
        """
        Format content by combining caption with hashtags based on platform requirements
        
        Args:
            caption: The main post content
            hashtags: List of hashtags (with or without # symbol)
            platform: Platform name for platform-specific formatting
            
        Returns:
            Formatted content string
        """
        if not hashtags:
            return caption
            
        # Ensure hashtags start with #
        formatted_hashtags = []
        for tag in hashtags:
            if isinstance(tag, str):
                tag = tag.strip()
                if tag and not tag.startswith('#'):
                    tag = f"#{tag}"
                if tag and len(tag) > 1:  # Don't add empty hashtags
                    formatted_hashtags.append(tag)
        
        if not formatted_hashtags:
            return caption
            
        # Platform-specific formatting
        if platform.lower() == 'linkedin':
            # LinkedIn: Space between caption and hashtags
            hashtag_string = ' '.join(formatted_hashtags)
            return f"{caption}\n\n{hashtag_string}" if caption else hashtag_string
        elif platform.lower() == 'twitter':
            # Twitter: Be mindful of character limit
            hashtag_string = ' '.join(formatted_hashtags)
            combined = f"{caption} {hashtag_string}" if caption else hashtag_string
            # Twitter has 280 character limit
            if len(combined) > 280:
                # Try to fit as many hashtags as possible
                remaining_chars = 280 - len(caption) - 1  # -1 for space
                if remaining_chars > 0:
                    hashtag_string = ' '.join(formatted_hashtags)
                    if len(hashtag_string) <= remaining_chars:
                        return f"{caption} {hashtag_string}"                                                                                                                                                                  
                    else:
                        # Truncate hashtags to fit
                        truncated_hashtags = []
                        current_length = 0
                        for tag in formatted_hashtags:
                            if current_length + len(tag) + 1 <= remaining_chars:  # +1 for space
                                truncated_hashtags.append(tag)
                                current_length += len(tag) + 1
                            else:
                                break
                        if truncated_hashtags:
                            return f"{caption} {' '.join(truncated_hashtags)}"
                return caption
            return combined
        else:
            # Default format for Instagram, Facebook, etc.
            hashtag_string = ' '.join(formatted_hashtags)
            return f"{caption}\n\n{hashtag_string}" if caption else hashtag_string

    def load_scheduled_posts(self):
        """Load scheduled posts from file"""
        try:
            if os.path.exists('scheduled_posts.json'):
                with open('scheduled_posts.json', 'r') as f:
                    self.scheduled_posts = json.load(f)
        except Exception as e:
            logger.error(f"Error loading scheduled posts: {e}")
            self.scheduled_posts = {}
    
    def save_scheduled_posts(self):
        """Save scheduled posts to file"""
        try:
            with open('scheduled_posts.json', 'w') as f:
                json.dump(self.scheduled_posts, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving scheduled posts: {e}")
    
    # MongoDB Integration Methods
    def create_scheduled_post_in_db(self, user_id: str, content_data: Dict, 
                                   platforms: List[str], schedule_time_ist: str,
                                   ai_generation: Optional[Dict] = None) -> Optional[str]:
        """
        Create a scheduled post in MongoDB database
        
        Args:
            user_id: User ID from your authentication system
            content_data: Dictionary containing caption, hashtags, images
            platforms: List of platforms to schedule for
            schedule_time_ist: Schedule time in IST format 'YYYY-MM-DD HH:MM'
            ai_generation: Optional AI generation metadata
        
        Returns:
            MongoDB document ID if successful, None if failed
        """
        try:
            # Validate and convert schedule time from IST to UTC
            utc_schedule_time = convert_ist_to_utc(schedule_time_ist)
            
            if not utc_schedule_time:
                logger.error("Invalid schedule time provided")
                return None
            
            # Prepare content structure matching the required format
            content = {
                "caption": content_data.get('caption', ''),
                "hashtags": content_data.get('hashtags', []),
                "images": content_data.get('images', [])
            }
            
            # If images are provided as URLs or paths, convert to required format
            if 'image_urls' in content_data:
                images = []
                for url in content_data['image_urls']:
                    images.append({
                        "url": url,
                        "_id": ObjectId()
                    })
                content["images"] = images
            
            # Create the scheduled post in database
            db_post_id = MongoDBHelper.create_scheduled_post(
                user_id=user_id,
                content=content,
                platforms=platforms,
                schedule_time=utc_schedule_time,
                ai_generation=ai_generation
            )
            
            if db_post_id:
                logger.info(f"Scheduled post created in DB with ID: {db_post_id}")
                return db_post_id
            else:
                logger.error("Failed to create scheduled post in database")
                return None
                
        except Exception as e:
            logger.error(f"Error creating scheduled post in DB: {e}")
            return None
    
    def create_scheduled_post_in_db_with_times(self, user_id: str, content_data: Dict, 
                                   platforms: List[str], platform_schedule_times: Dict[str, Dict],
                                   ai_generation: Optional[Dict] = None) -> Optional[str]:
        """
        Create a scheduled post in MongoDB database with different times for each platform
        
        Args:
            user_id: User ID from your authentication system
            content_data: Dictionary containing caption, hashtags, images
            platforms: List of platforms to schedule for
            platform_schedule_times: Dict with platform as key and dict with ist_time and utc_time
            ai_generation: Optional AI generation metadata
        
        Returns:
            MongoDB document ID if successful, None if failed
        """
        try:
            # Prepare content structure matching the required format
            content = {
                "caption": content_data.get('caption', ''),
                "hashtags": content_data.get('hashtags', []),
                "images": content_data.get('images', [])
            }
            
            # If images are provided as URLs or paths, convert to required format
            if 'image_urls' in content_data:
                images = []
                for url in content_data['image_urls']:
                    images.append({
                        "url": url,
                        "_id": ObjectId()
                    })
                content["images"] = images
            
            # Create the scheduled post in database with platform-specific times
            db_post_id = MongoDBHelper.create_scheduled_post_with_times(
                user_id=user_id,
                content=content,
                platforms=platforms,
                platform_schedule_times=platform_schedule_times,
                ai_generation=ai_generation
            )
            
            if db_post_id:
                logger.info(f"Scheduled post with platform-specific times created in DB with ID: {db_post_id}")
                return db_post_id
            else:
                logger.error("Failed to create scheduled post with platform times in database")
                return None
                
        except Exception as e:
            logger.error(f"Error creating scheduled post with platform times in DB: {e}")
            return None
    
    def schedule_post_for_multiple_platforms(self, user_id: str, content_data: Dict, 
                                           platforms: List[str], schedule_time_ist: str,
                                           ai_generation: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Schedule a post for multiple platforms and store in MongoDB
        
        This is the main method that should be called from the API endpoint
        """
        try:
            # Validate platforms
            valid_platforms = ['facebook', 'twitter', 'instagram', 'linkedin']
            platforms = [p.lower() for p in platforms if p.lower() in valid_platforms]
            
            if not platforms:
                return {"error": "No valid platforms specified"}
            
            # Validate schedule time - use the first platform for validation rules
            primary_platform = platforms[0] if platforms else 'twitter'
            utc_schedule_time = validate_schedule_time_ist(schedule_time_ist, primary_platform)
            if not utc_schedule_time:
                return {"error": "Invalid schedule time. Must be in future and format: YYYY-MM-DD HH:MM"}
            
            # Create the post in MongoDB
            db_post_id = self.create_scheduled_post_in_db(
                user_id=user_id,
                content_data=content_data,
                platforms=platforms,
                schedule_time_ist=schedule_time_ist,
                ai_generation=ai_generation
            )
            
            if not db_post_id:
                return {"error": "Failed to create scheduled post in database"}
            
            # Note: We're NOT storing in JSON file anymore to prevent duplicate execution
            # The MongoDB system is now the primary scheduling system
            logger.info(f"Post scheduled in MongoDB only (not in JSON file) to prevent duplicate execution")
            
            return {
                "success": True,
                "message": f"Post scheduled successfully for {len(platforms)} platform(s)",
                "db_post_id": db_post_id,
                "platforms": platforms,
                "scheduled_time_ist": schedule_time_ist,
                "scheduled_time_utc": utc_schedule_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error scheduling post for multiple platforms: {e}")
            return {"error": str(e)}
    
    def schedule_post_for_multiple_platforms_with_times(self, user_id: str, content_data: Dict, 
                                           platforms: List[str], platform_schedule_times: Dict[str, str],
                                           ai_generation: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Schedule a post for multiple platforms with different schedule times for each platform
        
        Args:
            user_id: User ID from your authentication system
            content_data: Dictionary containing caption, hashtags, images
            platforms: List of platforms to schedule for
            platform_schedule_times: Dictionary mapping platform names to their schedule times
            ai_generation: Optional AI generation metadata
        """
        try:
            # Validate platforms
            valid_platforms = ['facebook', 'twitter', 'instagram', 'linkedin']
            platforms = [p.lower() for p in platforms if p.lower() in valid_platforms]
            
            if not platforms:
                return {"error": "No valid platforms specified"}
            
            # Validate schedule times for each platform
            validated_times = {}
            for platform in platforms:
                schedule_time = platform_schedule_times.get(platform)
                if not schedule_time:
                    return {"error": f"Missing schedule time for platform: {platform}"}
                
                utc_schedule_time = validate_schedule_time_ist(schedule_time, platform)
                if not utc_schedule_time:
                    return {"error": f"Invalid schedule time for {platform}. Must be in future and format: YYYY-MM-DD HH:MM"}
                
                validated_times[platform] = {
                    'ist_time': schedule_time,
                    'utc_time': utc_schedule_time
                }
            
            # Create the post in MongoDB with platform-specific times
            db_post_id = self.create_scheduled_post_in_db_with_times(
                user_id=user_id,
                content_data=content_data,
                platforms=platforms,
                platform_schedule_times=validated_times,
                ai_generation=ai_generation
            )
            
            if not db_post_id:
                return {"error": "Failed to create scheduled post in database"}
            
            logger.info(f"Post scheduled in MongoDB with different times for each platform")
            
            # Prepare response data
            platform_times = {}
            for platform in platforms:
                platform_times[platform] = {
                    'scheduledTimeIST': validated_times[platform]['ist_time'],
                    'scheduledTimeUTC': validated_times[platform]['utc_time'].isoformat()
                }
            
            return {
                "success": True,
                "message": f"Post scheduled successfully for {len(platforms)} platform(s) with different times",
                "db_post_id": db_post_id,
                "platforms": platforms,
                "platformScheduleTimes": platform_times
            }
            
        except Exception as e:
            logger.error(f"Error scheduling post for multiple platforms with times: {e}")
            return {"error": str(e)}
    
    def execute_scheduled_posts_from_db(self):
        """
        Execute scheduled posts from MongoDB that are due
        """
        try:
            # Get posts ready for execution
            posts_to_execute = MongoDBHelper.get_scheduled_posts_for_execution()
            
            for post in posts_to_execute:
                post_id = str(post['_id'])
                content = post.get('content', {})
                platforms = post.get('platform', [])
                
                logger.info(f"Executing scheduled post: {post_id}")
                
                for platform_info in platforms:
                    platform = platform_info.get('platform')
                    status = platform_info.get('status')
                    
                    if status != 'scheduled':
                        continue  # Skip if already processed
                    
                    # Check if this platform is ready for execution (for platform-specific times)
                    if post.get('hasCustomTimes') and platform_info.get('scheduledTime'):
                        current_time = datetime.now(pytz.UTC)
                        platform_scheduled_time = platform_info.get('scheduledTime')
                        
                        # Ensure platform_scheduled_time is timezone-aware
                        if platform_scheduled_time.tzinfo is None:
                            # Assume it's UTC if no timezone info
                            platform_scheduled_time = pytz.UTC.localize(platform_scheduled_time)
                        elif platform_scheduled_time.tzinfo != pytz.UTC:
                            # Convert to UTC if it's in a different timezone
                            platform_scheduled_time = platform_scheduled_time.astimezone(pytz.UTC)
                        
                        # Convert times to IST for logging
                        current_ist = current_time.astimezone(pytz.timezone('Asia/Kolkata'))
                        scheduled_ist = platform_scheduled_time.astimezone(pytz.timezone('Asia/Kolkata'))
                        
                        logger.info(f"Platform {platform} timezone comparison - Scheduled: {platform_scheduled_time} UTC ({scheduled_ist.strftime('%Y-%m-%d %H:%M:%S')} IST), Current: {current_time} UTC ({current_ist.strftime('%Y-%m-%d %H:%M:%S')} IST)")
                        
                        # Skip if this platform's time hasn't arrived yet
                        if platform_scheduled_time > current_time:
                            logger.info(f"Platform {platform} not ready yet. Scheduled for: {platform_scheduled_time} UTC ({scheduled_ist.strftime('%Y-%m-%d %H:%M:%S')} IST), Current: {current_time} UTC ({current_ist.strftime('%Y-%m-%d %H:%M:%S')} IST)")
                            continue
                    
                    if status != 'scheduled':
                        continue  # Skip if already processed
                    
                    # Mark as 'executing' immediately to prevent duplicate execution
                    MongoDBHelper.update_post_status(post_id, platform, 'executing')
                    
                    try:
                        # Prepare post data for execution
                        caption = content.get('caption', '')
                        hashtags = content.get('hashtags', [])
                        
                        # Format content with hashtags based on platform
                        formatted_message = self.format_content_with_hashtags(caption, hashtags, platform)
                        
                        post_data = {
                            'message': formatted_message,
                            'platform': platform
                        }
                        
                        # Add images if available
                        if content.get('images'):
                            first_image = content['images'][0]
                            image_url = first_image.get('url', '')
                            post_data['photo_url'] = image_url
                            
                            # For Twitter, use 'image_url' parameter
                            if platform == 'twitter':
                                post_data['image_url'] = image_url
                            
                            # For LinkedIn, we need to download and save the image locally first
                            elif platform == 'linkedin':
                                try:
                                    logger.info(f"Downloading image for LinkedIn: {image_url[:100]}...")
                                    import tempfile
                                    import requests
                                    
                                    # If it's a Google Cloud Storage URL, generate a signed URL first
                                    if 'storage.googleapis.com' in image_url:
                                        try:
                                            # Extract bucket and blob name from URL
                                            url_parts = image_url.replace('https://storage.googleapis.com/', '').split('/', 1)
                                            if len(url_parts) == 2:
                                                bucket_name = url_parts[0]
                                                blob_name = url_parts[1]
                                                
                                                # Generate signed URL
                                                from google.cloud import storage
                                                storage_client = storage.Client()
                                                bucket = storage_client.bucket(bucket_name)
                                                blob = bucket.blob(blob_name)
                                                
                                                # Generate a signed URL that's valid for 1 hour
                                                signed_url = blob.generate_signed_url(
                                                    version="v4",
                                                    expiration=datetime.now(pytz.UTC) + timedelta(hours=1),
                                                    method="GET"
                                                )
                                                image_url = signed_url
                                                logger.info(f"Generated signed URL for LinkedIn: {signed_url[:100]}...")
                                        except Exception as signed_url_error:
                                            logger.warning(f"Failed to generate signed URL, using original: {signed_url_error}")
                                    
                                    # Download the image
                                    response = requests.get(image_url, timeout=30)
                                    if response.status_code == 200:
                                        # Save to temporary file
                                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                                            tmp_file.write(response.content)
                                            temp_path = tmp_file.name
                                            
                                        post_data['photo_path'] = temp_path
                                        logger.info(f"Downloaded LinkedIn image to: {temp_path}")
                                    else:
                                        logger.error(f"Failed to download LinkedIn image: HTTP {response.status_code}")
                                except Exception as e:
                                    logger.error(f"Error downloading LinkedIn image: {e}")
                            
                            if platform == 'facebook':
                                post_data['type'] = 'photo'
                        
                        # Execute the post based on platform
                        result = None
                        if platform == 'facebook':
                            result = self.post_to_facebook(post_data)
                        elif platform == 'twitter':
                            post_data['text'] = post_data['message']
                            # Ensure image_url is passed correctly
                            result = self.post_to_twitter(post_data)
                        elif platform == 'instagram':
                            result = self.post_to_instagram(post_data)
                        elif platform == 'linkedin':
                            result = self.post_to_linkedin(post_data)
                        
                        # Update status in database based on result
                        if result and result.get('success'):
                            published_id = result.get('post_id') or result.get('tweet_id') or result.get('media_id') or result.get('id')
                            MongoDBHelper.update_post_status(
                                post_id, platform, 'published', published_id
                            )
                            logger.info(f"Successfully published to {platform} for post {post_id}")
                        else:
                            MongoDBHelper.update_post_status(post_id, platform, 'failed')
                            error_msg = result.get('error', 'Unknown error') if result else 'No response from platform'
                            logger.error(f"Failed to publish to {platform} for post {post_id}: {error_msg}")
                        
                        # Clean up temporary LinkedIn image file if it exists
                        if platform == 'linkedin' and 'photo_path' in post_data:
                            try:
                                import os
                                temp_path = post_data['photo_path']
                                if os.path.exists(temp_path):
                                    os.unlink(temp_path)
                                    logger.info(f"Cleaned up temporary LinkedIn image: {temp_path}")
                            except Exception as cleanup_error:
                                logger.warning(f"Failed to cleanup temporary file: {cleanup_error}")
                    
                    except Exception as e:
                        logger.error(f"Error executing post {post_id} for {platform}: {e}")
                        MongoDBHelper.update_post_status(post_id, platform, 'failed')
                        
                        # Clean up temporary LinkedIn image file if it exists
                        if platform == 'linkedin' and 'post_data' in locals() and 'photo_path' in post_data:
                            try:
                                import os
                                temp_path = post_data['photo_path']
                                if os.path.exists(temp_path):
                                    os.unlink(temp_path)
                                    logger.info(f"Cleaned up temporary LinkedIn image after error: {temp_path}")
                            except Exception as cleanup_error:
                                logger.warning(f"Failed to cleanup temporary file after error: {cleanup_error}")
                        
        except Exception as e:
            logger.error(f"Error executing scheduled posts from DB: {e}")
    
    def get_user_scheduled_posts_from_db(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get scheduled posts for a user from MongoDB"""
        try:
            posts = MongoDBHelper.get_user_scheduled_posts(user_id, limit)
            
            # Convert ObjectId to string for JSON serialization
            for post in posts:
                post['_id'] = str(post['_id'])
                if post.get('userId'):
                    post['userId'] = str(post['userId'])
                
                # Convert image ObjectIds to strings
                if post.get('content', {}).get('images'):
                    for image in post['content']['images']:
                        if image.get('_id'):
                            image['_id'] = str(image['_id'])
                
                # Convert platform ObjectIds to strings
                if post.get('platform'):
                    for platform_info in post['platform']:
                        if platform_info.get('_id'):
                            platform_info['_id'] = str(platform_info['_id'])
            
            return posts
            
        except Exception as e:
            logger.error(f"Error getting user scheduled posts from DB: {e}")
            return []
    
    # FACEBOOK METHODS
    def setup_facebook_page_access(self) -> bool:
        """Get Facebook page access token"""
        try:
            url = f"{self.fb_base_url}/me/accounts"
            params = {
                'access_token': self.fb_user_access_token,
                'fields': 'id,name,access_token'
            }
            
            response = requests.get(url, params=params)
            result = response.json()
            
            if response.status_code == 200:
                pages = result.get('data', [])
                if pages:
                    page = pages[0]
                    self.fb_page_id = page.get('id')
                    self.fb_page_access_token = page.get('access_token')
                    logger.info(f"Connected to Facebook page: {page.get('name')}")
                    return True
            
            logger.error(f"Facebook page setup failed: {result}")
            return False
        except Exception as e:
            logger.error(f"Facebook setup error: {e}")
            return False
    
    def upload_photo_to_facebook(self, photo_path: str) -> Optional[str]:
        """Upload a photo to Facebook and return the ID"""
        try:
            if not self.fb_page_access_token:
                if not self.setup_facebook_page_access():
                    raise Exception("Failed to setup Facebook page access")

            # Create multipart form data with the image file
            with open(photo_path, 'rb') as image_file:
                files = {
                    'source': image_file,
                }
                data = {
                    'access_token': self.fb_page_access_token,
                    'published': 'false'  # Don't publish immediately
                }
                
                url = f"{self.fb_base_url}/{self.fb_page_id}/photos"
                response = requests.post(url, files=files, data=data)
                result = response.json()

                if response.status_code == 200:
                    return result.get('id')  # Return the photo ID
                else:
                    logger.error(f"Failed to upload photo: {result}")
                    raise Exception(f"Failed to upload photo: {result.get('error', {}).get('message')}")

        except Exception as e:
            logger.error(f"Error uploading photo to Facebook: {e}")
            raise Exception(f"Failed to upload photo: {str(e)}")
    
    def post_to_facebook(self, post_data: Dict) -> Dict[str, Any]:
        """Post to Facebook immediately or scheduled"""
        try:
            if not self.fb_page_access_token:
                return {"error": "Facebook page access not configured"}
            
            post_type = post_data.get('type', 'text')
            message = post_data.get('message', '')
            
            # Handle scheduling
            scheduled_timestamp = None
            if 'scheduled_time' in post_data:
                # Use the _schedule_post method for scheduling
                return self._schedule_post(post_data)
            
            # For immediate posts, continue with the existing logic
            if post_type == 'photo':
                # Handle local file upload
                if 'photo_path' in post_data:
                    try:
                        # First upload the photo
                        photo_id = self.upload_photo_to_facebook(post_data['photo_path'])
                        if not photo_id:
                            return {"error": "Failed to upload photo"}
                        
                        # Create the post with the uploaded photo
                        url = f"{self.fb_base_url}/{self.fb_page_id}/feed"
                        data = {
                            'message': message,
                            'attached_media[0]': f'{{"media_fbid":"{photo_id}"}}',
                            'access_token': self.fb_page_access_token
                        }
                        
                        response = requests.post(url, data=data)
                        result = response.json()
                        
                        if response.status_code == 200:
                            return {
                                "success": True,
                                "post_id": result.get('id'),
                                "platform": "facebook",
                                "message": "Post published successfully"
                            }
                        else:
                            return {"error": result}
                            
                    except Exception as e:
                        return {"error": str(e)}
                        
                # Handle photo URL
                elif 'photo_url' in post_data:
                    return self._post_facebook_photo(message, post_data['photo_url'], None)
                else:
                    return {"error": "No photo provided"}
                    
            elif post_type == 'text':
                return self._post_facebook_text(message, None)
            elif post_type == 'link':
                link = post_data.get('link', '')
                return self._post_facebook_link(message, link, None)
            else:
                return {"error": "Invalid post type"}
                
        except Exception as e:
            logger.error(f"Facebook post error: {e}")
            return {"error": str(e)}
    
    def _post_facebook_text(self, message: str, scheduled_time: Optional[int]) -> Dict:
        """Post text to Facebook"""
        url = f"{self.fb_base_url}/{self.fb_page_id}/feed"
        data = {
            'message': message,
            'access_token': self.fb_page_access_token
        }
        
        if scheduled_time:
            data['published'] = 'false'
            data['scheduled_publish_time'] = scheduled_time
        
        response = requests.post(url, data=data)
        result = response.json()
        
        if response.status_code == 200:
            return {"success": True, "post_id": result.get('id'), "platform": "facebook"}
        else:
            return {"error": result}
    
    def _post_facebook_photo(self, message: str, photo_url: str, scheduled_time: Optional[int]) -> Dict:
        """Post photo to Facebook"""
        url = f"{self.fb_base_url}/{self.fb_page_id}/photos"
        data = {
            'message': message,
            'url': photo_url,
            'access_token': self.fb_page_access_token
        }
        
        if scheduled_time:
            data['published'] = 'false'
            data['scheduled_publish_time'] = scheduled_time
        
        response = requests.post(url, data=data)
        result = response.json()
        
        if response.status_code == 200:
            return {"success": True, "photo_id": result.get('id'), "platform": "facebook"}
        else:
            return {"error": result}
    
    def _post_facebook_link(self, message: str, link: str, scheduled_time: Optional[int]) -> Dict:
        """Post link to Facebook"""
        url = f"{self.fb_base_url}/{self.fb_page_id}/feed"
        data = {
            'message': message,
            'link': link,
            'access_token': self.fb_page_access_token
        }
        
        if scheduled_time:
            data['published'] = 'false'
            data['scheduled_publish_time'] = scheduled_time
        
        response = requests.post(url, data=data)
        result = response.json()
        
        if response.status_code == 200:
            return {"success": True, "post_id": result.get('id'), "platform": "facebook"}
        else:
            return {"error": result}
    
    # TWITTER METHODS
    def create_twitter_client(self):
        """Create Twitter API client"""
        try:
            client = tweepy.Client(
                bearer_token=self.tw_bearer_token,
                consumer_key=self.tw_consumer_key,
                consumer_secret=self.tw_consumer_secret,
                access_token=self.tw_access_token,
                access_token_secret=self.tw_access_token_secret
            )
            return client
        except Exception as e:
            logger.error(f"Error creating Twitter client: {e}")
            return None
    
    def post_to_twitter(self, post_data: Dict) -> Dict[str, Any]:
        """Post to Twitter immediately or schedule for later"""
        try:
            text = post_data.get('text', '')
            # Handle both image_url and photo_url for flexibility
            image_url = post_data.get('image_url', '') or post_data.get('photo_url', '')
            scheduled_time = post_data.get('scheduled_time')
            
            logger.info(f"Twitter post - Text: {text[:50]}..., Image URL: {image_url[:100] if image_url else 'None'}")
            
            if scheduled_time:
                # Schedule the post
                return self._schedule_twitter_post(text, image_url, scheduled_time)
            else:
                # Post immediately
                return self._post_twitter_immediately(text, image_url)
                
        except Exception as e:
            logger.error(f"Twitter post error: {e}")
            return {"error": str(e)}
    
    def _post_twitter_immediately(self, text: str, image_url: str = '') -> Dict[str, Any]:
        """Post to Twitter immediately"""
        try:
            logger.info(f"Starting Twitter post - Text: {text[:100]}..., Image: {'Yes' if image_url else 'No'}")
            
            client = self.create_twitter_client()
            if not client:
                return {"error": "Failed to create Twitter client"}
            
            # Create v1.1 API for media upload
            auth = tweepy.OAuth1UserHandler(self.tw_consumer_key, self.tw_consumer_secret)
            auth.set_access_token(self.tw_access_token, self.tw_access_token_secret)
            api = tweepy.API(auth)
            
            media_ids = []
            if image_url:
                logger.info(f"Processing image for Twitter: {image_url}")
                
                # Check if it's a local file path
                if os.path.exists(image_url):
                    try:
                        logger.info(f"Uploading local file: {image_url}")
                        # Upload local file directly
                        media = api.media_upload(filename=image_url)
                        media_ids.append(media.media_id)
                        logger.info(f"Successfully uploaded local image, media_id: {media.media_id}")
                    except Exception as e:
                        logger.error(f"Error uploading local image: {e}")
                        return {"error": f"Failed to upload local image: {str(e)}"}
                else:
                    # Handle remote URL
                    try:
                        logger.info(f"Downloading and uploading remote image: {image_url}")
                        import urllib.request
                        import tempfile
                        
                        # Create a temporary file with proper extension
                        url_parts = image_url.split('.')
                        file_ext = url_parts[-1] if len(url_parts) > 1 and url_parts[-1].lower() in ['jpg', 'jpeg', 'png', 'gif'] else 'jpg'
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp_file:
                            # Download image with proper headers
                            req = urllib.request.Request(image_url, headers={
                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                            })
                            
                            try:
                                with urllib.request.urlopen(req, timeout=30) as response:
                                    if response.status == 200:
                                        tmp_file.write(response.read())
                                        tmp_file.flush()
                                        logger.info(f"Downloaded image to temp file: {tmp_file.name}")
                                        
                                        # Verify the file exists and has content
                                        if os.path.exists(tmp_file.name) and os.path.getsize(tmp_file.name) > 0:
                                            # Upload to Twitter
                                            media = api.media_upload(filename=tmp_file.name)
                                            media_ids.append(media.media_id)
                                            logger.info(f"Successfully uploaded remote image, media_id: {media.media_id}")
                                        else:
                                            logger.error("Downloaded file is empty or doesn't exist")
                                            return {"error": "Downloaded image file is empty"}
                                    else:
                                        logger.error(f"Failed to download image: HTTP {response.status}")
                                        return {"error": f"Failed to download image: HTTP {response.status}"}
                            finally:
                                # Clean up temp file
                                try:
                                    if os.path.exists(tmp_file.name):
                                        os.unlink(tmp_file.name)
                                        logger.info("Cleaned up temporary file")
                                except Exception as cleanup_error:
                                    logger.warning(f"Failed to cleanup temp file: {cleanup_error}")
                            
                    except Exception as e:
                        logger.error(f"Error downloading/uploading remote image: {e}")
                        return {"error": f"Failed to upload remote image: {str(e)}"}
            
            # Prepare final text
            final_text = text.strip()
            if not final_text:
                final_text = "📸"  # Default emoji if no text
            
            # Add small random suffix to prevent duplicate content issues if text is very short
            if len(final_text) < 20:  # Only for very short texts
                import random
                timestamp_suffix = f" #{random.randint(100, 999)}"
                if len(final_text + timestamp_suffix) <= 280:  # Twitter character limit
                    final_text = final_text + timestamp_suffix
            
            # Post tweet
            try:
                logger.info(f"Posting tweet with {len(media_ids)} media items")
                
                if media_ids:
                    response = client.create_tweet(text=final_text, media_ids=media_ids)
                else:
                    response = client.create_tweet(text=final_text)
                
                # Handle response - try different ways to get tweet ID
                tweet_id = None
                
                # Try to get tweet ID from response object
                if response:
                    # Method 1: Try accessing as dictionary
                    if isinstance(response, dict):
                        tweet_id = response.get('id')
                    # Method 2: Try accessing data attribute
                    elif hasattr(response, 'data'):
                        data = getattr(response, 'data')
                        if isinstance(data, dict):
                            tweet_id = data.get('id')
                        elif hasattr(data, 'get'):
                            tweet_id = data.get('id')
                
                if tweet_id:
                    logger.info(f"Tweet posted successfully with ID: {tweet_id}")
                    return {
                        "success": True,
                        "tweet_id": str(tweet_id),
                        "platform": "twitter",
                        "message": "Tweet posted successfully",
                        "has_media": len(media_ids) > 0
                    }
                
                logger.error("Failed to get tweet ID from response")
                return {"error": "Failed to get tweet ID from response"}
                
            except tweepy.TweepyException as e:
                logger.error(f"Twitter API error: {e}")
                return {"error": f"Twitter API error: {str(e)}"}
                
        except Exception as e:
            logger.error(f"Twitter immediate post error: {e}")
            return {"error": str(e)}
    
    def _schedule_twitter_post(self, text: str, image_url: str, scheduled_time: str) -> Dict:
        """Schedule Twitter post for later"""
        try:
            # Validate datetime format
            try:
                target_datetime = datetime.strptime(scheduled_time, '%Y-%m-%d %H:%M')
            except ValueError:
                return {"error": "Invalid date format. Use YYYY-MM-DD HH:MM"}
            
            # Get current time in IST for comparison
            current_time = datetime.now(pytz.timezone('Asia/Kolkata')).replace(tzinfo=None)
            if target_datetime <= current_time:
                return {"error": "Scheduled time must be in the future"}
            
            # Create unique post ID
            post_id = f"twitter_{int(time.time())}_{random.randint(1000,9999)}"
            
            # Store post data
            self.scheduled_posts[post_id] = {
                'platform': 'twitter',
                'text': text,
                'image_url': image_url,
                'scheduled_time': scheduled_time,
                'status': 'scheduled'
            }
            
            self.save_scheduled_posts()
            
            return {
                "success": True, 
                "post_id": post_id, 
                "scheduled_time": scheduled_time,
                "platform": "twitter"
            }
            
        except Exception as e:
            logger.error(f"Twitter schedule error: {e}")
            return {"error": str(e)}
    
    # UNIFIED METHODS
    def _schedule_post(self, post_data: Dict) -> Dict[str, Any]:
        """Store a scheduled post"""
        try:
            # Generate unique post ID
            post_id = f"{post_data['platform']}_{int(time.time())}_{random.randint(1000,9999)}"
            
            # Create base post data
            scheduled_post = {
                'platform': post_data['platform'],
                'message': post_data.get('message', ''),
                'text': post_data.get('text', post_data.get('message', '')),  # Ensure text is always set
                'scheduled_time': post_data.get('scheduled_time'),
                'status': 'scheduled'
            }
            
            # Add platform-specific data
            if post_data['platform'] == 'facebook':
                scheduled_post['type'] = post_data.get('type', 'text')
                if post_data.get('type') == 'photo':
                    scheduled_post['photo_path'] = post_data.get('photo_path')
                    scheduled_post['photo_url'] = post_data.get('photo_url')
                elif post_data.get('type') == 'link':
                    scheduled_post['link'] = post_data.get('link')
            elif post_data['platform'] == 'instagram':
                # For Instagram, always store photo path or URL
                scheduled_post['photo_path'] = post_data.get('photo_path')
                scheduled_post['photo_url'] = post_data.get('photo_url')
            elif post_data['platform'] == 'linkedin':
                # For LinkedIn, store caption, hashtags and media
                scheduled_post['caption'] = post_data.get('caption', post_data.get('message', ''))
                scheduled_post['hashtags'] = post_data.get('hashtags', '')
                scheduled_post['photo_path'] = post_data.get('photo_path')
                scheduled_post['photo_url'] = post_data.get('photo_url')
            else:  # Twitter
                # Handle both image_url and photo_url for Twitter
                image_url = post_data.get('image_url') or post_data.get('photo_url')
                if image_url:
                    scheduled_post['image_url'] = image_url
            
            # Store the post
            self.scheduled_posts[post_id] = scheduled_post
            
            # Save to persistent storage
            self.save_scheduled_posts()
            
            # Log the scheduling
            logger.info(f"Scheduled new post: {post_id} for platform: {post_data['platform']}")
            logger.info(f"Current scheduled posts: {list(self.scheduled_posts.keys())}")
            
            return {
                "success": True,
                "message": "Post scheduled successfully",
                "post_id": post_id,
                "platform": post_data['platform'],
                "scheduled_time": post_data.get('scheduled_time')
            }
            
        except Exception as e:
            logger.error(f"Error scheduling post: {e}")
            return {"error": str(e)}

    def post_to_instagram(self, post_data: Dict) -> Dict[str, Any]:
        """Post to Instagram immediately or schedule for later"""
        try:
            if not self.ig_user_id or not self.ig_access_token:
                return {"error": "Instagram credentials not configured"}
            
            caption = post_data.get('message', '')
            image_path = post_data.get('photo_path')
            image_url = post_data.get('photo_url')
            
            # Handle scheduling
            if 'scheduled_time' in post_data:
                return self._schedule_post(post_data)
            
            # For immediate posts
            if not image_path and not image_url:
                return {"error": "Image is required for Instagram posts"}
            
            # Determine which image source to use
            image_source = None
            if image_path and os.path.exists(image_path):
                # Use local file if it exists
                image_source = image_path
                logger.info(f"Using local image file for Instagram post: {image_path}")
            elif image_url:
                # Use remote URL (including Google Cloud Storage URLs)
                image_source = image_url
                logger.info(f"Using remote image URL for Instagram post: {image_url[:100]}...")
            else:
                return {"error": "No valid image found for Instagram post"}
            
            # Post to Instagram using the instagram_post function
            # The function now handles both local files and remote URLs
            media_id = instagram_post(
                image_path=image_source,  # Can be either local path or URL
                caption=caption,
                user_id=self.ig_user_id,
                access_token=self.ig_access_token
            )
            
            if media_id:
                return {
                    "success": True,
                    "media_id": media_id,
                    "platform": "instagram",
                    "message": "Post published successfully"
                }
            else:
                return {"error": "Failed to post to Instagram"}
                
        except Exception as e:
            logger.error(f"Instagram post error: {e}")
            return {"error": str(e)}

    def post_to_linkedin(self, post_data: Dict) -> Dict[str, Any]:
        """Post to LinkedIn immediately or schedule for later"""
        try:
            if not self.linkedin_access_token:
                return {"error": "LinkedIn access token not configured"}

            # Get the text content and hashtags
            text = post_data.get('text', '') or post_data.get('caption', '') or post_data.get('message', '')
            hashtags = post_data.get('hashtags', '')
            
            # Format the post text - if text already contains hashtags, don't add them again
            post_text = text.strip()
            
            # Only process hashtags if they're provided separately and not already in the text
            if hashtags and not any(tag.strip() in post_text for tag in str(hashtags).split() if tag.strip().startswith('#')):
                # Split hashtags by comma and clean
                hashtag_list = []
                for tag in str(hashtags).split(','):
                    tag = tag.strip()
                    if tag:
                        # Ensure hashtag starts with #
                        if not tag.startswith('#'):
                            tag = '#' + tag
                        hashtag_list.append(tag)
                
                if hashtag_list:
                    # Add hashtags on new lines
                    post_text = f"{post_text}\n\n{' '.join(hashtag_list)}"
                    logger.info(f"Added hashtags to post: {' '.join(hashtag_list)}")
            else:
                logger.info(f"Hashtags already included in text or not provided separately")

            # Add invisible character for duplicate prevention
            post_text = f"{post_text}\u200c"

            # Handle scheduling
            if 'scheduled_time' in post_data:
                scheduled_post_data = {
                    'platform': 'linkedin',
                    'text': text,
                    'caption': text,
                    'hashtags': hashtags,
                    'scheduled_time': post_data['scheduled_time'],
                    'photo_path': post_data.get('photo_path'),
                    'photo_url': post_data.get('photo_url')
                }
                logger.info(f"Scheduling LinkedIn post with data: {scheduled_post_data}")
                return self._schedule_post(scheduled_post_data)

            # Prepare headers for LinkedIn API
            headers = {
                'Authorization': f'Bearer {self.linkedin_access_token}',
                'Content-Type': 'application/json',
                'X-Restli-Protocol-Version': '2.0.0'
            }

            # Handle media upload if present
            media_urn = None
            if post_data.get('photo_path'):
                try:
                    media_urn = self._upload_image_to_linkedin(post_data['photo_path'])
                except Exception as e:
                    logger.error(f"LinkedIn media upload error: {e}")
                    return {"error": f"Failed to upload media: {str(e)}"}
            elif post_data.get('photo_url'):
                try:
                    # Download remote image and upload to LinkedIn
                    media_urn = self._upload_remote_image_to_linkedin(post_data['photo_url'])
                except Exception as e:
                    logger.error(f"LinkedIn remote media upload error: {e}")
                    return {"error": f"Failed to upload remote media: {str(e)}"}

            # Create post payload
            post_payload = {
                "author": f"urn:li:person:{self.linkedin_user_id}",
                "lifecycleState": "PUBLISHED",
                "specificContent": {
                    "com.linkedin.ugc.ShareContent": {
                        "shareCommentary": {
                            "text": post_text
                        },
                        "shareMediaCategory": "IMAGE" if media_urn else "NONE"
                    }
                },
                "visibility": {
                    "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
                }
            }

            # Add media if present
            if media_urn:
                post_payload["specificContent"]["com.linkedin.ugc.ShareContent"]["media"] = [{
                    "status": "READY",
                    "description": {
                        "text": "Image"
                    },
                    "media": media_urn,
                    "title": {
                        "text": "Image"
                    }
                }]

            # Post to LinkedIn
            response = requests.post(
                f"{self.linkedin_api_base}/ugcPosts",
                json=post_payload,
                headers=headers
            )

            if response.status_code == 201:
                post_response = response.json()
                return {
                    "success": True,
                    "post_id": post_response.get('id', 'unknown'),
                    "platform": "linkedin",
                    "message": "Post published successfully"
                }
            else:
                return {"error": f"LinkedIn API error: {response.text}"}

        except Exception as e:
            logger.error(f"LinkedIn post error: {e}")
            return {"error": str(e)}

    def _upload_image_to_linkedin(self, image_path: str) -> Optional[str]:
        """Upload image to LinkedIn and return media URN"""
        try:
            headers = {
                'Authorization': f'Bearer {self.linkedin_access_token}',
                'Content-Type': 'application/json',
                'X-Restli-Protocol-Version': '2.0.0'
            }

            # Step 1: Register upload
            register_payload = {
                "registerUploadRequest": {
                    "recipes": ["urn:li:digitalmediaRecipe:feedshare-image"],
                    "owner": f"urn:li:person:{self.linkedin_user_id}",
                    "serviceRelationships": [{
                        "relationshipType": "OWNER",
                        "identifier": "urn:li:userGeneratedContent"
                    }]
                }
            }

            register_response = requests.post(
                f"{self.linkedin_api_base}/assets?action=registerUpload",
                json=register_payload,
                headers=headers
            )

            if register_response.status_code != 200:
                raise Exception(f"Failed to register upload: {register_response.text}")

            register_data = register_response.json()
            upload_url = register_data['value']['uploadMechanism']['com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest']['uploadUrl']
            asset_urn = register_data['value']['asset']

            # Step 2: Upload the image
            with open(image_path, 'rb') as f:
                image_data = f.read()

            upload_headers = {
                'Authorization': f'Bearer {self.linkedin_access_token}'
            }

            upload_response = requests.put(
                upload_url,
                data=image_data,
                headers=upload_headers
            )

            if upload_response.status_code not in [200, 201]:
                raise Exception(f"Failed to upload image: {upload_response.text}")

            return asset_urn

        except Exception as e:
            logger.error(f"LinkedIn image upload error: {e}")
            raise

    def _upload_remote_image_to_linkedin(self, image_url: str) -> Optional[str]:
        """Download remote image and upload to LinkedIn"""
        import tempfile
        import os
        
        try:
            logger.info(f"Downloading remote image for LinkedIn: {image_url}")
            
            # Download the image
            response = requests.get(image_url)
            if response.status_code != 200:
                raise Exception(f"Failed to download image: {response.status_code}")
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name
            
            try:
                # Upload to LinkedIn using the local file method
                media_urn = self._upload_image_to_linkedin(temp_file_path)
                logger.info(f"Successfully uploaded remote image to LinkedIn: {media_urn}")
                return media_urn
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"LinkedIn remote image upload error: {e}")
            raise

    def schedule_post(self, platform: str, post_data: Dict) -> Dict[str, Any]:
        """Schedule a post for any platform"""
        try:
            # Normalize platform name
            platform = platform.lower()
            post_data['platform'] = platform
            
            # For scheduled posts, use the unified scheduling system
            if 'scheduled_time' in post_data:
                # Validate schedule time
                scheduled_timestamp = validate_schedule_time(post_data['scheduled_time'], platform)
                if scheduled_timestamp is None:
                    return {"error": "Invalid schedule time. Must be at least 10 minutes in the future and within 6 months."}
                
                # Use unified scheduling method
                return self._schedule_post(post_data)
            
            # For immediate posts
            if platform == 'facebook':
                return self.post_to_facebook(post_data)
            elif platform == 'twitter':
                # Handle both image_url and photo_url for Twitter
                image_url = post_data.get('image_url') or post_data.get('photo_url', '')
                return self._post_twitter_immediately(
                    post_data.get('text', ''),
                    image_url
                )
            elif platform == 'instagram':
                return self.post_to_instagram(post_data)
            elif platform == 'linkedin':
                return self.post_to_linkedin(post_data)
            else:
                return {"error": "Unsupported platform. Use 'facebook', 'twitter', 'instagram', or 'linkedin'"}
                
        except Exception as e:
            logger.error(f"Schedule post error: {e}")
            return {"error": str(e)}
    
    def execute_scheduled_posts(self):
        """Execute all scheduled posts that are due"""
        current_time = datetime.now(pytz.UTC)
        
        for post_id, post_data in list(self.scheduled_posts.items()):
            try:
                if post_data.get('status') != 'scheduled':
                    continue
                
                # Skip posts that are managed by MongoDB to prevent duplicate execution
                if post_data.get('db_post_id'):
                    logger.info(f"Skipping post {post_id} - managed by MongoDB system")
                    continue
                
                scheduled_time = datetime.strptime(post_data['scheduled_time'], '%Y-%m-%d %H:%M')
                # Convert current_time to naive datetime for comparison (assuming IST)
                current_time_naive = current_time.astimezone(pytz.timezone('Asia/Kolkata')).replace(tzinfo=None)
                
                if scheduled_time <= current_time_naive:
                    logger.info(f"Executing scheduled post {post_id}")
                    
                    platform = post_data.get('platform')
                    if platform == 'twitter':
                        result = self._post_twitter_immediately(
                            post_data.get('text', ''),
                            post_data.get('image_url', '')
                        )
                    elif platform == 'facebook':
                        # Create post data for Facebook
                        fb_post_data = {
                            'type': post_data.get('type', 'text'),
                            'message': post_data.get('message', ''),
                            'text': post_data.get('text', '')
                        }
                        
                        # Add media data if present
                        if post_data.get('type') == 'photo':
                            if post_data.get('photo_path'):
                                fb_post_data['photo_path'] = post_data['photo_path']
                            elif post_data.get('photo_url'):
                                fb_post_data['photo_url'] = post_data['photo_url']
                        elif post_data.get('type') == 'link':
                            fb_post_data['link'] = post_data.get('link', '')
                        
                        # Post to Facebook
                        result = self.post_to_facebook(fb_post_data)
                    elif platform == 'instagram':
                        # Create post data for Instagram
                        ig_post_data = {
                            'message': post_data.get('message', ''),
                            'photo_path': post_data.get('photo_path'),
                            'photo_url': post_data.get('photo_url')  # Include photo URL as fallback
                        }
                        
                        # Log the Instagram post data for debugging
                        logger.info(f"Executing Instagram post with data: {ig_post_data}")
                        
                        # Verify image path exists
                        if ig_post_data.get('photo_path') and not os.path.exists(ig_post_data['photo_path']):
                            logger.error(f"Image file not found at path: {ig_post_data['photo_path']}")
                            if not ig_post_data.get('photo_url'):
                                logger.error("No fallback photo URL available")
                                continue
                            else:
                                # Remove invalid photo_path if we have a photo_url fallback
                                del ig_post_data['photo_path']
                        
                        # Post to Instagram
                        result = self.post_to_instagram(ig_post_data)
                    elif platform == 'linkedin':
                        # Create post data for LinkedIn
                        ln_post_data = {
                            'text': post_data.get('text', ''),  # Original text
                            'caption': post_data.get('caption', ''),  # Original caption
                            'hashtags': post_data.get('hashtags', ''),  # Original hashtags
                            'photo_path': post_data.get('photo_path'),
                            'photo_url': post_data.get('photo_url')
                        }
                        
                        # Log the LinkedIn post data for debugging
                        logger.info(f"Executing LinkedIn post with data: {ln_post_data}")
                        
                        # Verify image path exists
                        if ln_post_data.get('photo_path') and not os.path.exists(ln_post_data['photo_path']):
                            logger.error(f"Image file not found at path: {ln_post_data['photo_path']}")
                            if not ln_post_data.get('photo_url'):
                                logger.error("No fallback photo URL available")
                                continue
                            else:
                                # Remove invalid photo_path if we have a photo_url fallback
                                del ln_post_data['photo_path']
                        
                        # Post to LinkedIn
                        result = self.post_to_linkedin(ln_post_data)
                    else:
                        logger.error(f"Unknown platform for post {post_id}")
                        continue
                    
                    if result.get('success'):
                        self.scheduled_posts[post_id]['status'] = 'executed'
                        self.save_scheduled_posts()
                        logger.info(f"Post {post_id} executed successfully")
                    else:
                        self.scheduled_posts[post_id]['status'] = 'failed'
                        self.save_scheduled_posts()
                        logger.error(f"Failed to execute post {post_id}: {result}")
                        
            except Exception as e:
                logger.error(f"Error executing scheduled post: {str(e)}")
                continue
    
    def get_scheduled_posts(self) -> Dict[str, Any]:
        """Get all scheduled posts"""
        return {
            "scheduled_posts": self.scheduled_posts,
            "total_count": len(self.scheduled_posts)
        }
    
    def delete_scheduled_post(self, post_id: str) -> Dict[str, Any]:
        """Delete a scheduled post"""
        if post_id in self.scheduled_posts:
            del self.scheduled_posts[post_id]
            self.save_scheduled_posts()
            return {"success": True, "message": "Post deleted"}
        else:
            return {"error": "Post not found"}

# Initialize the scheduler
scheduler = SocialMediaScheduler()

# API ENDPOINTS
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now(pytz.UTC).isoformat()})

@app.route('/api/cors-test', methods=['GET', 'POST', 'OPTIONS'])
def cors_test():
    """Test endpoint to verify CORS configuration"""
    return jsonify({
        "message": "CORS is working!", 
        "origin": request.headers.get('Origin', 'No origin header'),
        "method": request.method,
        "timestamp": datetime.now(pytz.UTC).isoformat()
    })

@app.route('/api/upload-image', methods=['POST', 'OPTIONS'])
def upload_image():
    """Upload image to Google Cloud Storage using working example pattern"""
    try:
        # Handle preflight request
        if request.method == 'OPTIONS':
            return jsonify({'status': 'ok'}), 200

        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "error": "No image file provided"
            }), 400

        file = request.files['image']
        
        # Check if file is selected and has a filename
        if not file or not file.filename:
            return jsonify({
                "success": False,
                "error": "No file selected or invalid filename"
            }), 400

        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400

        # Upload to Google Cloud Storage using the working example approach
        file.seek(0)  # Reset file pointer
        public_url = upload_to_gcs(file, str(file.filename))
        
        if public_url:
            logger.info(f"✅ Image uploaded successfully: {public_url}")
            return jsonify({
                "success": True,
                "statusCode": 200,
                "data": {
                    "imageUrl": public_url,
                    "filename": file.filename
                },
                "message": "Image uploaded successfully"
            }), 200
        else:
            logger.error("❌ Failed to upload image to cloud storage")
            return jsonify({
                "success": False,
                "statusCode": 500,
                "error": "Failed to upload image to cloud storage"
            }), 500

    except Exception as e:
        logger.error(f"❌ Error in upload_image: {e}")
        return jsonify({
            "success": False,
            "statusCode": 500,
            "error": str(e)
        }), 500

@app.route('/api/schedule-multiple', methods=['POST', 'OPTIONS'])
def schedule_multiple_platforms():
    """Schedule a post for multiple platforms with MongoDB integration"""
    try:
        # Handle preflight request
        if request.method == 'OPTIONS':
            return jsonify({'status': 'ok'}), 200

        # Parse JSON data
        if not request.is_json:
            return jsonify({
                "success": False,
                "error": "Content-Type must be application/json"
            }), 400

        data = request.get_json()
        
        # Check if we have platformScheduleTimes (new format) or scheduledTime (old format)
        has_platform_times = 'platformScheduleTimes' in data
        
        if has_platform_times:
            # New format with different times for each platform
            required_fields = ['userId', 'content', 'platforms', 'platformScheduleTimes']
            for field in required_fields:
                if field not in data:
                    return jsonify({
                        "success": False,
                        "error": f"Missing required field: {field}"
                    }), 400

            user_id = data['userId']
            content_data = data['content']
            platforms = data['platforms']
            platform_schedule_times = data['platformScheduleTimes']  # Dict with platform: time
            ai_generation = data.get('aiGeneration')

            # Validate platform schedule times
            if not isinstance(platform_schedule_times, dict):
                return jsonify({
                    "success": False,
                    "error": "platformScheduleTimes must be a dictionary"
                }), 400

            # Validate that all platforms have schedule times
            for platform in platforms:
                if platform not in platform_schedule_times:
                    return jsonify({
                        "success": False,
                        "error": f"Missing schedule time for platform: {platform}"
                    }), 400

        else:
            # Old format with same time for all platforms
            required_fields = ['userId', 'content', 'platforms', 'scheduledTime']
            for field in required_fields:
                if field not in data:
                    return jsonify({
                        "success": False,
                        "error": f"Missing required field: {field}"
                    }), 400

            user_id = data['userId']
            content_data = data['content']
            platforms = data['platforms']
            scheduled_time_ist = data['scheduledTime']  # Expected format: "YYYY-MM-DD HH:MM"
            ai_generation = data.get('aiGeneration')
            
            # Convert to platform_schedule_times format for consistency
            platform_schedule_times = {platform: scheduled_time_ist for platform in platforms}

        # Validate content
        if not content_data.get('caption'):
            return jsonify({
                "success": False,
                "error": "Caption is required in content"
            }), 400

        # Validate platforms
        if not isinstance(platforms, list) or not platforms:
            return jsonify({
                "success": False,
                "error": "Platforms must be a non-empty list"
            }), 400

        # Schedule the post for multiple platforms with different times
        result = scheduler.schedule_post_for_multiple_platforms_with_times(
            user_id=user_id,
            content_data=content_data,
            platforms=platforms,
            platform_schedule_times=platform_schedule_times,
            ai_generation=ai_generation
        )

        if result.get('success'):
            return jsonify(result), 200
        else:
            return jsonify(result), 400

    except Exception as e:
        logger.error(f"Error in schedule_multiple_platforms: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/user-posts/<user_id>', methods=['GET', 'OPTIONS'])
def get_user_posts(user_id):
    """Get scheduled posts for a specific user from MongoDB"""
    try:
        # Add CORS headers
        response_headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization'
        }

        # Handle preflight request
        if request.method == 'OPTIONS':
            return ('', 204)

        limit = request.args.get('limit', 50, type=int)
        
        posts = scheduler.get_user_scheduled_posts_from_db(user_id, limit)
        
        return jsonify({
            "success": True,
            "posts": posts,
            "total_count": len(posts)
        }), 200

    except Exception as e:
        logger.error(f"Error getting user posts: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/schedule', methods=['POST'])
def schedule_post():
    """Schedule a post for Facebook, Twitter, or Instagram"""
    try:
        # Debug logging
        logger.info("Received schedule request")
        logger.info(f"Form data: {request.form}")
        logger.info(f"Files: {request.files}")

        # Add CORS headers
        response_headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        }

        # Handle preflight request
        if request.method == 'OPTIONS':
            return ('', 204)

        # Check if any data was sent
        if not request.form and not request.files:
            logger.error("No data received in request")
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400

        platform = request.form.get('platform', '').lower()
        if platform not in ['facebook', 'twitter', 'instagram', 'linkedin']:
            logger.error(f"Invalid platform: {platform}")
            return jsonify({
                "success": False,
                "error": "Platform must be 'facebook', 'twitter', 'instagram', or 'linkedin'"
            }), 400

        # Initialize post data with required type checking
        message = request.form.get('message', '')
        if not message:
            return jsonify({
                "success": False,
                "error": "Message is required"
            }), 400

        post_type = request.form.get('type', 'text')
        
        # Process hashtags - ensure they start with #
        hashtags = request.form.get('hashtags', '').strip()
        if hashtags:
            # Split by comma and clean each hashtag
            hashtag_list = [tag.strip() for tag in hashtags.split(',') if tag.strip()]
            # Ensure each hashtag starts with #
            hashtag_list = ['#' + tag.lstrip('#') for tag in hashtag_list]
            # Join back with commas for storage
            hashtags = ','.join(hashtag_list)
            logger.info(f"Processed hashtags: {hashtags}")
        
        post_data = {
            'platform': platform,
            'type': post_type,
            'message': message,
            'text': message,  # For Twitter compatibility
            'hashtags': hashtags  # Add processed hashtags
        }

        # Handle scheduling
        scheduled_time = request.form.get('scheduled_time')
        if scheduled_time:
            try:
                # Parse the input time as local time
                scheduled_dt = datetime.strptime(scheduled_time, '%Y-%m-%d %H:%M')
                # Store the time as is, without timezone adjustment
                post_data['scheduled_time'] = scheduled_dt.strftime('%Y-%m-%d %H:%M')
                logger.info(f"Scheduled time set to: {post_data['scheduled_time']}")
            except ValueError as e:
                return jsonify({
                    "success": False,
                    "error": f"Invalid date format: {str(e)}"
                }), 400

        # Handle file upload
        media_file = request.files.get('media_file')
        if media_file:
            logger.info(f"Processing media file: {media_file.filename}")
            if not allowed_file(media_file.filename):
                return jsonify({
                    "success": False,
                    "error": "Invalid file type. Allowed types: png, jpg, jpeg, gif"
                }), 400

            try:
                media_path = save_uploaded_file(media_file)
                if not media_path:
                    raise Exception("Failed to save uploaded file")

                logger.info(f"File saved successfully at: {media_path}")
                
                # Add the local file path to post data
                if platform in ['facebook', 'instagram', 'linkedin']:
                    post_data['photo_path'] = media_path
                else:
                    post_data['image_url'] = media_path

            except Exception as e:
                logger.error(f"File upload error: {str(e)}")
                return jsonify({
                    "success": False,
                    "error": f"File upload failed: {str(e)}"
                }), 400

        # Handle media URL if no file was uploaded
        elif request.form.get('photo_url') or request.form.get('image_url'):
            if platform in ['facebook', 'instagram', 'linkedin']:
                photo_url = request.form.get('photo_url')
                if photo_url:  # Only set if not None
                    post_data['photo_url'] = photo_url
            else:
                image_url = request.form.get('image_url')
                if image_url:  # Only set if not None
                    post_data['image_url'] = image_url

        # Handle link for Facebook link posts
        if platform == 'facebook' and post_data['type'] == 'link':
            link = request.form.get('link')
            if link:
                post_data['link'] = link

        # Validate Instagram requirements
        if platform == 'instagram' and not (media_file or request.form.get('photo_url')):
            return jsonify({
                "success": False,
                "error": "Instagram posts require an image"
            }), 400

        # Schedule the post
        logger.info(f"Sending post data to scheduler: {post_data}")
        result = scheduler.schedule_post(platform, post_data)

        if result.get('success'):
            logger.info("Post scheduled/published successfully")
            response_data = {
                "success": True,
                "message": result.get('message', 'Post processed successfully'),
                "post_id": result.get('post_id'),
                "platform": platform,
                "scheduled_time": post_data.get('scheduled_time')
            }
            return jsonify(response_data), 200
        else:
            logger.error(f"Scheduler error: {result}")
            return jsonify({
                "success": False,
                "error": result.get('error', 'Failed to process post')
            }), 400

    except Exception as e:
        logger.error(f"Unexpected error in schedule endpoint: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "details": "Server error processing request"
        }), 500

@app.route('/api/posts', methods=['GET', 'OPTIONS'])
def get_posts():
    """Get all scheduled posts"""
    try:
        # Add CORS headers
        response_headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        }

        # Handle preflight request
        if request.method == 'OPTIONS':
            return ('', 204)

        result = scheduler.get_scheduled_posts()
        
        # Ensure we have the correct structure even if there are no posts
        if 'scheduled_posts' not in result:
            result['scheduled_posts'] = {}
            
        response_data = {
            "success": True,
            "scheduled_posts": result.get('scheduled_posts', {}),
            "total_count": len(result.get('scheduled_posts', {}))
        }
        
        logger.info(f"Returning {response_data['total_count']} scheduled posts")
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Get posts error: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "scheduled_posts": {},
            "total_count": 0
        }), 500

@app.route('/api/posts/<post_id>', methods=['DELETE', 'OPTIONS'])
def delete_post(post_id):
    """Delete a scheduled post"""
    try:
        # Add CORS headers
        response_headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'DELETE, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        }

        # Handle preflight request
        if request.method == 'OPTIONS':
            return ('', 204)

        result = scheduler.delete_scheduled_post(post_id)
        
        if result.get('success'):
            return jsonify({
                "success": True,
                "message": "Post deleted successfully"
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": result.get('error', 'Post not found')
            }), 404
            
    except Exception as e:
        logger.error(f"Delete post error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/execute', methods=['POST'])
def execute_scheduled_posts():
    """Manually trigger execution of scheduled posts"""
    try:
        scheduler.execute_scheduled_posts()
        return jsonify({"message": "Scheduled posts execution completed"}), 200
    except Exception as e:
        logger.error(f"Execute posts error: {e}")
        return jsonify({"error": str(e)}), 500

# Background scheduler thread
def run_scheduler():
    """Background thread to execute scheduled posts from MongoDB only"""
    while True:
        try:
            # Only execute posts from MongoDB (new functionality)
            # The old JSON system is kept for backward compatibility but not executed automatically
            scheduler.execute_scheduled_posts_from_db()
            
            time.sleep(60)  # Check every minute
        except Exception as e:
            logger.error(f"Background scheduler error: {e}")
            time.sleep(60)

# Start background scheduler
scheduler_thread = Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()

# Add these new routes to serve the frontend files
@app.route('/')
def serve_html():
    return send_from_directory('.', 'social_scheduler_interface.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

@app.route('/api/schedule-single', methods=['POST', 'OPTIONS'])
def schedule_single_platform():
    """Schedule a post for a single platform with MongoDB integration"""
    try:
        # Add CORS headers
        response_headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization'
        }

        # Handle preflight request
        if request.method == 'OPTIONS':
            return ('', 204)

        # Parse JSON data
        if not request.is_json:
            return jsonify({
                "success": False,
                "error": "Content-Type must be application/json"
            }), 400

        data = request.get_json()
        
        # Validate required fields
        required_fields = ['userId', 'content', 'platform', 'scheduledTime']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "success": False,
                    "error": f"Missing required field: {field}"
                }), 400

        user_id = data['userId']
        content_data = data['content']
        platform = data['platform'].lower()
        scheduled_time = data['scheduledTime']  # Expected format: "YYYY-MM-DD HH:MM"
        ai_generation = data.get('aiGeneration')

        # Validate platform
        valid_platforms = ['facebook', 'twitter', 'instagram', 'linkedin']
        if platform not in valid_platforms:
            return jsonify({
                "success": False,
                "error": f"Invalid platform. Must be one of: {', '.join(valid_platforms)}"
            }), 400

        # Validate content
        if not content_data.get('caption'):
            return jsonify({
                "success": False,
                "error": "Caption is required in content"
            }), 400

        # Schedule for single platform (convert to list for consistency)
        platforms = [platform]
        
        # Use the scheduler to create the scheduled post
        result = scheduler.schedule_post_for_multiple_platforms(
            user_id=user_id,
            content_data=content_data,
            platforms=platforms,
            schedule_time_ist=scheduled_time,
            ai_generation=ai_generation
        )

        if result.get('success'):
            return jsonify({
                "success": True,
                "statusCode": 200,
                "data": {
                    "postId": result['db_post_id'],
                    "platform": platform,
                    "scheduledTime": result['scheduled_time_ist'],
                    "scheduledTimeUTC": result['scheduled_time_utc'],
                    "status": "scheduled"
                },
                "message": f"Post scheduled successfully for {platform}"
            }), 200
        else:
            return jsonify({
                "success": False,
                "statusCode": 400,
                "error": result.get('error', 'Failed to schedule post')
            }), 400

    except Exception as e:
        logger.error(f"Error in schedule_single_platform: {e}")
        return jsonify({
            "success": False,
            "statusCode": 500,
            "error": str(e)
        }), 500

@app.route('/api/schedule-all', methods=['POST', 'OPTIONS'])
def schedule_all_platforms():
    """Schedule a post for all platforms with MongoDB integration"""
    try:
        # Add CORS headers
        response_headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization'
        }

        # Handle preflight request
        if request.method == 'OPTIONS':
            return ('', 204)

        # Parse JSON data
        if not request.is_json:
            return jsonify({
                "success": False,
                "error": "Content-Type must be application/json"
            }), 400

        data = request.get_json()
        
        # Validate required fields
        required_fields = ['userId', 'content', 'scheduledTime']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "success": False,
                    "error": f"Missing required field: {field}"
                }), 400

        user_id = data['userId']
        content_data = data['content']
        scheduled_time = data['scheduledTime']  # Expected format: "YYYY-MM-DD HH:MM"
        ai_generation = data.get('aiGeneration')

        # Validate content
        if not content_data.get('caption'):
            return jsonify({
                "success": False,
                "error": "Caption is required in content"
            }), 400

        # Schedule for all platforms
        all_platforms = ['facebook', 'twitter', 'instagram', 'linkedin']
        
        # Use the scheduler to create the scheduled post
        result = scheduler.schedule_post_for_multiple_platforms(
            user_id=user_id,
            content_data=content_data,
            platforms=all_platforms,
            schedule_time_ist=scheduled_time,
            ai_generation=ai_generation
        )

        if result.get('success'):
            return jsonify({
                "success": True,
                "statusCode": 200,
                "data": {
                    "postId": result['db_post_id'],
                    "platforms": all_platforms,
                    "scheduledTime": result['scheduled_time_ist'],
                    "scheduledTimeUTC": result['scheduled_time_utc'],
                    "status": "scheduled",
                    "totalPlatforms": len(all_platforms)
                },
                "message": f"Post scheduled successfully for all {len(all_platforms)} platforms"
            }), 200
        else:
            return jsonify({
                "success": False,
                "statusCode": 400,
                "error": result.get('error', 'Failed to schedule post')
            }), 400

    except Exception as e:
        logger.error(f"Error in schedule_all_platforms: {e}")
        return jsonify({
            "success": False,
            "statusCode": 500,
            "error": str(e)
        }), 500

@app.route('/api/clear-failed-posts', methods=['POST'])
def clear_failed_posts():
    """Clear all failed posts from the old JSON scheduling system"""
    try:
        # Load current scheduled posts
        if os.path.exists('scheduled_posts.json'):
            with open('scheduled_posts.json', 'r') as f:
                scheduled_posts = json.load(f)
            
            # Remove failed posts
            posts_to_remove = []
            for post_id, post_data in scheduled_posts.items():
                if post_data.get('status') == 'failed':
                    posts_to_remove.append(post_id)
            
            # Remove failed posts
            for post_id in posts_to_remove:
                del scheduled_posts[post_id]
            
            # Save updated posts
            with open('scheduled_posts.json', 'w') as f:
                json.dump(scheduled_posts, f, indent=2) 
            
            return jsonify({
                "success": True,
                "message": f"Cleared {len(posts_to_remove)} failed posts",
                "cleared_posts": posts_to_remove
            })
        else:
            return jsonify({
                "success": True,
                "message": "No scheduled_posts.json file found"
            })
            
    except Exception as e:
        logger.error(f"Clear failed posts error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/post-now', methods=['POST', 'OPTIONS'])
def post_now():
    """Post immediately to selected platforms"""
    try:
        # Add CORS headers
        response_headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization'
        }

        # Handle preflight request
        if request.method == 'OPTIONS':
            return jsonify({"status": "ok"}), 200

        # Parse JSON data
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400

        data = request.get_json()
        logger.info(f"Received post-now request: {data}")

        # Extract data
        platforms = data.get('platforms', [])
        content_data = {
            'caption': data.get('caption', ''),
            'hashtags': data.get('hashtags', []),
            'images': data.get('images', [])
        }
        user_id = data.get('userId', 'default_user')

        # Validate platforms
        valid_platforms = ['facebook', 'twitter', 'instagram', 'linkedin']
        platforms = [p.lower() for p in platforms if p.lower() in valid_platforms]
        
        if not platforms:
            return jsonify({"error": "At least one valid platform must be selected"}), 400

        # Validate content
        if not content_data.get('caption'):
            return jsonify({"error": "Caption is required"}), 400

        # Post to each platform immediately
        results = {}
        success_count = 0
        total_platforms = len(platforms)

        for platform in platforms:
            try:
                logger.info(f"Posting to {platform}...")
                
                # Prepare platform-specific content
                platform_content = scheduler.format_content_with_hashtags(
                    content_data['caption'], 
                    content_data.get('hashtags', []), 
                    platform
                )
                
                # Get image URL if available
                image_url = content_data.get('images', [None])[0] if content_data.get('images') else None
                
                # Post to specific platform
                if platform == 'facebook':
                    post_data = {
                        'type': 'photo' if image_url else 'text',
                        'message': platform_content,
                        'photo_url': image_url
                    }
                    result = scheduler.post_to_facebook(post_data)
                
                elif platform == 'twitter':
                    post_data = {
                        'text': platform_content,
                        'image_url': image_url
                    }
                    result = scheduler.post_to_twitter(post_data)
                
                elif platform == 'instagram':
                    if not image_url:
                        result = {"error": "Instagram requires an image"}
                    else:
                        post_data = {
                            'message': platform_content,
                            'photo_url': image_url
                        }
                        result = scheduler.post_to_instagram(post_data)
                
                elif platform == 'linkedin':
                    post_data = {
                        'text': platform_content,
                        'photo_url': image_url
                        # Don't pass hashtags separately - they're already in platform_content
                    }
                    result = scheduler.post_to_linkedin(post_data)
                
                # Store result
                results[platform] = result
                
                if result.get('success') or result.get('post_id') or result.get('photo_id') or result.get('tweet_id') or result.get('media_id'):
                    success_count += 1
                    logger.info(f"✅ Successfully posted to {platform}")
                else:
                    logger.error(f"❌ Failed to post to {platform}: {result}")
                    
            except Exception as e:
                logger.error(f"Error posting to {platform}: {e}")
                results[platform] = {"error": str(e)}

        # Prepare response
        if success_count == total_platforms:
            status_message = f"Successfully posted to all {total_platforms} platform(s)"
            status_code = 200
        elif success_count > 0:
            status_message = f"Posted to {success_count} out of {total_platforms} platform(s)"
            status_code = 207  # Partial success
        else:
            status_message = "Failed to post to any platform"
            status_code = 400

        response_data = {
            "success": success_count > 0,
            "message": status_message,
            "totalPlatforms": total_platforms,
            "successCount": success_count,
            "platforms": platforms,
            "results": results,
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }

        return jsonify(response_data), status_code

    except Exception as e:
        logger.error(f"Error in post_now endpoint: {e}")
        return jsonify({
            "error": f"Failed to post: {str(e)}",
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Unified Social Media Scheduler API with MongoDB Integration")
    print("📝 Available endpoints:")
    print("   GET  / - Frontend interface")
    print("   POST /api/schedule - Schedule a post for single platform")
    print("   POST /api/schedule-multiple - Schedule a post for multiple platforms (MongoDB)")
    print("   POST /api/post-now - Post immediately to selected platform(s)")
    print("   GET  /api/posts - Get all scheduled posts (JSON)")
    print("   GET  /api/user-posts/<user_id> - Get user scheduled posts (MongoDB)")
    print("   DELETE /api/posts/<post_id> - Delete a scheduled post")
    print("   POST /api/execute - Manually execute scheduled posts")
    print("   GET  /health - Health check")
    print("\n📊 Database: MongoDB (Al-SocialMedia)")
    print("🕒 Timezone: Indian Standard Time (IST)")
    print(f"\nServer will start on port {os.getenv('PORT', 5002)}")
    
    # Use PORT environment variable for deployment platforms like Render
    port = int(os.getenv('PORT', 5002))
    app.run(debug=False, host='0.0.0.0', port=port)
