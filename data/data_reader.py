from abc import ABC, abstractmethod
from functools import lru_cache
from io import BytesIO
import io
import logging
import os
import threading
import time
from typing import Union

import requests
from PIL import Image
import urllib3

Image.MAX_IMAGE_PIXELS = None
logger = logging.getLogger(__name__)


def download_image(url: str) -> Image.Image:
    try:
        response = requests.get(url, stream=True).raw
        image = Image.open(response)
        image.load()
        return image
    except Exception as e:
        logger.error(f"Failed to download image from {url}: {e}")
        return None


def read_general(path) -> Union[str, BytesIO]:
    if "s3://" in path:
        init_ceph_client_if_needed()
        file_bytes = BytesIO(client.get(path))
        return file_bytes
    else:
        return path


def init_ceph_client_if_needed():
    global client
    if client is None:
        logger.info(f"initializing ceph client ...")
        st = time.time()
        from petrel_client.client import Client  # noqa

        client = Client("./petreloss.conf")
        print("start read image ")
        ed = time.time()
        logger.info(f"initialize client cost {ed - st:.2f} s")


client = None




#############################################################################
#                            Data item Processor                            #
#############################################################################


class ItemProcessor(ABC):
    @abstractmethod
    def process_item(self, data_item, training_mode=False):
        raise NotImplementedError


class WorldModelItemProcessor(ItemProcessor):
    def __init__(self, transform=None, max_retries=3, timeout=10, 
                 connection_pool_size=100, image_cache_size=1000):
        """
        Initialize the ItemProcessor with optional image transformation.
        
        Args:
            transform: Optional function to transform the loaded image
            max_retries: Maximum number of download retries
            timeout: Timeout for HTTP requests in seconds
            connection_pool_size: Size of the connection pool
            image_cache_size: Size of the LRU cache for downloaded images
        """
        self.transform = transform
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Create a connection pool manager for efficient HTTP connections
        self.http = urllib3.PoolManager(
            maxsize=connection_pool_size,
            retries=urllib3.Retry(
                total=max_retries,
                backoff_factor=0.5,
                status_forcelist=[500, 502, 503, 504],
            ),
            timeout=timeout
        )
        
        # Thread-local storage for per-worker connection pools
        self.local = threading.local()
        
        # Create an LRU cache for images to avoid redundant downloads
        self.image_cache = lru_cache(maxsize=image_cache_size)(self._download_image)
        
    def _get_session(self):
        """Get or create a requests session for the current thread."""
        if not hasattr(self.local, 'session'):
            self.local.session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=20, 
                pool_maxsize=20,
                max_retries=self.max_retries
            )
            self.local.session.mount('http://', adapter)
            self.local.session.mount('https://', adapter)
        return self.local.session
        
    def _download_image(self, media_path):
        """
        Download an image from the given media path.
        
        Args:
            media_path: URL or file path to the image
            
        Returns:
            PIL Image object
        """
        # Handle local file paths
        if media_path.startswith('/') or media_path.startswith('./') or os.path.exists(media_path):
            return Image.open(media_path).convert('RGB')
        
        # Handle URLs
        try:
            session = self._get_session()
            response = session.get(media_path, timeout=self.timeout)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to download image from {media_path}: {e}")
            # Return a small black image as a fallback
            return Image.new('RGB', (224, 224), color='black')
    
    def process_item(self, data_item, training_mode=False):
        """
        Process a data item by downloading and optionally transforming the image.
        
        Args:
            data_item: Dictionary containing item metadata
            training_mode: Boolean indicating whether we're in training mode
            
        Returns:
            Dictionary with the processed item
        """
        result = {}
        
        # Copy metadata fields
        for key in ['_id', 'source_id', 'width', 'height', 'caption', 'source']:
            if key in data_item:
                result[key] = data_item[key]
        
        # Download and process the image
        try:
            # Use the cached download function
            image = self.image_cache(data_item['media_path'])
            
            # Apply transformation if provided
            if self.transform is not None:
                image = self.transform(image)
                
            result['image'] = image
        except Exception as e:
            if training_mode:
                # In training mode, log the error but don't fail
                logger.warning(f"Error processing image {data_item.get('_id', 'unknown')}: {e}")
                # Provide a fallback image
                result['image'] = Image.new('RGB', (224, 224), color='black')
            else:
                # In evaluation mode, propagate the error
                raise RuntimeError(f"Failed to process image: {e}")
        
        return result['image'], result['caption']
    
    def clear_cache(self):
        """Clear the image download cache."""
        self.image_cache.cache_clear()