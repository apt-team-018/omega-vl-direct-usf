"""
Multi-Modal Query Examples for Omega VLM API

This file demonstrates how to send text + image requests to the VLM API
using OpenAI-compatible format.
"""

import requests
import base64
import json

# API Configuration
API_URL = "http://localhost:8000/v1/chat/completions"
API_KEY = "EMPTY"  # Set to your API key if authentication is enabled

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


def example_1_image_url():
    """Example 1: OpenAI-compatible format with image URL"""
    payload = {
        "model": "model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/image.jpg"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.7,
        "max_tokens": 512
    }
    
    response = requests.post(API_URL, json=payload, headers=headers)
    print("Example 1 - Image URL:")
    print(json.dumps(response.json(), indent=2))
    print("\n" + "="*80 + "\n")


def example_2_base64_data_url():
    """Example 2: OpenAI-compatible format with base64 data URL"""
    # Read and encode an image
    with open("example_image.jpg", "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    payload = {
        "model": "model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image in detail"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.8,
        "max_tokens": 1024
    }
    
    response = requests.post(API_URL, json=payload, headers=headers)
    print("Example 2 - Base64 Data URL:")
    print(json.dumps(response.json(), indent=2))
    print("\n" + "="*80 + "\n")


def example_3_multiple_images():
    """Example 3: Multiple images in a single request"""
    payload = {
        "model": "model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Compare these two images and describe the differences"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/image1.jpg"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/image2.jpg"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.7,
        "max_tokens": 512
    }
    
    response = requests.post(API_URL, json=payload, headers=headers)
    print("Example 3 - Multiple Images:")
    print(json.dumps(response.json(), indent=2))
    print("\n" + "="*80 + "\n")


def example_4_conversation_with_images():
    """Example 4: Multi-turn conversation with images"""
    payload = {
        "model": "model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What do you see in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/scene.jpg"
                        }
                    }
                ]
            },
            {
                "role": "assistant",
                "content": "I see a beautiful sunset over mountains with vibrant colors."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What time of day was this photo likely taken?"
                    }
                ]
            }
        ],
        "temperature": 0.7,
        "max_tokens": 256
    }
    
    response = requests.post(API_URL, json=payload, headers=headers)
    print("Example 4 - Conversation with Images:")
    print(json.dumps(response.json(), indent=2))
    print("\n" + "="*80 + "\n")


def example_5_legacy_format():
    """Example 5: Legacy format (still supported for backward compatibility)"""
    payload = {
        "model": "model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this image"
                    },
                    {
                        "type": "image",
                        "image": "https://example.com/image.jpg"
                    }
                ]
            }
        ],
        "temperature": 0.7,
        "max_tokens": 512
    }
    
    response = requests.post(API_URL, json=payload, headers=headers)
    print("Example 5 - Legacy Format:")
    print(json.dumps(response.json(), indent=2))
    print("\n" + "="*80 + "\n")


def example_6_curl_command():
    """Example 6: cURL command for testing"""
    curl_command = """
curl -X POST http://localhost:8000/v1/chat/completions \\
  -H "Authorization: Bearer EMPTY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "model",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What is in this image?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://example.com/photo.jpg"
            }
          }
        ]
      }
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
    """
    print("Example 6 - cURL Command:")
    print(curl_command)
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("OMEGA VLM API - Multi-Modal Query Examples")
    print("="*80 + "\n")
    
    # Note: Uncomment the examples you want to run
    # Make sure to replace example URLs with actual image URLs
    
    # example_1_image_url()
    # example_2_base64_data_url()
    # example_3_multiple_images()
    # example_4_conversation_with_images()
    # example_5_legacy_format()
    example_6_curl_command()
    
    print("\nNOTES:")
    print("- Supported formats: JPEG, PNG, GIF, WEBP")
    print("- Max images per request: 10 (configurable)")
    print("- Max image size: 1024px (auto-resized)")
    print("- Both URL and base64 images are supported")
    print("- Use 'image_url' type for OpenAI compatibility")
    print("- Legacy 'image' type is still supported")