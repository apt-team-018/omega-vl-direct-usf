"""
OpenAI-Compatible Streaming Examples for Omega VLM API

This file demonstrates how to use the streaming feature with text and images,
compatible with OpenAI's streaming format.
"""

import requests
import json
import base64
import sseclient  # pip install sseclient-py


# API Configuration
API_URL = "http://localhost:8000/v1/chat/completions"
API_KEY = "EMPTY"  # Set to your API key if authentication is enabled

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


def example_1_text_streaming():
    """Example 1: Basic text streaming"""
    print("=" * 80)
    print("Example 1: Basic Text Streaming")
    print("=" * 80)
    
    payload = {
        "model": "model",
        "messages": [
            {
                "role": "user",
                "content": "Write a short poem about AI"
            }
        ],
        "stream": True,
        "temperature": 0.8,
        "max_tokens": 256
    }
    
    response = requests.post(API_URL, json=payload, headers=headers, stream=True)
    
    print("\nStreaming response:")
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = line[6:]  # Remove 'data: ' prefix
                if data == '[DONE]':
                    print("\n[Stream completed]")
                    break
                try:
                    chunk = json.loads(data)
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        delta = chunk['choices'][0].get('delta', {})
                        if 'content' in delta:
                            print(delta['content'], end='', flush=True)
                except json.JSONDecodeError:
                    pass
    print("\n")


def example_2_image_streaming():
    """Example 2: Streaming with image input"""
    print("=" * 80)
    print("Example 2: Streaming with Image")
    print("=" * 80)
    
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
                            "url": "https://example.com/image.jpg"
                        }
                    }
                ]
            }
        ],
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 512
    }
    
    response = requests.post(API_URL, json=payload, headers=headers, stream=True)
    
    print("\nStreaming response:")
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = line[6:]
                if data == '[DONE]':
                    print("\n[Stream completed]")
                    break
                try:
                    chunk = json.loads(data)
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        delta = chunk['choices'][0].get('delta', {})
                        if 'content' in delta:
                            print(delta['content'], end='', flush=True)
                        # Print usage info if present
                        if 'usage' in chunk:
                            print(f"\n\nUsage: {chunk['usage']}")
                except json.JSONDecodeError:
                    pass
    print("\n")


def example_3_sseclient_library():
    """Example 3: Using sseclient library for cleaner parsing"""
    print("=" * 80)
    print("Example 3: Using sseclient Library")
    print("=" * 80)
    
    payload = {
        "model": "model",
        "messages": [
            {
                "role": "user",
                "content": "Explain quantum computing in simple terms"
            }
        ],
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 300
    }
    
    response = requests.post(API_URL, json=payload, headers=headers, stream=True)
    client = sseclient.SSEClient(response)
    
    print("\nStreaming response:")
    for event in client.events():
        if event.data == '[DONE]':
            print("\n[Stream completed]")
            break
        try:
            chunk = json.loads(event.data)
            if 'choices' in chunk and len(chunk['choices']) > 0:
                delta = chunk['choices'][0].get('delta', {})
                if 'content' in delta:
                    print(delta['content'], end='', flush=True)
                # Check for completion
                finish_reason = chunk['choices'][0].get('finish_reason')
                if finish_reason:
                    print(f"\n[Finished: {finish_reason}]")
                # Print usage if present
                if 'usage' in chunk:
                    usage = chunk['usage']
                    print(f"\nTokens - Prompt: {usage['prompt_tokens']}, "
                          f"Completion: {usage['completion_tokens']}, "
                          f"Total: {usage['total_tokens']}")
        except json.JSONDecodeError:
            pass
    print("\n")


def example_4_base64_image_streaming():
    """Example 4: Streaming with base64 image"""
    print("=" * 80)
    print("Example 4: Streaming with Base64 Image")
    print("=" * 80)
    
    # Read and encode an image (replace with actual image path)
    try:
        with open("example_image.jpg", "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print("Note: example_image.jpg not found, using placeholder")
        base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
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
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 400
    }
    
    response = requests.post(API_URL, json=payload, headers=headers, stream=True)
    
    print("\nStreaming response:")
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = line[6:]
                if data == '[DONE]':
                    print("\n[Stream completed]")
                    break
                try:
                    chunk = json.loads(data)
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        delta = chunk['choices'][0].get('delta', {})
                        if 'content' in delta:
                            print(delta['content'], end='', flush=True)
                except json.JSONDecodeError:
                    pass
    print("\n")


def example_5_conversation_streaming():
    """Example 5: Multi-turn conversation with streaming"""
    print("=" * 80)
    print("Example 5: Multi-turn Conversation Streaming")
    print("=" * 80)
    
    payload = {
        "model": "model",
        "messages": [
            {
                "role": "user",
                "content": "What are the three laws of robotics?"
            },
            {
                "role": "assistant",
                "content": "The Three Laws of Robotics by Isaac Asimov are:\n1. A robot may not injure a human being or allow one to come to harm\n2. A robot must obey human orders unless it conflicts with the First Law\n3. A robot must protect its own existence unless it conflicts with the First or Second Law"
            },
            {
                "role": "user",
                "content": "Can you explain the first law in more detail?"
            }
        ],
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 300
    }
    
    response = requests.post(API_URL, json=payload, headers=headers, stream=True)
    
    print("\nAssistant (streaming):")
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = line[6:]
                if data == '[DONE]':
                    print("\n[Stream completed]")
                    break
                try:
                    chunk = json.loads(data)
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        delta = chunk['choices'][0].get('delta', {})
                        if 'content' in delta:
                            print(delta['content'], end='', flush=True)
                except json.JSONDecodeError:
                    pass
    print("\n")


def example_6_curl_command():
    """Example 6: cURL command for streaming"""
    curl_command = """
# Basic streaming request
curl -X POST http://localhost:8000/v1/chat/completions \\
  -H "Authorization: Bearer EMPTY" \\
  -H "Content-Type: application/json" \\
  -N \\
  -d '{
    "model": "model",
    "messages": [
      {
        "role": "user",
        "content": "Tell me a short story"
      }
    ],
    "stream": true,
    "temperature": 0.8,
    "max_tokens": 256
  }'

# Streaming with image
curl -X POST http://localhost:8000/v1/chat/completions \\
  -H "Authorization: Bearer EMPTY" \\
  -H "Content-Type: application/json" \\
  -N \\
  -d '{
    "model": "model",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe this image"
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
    "stream": true,
    "temperature": 0.7,
    "max_tokens": 512
  }'
    """
    print("=" * 80)
    print("Example 6: cURL Commands for Streaming")
    print("=" * 80)
    print(curl_command)
    print()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("OMEGA VLM API - Streaming Examples")
    print("OpenAI-Compatible Streaming Format")
    print("=" * 80 + "\n")
    
    print("Available examples:")
    print("1. Basic text streaming")
    print("2. Streaming with image input")
    print("3. Using sseclient library")
    print("4. Streaming with base64 image")
    print("5. Multi-turn conversation streaming")
    print("6. cURL commands\n")
    
    # Run examples (uncomment to execute)
    # example_1_text_streaming()
    # example_2_image_streaming()
    # example_3_sseclient_library()
    # example_4_base64_image_streaming()
    # example_5_conversation_streaming()
    example_6_curl_command()
    
    print("\nNOTES:")
    print("- Streaming uses Server-Sent Events (SSE) format")
    print("- Each chunk contains 'delta' with incremental content")
    print("- Final chunk includes 'finish_reason' and 'usage' statistics")
    print("- Stream ends with 'data: [DONE]'")
    print("- Compatible with OpenAI's streaming format")
    print("- Supports both text-only and multimodal (text + images)")
    print("\nDependencies:")
    print("- requests: pip install requests")
    print("- sseclient-py: pip install sseclient-py (optional, for cleaner parsing)")