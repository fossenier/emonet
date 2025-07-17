import requests
import base64
import json
import sys
from pathlib import Path
from typing import Optional


class EmotionAPITester:
    """Test client for the Emotion Detection API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def test_health(self) -> dict:
        """Test the health endpoint."""
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Health check failed: {e}")
            return None

    def detect_emotion_from_file(self, image_path: str) -> Optional[dict]:
        """
        Test emotion detection using file upload.

        Args:
            image_path: Path to the image file

        Returns:
            API response or None if failed
        """
        image_path = Path(image_path)

        if not image_path.exists():
            print(f"Error: Image file not found: {image_path}")
            return None

        if not image_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            print(f"Warning: File extension {image_path.suffix} might not be supported")

        try:
            with open(image_path, "rb") as image_file:
                files = {"file": (image_path.name, image_file, "image/jpeg")}
                response = requests.post(f"{self.base_url}/detect_emotion", files=files)
                response.raise_for_status()
                return response.json()

        except requests.exceptions.RequestException as e:
            print(f"File upload request failed: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response text: {e.response.text}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    def detect_emotion_from_base64(self, image_path: str) -> Optional[dict]:
        """
        Test emotion detection using base64 encoding.

        Args:
            image_path: Path to the image file

        Returns:
            API response or None if failed
        """
        image_path = Path(image_path)

        if not image_path.exists():
            print(f"Error: Image file not found: {image_path}")
            return None

        try:
            # Read and encode image
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                base64_string = base64.b64encode(image_data).decode("utf-8")

            # Send request
            payload = {"image": base64_string}
            headers = {"Content-Type": "application/json"}

            response = requests.post(
                f"{self.base_url}/detect_emotion_base64", json=payload, headers=headers
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Base64 request failed: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response text: {e.response.text}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    def print_result(self, result: dict, method: str):
        """Pretty print the API response."""
        if not result:
            print(f"❌ {method} method failed")
            return

        print(f"\n✅ {method} method successful!")
        print("=" * 50)

        if result.get("face_detected"):
            print(f"😊 Primary Emotion: {result.get('emotion', 'Unknown')}")
            print(
                f"😌 Valence: {result.get('valence', 'N/A'):.3f}"
                if result.get("valence") is not None
                else "😌 Valence: N/A"
            )
            print(
                f"⚡ Arousal: {result.get('arousal', 'N/A'):.3f}"
                if result.get("arousal") is not None
                else "⚡ Arousal: N/A"
            )

            if result.get("face_bbox"):
                bbox = result["face_bbox"]
                print(
                    f"📦 Face Bounding Box: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
                )

            if result.get("emotion_probabilities"):
                print("\n🎭 Emotion Probabilities:")
                for emotion, prob in sorted(
                    result["emotion_probabilities"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                ):
                    print(f"  {emotion}: {prob:.3f}")
        else:
            print("❌ No face detected in the image")

        if result.get("message"):
            print(f"\n💬 Message: {result['message']}")


def main():
    """Main function to run the tests."""
    # Default image path - update this to your image file
    default_image_path = "/Users/admin/projects/monorepo/ise-emonet/captures/capture-2025-06-04T15-36-55-737Z.png"

    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = default_image_path
        print(f"No image path provided. Using default: {image_path}")
        print("Usage: python test_emotion_api.py <path_to_image>")

    # Initialize tester
    tester = EmotionAPITester()

    # Test health endpoint
    print("🔍 Testing API health...")
    health_result = tester.test_health()
    if health_result:
        print("✅ API is healthy!")
        print(f"Device: {health_result.get('device')}")
        print(f"Model loaded: {health_result.get('model_loaded')}")
        print(f"Number of classes: {health_result.get('n_classes')}")
    else:
        print(
            "❌ API health check failed. Make sure the server is running on localhost:8000"
        )
        return

    # Test both endpoints
    print(f"\n🖼️  Testing with image: {image_path}")

    # Test file upload method
    print("\n📤 Testing file upload method...")
    file_result = tester.detect_emotion_from_file(image_path)
    tester.print_result(file_result, "File upload")

    # Test base64 method
    print("\n📝 Testing base64 method...")
    base64_result = tester.detect_emotion_from_base64(image_path)
    tester.print_result(base64_result, "Base64")

    # Compare results
    if file_result and base64_result:
        if file_result.get("emotion") == base64_result.get(
            "emotion"
        ) and file_result.get("face_detected") == base64_result.get("face_detected"):
            print("\n✅ Both methods returned consistent results!")
        else:
            print("\n⚠️  Methods returned different results")


if __name__ == "__main__":
    main()
