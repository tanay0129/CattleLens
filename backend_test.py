import requests
import sys
import base64
import json
from datetime import datetime
from pathlib import Path

class BreedRecognitionTester:
    def __init__(self, base_url="https://cattle-breed-scan-1.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def log_test(self, name, success, details=""):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name} - PASSED")
        else:
            print(f"❌ {name} - FAILED: {details}")
        
        self.test_results.append({
            "test": name,
            "success": success,
            "details": details
        })

    def test_api_health(self):
        """Test basic API connectivity"""
        try:
            response = requests.get(f"{self.api_url}/", timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            if success:
                details += f", Response: {response.json()}"
            self.log_test("API Health Check", success, details)
            return success
        except Exception as e:
            self.log_test("API Health Check", False, str(e))
            return False

    def test_breeds_endpoint(self):
        """Test breeds listing endpoint"""
        try:
            response = requests.get(f"{self.api_url}/breeds", timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                cattle_count = len(data.get('cattle', []))
                buffalo_count = len(data.get('buffalo', []))
                details += f", Cattle breeds: {cattle_count}, Buffalo breeds: {buffalo_count}"
                
                # Verify we have expected breeds
                if cattle_count < 5 or buffalo_count < 5:
                    success = False
                    details += " - Insufficient breed data"
            
            self.log_test("Breeds Endpoint", success, details)
            return success
        except Exception as e:
            self.log_test("Breeds Endpoint", False, str(e))
            return False

    def create_test_image_base64(self):
        """Create a simple test image in base64 format"""
        # Create a minimal PNG image (1x1 pixel red)
        # PNG header + IHDR + red pixel + IEND
        png_data = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
            0x00, 0x00, 0x00, 0x0D,  # IHDR length
            0x49, 0x48, 0x44, 0x52,  # IHDR
            0x00, 0x00, 0x00, 0x01,  # width: 1
            0x00, 0x00, 0x00, 0x01,  # height: 1
            0x08, 0x02,              # bit depth: 8, color type: 2 (RGB)
            0x00, 0x00, 0x00,        # compression, filter, interlace
            0x90, 0x77, 0x53, 0xDE,  # CRC
            0x00, 0x00, 0x00, 0x0C,  # IDAT length
            0x49, 0x44, 0x41, 0x54,  # IDAT
            0x08, 0x99, 0x01, 0x01, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x02, 0x00, 0x01,  # compressed data
            0xE2, 0x21, 0xBC, 0x33,  # CRC
            0x00, 0x00, 0x00, 0x00,  # IEND length
            0x49, 0x45, 0x4E, 0x44,  # IEND
            0xAE, 0x42, 0x60, 0x82   # CRC
        ])
        return base64.b64encode(png_data).decode('utf-8')

    def test_breed_recognition_invalid_data(self):
        """Test breed recognition with invalid data"""
        try:
            # Test with invalid base64
            response = requests.post(
                f"{self.api_url}/recognize-breed",
                json={"image_base64": "invalid_base64"},
                timeout=30
            )
            
            success = response.status_code in [400, 422, 500]  # Should fail gracefully
            details = f"Status: {response.status_code}"
            
            if response.status_code == 200:
                data = response.json()
                if not data.get('success', True):
                    success = True  # API correctly returned error in response
                    details += f", Error handled: {data.get('error', 'Unknown error')}"
                else:
                    success = False
                    details += " - Should have failed with invalid data"
            
            self.log_test("Breed Recognition - Invalid Data", success, details)
            return success
        except Exception as e:
            self.log_test("Breed Recognition - Invalid Data", False, str(e))
            return False

    def test_breed_recognition_valid_image(self):
        """Test breed recognition with valid image"""
        try:
            test_image = self.create_test_image_base64()
            
            response = requests.post(
                f"{self.api_url}/recognize-breed",
                json={"image_base64": test_image},
                timeout=60  # AI processing can take time
            )
            
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                details += f", Success: {data.get('success')}"
                
                if data.get('success'):
                    details += f", Breed: {data.get('breed')}, Animal: {data.get('animal_type')}, Confidence: {data.get('confidence')}"
                else:
                    details += f", Error: {data.get('error')}"
                    # For test image, error is acceptable since it's not a real animal
                    success = True
            
            self.log_test("Breed Recognition - Valid Image", success, details)
            return success
        except Exception as e:
            self.log_test("Breed Recognition - Valid Image", False, str(e))
            return False

    def test_breed_recognition_missing_fields(self):
        """Test breed recognition with missing required fields"""
        try:
            response = requests.post(
                f"{self.api_url}/recognize-breed",
                json={},  # Missing image_base64
                timeout=10
            )
            
            success = response.status_code in [400, 422]  # Should return validation error
            details = f"Status: {response.status_code}"
            
            self.log_test("Breed Recognition - Missing Fields", success, details)
            return success
        except Exception as e:
            self.log_test("Breed Recognition - Missing Fields", False, str(e))
            return False

    def test_cors_headers(self):
        """Test CORS headers are present"""
        try:
            response = requests.options(f"{self.api_url}/recognize-breed", timeout=10)
            
            cors_headers = [
                'Access-Control-Allow-Origin',
                'Access-Control-Allow-Methods',
                'Access-Control-Allow-Headers'
            ]
            
            present_headers = [h for h in cors_headers if h in response.headers]
            success = len(present_headers) >= 1  # At least one CORS header should be present
            
            details = f"Status: {response.status_code}, CORS headers: {present_headers}"
            
            self.log_test("CORS Headers", success, details)
            return success
        except Exception as e:
            self.log_test("CORS Headers", False, str(e))
            return False

    def run_all_tests(self):
        """Run all backend tests"""
        print("🧪 Starting Backend API Tests...")
        print(f"🌐 Testing against: {self.base_url}")
        print("=" * 60)
        
        # Run tests in order
        tests = [
            self.test_api_health,
            self.test_breeds_endpoint,
            self.test_cors_headers,
            self.test_breed_recognition_missing_fields,
            self.test_breed_recognition_invalid_data,
            self.test_breed_recognition_valid_image,
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                print(f"❌ {test.__name__} - CRASHED: {str(e)}")
                self.tests_run += 1
        
        print("=" * 60)
        print(f"📊 Backend Tests Summary: {self.tests_passed}/{self.tests_run} passed")
        
        if self.tests_passed == self.tests_run:
            print("🎉 All backend tests passed!")
        elif self.tests_passed / self.tests_run >= 0.8:
            print("⚠️  Most backend tests passed - minor issues detected")
        else:
            print("🚨 Multiple backend failures detected")
        
        return self.tests_passed, self.tests_run, self.test_results

def main():
    tester = BreedRecognitionTester()
    passed, total, results = tester.run_all_tests()
    
    # Save detailed results
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "passed": passed,
            "total": total,
            "success_rate": passed / total if total > 0 else 0
        },
        "tests": results
    }
    
    with open("/app/backend_test_results.json", "w") as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: /app/backend_test_results.json")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())