from src.inference import MediVisionPipeline
import os

def test_pipeline():
    print("Initializing Pipeline...")
    pipeline = MediVisionPipeline()
    print("Pipeline Initialized.")
    
    image_path = None # Testing without image first
    symptoms = "I have a red rash on my arm that is itchy."
    
    print(f"Testing with symptoms: {symptoms}")
    result = pipeline.process(image_path, symptoms, lang="en")
    print(f"Result (EN): {result}")
    
    result_vn = pipeline.process(image_path, symptoms, lang="vn")
    print(f"Result (VN): {result_vn}")

if __name__ == "__main__":
    test_pipeline()
