import requests
import json
import sys
import time

# ================================
# Configuration
# ================================
API_BASE_URL = "http://localhost:8000"

# Expanded list of test cases to cover more semantic relationships
TEST_CASES = [
    # --- Exact Matches ---
    ("happy", "happy", "Exact Match"),
    ("car", "car", "Exact Match"),
    ("rich", "rich", "Exact Match"),
    ("student", "student", "Exact Match"),
    ("dog", "dog", "Exact Match"),
    
    # --- High Similarity (Synonyms & Direct Relationships) ---
    ("happy", "joyful", "High (Synonym)"),
    ("car", "vehicle", "High (Synonym)"),
    ("rich", "wealthy", "High (Synonym)"),
    ("student", "pupil", "High (Synonym)"),
    ("dog", "puppy", "High (Direct relationship)"),
    ("king", "queen", "High (Strongly related concepts)"),
    ("idea", "thought", "High (Abstract synonym)"),

    # --- Medium Similarity (Contextual & Functional Relationships) ---
    ("doctor", "hospital", "Medium (Person and Place)"),
    ("book", "library", "Medium (Object and Place)"),
    ("computer", "keyboard", "Medium (System and Component)"),
    ("japan", "tokyo", "Medium (Country and Capital)"),
    ("student", "exam", "Medium (Person and Event)"),
    ("drive", "road", "Medium (Action and Location)"),

    # --- Part/Whole Relationships (Meronyms/Holonyms) ---
    ("finger", "hand", "Part/Whole"),
    ("wheel", "bicycle", "Part/Whole"),
    ("kitchen", "house", "Part/Whole"),

    # --- Category/Instance Relationships (Hypernyms/Hyponyms) ---
    ("animal", "lion", "Category/Instance"),
    ("color", "blue", "Category/Instance"),
    ("emotion", "happiness", "Category/Instance"),
    
    # --- Antonyms (Opposites) - Expect low-to-medium scores ---
    # Note: Antonyms often appear in similar contexts, so they are not 0.
    ("hot", "cold", "Antonym"),
    ("happy", "sad", "Antonym"),
    ("love", "hate", "Antonym"),
    ("light", "dark", "Antonym"),

    # --- Low / Distant Similarity ---
    ("tree", "sky", "Low (Co-occur in scenes)"),
    ("sun", "water", "Low (General nature terms)"),
    ("king", "castle", "Low (Related but distinct)"),
    ("japan", "game", "Low (User's test case)"),

    # --- Very Low / No Similarity ---
    ("music", "algorithm", "Very Low (Unrelated abstract)"),
    ("car", "philosophy", "Very Low (Unrelated concrete/abstract)"),
    ("moon", "cheese", "Very Low (Classic unrelated pair)"),
    ("science", "religion", "Very Low (Opposing domains)"),

    # --- Invalid Input Tests ---
    ("house", "brillig", "Invalid Word (brillig)"),
    ("asdfgh", "zxcvbn", "Invalid Word (nonsense)"),
    ("hi", "there", "Invalid Length"),
    ("car", "go", "Invalid Length")
]

def run_test(word1: str, word2: str, description: str):
    """
    Calls the /similarity endpoint for a single word pair and prints the result.
    Returns True on success, False on failure.
    """
    endpoint_url = f"{API_BASE_URL}/similarity"
    params = {"word1": word1, "word2": word2}
    
    print(f"▶️  Testing: '{word1}' vs '{word2}'  ({description})")

    try:
        response = requests.get(endpoint_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        required_keys = ["similarity", "progress_score", "isValidGuess", "reason"]
        if not all(key in data for key in required_keys):
            print(f"  ❌ FAILED: Response is missing required keys.")
            print(f"     Required: {required_keys}, Received: {list(data.keys())}")
            return False

        sim = data['similarity']
        progress = data['progress_score']
        is_valid = data['isValidGuess']
        reason = data['reason']
        
        sim_formatted = f"{sim:.4f}" if sim != -1 else "-1"
        
        print(f"  ✅ SUCCESS | Raw Sim: {sim_formatted} | Progress: {progress}/100 | Valid: {is_valid} | Reason: {reason}")
        return True

    except requests.exceptions.ConnectionError:
        print(f"  ❌ FAILED: Connection Error. Is the server running at {API_BASE_URL}?")
        return False

    except requests.exceptions.HTTPError as e:
        print(f"  ❌ FAILED: HTTP Error. Status Code: {e.response.status_code}")
        try:
            print(f"     Server response: {json.dumps(e.response.json(), indent=4)}")
        except json.JSONDecodeError:
            print(f"     Server response: {e.response.text}")
        return False

    except requests.exceptions.RequestException as e:
        print(f"  ❌ FAILED: An unexpected request error occurred: {e}")
        return False
    finally:
        time.sleep(0.1)


def main():
    """
    Main function to run all test cases and print a summary.
    """
    print("="*80)
    print("  Running Final Word Similarity API Validation Suite (Expanded)")
    print("="*80)
    
    success_count, failure_count = 0, 0

    for word1, word2, description in TEST_CASES:
        if run_test(word1, word2, description):
            success_count += 1
        else:
            failure_count += 1
        print("-" * 80)

    print("🏁 Validation Complete 🏁\n")
    print("--- Summary ---")
    print(f"  Total Tests: {len(TEST_CASES)}")
    print(f"  ✅ Successes: {success_count}")
    print(f"  ❌ Failures:  {failure_count}")
    print("---------------")

    if failure_count > 0:
        print("\nSome tests failed. Please review the log.")
        sys.exit(1)
    else:
        print("\nAll tests passed successfully! The API is ready.")
        sys.exit(0)

if __name__ == "__main__":
    main()