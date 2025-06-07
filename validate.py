import requests
import json
import sys
import time

# ================================
# Configuration
# ================================
API_BASE_URL = "http://localhost:8000"

# A list of word pairs to test, with a description of the expected relationship.
# This list covers various semantic relationships to test the model's nuances.
TEST_CASES = [
    # --- High Similarity ---
    ("car", "vehicle", "High (Synonym-like)"),
    ("king", "queen", "High (Strongly related concepts)"),
    ("dog", "puppy", "High (Direct relationship)"),

    # --- Medium Similarity ---
    ("doctor", "hospital", "Medium (Person and Place)"),
    ("book", "library", "Medium (Object and Place)"),
    ("computer", "keyboard", "Medium (System and Component)"),
    ("japan", "tokyo", "Medium (Country and Capital)"),

    # --- Low Similarity ---
    ("tree", "sky", "Low (Co-occur in scenes, but unrelated)"),
    ("sun", "water", "Low (General nature terms)"),
    ("japan", "game", "Low (User's test case, abstract link)"),

    # --- Very Low / No Similarity ---
    ("music", "algorithm", "Very Low (Unrelated abstract concepts)"),
    ("car", "philosophy", "Very Low (Unrelated concrete and abstract)"),
    ("moon", "cheese", "Very Low (Classic unrelated pair)"),

    # --- Invalid Word Test ---
    # 'brillig' is a nonsense word from the poem "Jabberwocky" and should not be in the model.
    ("house", "brillig", "Invalid Word (Should fail validation)")
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

        print(f"  ✅ SUCCESS | Score: {data.get('similarity_score', 'N/A')}, Valid Guess: {data.get('valid_guess', 'N/A')}")
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
        # Add a small delay to avoid spamming the server
        time.sleep(0.1)


def main():
    """
    Main function to run all test cases and print a summary.
    """
    print("="*60)
    print("  Running Word Similarity API Validation Suite")
    print("="*60)
    
    success_count = 0
    failure_count = 0

    for word1, word2, description in TEST_CASES:
        if run_test(word1, word2, description):
            success_count += 1
        else:
            failure_count += 1
        print("-" * 60)

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
        print("\nAll tests passed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()