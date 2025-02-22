import requests
import json

# Payload
json_payload = json.dumps({
    "invoice_id": [4919, 10423],
    "country": "CL"
})



def predict(payload: str):
    try:
        # URL
        base_url = "https://api-demo-cicd-807455101353.us-central1.run.app"  # Replace URL
        url = f"{base_url}/invoice_risk"
        # Header
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.post(url, headers=headers, data=payload, timeout=10)

        # Check for successful response (200 OK)
        if response.status_code == 200:
            print("Predicts:")
            print(f"Response: {response.text}") # Print the response body if needed
            return True
        else:
            print(f"API returned an unexpected status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return False


# Generate predictions
predict(json_payload)
