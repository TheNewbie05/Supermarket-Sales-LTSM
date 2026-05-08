import httpx

def test_forecast_endpoint():
    with httpx.Client(base_url="http://localhost:8000") as client:
        response = client.get("/forecast?days=5")
        assert response.status_code == 200
        data = response.json()
        assert len(data["forecast"]) == 5
        print("✅ API Test Passed!")

if __name__ == "__main__":
    test_forecast_endpoint()