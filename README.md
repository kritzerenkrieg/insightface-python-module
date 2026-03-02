# Face Recognition API

This project provides an API for image-based registration and recognition using the InsightFace library. It allows users to register images with associated names and recognize images against a database of registered images.

## Project Structure

```
face-recog-api
├── src
│   ├── api
│   │   ├── main.py          # Entry point for the FastAPI application
│   │   ├── routers
│   │   │   ├── register.py  # API endpoint for registering images
│   │   │   └── recognize.py  # API endpoint for recognizing images
│   │   └── schemas
│   │       └── index.py     # Pydantic models for request and response schemas
│   ├── face_recog
│   │   ├── __init__.py      # Package initialization
│   │   ├── register.py      # Logic for registering images
│   │   ├── recognize.py     # Logic for recognizing images
│   │   └── cli.py           # Command-line interface for image operations
├── tests
│   └── test_api.py          # Unit tests for API endpoints
├── Dockerfile                # Instructions to build the Docker image
├── docker-compose.yml        # Service orchestration for the application
├── .dockerignore             # Files to ignore when building the Docker image
├── requirements.txt          # Python dependencies for the project
└── README.md                 # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd face-recog-api
   ```

2. **Build the Docker image:**
   ```
   docker build -t face-recog-api .
   ```

3. **Run the application using Docker Compose:**
   ```
   docker-compose up
   ```

4. **Access the API:**
   The API will be available at `http://localhost:8000`. You can use tools like Postman or curl to interact with the endpoints.

## API Endpoints

### Register an Image

- **Endpoint:** `POST /register`
- **Request Body:**
  ```json
  {
    "image": "base64_encoded_image_string",
    "name": "person_name"
  }
  ```
- **Response:**
  - Success: `{"message": "Registered successfully", "id": "unique_id"}`
  - Error: `{"error": "Error message"}`

### Recognize an Image

- **Endpoint:** `POST /recognize`
- **Request Body:**
  ```json
  {
    "image": "base64_encoded_image_string",
    "topk": 1,
    "threshold": 0.5
  }
  ```
- **Response:**
  - Success: `{"matches": [{"name": "person_name", "id": "unique_id", "score": score}]}` or `{"matches": []}`
  - Error: `{"error": "Error message"}`

## Testing

To run the tests, use the following command:
```
pytest tests/test_api.py
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.