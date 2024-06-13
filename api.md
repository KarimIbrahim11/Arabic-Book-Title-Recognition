# API Documentation

## Introduction
This is the API documentation for [BOOK-OCR] and how to use it.


## Error Codes

- 200: Image received and processed successfully 
- 400: Bad Request
- 404: Image Not Found
- 500: Internal Server Error

## Endpoints

### 1. Get Book Title from Image

**URL**: `http://127.0.0.1:5000/predict:5000`
`

**Method**: `POST`

**Requests**:

```form-data
{
  "Key": "file"
  "Value": Choose file -> upload file as <FileStorage (img/{png,jpg,jpg})
}
```
**Response**:
```json
{
    "bounding_boxes": [
        [
            387,
            833,
            320,
            410
        ]
    ],
    "title": "دورالقرآن الكريم"
}
```
