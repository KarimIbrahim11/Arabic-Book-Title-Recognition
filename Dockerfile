# Use an official Python runtime as a parent image
FROM ubuntu:22.04

# # Install dependencies
RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx python3-pip\
    libglib2.0-0 -y \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /book-ocr

# Copy the current directory contents into the container at /app
COPY . /book-ocr

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Define environment variable
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

#Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["flask", "run"]
