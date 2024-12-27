# Use an official Python runtime as a parent image
FROM python:3.11.8-bullseye

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
#CMD ["python", "app.py"]

ENV FLASK_APP=app.py
CMD ["flask", "run", "--host", "0.0.0.0"]

#ToDO:  FIX Warning:
#WARNING: This is a development server. 
#Do not use it in a production deployment. Use a production WSGI server instead.
