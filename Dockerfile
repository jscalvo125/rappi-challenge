# Use Python38
FROM python:3.8
# Copy requirements.txt to the docker image and install packages
COPY requirements.txt /
RUN pip install -r requirements.txt
# Set the WORKDIR to be the folder
COPY . .
# Expose port 80
EXPOSE 80
ENV PORT 80
WORKDIR /
# Use gunicorn as the entrypoint
CMD ["python", "main.py"]