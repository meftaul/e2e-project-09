
As an end-user / consumer, you can run this project either manually on your local machine or by using Docker. Follow the steps below based on your preferred method.

# Manual Steps:
01. OS
02. Python
04. Install Dependencies
   ```bash
    pip install -r requirements.txt
    ```
04. Run the application
```bash
streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0
```

# Docker Steps:
01. OS
02. Install Docker
03. Download docker image
   ```bash
   docker pull your-docker-image-name
   ```
04. Run the Docker container
   ```bash
   docker run -d -p 8501:8501 your-docker-image-name
   ```