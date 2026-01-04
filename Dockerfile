FROM apache/airflow:2.8.0

# Copy requirements and install
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Set user back to airflow
USER airflow