FROM python:3.9-slim


# Copy and install the installlation files that don't change often
COPY ./requirements-apis.txt  /app/requirements-apis.txt

RUN apt-get update && \
    apt-get install -y build-essential && \
    pip install --no-cache-dir -r /app/requirements-apis.txt && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache/pip/ && \
    rm -rf /app/requirements-apis.txt &&\
    apt-get remove --purge -y  build-essential && \
    apt-get autoremove -y 

# Copy and install self as a package
COPY ./rnd /app/rnd
COPY ./setup.cfg /app/setup.cfg
COPY ./setup.py /app/setup.py
COPY ./gcp_token.json /app/gcp_token.json
COPY ./gcp_credentials.json /app/gcp_credentials.json
WORKDIR /app
RUN pip install --no-cache-dir .

# When the container launches
EXPOSE 80

# Single worker
CMD ["python", "rnd/apis/manage.py", "run-api-server"]