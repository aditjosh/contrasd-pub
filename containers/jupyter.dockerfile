# Use a base Jupyter Notebook image
## See https://hub.docker.com/layers/jupyter/base-notebook/x86_64-ubuntu-22.04/images/sha256-2d799fdfda8926ccec5c0791a8f01b456592e01b1e6bbf88fdb732888ae78b49
FROM jupyter/base-notebook:x86_64-ubuntu-22.04

USER root

# Install Java
RUN apt-get update && \
    apt-get install -y openjdk-17-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME
ENV JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64" 
ENV PATH="$PATH:$JAVA_HOME/bin"

# Python
# You'll need to replace these paths with your actual host paths
ARG CUSTOM_PYTHONHOME
ENV PATH="${CUSTOM_PYTHONHOME}/bin:${PATH}"

COPY containers/requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

EXPOSE 8888

# Set the working directory inside the container
WORKDIR /home_dir

# Run Jupyter Notebook when the container launches
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

