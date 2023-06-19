# Include the base image for the docker
# You can use an image based on PyTorch, Tensorflow, MXNet etc, depending on your prefered machine learning tool.
FROM python:3.7
#FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime 
# Setting working directory to /opt, rather than doing all the work in root.
# Copying the /code directory into /opt
WORKDIR /opt
COPY ./src /opt

# Running pip install to download required packages
#RUN pip install -r requirements.txt

COPY requirements.txt ./
RUN pip install -i  https://pypi.tuna.tsinghua.edu.cn/simple --default-timeout=100 --no-cache-dir -r requirements.txt
#RUN pip install-i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python


# Setting the default code to run when a container is launced with this image.
ENTRYPOINT [ "/bin/bash", "/opt/run.sh" ]
