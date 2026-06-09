FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# Install any python packages you need
COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install jupyter

RUN apt-get update && apt-get install -y git graphviz build-essential && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install git+https://github.com/ContinualAI/avalanche.git

CMD [ "jupyter", "notebook", "--ip", "0.0.0.0", "--port", "8889", \
      "--no-browser", "--allow-root", \
      "--NotebookApp.token=''","--NotebookApp.password=''" ]