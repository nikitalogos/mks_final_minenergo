FROM ubuntu

ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update -y
RUN apt-get install -y python3-pip python3.7 build-essential libgl1-mesa-glx ffmpeg libsm6 libxext6 

COPY . .

WORKDIR .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt && pip3 install -r /yolov5/requirements.txt


EXPOSE 8501
CMD streamlit run main.py

