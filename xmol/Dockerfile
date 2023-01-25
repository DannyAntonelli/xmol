FROM python:3.9

WORKDIR /code

COPY requirements.txt requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip3 install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
RUN pip3 install dive-into-graphs==0.2.0

EXPOSE 8000

COPY . .

CMD ["gunicorn", "xmol.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "3"]
