FROM spectreteam/python_msi:v5.0.0

COPY requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

RUN mkdir -p /root/.config/matplotlib &&\
  echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc

COPY . /app

RUN python -m unittest discover

RUN python setup.py install

EXPOSE 8050

VOLUME /data

WORKDIR /data

RUN rm -rf /app
