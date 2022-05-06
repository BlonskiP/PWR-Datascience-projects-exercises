FROM jupyter/scipy-notebook

RUN pip install tqdm more-itertools==8.0.2
RUN pip install geopandas
RUN pip install country_converter --upgrade

ENV WORK_DIR ${HOME}/work
USER jovyan

