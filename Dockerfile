FROM jayaneetha/images:tf2.1.0-gpu-py3.6.8-base

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py ./

RUN sudo chown user *.py


#CMD ["python rl_run.py --data-version=esd --policy=ZetaPolicy --pre-train-dataset=esd --pre-train=true --env-name=Zeta2.0 --disable-wandb=True"]